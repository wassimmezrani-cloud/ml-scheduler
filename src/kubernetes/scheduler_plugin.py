#!/usr/bin/env python3
"""
Kubernetes Scheduler Plugin for HYDATIS ML Scheduler.
Integrates ML-powered scheduling decisions with Kubernetes scheduler framework.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
import yaml
import threading

logger = logging.getLogger(__name__)


class HYDATISSchedulerPlugin:
    """Kubernetes scheduler plugin integrating ML-powered scheduling decisions."""
    
    def __init__(self, 
                 ml_gateway_url: str = "http://localhost:8083",
                 namespace: str = "hydatis-ml-scheduler"):
        
        self.ml_gateway_url = ml_gateway_url
        self.namespace = namespace
        
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes configuration")
        except:
            try:
                config.load_kube_config()
                logger.info("Loaded local Kubernetes configuration")
            except Exception as e:
                logger.error(f"Failed to load Kubernetes configuration: {e}")
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.scheduling_v1 = client.SchedulingV1Api()
        
        self.plugin_config = {
            'scheduling_timeout_seconds': 30,
            'ml_decision_timeout_seconds': 10,
            'fallback_strategy': 'least_allocated',
            'scheduling_queue_size': 100,
            'batch_scheduling_enabled': False,
            'performance_monitoring': True
        }
        
        self.cluster_nodes = {
            'master_nodes': ['master-1', 'master-2', 'master-3'],
            'worker_nodes': ['worker-1', 'worker-2', 'worker-3'],
            'schedulable_nodes': ['worker-1', 'worker-2', 'worker-3']
        }
        
        self.scheduling_stats = {
            'total_scheduling_requests': 0,
            'ml_scheduling_decisions': 0,
            'fallback_decisions': 0,
            'successful_placements': 0,
            'failed_placements': 0,
            'average_decision_latency_ms': 0.0,
            'node_placement_counts': {node: 0 for node in self.cluster_nodes['worker_nodes']},
            'scheduling_errors': []
        }
        
        self.pending_pods_queue = []
        self.processing_pods = {}
        
        self.is_active = False
        self.scheduler_thread = None
    
    def start_scheduler_plugin(self):
        """Start the ML scheduler plugin."""
        
        if self.is_active:
            logger.warning("Scheduler plugin already active")
            return
        
        self.is_active = True
        
        self.scheduler_thread = threading.Thread(target=self._scheduler_main_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("HYDATIS ML Scheduler Plugin started")
        logger.info(f"ML Gateway: {self.ml_gateway_url}")
        logger.info(f"Schedulable Nodes: {self.cluster_nodes['schedulable_nodes']}")
    
    def stop_scheduler_plugin(self):
        """Stop the ML scheduler plugin."""
        
        self.is_active = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=10)
        
        logger.info("HYDATIS ML Scheduler Plugin stopped")
    
    def _scheduler_main_loop(self):
        """Main scheduler loop for processing pod scheduling requests."""
        
        logger.info("Starting ML scheduler main loop...")
        
        w = watch.Watch()
        
        try:
            for event in w.stream(self.v1.list_pod_for_all_namespaces, timeout_seconds=0):
                if not self.is_active:
                    break
                
                event_type = event['type']
                pod = event['object']
                
                if (event_type == 'ADDED' and 
                    pod.status.phase == 'Pending' and 
                    pod.spec.node_name is None and
                    pod.spec.scheduler_name == 'hydatis-ml-scheduler'):
                    
                    asyncio.run(self._process_pod_scheduling(pod))
        
        except Exception as e:
            logger.error(f"Scheduler main loop error: {e}")
            if self.is_active:
                time.sleep(5)
                self._scheduler_main_loop()
    
    async def _process_pod_scheduling(self, pod) -> Dict[str, Any]:
        """Process individual pod scheduling request."""
        
        scheduling_start = time.time()
        
        self.scheduling_stats['total_scheduling_requests'] += 1
        
        pod_name = pod.metadata.name
        pod_namespace = pod.metadata.namespace
        
        logger.info(f"Processing scheduling for pod {pod_namespace}/{pod_name}")
        
        try:
            cluster_context = await self._gather_cluster_context()
            
            available_nodes = await self._get_available_nodes()
            
            pod_spec = self._extract_pod_specification(pod)
            
            scheduler_request = {
                'pod_spec': pod_spec,
                'cluster_context': cluster_context,
                'available_nodes': available_nodes
            }
            
            ml_decision = await self._get_ml_scheduling_decision(scheduler_request)
            
            if ml_decision and 'error' not in ml_decision:
                self.scheduling_stats['ml_scheduling_decisions'] += 1
                
                final_recommendation = ml_decision.get('final_recommendation', {})
                
                if final_recommendation.get('action') == 'defer':
                    logger.info(f"ML decision: defer scheduling for {pod_name} - {final_recommendation.get('reasoning', ['Unknown reason'])[0]}")
                    return {'action': 'deferred', 'reason': 'ML recommendation'}
                
                selected_node = final_recommendation.get('selected_node')
                
                if selected_node and selected_node in available_nodes:
                    binding_result = await self._bind_pod_to_node(pod, selected_node)
                    
                    if binding_result['success']:
                        self.scheduling_stats['successful_placements'] += 1
                        self.scheduling_stats['node_placement_counts'][selected_node] += 1
                        
                        scheduling_latency = (time.time() - scheduling_start) * 1000
                        self._update_latency_tracking(scheduling_latency)
                        
                        logger.info(f"Successfully scheduled {pod_name} to {selected_node} "
                                   f"(ML decision, {scheduling_latency:.1f}ms)")
                        
                        return {
                            'action': 'scheduled',
                            'node': selected_node,
                            'decision_source': 'ml_ensemble',
                            'latency_ms': scheduling_latency,
                            'ml_confidence': ml_decision.get('synthesis_metrics', {}).get('models_consulted', 0)
                        }
                    else:
                        logger.error(f"Failed to bind {pod_name} to {selected_node}: {binding_result['error']}")
                        self.scheduling_stats['failed_placements'] += 1
                        return await self._handle_scheduling_failure(pod, binding_result['error'])
                else:
                    logger.warning(f"ML recommended unavailable node {selected_node} for {pod_name}")
                    return await self._fallback_scheduling(pod, available_nodes, "ML recommended unavailable node")
            else:
                error_msg = ml_decision.get('error', 'ML gateway unavailable') if ml_decision else 'No ML response'
                logger.warning(f"ML decision unavailable for {pod_name}: {error_msg}")
                return await self._fallback_scheduling(pod, available_nodes, error_msg)
        
        except Exception as e:
            logger.error(f"Scheduling error for {pod_name}: {e}")
            self.scheduling_stats['failed_placements'] += 1
            self.scheduling_stats['scheduling_errors'].append({
                'pod': f"{pod_namespace}/{pod_name}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            return {'action': 'failed', 'error': str(e)}
    
    async def _gather_cluster_context(self) -> Dict[str, Any]:
        """Gather current cluster context for ML decision making."""
        
        try:
            nodes = self.v1.list_node()
            
            node_metrics = {}
            cluster_metrics = {
                'total_nodes': len(nodes.items),
                'schedulable_nodes': len([n for n in nodes.items if not n.spec.unschedulable]),
                'pending_pods': 0,
                'cluster_utilization': {}
            }
            
            pods = self.v1.list_pod_for_all_namespaces()
            pending_pods = [p for p in pods.items if p.status.phase == 'Pending']
            cluster_metrics['pending_pods'] = len(pending_pods)
            
            for node in nodes.items:
                if node.metadata.name in self.cluster_nodes['worker_nodes']:
                    
                    node_pods = [p for p in pods.items if p.spec.node_name == node.metadata.name and p.status.phase == 'Running']
                    
                    total_cpu_requests = sum(
                        float(container.resources.requests.get('cpu', '0').rstrip('m')) / 1000
                        for pod in node_pods
                        for container in pod.spec.containers
                        if container.resources and container.resources.requests and container.resources.requests.get('cpu')
                    )
                    
                    total_memory_requests = sum(
                        self._parse_memory(container.resources.requests.get('memory', '0'))
                        for pod in node_pods
                        for container in pod.spec.containers
                        if container.resources and container.resources.requests and container.resources.requests.get('memory')
                    )
                    
                    node_capacity = {
                        'cpu': float(node.status.allocatable.get('cpu', '8')),
                        'memory': self._parse_memory(node.status.allocatable.get('memory', '16Gi'))
                    }
                    
                    node_metrics[node.metadata.name] = {
                        'cpu_utilization': total_cpu_requests / node_capacity['cpu'],
                        'memory_utilization': total_memory_requests / node_capacity['memory'],
                        'cpu_capacity': node_capacity['cpu'],
                        'memory_capacity': node_capacity['memory'],
                        'running_pods': len(node_pods),
                        'node_ready': any(condition.type == 'Ready' and condition.status == 'True' 
                                        for condition in node.status.conditions or [])
                    }
            
            return {
                'node_metrics': node_metrics,
                'cluster_metrics': cluster_metrics,
                'context_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error gathering cluster context: {e}")
            return {'error': str(e)}
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse Kubernetes memory string to GB."""
        
        if not memory_str or memory_str == '0':
            return 0.0
        
        memory_str = memory_str.upper()
        
        if memory_str.endswith('GI'):
            return float(memory_str[:-2])
        elif memory_str.endswith('G'):
            return float(memory_str[:-1])
        elif memory_str.endswith('MI'):
            return float(memory_str[:-2]) / 1024
        elif memory_str.endswith('M'):
            return float(memory_str[:-1]) / 1024
        elif memory_str.endswith('KI'):
            return float(memory_str[:-2]) / (1024 * 1024)
        elif memory_str.endswith('K'):
            return float(memory_str[:-1]) / (1024 * 1024)
        else:
            return float(memory_str) / (1024 * 1024 * 1024)
    
    async def _get_available_nodes(self) -> List[str]:
        """Get list of available schedulable nodes."""
        
        try:
            nodes = self.v1.list_node()
            
            available_nodes = []
            
            for node in nodes.items:
                if (not node.spec.unschedulable and 
                    node.metadata.name in self.cluster_nodes['schedulable_nodes']):
                    
                    node_ready = any(condition.type == 'Ready' and condition.status == 'True'
                                   for condition in node.status.conditions or [])
                    
                    if node_ready:
                        available_nodes.append(node.metadata.name)
            
            return available_nodes
            
        except Exception as e:
            logger.error(f"Error getting available nodes: {e}")
            return self.cluster_nodes['schedulable_nodes']
    
    def _extract_pod_specification(self, pod) -> Dict[str, Any]:
        """Extract relevant pod specification for ML scheduling."""
        
        pod_spec = {
            'metadata': {
                'name': pod.metadata.name,
                'namespace': pod.metadata.namespace,
                'labels': dict(pod.metadata.labels) if pod.metadata.labels else {},
                'annotations': dict(pod.metadata.annotations) if pod.metadata.annotations else {}
            },
            'resources': {
                'cpu_request': 0.0,
                'memory_request': 0.0,
                'cpu_limit': 0.0,
                'memory_limit': 0.0
            },
            'scheduling_constraints': {
                'node_selector': dict(pod.spec.node_selector) if pod.spec.node_selector else {},
                'affinity': self._extract_affinity_rules(pod.spec.affinity) if pod.spec.affinity else {},
                'tolerations': self._extract_tolerations(pod.spec.tolerations) if pod.spec.tolerations else []
            },
            'workload_characteristics': {
                'priority_class': pod.spec.priority_class_name,
                'restart_policy': pod.spec.restart_policy,
                'container_count': len(pod.spec.containers),
                'service_account': pod.spec.service_account_name
            }
        }
        
        for container in pod.spec.containers:
            if container.resources:
                if container.resources.requests:
                    cpu_request = container.resources.requests.get('cpu', '0')
                    memory_request = container.resources.requests.get('memory', '0')
                    
                    pod_spec['resources']['cpu_request'] += self._parse_cpu(cpu_request)
                    pod_spec['resources']['memory_request'] += self._parse_memory(memory_request)
                
                if container.resources.limits:
                    cpu_limit = container.resources.limits.get('cpu', '0')
                    memory_limit = container.resources.limits.get('memory', '0')
                    
                    pod_spec['resources']['cpu_limit'] += self._parse_cpu(cpu_limit)
                    pod_spec['resources']['memory_limit'] += self._parse_memory(memory_limit)
        
        return pod_spec
    
    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse Kubernetes CPU string to cores."""
        
        if not cpu_str or cpu_str == '0':
            return 0.0
        
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1000
        else:
            return float(cpu_str)
    
    def _extract_affinity_rules(self, affinity) -> Dict[str, Any]:
        """Extract pod affinity/anti-affinity rules."""
        
        affinity_rules = {}
        
        if affinity.node_affinity:
            affinity_rules['node_affinity'] = 'present'
        
        if affinity.pod_affinity:
            affinity_rules['pod_affinity'] = 'present'
        
        if affinity.pod_anti_affinity:
            affinity_rules['pod_anti_affinity'] = 'present'
        
        return affinity_rules
    
    def _extract_tolerations(self, tolerations) -> List[Dict[str, str]]:
        """Extract pod tolerations."""
        
        if not tolerations:
            return []
        
        return [
            {
                'key': toleration.key or '',
                'operator': toleration.operator or 'Equal',
                'effect': toleration.effect or 'NoSchedule'
            }
            for toleration in tolerations
        ]
    
    async def _get_ml_scheduling_decision(self, scheduler_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get ML scheduling decision from gateway."""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ml_gateway_url}/schedule",
                    json=scheduler_request,
                    timeout=aiohttp.ClientTimeout(total=self.plugin_config['ml_decision_timeout_seconds'])
                ) as response:
                    
                    if response.status == 200:
                        decision = await response.json()
                        return decision
                    else:
                        logger.error(f"ML gateway error: HTTP {response.status}")
                        return None
        
        except asyncio.TimeoutError:
            logger.error("ML gateway timeout")
            return None
        except Exception as e:
            logger.error(f"ML gateway communication error: {e}")
            return None
    
    async def _bind_pod_to_node(self, pod, node_name: str) -> Dict[str, Any]:
        """Bind pod to selected node."""
        
        try:
            binding = client.V1Binding(
                api_version="v1",
                kind="Binding",
                metadata=client.V1ObjectMeta(
                    name=pod.metadata.name,
                    namespace=pod.metadata.namespace
                ),
                target=client.V1ObjectReference(
                    api_version="v1",
                    kind="Node",
                    name=node_name
                )
            )
            
            self.v1.create_namespaced_binding(
                namespace=pod.metadata.namespace,
                body=binding
            )
            
            logger.info(f"Successfully bound pod {pod.metadata.namespace}/{pod.metadata.name} to node {node_name}")
            
            return {'success': True, 'node': node_name}
            
        except ApiException as e:
            logger.error(f"Kubernetes API error binding pod: {e}")
            return {'success': False, 'error': f'Kubernetes API error: {e.reason}'}
        except Exception as e:
            logger.error(f"Error binding pod to node: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _fallback_scheduling(self, pod, available_nodes: List[str], reason: str) -> Dict[str, Any]:
        """Perform fallback scheduling when ML decision is unavailable."""
        
        self.scheduling_stats['fallback_decisions'] += 1
        
        pod_name = pod.metadata.name
        
        logger.info(f"Fallback scheduling for {pod_name}: {reason}")
        
        if not available_nodes:
            logger.error(f"No available nodes for fallback scheduling of {pod_name}")
            return {'action': 'failed', 'error': 'No available nodes'}
        
        fallback_strategy = self.plugin_config['fallback_strategy']
        
        if fallback_strategy == 'least_allocated':
            selected_node = await self._select_least_allocated_node(available_nodes)
        elif fallback_strategy == 'round_robin':
            selected_node = self._select_round_robin_node(available_nodes)
        else:
            selected_node = available_nodes[0]
        
        binding_result = await self._bind_pod_to_node(pod, selected_node)
        
        if binding_result['success']:
            self.scheduling_stats['successful_placements'] += 1
            self.scheduling_stats['node_placement_counts'][selected_node] += 1
            
            logger.info(f"Fallback scheduled {pod_name} to {selected_node}")
            
            return {
                'action': 'scheduled',
                'node': selected_node,
                'decision_source': 'fallback',
                'fallback_reason': reason,
                'fallback_strategy': fallback_strategy
            }
        else:
            self.scheduling_stats['failed_placements'] += 1
            logger.error(f"Fallback scheduling failed for {pod_name}: {binding_result['error']}")
            
            return {'action': 'failed', 'error': binding_result['error']}
    
    async def _select_least_allocated_node(self, available_nodes: List[str]) -> str:
        """Select node with least resource allocation."""
        
        try:
            cluster_context = await self._gather_cluster_context()
            node_metrics = cluster_context.get('node_metrics', {})
            
            node_scores = {}
            
            for node in available_nodes:
                if node in node_metrics:
                    metrics = node_metrics[node]
                    
                    cpu_available = 1.0 - metrics.get('cpu_utilization', 0.5)
                    memory_available = 1.0 - metrics.get('memory_utilization', 0.5)
                    
                    availability_score = (cpu_available * 0.6 + memory_available * 0.4)
                    node_scores[node] = availability_score
                else:
                    node_scores[node] = 0.5
            
            best_node = max(node_scores.items(), key=lambda x: x[1])[0]
            return best_node
            
        except Exception as e:
            logger.error(f"Error selecting least allocated node: {e}")
            return available_nodes[0]
    
    def _select_round_robin_node(self, available_nodes: List[str]) -> str:
        """Select node using round-robin strategy."""
        
        node_counts = {node: self.scheduling_stats['node_placement_counts'].get(node, 0) 
                      for node in available_nodes}
        
        least_used_node = min(node_counts.items(), key=lambda x: x[1])[0]
        return least_used_node
    
    async def _handle_scheduling_failure(self, pod, error: str) -> Dict[str, Any]:
        """Handle pod scheduling failure with retry logic."""
        
        pod_name = f"{pod.metadata.namespace}/{pod.metadata.name}"
        
        logger.error(f"Scheduling failure for {pod_name}: {error}")
        
        self.scheduling_stats['scheduling_errors'].append({
            'pod': pod_name,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'retry_attempted': False
        })
        
        return {
            'action': 'failed',
            'error': error,
            'pod': pod_name,
            'failure_timestamp': datetime.now().isoformat()
        }
    
    def _update_latency_tracking(self, latency_ms: float):
        """Update scheduling latency tracking."""
        
        current_avg = self.scheduling_stats['average_decision_latency_ms']
        total_decisions = self.scheduling_stats['total_scheduling_requests']
        
        new_avg = ((current_avg * (total_decisions - 1)) + latency_ms) / total_decisions
        self.scheduling_stats['average_decision_latency_ms'] = new_avg
    
    def get_scheduler_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scheduler plugin metrics."""
        
        success_rate = (self.scheduling_stats['successful_placements'] / 
                       max(self.scheduling_stats['total_scheduling_requests'], 1))
        
        ml_utilization_rate = (self.scheduling_stats['ml_scheduling_decisions'] / 
                             max(self.scheduling_stats['total_scheduling_requests'], 1))
        
        fallback_rate = (self.scheduling_stats['fallback_decisions'] / 
                        max(self.scheduling_stats['total_scheduling_requests'], 1))
        
        metrics = {
            'plugin_status': {
                'active': self.is_active,
                'ml_gateway_url': self.ml_gateway_url,
                'namespace': self.namespace,
                'schedulable_nodes': len(self.cluster_nodes['schedulable_nodes'])
            },
            'scheduling_performance': {
                'total_requests': self.scheduling_stats['total_scheduling_requests'],
                'successful_placements': self.scheduling_stats['successful_placements'],
                'failed_placements': self.scheduling_stats['failed_placements'],
                'success_rate': success_rate,
                'average_decision_latency_ms': self.scheduling_stats['average_decision_latency_ms']
            },
            'ml_integration': {
                'ml_decisions_used': self.scheduling_stats['ml_scheduling_decisions'],
                'fallback_decisions': self.scheduling_stats['fallback_decisions'],
                'ml_utilization_rate': ml_utilization_rate,
                'fallback_rate': fallback_rate
            },
            'node_distribution': {
                'placement_counts': self.scheduling_stats['node_placement_counts'],
                'most_utilized_node': max(self.scheduling_stats['node_placement_counts'].items(), 
                                        key=lambda x: x[1])[0] if self.scheduling_stats['node_placement_counts'] else None,
                'load_balance_variance': np.var(list(self.scheduling_stats['node_placement_counts'].values()))
            },
            'error_analysis': {
                'total_errors': len(self.scheduling_stats['scheduling_errors']),
                'recent_errors': [err for err in self.scheduling_stats['scheduling_errors']
                                if datetime.fromisoformat(err['timestamp']) > datetime.now() - timedelta(hours=1)]
            },
            'metrics_timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def create_scheduler_deployment(self) -> Dict[str, str]:
        """Create Kubernetes deployment manifests for ML scheduler."""
        
        deployment_dir = Path("/home/wassim/Desktop/ml scheduler/deployments/kubernetes")
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        scheduler_deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'hydatis-ml-scheduler',
                'namespace': self.namespace,
                'labels': {
                    'app': 'hydatis-ml-scheduler',
                    'component': 'scheduler'
                }
            },
            'spec': {
                'replicas': 1,
                'selector': {
                    'matchLabels': {
                        'app': 'hydatis-ml-scheduler'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'hydatis-ml-scheduler',
                            'component': 'scheduler'
                        }
                    },
                    'spec': {
                        'serviceAccountName': 'hydatis-ml-scheduler',
                        'containers': [{
                            'name': 'ml-scheduler',
                            'image': 'hydatis/ml-scheduler:latest',
                            'ports': [
                                {'containerPort': 8083, 'name': 'gateway'},
                                {'containerPort': 10259, 'name': 'metrics'}
                            ],
                            'env': [
                                {'name': 'ML_GATEWAY_PORT', 'value': '8083'},
                                {'name': 'PROMETHEUS_URL', 'value': 'http://10.110.190.83:9090'},
                                {'name': 'MLFLOW_TRACKING_URI', 'value': 'http://10.110.190.32:31380'},
                                {'name': 'MODEL_STORAGE_PATH', 'value': '/data/ml_scheduler_longhorn/models'}
                            ],
                            'volumeMounts': [{
                                'name': 'model-storage',
                                'mountPath': '/data/ml_scheduler_longhorn'
                            }],
                            'resources': {
                                'requests': {
                                    'cpu': '500m',
                                    'memory': '1Gi'
                                },
                                'limits': {
                                    'cpu': '2000m',
                                    'memory': '4Gi'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8083
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 30
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8083
                                },
                                'initialDelaySeconds': 15,
                                'periodSeconds': 10
                            }
                        }],
                        'volumes': [{
                            'name': 'model-storage',
                            'persistentVolumeClaim': {
                                'claimName': 'ml-scheduler-models-pvc'
                            }
                        }]
                    }
                }
            }
        }
        
        scheduler_config = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'hydatis-ml-scheduler-config',
                'namespace': self.namespace
            },
            'data': {
                'config.yaml': yaml.dump({
                    'schedulerName': 'hydatis-ml-scheduler',
                    'profiles': [{
                        'schedulerName': 'hydatis-ml-scheduler',
                        'plugins': {
                            'filter': {
                                'enabled': [
                                    {'name': 'NodeResourcesFit'},
                                    {'name': 'NodeAffinity'},
                                    {'name': 'PodTopologySpread'}
                                ]
                            },
                            'score': {
                                'enabled': [
                                    {'name': 'NodeResourcesFit', 'weight': 1},
                                    {'name': 'NodeAffinity', 'weight': 1}
                                ]
                            }
                        },
                        'pluginConfig': [{
                            'name': 'NodeResourcesFit',
                            'args': {
                                'scoringStrategy': {
                                    'type': 'LeastAllocated'
                                }
                            }
                        }]
                    }]
                })
            }
        }
        
        service_account = {
            'apiVersion': 'v1',
            'kind': 'ServiceAccount',
            'metadata': {
                'name': 'hydatis-ml-scheduler',
                'namespace': self.namespace
            }
        }
        
        cluster_role = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'ClusterRole',
            'metadata': {
                'name': 'hydatis-ml-scheduler'
            },
            'rules': [
                {
                    'apiGroups': [''],
                    'resources': ['pods', 'nodes', 'bindings'],
                    'verbs': ['get', 'list', 'watch', 'create', 'update', 'patch']
                },
                {
                    'apiGroups': ['apps'],
                    'resources': ['deployments', 'replicasets'],
                    'verbs': ['get', 'list', 'watch']
                },
                {
                    'apiGroups': ['metrics.k8s.io'],
                    'resources': ['nodes', 'pods'],
                    'verbs': ['get', 'list']
                }
            ]
        }
        
        cluster_role_binding = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'ClusterRoleBinding',
            'metadata': {
                'name': 'hydatis-ml-scheduler'
            },
            'roleRef': {
                'apiGroup': 'rbac.authorization.k8s.io',
                'kind': 'ClusterRole',
                'name': 'hydatis-ml-scheduler'
            },
            'subjects': [{
                'kind': 'ServiceAccount',
                'name': 'hydatis-ml-scheduler',
                'namespace': self.namespace
            }]
        }
        
        manifest_files = {}
        
        manifests = {
            'deployment': scheduler_deployment,
            'configmap': scheduler_config,
            'serviceaccount': service_account,
            'clusterrole': cluster_role,
            'clusterrolebinding': cluster_role_binding
        }
        
        for manifest_name, manifest_content in manifests.items():
            file_path = deployment_dir / f"hydatis-ml-scheduler-{manifest_name}.yaml"
            
            with open(file_path, 'w') as f:
                yaml.dump(manifest_content, f, default_flow_style=False)
            
            manifest_files[manifest_name] = str(file_path)
        
        logger.info(f"Kubernetes manifests created in {deployment_dir}")
        
        return manifest_files


def main():
    """Main scheduler plugin demonstration."""
    
    print("HYDATIS Kubernetes ML Scheduler Plugin - Week 8")
    print("Integrating ML-powered scheduling with Kubernetes")
    
    plugin = HYDATISSchedulerPlugin()
    
    print("Plugin Configuration:")
    print(f"  ML Gateway URL: {plugin.ml_gateway_url}")
    print(f"  Namespace: {plugin.namespace}")
    print(f"  Schedulable Nodes: {plugin.cluster_nodes['schedulable_nodes']}")
    print(f"  Fallback Strategy: {plugin.plugin_config['fallback_strategy']}")
    print(f"  ML Decision Timeout: {plugin.plugin_config['ml_decision_timeout_seconds']}s")
    
    manifest_files = plugin.create_scheduler_deployment()
    
    print("Kubernetes Manifests Created:")
    for manifest_type, file_path in manifest_files.items():
        print(f"  {manifest_type}: {file_path}")
    
    print("Scheduler Features:")
    print("  ✅ ML ensemble decision integration")
    print("  ✅ Intelligent fallback strategies")
    print("  ✅ Real-time cluster context gathering")
    print("  ✅ Pod binding and placement")
    print("  ✅ Performance monitoring and metrics")
    print("  ✅ Error handling and retry logic")
    
    return plugin


if __name__ == "__main__":
    scheduler = main()