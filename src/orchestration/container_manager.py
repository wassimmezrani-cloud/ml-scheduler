#!/usr/bin/env python3

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import docker
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException

class ContainerHealth(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    UNKNOWN = "unknown"

@dataclass
class ContainerMetrics:
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    network_rx_bytes: int
    network_tx_bytes: int
    restart_count: int
    uptime_seconds: int

@dataclass
class ContainerStatus:
    name: str
    namespace: str
    pod_name: str
    node_name: str
    health: ContainerHealth
    metrics: Optional[ContainerMetrics]
    last_restart: Optional[datetime]
    error_logs: List[str]

class ContainerManager:
    def __init__(self):
        self.logger = self._setup_logging()
        self.k8s_client = self._initialize_k8s_client()
        self.docker_client = self._initialize_docker_client()
        
        self.hydatis_config = {
            'namespace': 'ml-scheduler',
            'cluster_name': 'HYDATIS',
            'prometheus_endpoint': 'http://10.110.190.83:9090'
        }
        
        self.container_registry = {}
        self.health_thresholds = {
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'restart_threshold': 5,
            'response_timeout': 10
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for container manager."""
        logger = logging.getLogger('container_manager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _initialize_k8s_client(self):
        """Initialize Kubernetes client."""
        try:
            config.load_kube_config()
            return client.ApiClient()
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise

    def _initialize_docker_client(self):
        """Initialize Docker client for local operations."""
        try:
            return docker.from_env()
        except Exception as e:
            self.logger.warning(f"Docker client not available: {e}")
            return None

    async def monitor_container_health(self) -> Dict[str, ContainerStatus]:
        """Monitor health of all ML scheduler containers."""
        core_v1 = client.CoreV1Api(self.k8s_client)
        container_statuses = {}
        
        try:
            pods = core_v1.list_namespaced_pod(
                namespace=self.hydatis_config['namespace'],
                label_selector="cluster=hydatis"
            )
            
            monitoring_tasks = []
            for pod in pods.items:
                for container in pod.spec.containers:
                    task = self._monitor_single_container(pod, container)
                    monitoring_tasks.append(task)
            
            results = await asyncio.gather(*monitoring_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ContainerStatus):
                    container_statuses[result.name] = result
                elif isinstance(result, Exception):
                    self.logger.error(f"Container monitoring failed: {result}")
            
            return container_statuses
            
        except Exception as e:
            self.logger.error(f"Failed to monitor container health: {e}")
            return {}

    async def _monitor_single_container(self, pod, container) -> ContainerStatus:
        """Monitor single container health and metrics."""
        try:
            container_name = f"{pod.metadata.name}/{container.name}"
            
            health = await self._check_container_health(pod, container)
            
            metrics = await self._collect_container_metrics(pod, container)
            
            error_logs = await self._get_container_error_logs(pod, container)
            
            last_restart = self._get_last_restart_time(pod, container)
            
            return ContainerStatus(
                name=container_name,
                namespace=pod.metadata.namespace,
                pod_name=pod.metadata.name,
                node_name=pod.spec.node_name or 'unknown',
                health=health,
                metrics=metrics,
                last_restart=last_restart,
                error_logs=error_logs
            )
            
        except Exception as e:
            self.logger.error(f"Failed to monitor container {container.name}: {e}")
            return ContainerStatus(
                name=f"{pod.metadata.name}/{container.name}",
                namespace=pod.metadata.namespace,
                pod_name=pod.metadata.name,
                node_name='unknown',
                health=ContainerHealth.UNKNOWN,
                metrics=None,
                last_restart=None,
                error_logs=[str(e)]
            )

    async def _check_container_health(self, pod, container) -> ContainerHealth:
        """Check individual container health status."""
        try:
            for container_status in pod.status.container_statuses or []:
                if container_status.name == container.name:
                    if not container_status.ready:
                        return ContainerHealth.STARTING if container_status.restart_count == 0 else ContainerHealth.UNHEALTHY
                    
                    if container_status.restart_count > self.health_thresholds['restart_threshold']:
                        return ContainerHealth.UNHEALTHY
                    
                    if hasattr(container, 'liveness_probe') and container.liveness_probe:
                        health_check_result = await self._perform_health_check(pod, container)
                        return health_check_result
                    
                    return ContainerHealth.HEALTHY
            
            return ContainerHealth.UNKNOWN
            
        except Exception as e:
            self.logger.error(f"Health check failed for {container.name}: {e}")
            return ContainerHealth.UNKNOWN

    async def _perform_health_check(self, pod, container) -> ContainerHealth:
        """Perform HTTP health check on container."""
        try:
            if not container.liveness_probe or not container.liveness_probe.http_get:
                return ContainerHealth.HEALTHY
            
            health_path = container.liveness_probe.http_get.path
            health_port = container.liveness_probe.http_get.port
            
            service_name = f"{pod.metadata.labels.get('app', pod.metadata.name)}-service"
            health_url = f"http://{service_name}.{pod.metadata.namespace}.svc.cluster.local:{health_port}{health_path}"
            
            response = requests.get(health_url, timeout=self.health_thresholds['response_timeout'])
            
            if response.status_code == 200:
                return ContainerHealth.HEALTHY
            else:
                return ContainerHealth.UNHEALTHY
                
        except requests.exceptions.RequestException:
            return ContainerHealth.UNHEALTHY
        except Exception as e:
            self.logger.error(f"Health check error for {container.name}: {e}")
            return ContainerHealth.UNKNOWN

    async def _collect_container_metrics(self, pod, container) -> Optional[ContainerMetrics]:
        """Collect container resource metrics from Prometheus."""
        try:
            prometheus_url = self.hydatis_config['prometheus_endpoint']
            pod_name = pod.metadata.name
            container_name = container.name
            
            metrics_queries = {
                'cpu_usage': f'rate(container_cpu_usage_seconds_total{{pod="{pod_name}",container="{container_name}"}}[5m]) * 100',
                'memory_usage': f'container_memory_working_set_bytes{{pod="{pod_name}",container="{container_name}"}}',
                'memory_limit': f'container_spec_memory_limit_bytes{{pod="{pod_name}",container="{container_name}"}}',
                'network_rx': f'container_network_receive_bytes_total{{pod="{pod_name}"}}',
                'network_tx': f'container_network_transmit_bytes_total{{pod="{pod_name}"}}'
            }
            
            metrics_data = {}
            
            for metric_name, query in metrics_queries.items():
                try:
                    response = requests.get(
                        f"{prometheus_url}/api/v1/query",
                        params={'query': query},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data['data']['result']:
                            metrics_data[metric_name] = float(data['data']['result'][0]['value'][1])
                        else:
                            metrics_data[metric_name] = 0.0
                    
                except Exception as e:
                    self.logger.warning(f"Failed to collect {metric_name} for {container_name}: {e}")
                    metrics_data[metric_name] = 0.0
            
            restart_count = 0
            uptime_seconds = 0
            
            for container_status in pod.status.container_statuses or []:
                if container_status.name == container.name:
                    restart_count = container_status.restart_count
                    if container_status.state.running:
                        start_time = container_status.state.running.started_at
                        if start_time:
                            uptime_seconds = (datetime.now(start_time.tzinfo) - start_time).total_seconds()
            
            return ContainerMetrics(
                cpu_usage_percent=metrics_data.get('cpu_usage', 0.0),
                memory_usage_mb=metrics_data.get('memory_usage', 0.0) / (1024 * 1024),
                memory_limit_mb=metrics_data.get('memory_limit', 0.0) / (1024 * 1024),
                network_rx_bytes=int(metrics_data.get('network_rx', 0)),
                network_tx_bytes=int(metrics_data.get('network_tx', 0)),
                restart_count=restart_count,
                uptime_seconds=int(uptime_seconds)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics for {container.name}: {e}")
            return None

    async def _get_container_error_logs(self, pod, container, lines: int = 50) -> List[str]:
        """Get recent error logs from container."""
        try:
            core_v1 = client.CoreV1Api(self.k8s_client)
            
            log_response = core_v1.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=pod.metadata.namespace,
                container=container.name,
                tail_lines=lines
            )
            
            log_lines = log_response.split('\n')
            error_lines = [
                line for line in log_lines 
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'critical'])
            ]
            
            return error_lines[-10:] if len(error_lines) > 10 else error_lines
            
        except Exception as e:
            self.logger.error(f"Failed to get logs for {container.name}: {e}")
            return [f"Log retrieval failed: {str(e)}"]

    def _get_last_restart_time(self, pod, container) -> Optional[datetime]:
        """Get timestamp of last container restart."""
        try:
            for container_status in pod.status.container_statuses or []:
                if container_status.name == container.name:
                    if container_status.last_state and container_status.last_state.terminated:
                        return container_status.last_state.terminated.finished_at
            return None
        except Exception:
            return None

    async def restart_unhealthy_containers(self) -> Dict[str, Any]:
        """Restart containers that are unhealthy."""
        container_statuses = await self.monitor_container_health()
        restart_results = {}
        
        for container_name, status in container_statuses.items():
            if status.health == ContainerHealth.UNHEALTHY:
                try:
                    self.logger.info(f"Restarting unhealthy container: {container_name}")
                    
                    result = await self._restart_container(status.pod_name, status.namespace)
                    restart_results[container_name] = result
                    
                except Exception as e:
                    restart_results[container_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        return restart_results

    async def _restart_container(self, pod_name: str, namespace: str) -> Dict[str, Any]:
        """Restart specific container by deleting pod."""
        try:
            core_v1 = client.CoreV1Api(self.k8s_client)
            
            core_v1.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace
            )
            
            await asyncio.sleep(5)
            
            max_wait = 120
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                try:
                    pod = core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)
                    
                    if pod.status.phase == 'Running':
                        all_ready = all(
                            container_status.ready 
                            for container_status in pod.status.container_statuses or []
                        )
                        
                        if all_ready:
                            return {
                                'status': 'success',
                                'restart_time': datetime.now().isoformat()
                            }
                    
                except ApiException as e:
                    if e.status == 404:
                        await asyncio.sleep(2)
                        continue
                    else:
                        raise
                
                await asyncio.sleep(3)
            
            return {
                'status': 'timeout',
                'error': f"Container restart timed out after {max_wait}s"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def auto_scale_based_on_metrics(self) -> Dict[str, Any]:
        """Auto-scale components based on resource metrics."""
        container_statuses = await self.monitor_container_health()
        scaling_decisions = {}
        
        component_metrics = {}
        for container_name, status in container_statuses.items():
            if status.metrics:
                component = container_name.split('/')[0]
                if component not in component_metrics:
                    component_metrics[component] = []
                component_metrics[component].append(status.metrics)
        
        for component, metrics_list in component_metrics.items():
            try:
                avg_cpu = sum(m.cpu_usage_percent for m in metrics_list) / len(metrics_list)
                avg_memory = sum(m.memory_usage_mb / m.memory_limit_mb * 100 for m in metrics_list if m.memory_limit_mb > 0) / len(metrics_list)
                
                current_replicas = len(metrics_list)
                
                scaling_decision = self._calculate_scaling_decision(
                    component, current_replicas, avg_cpu, avg_memory
                )
                
                if scaling_decision['action'] != 'none':
                    result = await self._execute_scaling(component, scaling_decision['target_replicas'])
                    scaling_decisions[component] = {
                        'action': scaling_decision['action'],
                        'target_replicas': scaling_decision['target_replicas'],
                        'reason': scaling_decision['reason'],
                        'result': result
                    }
                
            except Exception as e:
                self.logger.error(f"Auto-scaling failed for {component}: {e}")
                scaling_decisions[component] = {
                    'action': 'error',
                    'error': str(e)
                }
        
        return scaling_decisions

    def _calculate_scaling_decision(self, component: str, current_replicas: int, 
                                  avg_cpu: float, avg_memory: float) -> Dict[str, Any]:
        """Calculate scaling decision based on metrics."""
        
        cpu_threshold_high = self.health_thresholds['cpu_threshold']
        memory_threshold_high = self.health_thresholds['memory_threshold']
        cpu_threshold_low = cpu_threshold_high * 0.3
        memory_threshold_low = memory_threshold_high * 0.3
        
        max_replicas = self._get_max_replicas_for_component(component)
        min_replicas = self._get_min_replicas_for_component(component)
        
        if (avg_cpu > cpu_threshold_high or avg_memory > memory_threshold_high) and current_replicas < max_replicas:
            target_replicas = min(current_replicas + 1, max_replicas)
            return {
                'action': 'scale_up',
                'target_replicas': target_replicas,
                'reason': f"High resource usage: CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%"
            }
        
        elif (avg_cpu < cpu_threshold_low and avg_memory < memory_threshold_low) and current_replicas > min_replicas:
            target_replicas = max(current_replicas - 1, min_replicas)
            return {
                'action': 'scale_down',
                'target_replicas': target_replicas,
                'reason': f"Low resource usage: CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%"
            }
        
        else:
            return {
                'action': 'none',
                'target_replicas': current_replicas,
                'reason': f"Metrics within acceptable range: CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%"
            }

    def _get_max_replicas_for_component(self, component: str) -> int:
        """Get maximum replicas allowed for component."""
        component_limits = {
            'xgboost-load-predictor': 5,
            'qlearning-placement-optimizer': 4,
            'isolation-forest-detector': 4,
            'ml-scheduler-gateway': 6,
            'hydatis-scheduler-plugin': 1,
            'unified-monitoring-dashboard': 2,
            'alert-manager': 1
        }
        return component_limits.get(component, 3)

    def _get_min_replicas_for_component(self, component: str) -> int:
        """Get minimum replicas required for component."""
        component_minimums = {
            'xgboost-load-predictor': 1,
            'qlearning-placement-optimizer': 1,
            'isolation-forest-detector': 1,
            'ml-scheduler-gateway': 2,
            'hydatis-scheduler-plugin': 1,
            'unified-monitoring-dashboard': 1,
            'alert-manager': 1
        }
        return component_minimums.get(component, 1)

    async def _execute_scaling(self, component: str, target_replicas: int) -> Dict[str, Any]:
        """Execute scaling operation for component."""
        try:
            apps_v1 = client.AppsV1Api(self.k8s_client)
            
            apps_v1.patch_namespaced_deployment_scale(
                name=component,
                namespace=self.hydatis_config['namespace'],
                body={'spec': {'replicas': target_replicas}}
            )
            
            self.logger.info(f"Scaled {component} to {target_replicas} replicas")
            
            return {
                'status': 'success',
                'target_replicas': target_replicas
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def rolling_update_component(self, component: str, new_image: str) -> Dict[str, Any]:
        """Perform rolling update of component with new image."""
        try:
            apps_v1 = client.AppsV1Api(self.k8s_client)
            
            deployment = apps_v1.read_namespaced_deployment(
                name=component,
                namespace=self.hydatis_config['namespace']
            )
            
            old_image = deployment.spec.template.spec.containers[0].image
            
            deployment.spec.template.spec.containers[0].image = new_image
            
            apps_v1.patch_namespaced_deployment(
                name=component,
                namespace=self.hydatis_config['namespace'],
                body=deployment
            )
            
            self.logger.info(f"Started rolling update for {component}: {old_image} -> {new_image}")
            
            max_wait = 300
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                updated_deployment = apps_v1.read_namespaced_deployment(
                    name=component,
                    namespace=self.hydatis_config['namespace']
                )
                
                if (updated_deployment.status.ready_replicas == updated_deployment.spec.replicas and
                    updated_deployment.status.updated_replicas == updated_deployment.spec.replicas):
                    
                    return {
                        'status': 'success',
                        'component': component,
                        'old_image': old_image,
                        'new_image': new_image,
                        'update_duration': time.time() - start_time
                    }
                
                await asyncio.sleep(5)
            
            return {
                'status': 'timeout',
                'component': component,
                'error': f"Rolling update timed out after {max_wait}s"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'component': component,
                'error': str(e)
            }

    async def generate_container_report(self) -> Dict[str, Any]:
        """Generate comprehensive container health and metrics report."""
        container_statuses = await self.monitor_container_health()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'cluster': self.hydatis_config['cluster_name'],
            'namespace': self.hydatis_config['namespace'],
            'total_containers': len(container_statuses),
            'healthy_containers': len([s for s in container_statuses.values() if s.health == ContainerHealth.HEALTHY]),
            'unhealthy_containers': len([s for s in container_statuses.values() if s.health == ContainerHealth.UNHEALTHY]),
            'container_details': {},
            'resource_summary': {},
            'recommendations': []
        }
        
        total_cpu = 0
        total_memory = 0
        total_restarts = 0
        
        for container_name, status in container_statuses.items():
            if status.metrics:
                total_cpu += status.metrics.cpu_usage_percent
                total_memory += status.metrics.memory_usage_mb
                total_restarts += status.metrics.restart_count
            
            report['container_details'][container_name] = {
                'health': status.health.value,
                'pod_name': status.pod_name,
                'node_name': status.node_name,
                'metrics': status.metrics.__dict__ if status.metrics else None,
                'error_count': len(status.error_logs),
                'last_restart': status.last_restart.isoformat() if status.last_restart else None
            }
        
        report['resource_summary'] = {
            'total_cpu_usage_percent': total_cpu,
            'total_memory_usage_mb': total_memory,
            'total_restart_count': total_restarts,
            'average_cpu_per_container': total_cpu / len(container_statuses) if container_statuses else 0,
            'average_memory_per_container': total_memory / len(container_statuses) if container_statuses else 0
        }
        
        if report['unhealthy_containers'] > 0:
            report['recommendations'].append("Investigate unhealthy containers and consider restarting")
        
        if total_restarts > 10:
            report['recommendations'].append("High restart count detected - review resource limits and health checks")
        
        if total_cpu / len(container_statuses) > 70:
            report['recommendations'].append("Consider scaling up high CPU usage components")
        
        return report

async def main():
    """Main container management entry point."""
    manager = ContainerManager()
    
    try:
        log_info = manager.logger.info
        
        log_info("Starting container health monitoring...")
        
        container_statuses = await manager.monitor_container_health()
        log_info(f"Monitored {len(container_statuses)} containers")
        
        unhealthy_containers = [
            name for name, status in container_statuses.items()
            if status.health == ContainerHealth.UNHEALTHY
        ]
        
        if unhealthy_containers:
            log_info(f"Found {len(unhealthy_containers)} unhealthy containers")
            restart_results = await manager.restart_unhealthy_containers()
            log_info(f"Restart operations completed: {restart_results}")
        
        scaling_results = await manager.auto_scale_based_on_metrics()
        if scaling_results:
            log_info(f"Auto-scaling completed: {scaling_results}")
        
        report = await manager.generate_container_report()
        
        with open('/tmp/hydatis_container_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        log_info("Container management report saved to /tmp/hydatis_container_report.json")
        
    except Exception as e:
        manager.logger.error(f"Container management failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())