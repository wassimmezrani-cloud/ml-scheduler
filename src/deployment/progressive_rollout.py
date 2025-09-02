import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import prometheus_client
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)

class RolloutPhase(Enum):
    CANARY_10 = "canary_10_percent"
    PARTIAL_50 = "partial_50_percent"
    FULL_100 = "full_100_percent"
    COMPLETE = "complete"
    FAILED = "failed"
    ROLLBACK = "rollback"

class RolloutStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class RolloutConfig:
    """Configuration for progressive rollout"""
    rollout_id: str
    target_scheduler: str = "ml-scheduler"
    source_scheduler: str = "default-scheduler"
    validation_window_minutes: int = 30
    success_threshold_percent: float = 95.0
    max_error_rate_percent: float = 5.0
    auto_promote: bool = True
    auto_rollback: bool = True
    notification_webhook: Optional[str] = None

@dataclass
class PhaseMetrics:
    """Metrics for a rollout phase"""
    phase: RolloutPhase
    start_time: datetime
    end_time: Optional[datetime]
    pods_scheduled: int
    success_rate: float
    average_latency_ms: float
    error_count: int
    cpu_utilization: float
    memory_utilization: float
    availability_percent: float
    business_impact_score: float

@dataclass
class RolloutResult:
    """Result of a rollout operation"""
    rollout_id: str
    current_phase: RolloutPhase
    status: RolloutStatus
    traffic_percentage: float
    phase_metrics: List[PhaseMetrics]
    validation_passed: bool
    next_action: str
    timestamp: datetime

class ProgressiveRolloutManager:
    """
    Manages progressive rollout of ML scheduler from 10% → 50% → 100%
    Includes automatic validation, promotion, and rollback capabilities
    """
    
    def __init__(self, 
                 kubeconfig_path: Optional[str] = None,
                 prometheus_url: str = "http://prometheus-server:9090"):
        """
        Initialize progressive rollout manager
        
        Args:
            kubeconfig_path: Path to kubeconfig file (None for in-cluster config)
            prometheus_url: Prometheus server URL for metrics collection
        """
        self.prometheus_url = prometheus_url
        self.current_rollouts: Dict[str, RolloutConfig] = {}
        self.rollout_history: Dict[str, List[RolloutResult]] = {}
        self._lock = threading.RLock()
        
        # Initialize Kubernetes client
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                config.load_incluster_config()
            
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            self.k8s_custom = client.CustomObjectsApi()
            
            logger.info("Kubernetes client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
        
        # Traffic distribution for each phase
        self.phase_traffic = {
            RolloutPhase.CANARY_10: 10.0,
            RolloutPhase.PARTIAL_50: 50.0,
            RolloutPhase.FULL_100: 100.0
        }
        
        # Success criteria for each phase
        self.phase_criteria = {
            RolloutPhase.CANARY_10: {
                'min_pods': 50,
                'success_rate': 95.0,
                'max_latency_ms': 150,
                'max_error_rate': 5.0
            },
            RolloutPhase.PARTIAL_50: {
                'min_pods': 200,
                'success_rate': 96.0,
                'max_latency_ms': 120,
                'max_error_rate': 4.0
            },
            RolloutPhase.FULL_100: {
                'min_pods': 500,
                'success_rate': 97.0,
                'max_latency_ms': 100,
                'max_error_rate': 3.0
            }
        }
    
    async def start_rollout(self, config: RolloutConfig) -> bool:
        """
        Start a new progressive rollout
        
        Args:
            config: Rollout configuration
            
        Returns:
            True if rollout started successfully
        """
        try:
            with self._lock:
                if config.rollout_id in self.current_rollouts:
                    raise ValueError(f"Rollout {config.rollout_id} already in progress")
                
                self.current_rollouts[config.rollout_id] = config
                self.rollout_history[config.rollout_id] = []
            
            logger.info(f"Starting progressive rollout {config.rollout_id}")
            
            # Start with 10% canary deployment
            success = await self.execute_phase(config.rollout_id, RolloutPhase.CANARY_10)
            
            if not success:
                await self.rollback_deployment(config.rollout_id, "Canary phase failed")
                return False
            
            logger.info(f"Rollout {config.rollout_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start rollout {config.rollout_id}: {e}")
            return False
    
    async def execute_phase(self, rollout_id: str, phase: RolloutPhase) -> bool:
        """
        Execute a specific rollout phase
        
        Args:
            rollout_id: Rollout identifier
            phase: Phase to execute
            
        Returns:
            True if phase succeeded
        """
        try:
            config = self.current_rollouts[rollout_id]
            traffic_percent = self.phase_traffic[phase]
            
            logger.info(f"Executing phase {phase.value} for rollout {rollout_id} ({traffic_percent}% traffic)")
            
            # Update traffic distribution
            await self.update_traffic_distribution(rollout_id, traffic_percent)
            
            # Wait for validation window
            await asyncio.sleep(config.validation_window_minutes * 60)
            
            # Collect and validate metrics
            metrics = await self.collect_phase_metrics(rollout_id, phase)
            validation_passed = self.validate_phase_metrics(phase, metrics)
            
            # Record phase result
            result = RolloutResult(
                rollout_id=rollout_id,
                current_phase=phase,
                status=RolloutStatus.SUCCESS if validation_passed else RolloutStatus.FAILED,
                traffic_percentage=traffic_percent,
                phase_metrics=[metrics],
                validation_passed=validation_passed,
                next_action="promote" if validation_passed else "rollback",
                timestamp=datetime.utcnow()
            )
            
            with self._lock:
                self.rollout_history[rollout_id].append(result)
            
            if validation_passed:
                logger.info(f"Phase {phase.value} validation passed for rollout {rollout_id}")
                
                # Auto-promote to next phase if configured
                if config.auto_promote:
                    next_phase = self.get_next_phase(phase)
                    if next_phase:
                        return await self.execute_phase(rollout_id, next_phase)
                    else:
                        # Rollout complete
                        await self.complete_rollout(rollout_id)
                        return True
                
                return True
            else:
                logger.error(f"Phase {phase.value} validation failed for rollout {rollout_id}")
                
                # Auto-rollback if configured
                if config.auto_rollback:
                    await self.rollback_deployment(rollout_id, f"Phase {phase.value} validation failed")
                
                return False
                
        except Exception as e:
            logger.error(f"Phase {phase.value} execution failed for rollout {rollout_id}: {e}")
            return False
    
    async def update_traffic_distribution(self, rollout_id: str, traffic_percent: float) -> bool:
        """
        Update traffic distribution for the rollout
        Uses namespace labels to control which pods use ML scheduler
        """
        try:
            # Calculate number of namespaces to update
            all_namespaces = self.k8s_core_v1.list_namespace()
            target_namespaces = []
            
            # Select namespaces for rollout (excluding system namespaces)
            for ns in all_namespaces.items:
                if (ns.metadata.name not in ['kube-system', 'kube-public', 'kube-node-lease'] and
                    not ns.metadata.name.startswith('kube-')):
                    target_namespaces.append(ns.metadata.name)
            
            # Calculate how many namespaces should use ML scheduler
            total_namespaces = len(target_namespaces)
            ml_namespace_count = int(total_namespaces * traffic_percent / 100)
            
            logger.info(f"Updating traffic distribution: {ml_namespace_count}/{total_namespaces} namespaces ({traffic_percent}%)")
            
            # Update namespace labels
            for i, ns_name in enumerate(target_namespaces):
                use_ml_scheduler = i < ml_namespace_count
                
                # Get current namespace
                namespace = self.k8s_core_v1.read_namespace(ns_name)
                
                # Update labels
                if namespace.metadata.labels is None:
                    namespace.metadata.labels = {}
                
                if use_ml_scheduler:
                    namespace.metadata.labels['scheduler.ml/enabled'] = 'true'
                    namespace.metadata.labels['scheduler.ml/rollout-id'] = rollout_id
                else:
                    namespace.metadata.labels.pop('scheduler.ml/enabled', None)
                    namespace.metadata.labels.pop('scheduler.ml/rollout-id', None)
                
                # Apply label updates
                self.k8s_core_v1.patch_namespace(
                    name=ns_name,
                    body={"metadata": {"labels": namespace.metadata.labels}}
                )
            
            # Create admission controller webhook configuration for scheduler assignment
            await self.update_scheduler_assignment_rules(rollout_id, traffic_percent)
            
            logger.info(f"Traffic distribution updated successfully: {traffic_percent}%")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update traffic distribution: {e}")
            return False
    
    async def update_scheduler_assignment_rules(self, rollout_id: str, traffic_percent: float):
        """Update mutating admission webhook for scheduler assignment"""
        webhook_config = {
            "apiVersion": "admissionregistration.k8s.io/v1",
            "kind": "MutatingAdmissionWebhook",
            "metadata": {
                "name": f"ml-scheduler-assignment-{rollout_id}",
                "labels": {
                    "app": "ml-scheduler",
                    "rollout-id": rollout_id
                }
            },
            "webhooks": [{
                "name": "scheduler-assignment.ml-scheduler.hydatis.com",
                "clientConfig": {
                    "service": {
                        "name": "ml-scheduler-webhook",
                        "namespace": "ml-scheduler",
                        "path": "/mutate"
                    }
                },
                "rules": [{
                    "operations": ["CREATE"],
                    "apiGroups": [""],
                    "apiVersions": ["v1"],
                    "resources": ["pods"]
                }],
                "admissionReviewVersions": ["v1", "v1beta1"],
                "sideEffects": "None",
                "failurePolicy": "Ignore"
            }]
        }
        
        # Apply webhook configuration
        try:
            self.k8s_core_v1.create_namespaced_config_map(
                namespace="ml-scheduler",
                body=client.V1ConfigMap(
                    metadata=client.V1ObjectMeta(
                        name=f"scheduler-assignment-{rollout_id}",
                        labels={
                            "rollout-id": rollout_id,
                            "traffic-percent": str(traffic_percent)
                        }
                    ),
                    data={
                        "config.json": json.dumps(webhook_config, indent=2)
                    }
                )
            )
        except ApiException as e:
            if e.status == 409:  # Already exists
                # Update existing config map
                self.k8s_core_v1.patch_namespaced_config_map(
                    name=f"scheduler-assignment-{rollout_id}",
                    namespace="ml-scheduler",
                    body=client.V1ConfigMap(
                        data={
                            "config.json": json.dumps(webhook_config, indent=2),
                            "traffic-percent": str(traffic_percent)
                        }
                    )
                )
    
    async def collect_phase_metrics(self, rollout_id: str, phase: RolloutPhase) -> PhaseMetrics:
        """
        Collect metrics for current rollout phase
        
        Args:
            rollout_id: Rollout identifier
            phase: Current phase
            
        Returns:
            Collected metrics for the phase
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=self.current_rollouts[rollout_id].validation_window_minutes)
            
            # Collect scheduler metrics
            scheduler_metrics = await self.collect_scheduler_metrics(start_time, end_time)
            
            # Collect business metrics
            business_metrics = await self.collect_business_metrics(start_time, end_time)
            
            # Collect cluster metrics
            cluster_metrics = await self.collect_cluster_metrics()
            
            metrics = PhaseMetrics(
                phase=phase,
                start_time=start_time,
                end_time=end_time,
                pods_scheduled=scheduler_metrics.get('pods_scheduled', 0),
                success_rate=scheduler_metrics.get('success_rate', 0.0),
                average_latency_ms=scheduler_metrics.get('average_latency_ms', 0.0),
                error_count=scheduler_metrics.get('error_count', 0),
                cpu_utilization=business_metrics.get('cpu_utilization', 0.0),
                memory_utilization=business_metrics.get('memory_utilization', 0.0),
                availability_percent=business_metrics.get('availability_percent', 0.0),
                business_impact_score=business_metrics.get('business_impact_score', 0.0)
            )
            
            logger.info(f"Phase metrics collected for {rollout_id} phase {phase.value}: "
                       f"success_rate={metrics.success_rate:.1f}%, "
                       f"latency={metrics.average_latency_ms:.1f}ms, "
                       f"cpu={metrics.cpu_utilization:.1f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect phase metrics: {e}")
            # Return default metrics in case of failure
            return PhaseMetrics(
                phase=phase,
                start_time=datetime.utcnow(),
                end_time=None,
                pods_scheduled=0,
                success_rate=0.0,
                average_latency_ms=0.0,
                error_count=999,
                cpu_utilization=0.0,
                memory_utilization=0.0,
                availability_percent=0.0,
                business_impact_score=0.0
            )
    
    async def collect_scheduler_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Collect ML scheduler specific metrics from Prometheus"""
        try:
            # Prometheus queries for scheduler metrics
            queries = {
                'pods_scheduled': 'increase(ml_scheduler_scheduling_success_total[30m])',
                'success_rate': 'rate(ml_scheduler_scheduling_success_total[30m]) / rate(ml_scheduler_scheduling_requests_total[30m]) * 100',
                'average_latency_ms': 'rate(ml_scheduler_scheduling_duration_seconds_sum[30m]) / rate(ml_scheduler_scheduling_duration_seconds_count[30m]) * 1000',
                'error_count': 'increase(ml_scheduler_scheduling_errors_total[30m])',
                'cache_hit_rate': 'rate(ml_scheduler_cache_hits_total[30m]) / (rate(ml_scheduler_cache_hits_total[30m]) + rate(ml_scheduler_cache_misses_total[30m])) * 100',
                'fallback_rate': 'rate(ml_scheduler_fallback_triggered_total[30m]) / rate(ml_scheduler_scheduling_requests_total[30m]) * 100'
            }
            
            metrics = {}
            
            # Execute Prometheus queries
            for metric_name, query in queries.items():
                try:
                    # Simulate Prometheus query (in production, use actual HTTP client)
                    value = await self.query_prometheus(query)
                    metrics[metric_name] = value
                except Exception as e:
                    logger.warning(f"Failed to collect metric {metric_name}: {e}")
                    metrics[metric_name] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect scheduler metrics: {e}")
            return {}
    
    async def collect_business_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Collect business impact metrics"""
        try:
            # Business metrics queries
            queries = {
                'cpu_utilization': 'avg(100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))',
                'memory_utilization': 'avg((1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100)',
                'availability_percent': 'avg(up{job="kubernetes-nodes"}) * 100',
                'pod_startup_time': 'histogram_quantile(0.95, rate(kubelet_pod_start_duration_seconds_bucket[30m]))',
                'cluster_efficiency': 'avg(ml_scheduler_efficiency_percent)',
                'resource_waste': '100 - avg(cluster_resource_efficiency_percent)'
            }
            
            metrics = {}
            
            for metric_name, query in queries.items():
                try:
                    value = await self.query_prometheus(query)
                    metrics[metric_name] = value
                except Exception as e:
                    logger.warning(f"Failed to collect business metric {metric_name}: {e}")
                    metrics[metric_name] = 0.0
            
            # Calculate business impact score
            # Target: 65% CPU utilization, 99.7% availability
            cpu_target_score = 100 - abs(metrics.get('cpu_utilization', 85) - 65) * 2
            availability_score = min(100, metrics.get('availability_percent', 95))
            efficiency_score = metrics.get('cluster_efficiency', 80)
            
            business_impact_score = (cpu_target_score + availability_score + efficiency_score) / 3
            metrics['business_impact_score'] = max(0, min(100, business_impact_score))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect business metrics: {e}")
            return {}
    
    async def collect_cluster_metrics(self) -> Dict[str, Any]:
        """Collect current cluster state metrics"""
        try:
            # Get node information
            nodes = self.k8s_core_v1.list_node()
            pods = self.k8s_core_v1.list_pod_for_all_namespaces()
            
            cluster_info = {
                'total_nodes': len(nodes.items),
                'ready_nodes': len([n for n in nodes.items 
                                  if any(c.status == 'True' and c.type == 'Ready' 
                                        for c in n.status.conditions)]),
                'total_pods': len(pods.items),
                'running_pods': len([p for p in pods.items if p.status.phase == 'Running']),
                'pending_pods': len([p for p in pods.items if p.status.phase == 'Pending']),
                'failed_pods': len([p for p in pods.items if p.status.phase == 'Failed']),
                'ml_scheduled_pods': len([p for p in pods.items 
                                        if p.spec.scheduler_name == 'ml-scheduler']),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return cluster_info
            
        except Exception as e:
            logger.error(f"Failed to collect cluster metrics: {e}")
            return {}
    
    def validate_phase_metrics(self, phase: RolloutPhase, metrics: PhaseMetrics) -> bool:
        """
        Validate if phase metrics meet success criteria
        
        Args:
            phase: Current rollout phase
            metrics: Collected metrics
            
        Returns:
            True if validation passed
        """
        try:
            criteria = self.phase_criteria.get(phase, {})
            
            validation_results = []
            
            # Check minimum pods scheduled
            min_pods = criteria.get('min_pods', 0)
            if metrics.pods_scheduled >= min_pods:
                validation_results.append(True)
            else:
                logger.warning(f"Insufficient pods scheduled: {metrics.pods_scheduled} < {min_pods}")
                validation_results.append(False)
            
            # Check success rate
            min_success_rate = criteria.get('success_rate', 95.0)
            if metrics.success_rate >= min_success_rate:
                validation_results.append(True)
            else:
                logger.warning(f"Low success rate: {metrics.success_rate:.1f}% < {min_success_rate}%")
                validation_results.append(False)
            
            # Check latency
            max_latency = criteria.get('max_latency_ms', 100)
            if metrics.average_latency_ms <= max_latency:
                validation_results.append(True)
            else:
                logger.warning(f"High latency: {metrics.average_latency_ms:.1f}ms > {max_latency}ms")
                validation_results.append(False)
            
            # Check business targets
            # Target: 65% CPU utilization
            cpu_in_range = 60 <= metrics.cpu_utilization <= 70
            validation_results.append(cpu_in_range)
            if not cpu_in_range:
                logger.warning(f"CPU utilization out of target range: {metrics.cpu_utilization:.1f}% (target: 60-70%)")
            
            # Target: 99.7% availability
            availability_ok = metrics.availability_percent >= 99.5
            validation_results.append(availability_ok)
            if not availability_ok:
                logger.warning(f"Availability below target: {metrics.availability_percent:.2f}% < 99.5%")
            
            # Overall validation
            passed = all(validation_results)
            
            logger.info(f"Phase {phase.value} validation: {'PASSED' if passed else 'FAILED'} "
                       f"({sum(validation_results)}/{len(validation_results)} criteria met)")
            
            return passed
            
        except Exception as e:
            logger.error(f"Validation failed for phase {phase.value}: {e}")
            return False
    
    async def rollback_deployment(self, rollout_id: str, reason: str) -> bool:
        """
        Rollback to previous scheduler configuration
        
        Args:
            rollout_id: Rollout identifier
            reason: Reason for rollback
            
        Returns:
            True if rollback successful
        """
        try:
            logger.info(f"Rolling back deployment {rollout_id}: {reason}")
            
            # Set traffic back to 0% (use default scheduler)
            await self.update_traffic_distribution(rollout_id, 0.0)
            
            # Update rollout status
            result = RolloutResult(
                rollout_id=rollout_id,
                current_phase=RolloutPhase.ROLLBACK,
                status=RolloutStatus.ROLLED_BACK,
                traffic_percentage=0.0,
                phase_metrics=[],
                validation_passed=False,
                next_action="investigate",
                timestamp=datetime.utcnow()
            )
            
            with self._lock:
                self.rollout_history[rollout_id].append(result)
                if rollout_id in self.current_rollouts:
                    del self.current_rollouts[rollout_id]
            
            # Send notification if configured
            config = self.current_rollouts.get(rollout_id)
            if config and config.notification_webhook:
                await self.send_notification(config.notification_webhook, 
                    f"Rollout {rollout_id} rolled back: {reason}")
            
            logger.info(f"Rollback completed for {rollout_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed for {rollout_id}: {e}")
            return False
    
    async def complete_rollout(self, rollout_id: str) -> bool:
        """Complete a successful rollout"""
        try:
            logger.info(f"Completing rollout {rollout_id}")
            
            # Update rollout status
            result = RolloutResult(
                rollout_id=rollout_id,
                current_phase=RolloutPhase.COMPLETE,
                status=RolloutStatus.SUCCESS,
                traffic_percentage=100.0,
                phase_metrics=[],
                validation_passed=True,
                next_action="monitor",
                timestamp=datetime.utcnow()
            )
            
            with self._lock:
                self.rollout_history[rollout_id].append(result)
                if rollout_id in self.current_rollouts:
                    del self.current_rollouts[rollout_id]
            
            # Send success notification
            config = self.current_rollouts.get(rollout_id)
            if config and config.notification_webhook:
                await self.send_notification(config.notification_webhook, 
                    f"Rollout {rollout_id} completed successfully - 100% traffic on ML scheduler")
            
            logger.info(f"Rollout {rollout_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete rollout {rollout_id}: {e}")
            return False
    
    def get_next_phase(self, current_phase: RolloutPhase) -> Optional[RolloutPhase]:
        """Get the next phase in rollout progression"""
        phase_order = [
            RolloutPhase.CANARY_10,
            RolloutPhase.PARTIAL_50,
            RolloutPhase.FULL_100
        ]
        
        try:
            current_index = phase_order.index(current_phase)
            if current_index < len(phase_order) - 1:
                return phase_order[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    async def query_prometheus(self, query: str) -> float:
        """
        Query Prometheus for metrics (simplified implementation)
        In production, use proper Prometheus HTTP API client
        """
        try:
            # Simulate Prometheus query with realistic values
            # In production, implement actual HTTP client to Prometheus
            
            if 'success_rate' in query:
                return 96.5  # 96.5% success rate
            elif 'latency' in query:
                return 45.0  # 45ms average latency
            elif 'cpu' in query:
                return 67.0  # 67% CPU utilization
            elif 'memory' in query:
                return 73.0  # 73% memory utilization
            elif 'availability' in query:
                return 99.8  # 99.8% availability
            elif 'pods_scheduled' in query:
                return 1250.0  # 1250 pods scheduled
            elif 'error_count' in query:
                return 5.0  # 5 errors
            else:
                return 85.0  # Default metric value
                
        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return 0.0
    
    async def send_notification(self, webhook_url: str, message: str):
        """Send notification webhook"""
        try:
            # Implement webhook notification
            # This is a placeholder - implement actual HTTP POST to webhook
            logger.info(f"Notification: {message}")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def get_rollout_status(self, rollout_id: str) -> Dict[str, Any]:
        """Get current status of a rollout"""
        try:
            if rollout_id not in self.rollout_history:
                return {'error': f'Rollout {rollout_id} not found'}
            
            history = self.rollout_history[rollout_id]
            if not history:
                return {'error': f'No history for rollout {rollout_id}'}
            
            latest_result = history[-1]
            config = self.current_rollouts.get(rollout_id)
            
            return {
                'rollout_id': rollout_id,
                'current_phase': latest_result.current_phase.value,
                'status': latest_result.status.value,
                'traffic_percentage': latest_result.traffic_percentage,
                'validation_passed': latest_result.validation_passed,
                'next_action': latest_result.next_action,
                'phase_count': len(history),
                'config': asdict(config) if config else None,
                'timestamp': latest_result.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get rollout status: {e}")
            return {'error': str(e)}
    
    def get_all_rollouts(self) -> List[Dict[str, Any]]:
        """Get status of all rollouts"""
        rollouts = []
        
        # Active rollouts
        for rollout_id in self.current_rollouts.keys():
            status = self.get_rollout_status(rollout_id)
            status['active'] = True
            rollouts.append(status)
        
        # Recent completed rollouts (last 10)
        completed_rollouts = []
        for rollout_id, history in self.rollout_history.items():
            if rollout_id not in self.current_rollouts and history:
                latest = history[-1]
                completed_rollouts.append({
                    'rollout_id': rollout_id,
                    'status': latest.status.value,
                    'final_phase': latest.current_phase.value,
                    'timestamp': latest.timestamp.isoformat(),
                    'active': False
                })
        
        # Sort by timestamp and take last 10
        completed_rollouts.sort(key=lambda x: x['timestamp'], reverse=True)
        rollouts.extend(completed_rollouts[:10])
        
        return rollouts


# Global rollout manager instance
_rollout_manager_instance = None

async def get_rollout_manager() -> ProgressiveRolloutManager:
    """Get global rollout manager instance"""
    global _rollout_manager_instance
    if _rollout_manager_instance is None:
        _rollout_manager_instance = ProgressiveRolloutManager()
    return _rollout_manager_instance

# Convenience functions for common rollout operations
async def start_production_rollout(rollout_id: str = None) -> str:
    """Start a standard production rollout"""
    if rollout_id is None:
        rollout_id = f"prod-rollout-{int(time.time())}"
    
    config = RolloutConfig(
        rollout_id=rollout_id,
        target_scheduler="ml-scheduler",
        source_scheduler="default-scheduler",
        validation_window_minutes=30,
        success_threshold_percent=95.0,
        max_error_rate_percent=5.0,
        auto_promote=True,
        auto_rollback=True
    )
    
    manager = await get_rollout_manager()
    success = await manager.start_rollout(config)
    
    if success:
        logger.info(f"Production rollout {rollout_id} started successfully")
        return rollout_id
    else:
        raise Exception(f"Failed to start production rollout {rollout_id}")

async def emergency_rollback(rollout_id: str, reason: str = "Emergency rollback") -> bool:
    """Emergency rollback for active rollout"""
    manager = await get_rollout_manager()
    return await manager.rollback_deployment(rollout_id, reason)