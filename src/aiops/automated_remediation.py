"""
AIOps Automated Remediation System for HYDATIS Cluster
Implements intelligent automated incident response and self-healing capabilities
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import yaml
from kubernetes import client, config as k8s_config
from prometheus_client import Counter, Histogram, Gauge
import uuid

logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RemediationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class RemediationType(Enum):
    SCALING = "scaling"
    RESTART = "restart"
    CONFIGURATION_CHANGE = "configuration_change"
    TRAFFIC_REROUTING = "traffic_rerouting"
    FALLBACK_ACTIVATION = "fallback_activation"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    CACHE_CLEANUP = "cache_cleanup"
    MODEL_RELOAD = "model_reload"

@dataclass
class IncidentContext:
    """Context information for an incident"""
    incident_id: str
    alert_name: str
    severity: IncidentSeverity
    affected_component: str
    metrics: Dict[str, float]
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RemediationAction:
    """Automated remediation action"""
    action_id: str
    type: RemediationType
    description: str
    command: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    rollback_command: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

@dataclass
class RemediationResult:
    """Result of a remediation action"""
    action_id: str
    status: RemediationStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    output: str = ""
    error: Optional[str] = None
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    rollback_performed: bool = False

class KubernetesRemediator:
    """Kubernetes-specific remediation actions"""
    
    def __init__(self):
        # Load Kubernetes config
        try:
            k8s_config.load_incluster_config()
        except:
            k8s_config.load_kube_config()
            
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.autoscaling_v1 = client.AutoscalingV1Api()
        
    async def scale_deployment(self, namespace: str, deployment_name: str, 
                             replicas: int) -> Dict[str, Any]:
        """Scale a deployment to specified replica count"""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name, namespace=namespace)
            
            current_replicas = deployment.spec.replicas
            
            # Update replica count
            deployment.spec.replicas = replicas
            
            # Apply the change
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Scaled {deployment_name} from {current_replicas} to {replicas} replicas")
            
            return {
                'success': True,
                'previous_replicas': current_replicas,
                'new_replicas': replicas,
                'action': 'scale_deployment'
            }
            
        except Exception as e:
            logger.error(f"Failed to scale deployment {deployment_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'scale_deployment'
            }
    
    async def restart_deployment(self, namespace: str, deployment_name: str) -> Dict[str, Any]:
        """Restart a deployment by updating annotation"""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name, namespace=namespace)
            
            # Add restart annotation
            if not deployment.spec.template.metadata.annotations:
                deployment.spec.template.metadata.annotations = {}
                
            deployment.spec.template.metadata.annotations['kubectl.kubernetes.io/restartedAt'] = \
                datetime.utcnow().isoformat()
            
            # Apply the change
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Restarted deployment {deployment_name}")
            
            return {
                'success': True,
                'action': 'restart_deployment',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to restart deployment {deployment_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'restart_deployment'
            }
    
    async def drain_node(self, node_name: str) -> Dict[str, Any]:
        """Safely drain a problematic node"""
        try:
            # Mark node as unschedulable
            node = self.v1.read_node(name=node_name)
            node.spec.unschedulable = True
            
            self.v1.patch_node(name=node_name, body=node)
            
            # Wait for pods to be evicted (simplified - production would use kubectl drain)
            await asyncio.sleep(60)
            
            logger.info(f"Drained node {node_name}")
            
            return {
                'success': True,
                'action': 'drain_node',
                'node': node_name
            }
            
        except Exception as e:
            logger.error(f"Failed to drain node {node_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'drain_node'
            }
    
    async def update_scheduler_config(self, config_name: str, 
                                    new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update ML scheduler configuration"""
        try:
            # Get current ConfigMap
            config_map = self.v1.read_namespaced_config_map(
                name=config_name, namespace='ml-scheduler')
            
            # Update configuration
            if 'scheduler-config.yaml' in config_map.data:
                current_config = yaml.safe_load(config_map.data['scheduler-config.yaml'])
                current_config.update(new_config)
                config_map.data['scheduler-config.yaml'] = yaml.dump(current_config)
            
            # Apply the change
            self.v1.patch_namespaced_config_map(
                name=config_name,
                namespace='ml-scheduler',
                body=config_map
            )
            
            logger.info(f"Updated scheduler configuration {config_name}")
            
            return {
                'success': True,
                'action': 'update_scheduler_config',
                'config_name': config_name
            }
            
        except Exception as e:
            logger.error(f"Failed to update scheduler config {config_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'update_scheduler_config'
            }

class MLServiceRemediator:
    """ML service specific remediation actions"""
    
    def __init__(self, kserve_url: str, mlflow_url: str):
        self.kserve_url = kserve_url
        self.mlflow_url = mlflow_url
        self.k8s_remediator = KubernetesRemediator()
    
    async def reload_model(self, model_name: str, model_version: str = "latest") -> Dict[str, Any]:
        """Reload ML model to address performance issues"""
        try:
            # Get model configuration
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.kserve_url}/v1/models/{model_name}") as response:
                    if response.status != 200:
                        raise Exception(f"Model {model_name} not found")
                    
                    model_config = await response.json()
            
            # Update model version if specified
            if model_version != "latest":
                model_config['spec']['predictor']['model']['modelUri'] = \
                    f"models:/{model_name}/{model_version}"
            
            # Add reload annotation
            if 'annotations' not in model_config['metadata']:
                model_config['metadata']['annotations'] = {}
                
            model_config['metadata']['annotations']['ml-scheduler/reloaded-at'] = \
                datetime.utcnow().isoformat()
            
            # Apply the change
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.kserve_url}/v1/models/{model_name}",
                    json=model_config
                ) as response:
                    if response.status == 200:
                        logger.info(f"Reloaded model {model_name}")
                        return {
                            'success': True,
                            'action': 'reload_model',
                            'model_name': model_name,
                            'model_version': model_version
                        }
                    else:
                        raise Exception(f"Failed to reload model: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to reload model {model_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'reload_model'
            }
    
    async def adjust_model_traffic(self, model_name: str, traffic_percent: int) -> Dict[str, Any]:
        """Adjust traffic percentage to a model"""
        try:
            # Update KServe InferenceService traffic configuration
            traffic_config = {
                'apiVersion': 'serving.kserve.io/v1beta1',
                'kind': 'InferenceService',
                'metadata': {
                    'name': model_name,
                    'annotations': {
                        'ml-scheduler/traffic-adjusted-at': datetime.utcnow().isoformat(),
                        'ml-scheduler/traffic-reason': 'automated_remediation'
                    }
                },
                'spec': {
                    'predictor': {
                        'canary': {
                            'trafficPercent': traffic_percent
                        }
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    f"{self.kserve_url}/v1/namespaces/ml-scheduler/inferenceservices/{model_name}",
                    json=traffic_config
                ) as response:
                    if response.status == 200:
                        logger.info(f"Adjusted {model_name} traffic to {traffic_percent}%")
                        return {
                            'success': True,
                            'action': 'adjust_model_traffic',
                            'model_name': model_name,
                            'traffic_percent': traffic_percent
                        }
                    else:
                        raise Exception(f"Failed to adjust traffic: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to adjust traffic for {model_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'adjust_model_traffic'
            }
    
    async def activate_fallback_mode(self, component: str) -> Dict[str, Any]:
        """Activate fallback mode for specific component"""
        try:
            # Update scheduler configuration to enable fallback
            fallback_config = {
                'fallback': {
                    'enabled': True,
                    'component': component,
                    'activated_at': datetime.utcnow().isoformat(),
                    'reason': 'automated_remediation'
                }
            }
            
            result = await self.k8s_remediator.update_scheduler_config(
                'ml-scheduler-config', fallback_config)
            
            if result['success']:
                logger.info(f"Activated fallback mode for {component}")
                return {
                    'success': True,
                    'action': 'activate_fallback_mode',
                    'component': component
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Failed to activate fallback mode for {component}: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'activate_fallback_mode'
            }

class CacheRemediator:
    """Redis cache remediation actions"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
    
    async def clear_cache_keys(self, pattern: str) -> Dict[str, Any]:
        """Clear Redis cache keys matching pattern"""
        try:
            # Connect to Redis and clear keys
            import redis.asyncio as redis
            
            redis_client = redis.from_url(self.redis_url)
            keys = await redis_client.keys(pattern)
            
            if keys:
                deleted_count = await redis_client.delete(*keys)
                logger.info(f"Cleared {deleted_count} cache keys matching {pattern}")
                
                return {
                    'success': True,
                    'action': 'clear_cache_keys',
                    'pattern': pattern,
                    'deleted_count': deleted_count
                }
            else:
                return {
                    'success': True,
                    'action': 'clear_cache_keys',
                    'pattern': pattern,
                    'deleted_count': 0
                }
                
        except Exception as e:
            logger.error(f"Failed to clear cache keys {pattern}: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'clear_cache_keys'
            }
    
    async def restart_redis(self) -> Dict[str, Any]:
        """Restart Redis service"""
        try:
            k8s_remediator = KubernetesRemediator()
            
            # Restart Redis deployment
            result = await k8s_remediator.restart_deployment('ml-scheduler', 'redis')
            
            if result['success']:
                # Wait for Redis to be ready
                await asyncio.sleep(30)
                
                # Verify Redis is responding
                import redis.asyncio as redis
                redis_client = redis.from_url(self.redis_url)
                await redis_client.ping()
                
                logger.info("Redis restart completed successfully")
                return {
                    'success': True,
                    'action': 'restart_redis'
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Failed to restart Redis: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'restart_redis'
            }

class RemediationEngine:
    """Main remediation engine that orchestrates automated responses"""
    
    def __init__(self, config_path: str, prometheus_url: str, 
                 kserve_url: str, redis_url: str):
        self.config = self._load_config(config_path)
        self.prometheus_url = prometheus_url
        
        # Initialize remediators
        self.k8s_remediator = KubernetesRemediator()
        self.ml_remediator = MLServiceRemediator(kserve_url, redis_url)
        self.cache_remediator = CacheRemediator(redis_url)
        
        # Active remediations tracking
        self.active_remediations: Dict[str, RemediationResult] = {}
        
        # Metrics
        self.remediations_counter = Counter('aiops_remediations_total',
                                          'Total automated remediations performed',
                                          ['type', 'status'])
        self.remediation_duration = Histogram('aiops_remediation_duration_seconds',
                                            'Time spent on automated remediations',
                                            ['type'])
        self.incidents_handled_counter = Counter('aiops_incidents_handled_total',
                                               'Total incidents handled by AIOps',
                                               ['severity', 'component'])
        self.success_rate_gauge = Gauge('aiops_remediation_success_rate',
                                      'Success rate of automated remediations',
                                      ['type'])
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load AIOps configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default AIOps configuration"""
        return {
            'remediation_rules': {
                'CPUUtilizationOutOfTarget': {
                    'severity_threshold': 'warning',
                    'actions': [
                        {
                            'type': 'configuration_change',
                            'description': 'Adjust ML scheduler aggressiveness',
                            'command': 'update_scheduler_aggressiveness',
                            'parameters': {'target_cpu': 65.0}
                        }
                    ]
                },
                'SchedulingLatencyHigh': {
                    'severity_threshold': 'warning',
                    'actions': [
                        {
                            'type': 'scaling',
                            'description': 'Scale ML predictor service',
                            'command': 'scale_ml_service',
                            'parameters': {'service': 'xgboost-predictor', 'replicas': 3}
                        },
                        {
                            'type': 'cache_cleanup',
                            'description': 'Clear stale cache entries',
                            'command': 'clear_cache',
                            'parameters': {'pattern': 'ml:predictions:*'}
                        }
                    ]
                },
                'MLModelServiceDown': {
                    'severity_threshold': 'critical',
                    'actions': [
                        {
                            'type': 'restart',
                            'description': 'Restart ML model service',
                            'command': 'restart_ml_service',
                            'parameters': {'service_name': 'from_alert_labels'}
                        },
                        {
                            'type': 'fallback_activation',
                            'description': 'Activate scheduler fallback mode',
                            'command': 'activate_fallback',
                            'parameters': {'component': 'ml_scoring'}
                        }
                    ]
                }
            },
            'global_settings': {
                'max_concurrent_remediations': 3,
                'remediation_timeout_seconds': 300,
                'cooldown_between_actions_seconds': 30,
                'max_retries': 2
            }
        }
    
    async def handle_incident(self, incident: IncidentContext) -> List[RemediationResult]:
        """Handle incident with automated remediation"""
        logger.info(f"Handling incident: {incident.alert_name} (severity: {incident.severity.value})")
        
        # Get remediation rules for this alert
        remediation_rules = self.config['remediation_rules'].get(incident.alert_name, {})
        
        if not remediation_rules:
            logger.info(f"No remediation rules found for {incident.alert_name}")
            return []
        
        # Check severity threshold
        severity_threshold = remediation_rules.get('severity_threshold', 'critical')
        if not self._should_remediate(incident.severity, severity_threshold):
            logger.info(f"Incident severity {incident.severity.value} below threshold {severity_threshold}")
            return []
        
        # Execute remediation actions
        actions = remediation_rules.get('actions', [])
        results = []
        
        self.incidents_handled_counter.labels(
            severity=incident.severity.value,
            component=incident.affected_component
        ).inc()
        
        for action_config in actions:
            result = await self._execute_remediation_action(incident, action_config)
            results.append(result)
            
            # If action failed and it's critical, stop further actions
            if result.status == RemediationStatus.FAILED and incident.severity == IncidentSeverity.CRITICAL:
                logger.error(f"Critical remediation failed for {incident.incident_id}, stopping further actions")
                break
            
            # Cooldown between actions
            await asyncio.sleep(self.config['global_settings']['cooldown_between_actions_seconds'])
        
        return results
    
    async def _execute_remediation_action(self, incident: IncidentContext,
                                        action_config: Dict[str, Any]) -> RemediationResult:
        """Execute a single remediation action"""
        action_id = str(uuid.uuid4())
        action = RemediationAction(
            action_id=action_id,
            type=RemediationType(action_config['type']),
            description=action_config['description'],
            command=action_config['command'],
            parameters=action_config.get('parameters', {}),
            timeout_seconds=action_config.get('timeout_seconds', 300)
        )
        
        result = RemediationResult(
            action_id=action_id,
            status=RemediationStatus.IN_PROGRESS,
            start_time=datetime.utcnow()
        )
        
        self.active_remediations[action_id] = result
        
        try:
            with self.remediation_duration.labels(type=action.type.value).time():
                # Collect metrics before remediation
                result.metrics_before = await self._collect_incident_metrics(incident)
                
                # Execute the action
                action_result = await self._dispatch_action(action, incident)
                
                if action_result.get('success', False):
                    result.status = RemediationStatus.COMPLETED
                    result.output = json.dumps(action_result)
                    
                    # Wait for metrics to stabilize
                    await asyncio.sleep(60)
                    
                    # Collect metrics after remediation
                    result.metrics_after = await self._collect_incident_metrics(incident)
                    
                    # Validate remediation success
                    if await self._validate_remediation_success(incident, result):
                        logger.info(f"Remediation {action_id} completed successfully")
                    else:
                        logger.warning(f"Remediation {action_id} completed but metrics not improved")
                        
                else:
                    result.status = RemediationStatus.FAILED
                    result.error = action_result.get('error', 'Unknown error')
                    
            result.end_time = datetime.utcnow()
            
            # Update metrics
            self.remediations_counter.labels(
                type=action.type.value, 
                status=result.status.value
            ).inc()
            
        except Exception as e:
            result.status = RemediationStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.utcnow()
            logger.error(f"Error executing remediation {action_id}: {e}")
        
        finally:
            # Clean up active remediations
            if action_id in self.active_remediations:
                del self.active_remediations[action_id]
        
        return result
    
    async def _dispatch_action(self, action: RemediationAction, 
                             incident: IncidentContext) -> Dict[str, Any]:
        """Dispatch remediation action to appropriate handler"""
        command = action.command
        params = action.parameters.copy()
        
        # Substitute incident context into parameters
        params = self._substitute_incident_context(params, incident)
        
        if command == 'scale_ml_service':
            service_name = params['service']
            replicas = params['replicas']
            return await self.k8s_remediator.scale_deployment('ml-scheduler', service_name, replicas)
            
        elif command == 'restart_ml_service':
            service_name = params['service_name']
            return await self.k8s_remediator.restart_deployment('ml-scheduler', service_name)
            
        elif command == 'update_scheduler_aggressiveness':
            target_cpu = params['target_cpu']
            config_update = {
                'scheduler': {
                    'aggressiveness': self._calculate_aggressiveness(target_cpu, incident.metrics)
                }
            }
            return await self.k8s_remediator.update_scheduler_config('ml-scheduler-config', config_update)
            
        elif command == 'activate_fallback':
            component = params['component']
            return await self.ml_remediator.activate_fallback_mode(component)
            
        elif command == 'clear_cache':
            pattern = params['pattern']
            return await self.cache_remediator.clear_cache_keys(pattern)
            
        elif command == 'reload_model':
            model_name = params.get('model_name', incident.affected_component)
            return await self.ml_remediator.reload_model(model_name)
            
        elif command == 'adjust_model_traffic':
            model_name = params.get('model_name', incident.affected_component)
            traffic_percent = params['traffic_percent']
            return await self.ml_remediator.adjust_model_traffic(model_name, traffic_percent)
            
        else:
            logger.error(f"Unknown remediation command: {command}")
            return {
                'success': False,
                'error': f'Unknown command: {command}',
                'action': 'dispatch_action'
            }
    
    def _substitute_incident_context(self, params: Dict[str, Any], 
                                   incident: IncidentContext) -> Dict[str, Any]:
        """Substitute incident context variables in parameters"""
        result = params.copy()
        
        for key, value in result.items():
            if isinstance(value, str):
                if value == 'from_alert_labels':
                    # Extract from incident labels
                    result[key] = incident.labels.get(key, incident.affected_component)
                elif value.startswith('${') and value.endswith('}'):
                    # Extract metric value
                    metric_name = value[2:-1]
                    result[key] = incident.metrics.get(metric_name, 0.0)
        
        return result
    
    def _calculate_aggressiveness(self, target_cpu: float, 
                                current_metrics: Dict[str, float]) -> float:
        """Calculate scheduler aggressiveness based on current CPU"""
        current_cpu = current_metrics.get('cpu_utilization', 85.0)
        
        if current_cpu < target_cpu:
            # Increase aggressiveness to reach target
            return min(1.0, 0.5 + (target_cpu - current_cpu) / 100.0)
        else:
            # Decrease aggressiveness to reduce CPU usage
            return max(0.1, 0.5 - (current_cpu - target_cpu) / 100.0)
    
    async def _collect_incident_metrics(self, incident: IncidentContext) -> Dict[str, float]:
        """Collect current metrics related to the incident"""
        metrics = {}
        
        try:
            # Define metric queries based on incident type
            metric_queries = {
                'cpu_utilization': 'avg(100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))',
                'availability': 'avg(up{job="kubernetes-nodes"}) * 100',
                'scheduling_latency': 'histogram_quantile(0.99, rate(ml_scheduler_scheduling_duration_seconds_bucket[5m])) * 1000',
                'success_rate': 'rate(ml_scheduler_scheduling_success_total[5m]) / rate(ml_scheduler_scheduling_requests_total[5m]) * 100',
                'cache_hit_rate': 'rate(redis_cache_hits_total[5m]) / (rate(redis_cache_hits_total[5m]) + rate(redis_cache_misses_total[5m])) * 100'
            }
            
            # Query Prometheus for current values
            async with aiohttp.ClientSession() as session:
                for metric_name, query in metric_queries.items():
                    try:
                        async with session.get(f"{self.prometheus_url}/api/v1/query",
                                             params={'query': query}) as response:
                            data = await response.json()
                            results = data.get('data', {}).get('result', [])
                            if results:
                                metrics[metric_name] = float(results[0]['value'][1])
                    except Exception as e:
                        logger.warning(f"Failed to collect metric {metric_name}: {e}")
                        metrics[metric_name] = 0.0
            
        except Exception as e:
            logger.error(f"Error collecting incident metrics: {e}")
        
        return metrics
    
    async def _validate_remediation_success(self, incident: IncidentContext,
                                          result: RemediationResult) -> bool:
        """Validate if remediation was successful"""
        # Compare before/after metrics
        before_metrics = result.metrics_before
        after_metrics = result.metrics_after
        
        # Define success criteria based on incident type
        if 'CPU' in incident.alert_name:
            before_cpu = before_metrics.get('cpu_utilization', 0)
            after_cpu = after_metrics.get('cpu_utilization', 0)
            target_cpu = 65.0
            
            # Success if CPU moved closer to target
            before_distance = abs(before_cpu - target_cpu)
            after_distance = abs(after_cpu - target_cpu)
            return after_distance < before_distance
            
        elif 'Latency' in incident.alert_name:
            before_latency = before_metrics.get('scheduling_latency', 0)
            after_latency = after_metrics.get('scheduling_latency', 0)
            
            # Success if latency decreased
            return after_latency < before_latency
            
        elif 'Availability' in incident.alert_name:
            before_availability = before_metrics.get('availability', 0)
            after_availability = after_metrics.get('availability', 0)
            
            # Success if availability increased
            return after_availability > before_availability
            
        elif 'Success' in incident.alert_name:
            before_success = before_metrics.get('success_rate', 0)
            after_success = after_metrics.get('success_rate', 0)
            
            # Success if success rate improved
            return after_success > before_success
        
        # Default: consider successful if no errors occurred
        return result.status == RemediationStatus.COMPLETED
    
    def _should_remediate(self, incident_severity: IncidentSeverity, 
                         threshold: str) -> bool:
        """Determine if incident meets remediation threshold"""
        severity_levels = {
            'info': 0,
            'warning': 1,
            'critical': 2,
            'emergency': 3
        }
        
        incident_level = severity_levels.get(incident_severity.value, 0)
        threshold_level = severity_levels.get(threshold, 2)
        
        return incident_level >= threshold_level

class IncidentCorrelator:
    """Correlates related incidents for holistic remediation"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        self.active_incidents: Dict[str, IncidentContext] = {}
        self.correlation_window_minutes = 15
    
    async def correlate_incidents(self, new_incident: IncidentContext) -> List[IncidentContext]:
        """Find correlated incidents within time window"""
        correlated = [new_incident]
        
        # Find incidents within correlation window
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.correlation_window_minutes)
        
        for incident_id, incident in self.active_incidents.items():
            if incident.timestamp >= cutoff_time:
                if self._are_incidents_correlated(new_incident, incident):
                    correlated.append(incident)
        
        # Add new incident to active tracking
        self.active_incidents[new_incident.incident_id] = new_incident
        
        # Clean up old incidents
        self._cleanup_old_incidents()
        
        return correlated
    
    def _are_incidents_correlated(self, incident1: IncidentContext, 
                                incident2: IncidentContext) -> bool:
        """Determine if two incidents are correlated"""
        # Same component correlation
        if incident1.affected_component == incident2.affected_component:
            return True
        
        # Related components correlation
        related_components = {
            'ml-scheduler': ['xgboost-predictor', 'qlearning-optimizer', 'anomaly-detector'],
            'xgboost-predictor': ['ml-scheduler', 'redis'],
            'qlearning-optimizer': ['ml-scheduler', 'redis'],
            'anomaly-detector': ['ml-scheduler', 'redis'],
            'redis': ['ml-scheduler', 'xgboost-predictor', 'qlearning-optimizer', 'anomaly-detector']
        }
        
        component1_related = related_components.get(incident1.affected_component, [])
        if incident2.affected_component in component1_related:
            return True
        
        # Cascade failure patterns
        cascade_patterns = [
            ['SchedulingLatencyHigh', 'MLModelLatencyHigh'],
            ['MLModelServiceDown', 'FallbackRateHigh'],
            ['CacheHitRateLow', 'SchedulingLatencyHigh'],
            ['CPUUtilizationOutOfTarget', 'SchedulingLatencyHigh']
        ]
        
        for pattern in cascade_patterns:
            if incident1.alert_name in pattern and incident2.alert_name in pattern:
                return True
        
        return False
    
    def _cleanup_old_incidents(self):
        """Remove incidents older than correlation window"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.correlation_window_minutes * 2)
        
        old_incidents = [
            incident_id for incident_id, incident in self.active_incidents.items()
            if incident.timestamp < cutoff_time
        ]
        
        for incident_id in old_incidents:
            del self.active_incidents[incident_id]

class SelfHealingOrchestrator:
    """Orchestrates self-healing workflows"""
    
    def __init__(self, config_path: str, prometheus_url: str, 
                 kserve_url: str, redis_url: str):
        self.remediation_engine = RemediationEngine(config_path, prometheus_url, kserve_url, redis_url)
        self.incident_correlator = IncidentCorrelator(prometheus_url)
        self.prometheus_url = prometheus_url
        
        # Self-healing metrics
        self.self_healing_cycles_counter = Counter('aiops_self_healing_cycles_total',
                                                 'Self-healing cycles executed')
        self.incidents_resolved_counter = Counter('aiops_incidents_resolved_total',
                                                'Incidents resolved by self-healing',
                                                ['resolution_type'])
    
    async def process_alert(self, alert_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming alert and trigger remediation"""
        try:
            # Parse alert into incident context
            incident = self._parse_alert_to_incident(alert_payload)
            
            # Correlate with existing incidents
            correlated_incidents = await self.incident_correlator.correlate_incidents(incident)
            
            # Execute remediation for primary incident
            remediation_results = await self.remediation_engine.handle_incident(incident)
            
            # If correlated incidents exist, handle holistically
            if len(correlated_incidents) > 1:
                holistic_results = await self._handle_correlated_incidents(correlated_incidents)
                remediation_results.extend(holistic_results)
            
            self.self_healing_cycles_counter.inc()
            
            # Determine overall resolution status
            resolution_type = self._determine_resolution_type(remediation_results)
            if resolution_type != 'failed':
                self.incidents_resolved_counter.labels(resolution_type=resolution_type).inc()
            
            return {
                'incident_id': incident.incident_id,
                'correlation_count': len(correlated_incidents),
                'remediation_actions': len(remediation_results),
                'successful_actions': len([r for r in remediation_results 
                                         if r.status == RemediationStatus.COMPLETED]),
                'resolution_type': resolution_type,
                'results': [self._serialize_result(r) for r in remediation_results]
            }
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
            return {
                'error': str(e),
                'incident_id': None,
                'resolution_type': 'failed'
            }
    
    def _parse_alert_to_incident(self, alert_payload: Dict[str, Any]) -> IncidentContext:
        """Parse Prometheus alert into incident context"""
        alert = alert_payload
        
        return IncidentContext(
            incident_id=str(uuid.uuid4()),
            alert_name=alert.get('alertname', 'UnknownAlert'),
            severity=IncidentSeverity(alert.get('severity', 'warning')),
            affected_component=alert.get('component', 'unknown'),
            metrics=self._extract_metrics_from_alert(alert),
            description=alert.get('description', alert.get('summary', '')),
            labels=alert.get('labels', {}),
            timestamp=datetime.utcnow()
        )
    
    def _extract_metrics_from_alert(self, alert: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical metrics from alert payload"""
        metrics = {}
        
        # Extract from alert annotations and labels
        for key, value in alert.get('annotations', {}).items():
            try:
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
                elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    metrics[key] = float(value)
            except (ValueError, TypeError):
                pass
        
        # Extract from labels
        for key, value in alert.get('labels', {}).items():
            try:
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
            except (ValueError, TypeError):
                pass
        
        return metrics
    
    async def _handle_correlated_incidents(self, incidents: List[IncidentContext]) -> List[RemediationResult]:
        """Handle correlated incidents with holistic approach"""
        # Group incidents by component
        component_groups = {}
        for incident in incidents:
            component = incident.affected_component
            if component not in component_groups:
                component_groups[component] = []
            component_groups[component].append(incident)
        
        # Execute component-level remediation
        results = []
        
        # Handle ML service issues holistically
        if 'ml-scheduler' in component_groups and len(component_groups['ml-scheduler']) > 1:
            # Multiple scheduler issues - comprehensive restart
            result = await self._execute_comprehensive_scheduler_restart()
            results.append(result)
        
        # Handle model service cascade failures
        model_services = ['xgboost-predictor', 'qlearning-optimizer', 'anomaly-detector']
        affected_models = [comp for comp in component_groups.keys() if comp in model_services]
        
        if len(affected_models) >= 2:
            # Multiple model services affected - activate global fallback
            result = await self._execute_global_fallback_activation()
            results.append(result)
        
        return results
    
    async def _execute_comprehensive_scheduler_restart(self) -> RemediationResult:
        """Execute comprehensive scheduler restart for multiple issues"""
        action_id = str(uuid.uuid4())
        
        result = RemediationResult(
            action_id=action_id,
            status=RemediationStatus.IN_PROGRESS,
            start_time=datetime.utcnow()
        )
        
        try:
            # 1. Scale down scheduler temporarily
            scale_result = await self.remediation_engine.k8s_remediator.scale_deployment(
                'ml-scheduler', 'ml-scheduler', 0)
            
            # 2. Clear all caches
            cache_result = await self.remediation_engine.cache_remediator.clear_cache_keys('ml:*')
            
            # 3. Wait for cleanup
            await asyncio.sleep(30)
            
            # 4. Scale back up
            restore_result = await self.remediation_engine.k8s_remediator.scale_deployment(
                'ml-scheduler', 'ml-scheduler', 3)
            
            if all(r.get('success', False) for r in [scale_result, cache_result, restore_result]):
                result.status = RemediationStatus.COMPLETED
                result.output = "Comprehensive scheduler restart completed"
            else:
                result.status = RemediationStatus.FAILED
                result.error = "One or more restart steps failed"
                
        except Exception as e:
            result.status = RemediationStatus.FAILED
            result.error = str(e)
        
        result.end_time = datetime.utcnow()
        return result
    
    async def _execute_global_fallback_activation(self) -> RemediationResult:
        """Activate global fallback mode for widespread ML service issues"""
        action_id = str(uuid.uuid4())
        
        result = RemediationResult(
            action_id=action_id,
            status=RemediationStatus.IN_PROGRESS,
            start_time=datetime.utcnow()
        )
        
        try:
            # Activate fallback for all ML components
            fallback_result = await self.remediation_engine.ml_remediator.activate_fallback_mode('all_ml_services')
            
            if fallback_result.get('success', False):
                result.status = RemediationStatus.COMPLETED
                result.output = "Global fallback mode activated"
                logger.critical("Global ML fallback mode activated due to widespread service issues")
            else:
                result.status = RemediationStatus.FAILED
                result.error = fallback_result.get('error', 'Failed to activate global fallback')
                
        except Exception as e:
            result.status = RemediationStatus.FAILED
            result.error = str(e)
        
        result.end_time = datetime.utcnow()
        return result
    
    def _determine_resolution_type(self, results: List[RemediationResult]) -> str:
        """Determine overall resolution type from remediation results"""
        if not results:
            return 'no_action'
        
        completed_count = len([r for r in results if r.status == RemediationStatus.COMPLETED])
        failed_count = len([r for r in results if r.status == RemediationStatus.FAILED])
        
        if completed_count == len(results):
            return 'fully_resolved'
        elif completed_count > 0:
            return 'partially_resolved'
        elif failed_count == len(results):
            return 'failed'
        else:
            return 'in_progress'
    
    def _serialize_result(self, result: RemediationResult) -> Dict[str, Any]:
        """Serialize remediation result for JSON output"""
        return {
            'action_id': result.action_id,
            'status': result.status.value,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'output': result.output,
            'error': result.error,
            'rollback_performed': result.rollback_performed
        }

class AIOpsWebhookHandler:
    """Handles incoming webhooks from Alertmanager"""
    
    def __init__(self, orchestrator: SelfHealingOrchestrator):
        self.orchestrator = orchestrator
        
        # Webhook metrics
        self.webhooks_received_counter = Counter('aiops_webhooks_received_total',
                                               'Webhooks received from Alertmanager')
        self.webhooks_processed_counter = Counter('aiops_webhooks_processed_total',
                                                'Webhooks successfully processed',
                                                ['alert_name'])
    
    async def handle_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming Alertmanager webhook"""
        self.webhooks_received_counter.inc()
        
        try:
            alerts = payload.get('alerts', [])
            results = []
            
            for alert in alerts:
                # Only process firing alerts
                if alert.get('status') == 'firing':
                    result = await self.orchestrator.process_alert(alert)
                    results.append(result)
                    
                    if result.get('incident_id'):
                        self.webhooks_processed_counter.labels(
                            alert_name=alert.get('alertname', 'unknown')).inc()
            
            return {
                'processed_alerts': len(results),
                'results': results,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

class AIOpsMetricsCollector:
    """Collects and reports AIOps effectiveness metrics"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        
        # Effectiveness metrics
        self.mttr_gauge = Gauge('aiops_mean_time_to_resolution_seconds',
                              'Mean time to resolution for incidents')
        self.prevention_rate_gauge = Gauge('aiops_incident_prevention_rate',
                                         'Rate of incidents prevented by AIOps')
        self.automation_rate_gauge = Gauge('aiops_automation_rate',
                                         'Percentage of incidents handled automatically')
    
    async def calculate_effectiveness_metrics(self) -> Dict[str, float]:
        """Calculate AIOps effectiveness metrics"""
        # Query remediation metrics from last 24 hours
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        metrics = {}
        
        # Mean Time to Resolution (MTTR)
        mttr = await self._calculate_mttr(start_time, end_time)
        metrics['mttr_seconds'] = mttr
        self.mttr_gauge.set(mttr)
        
        # Automation rate
        automation_rate = await self._calculate_automation_rate(start_time, end_time)
        metrics['automation_rate'] = automation_rate
        self.automation_rate_gauge.set(automation_rate)
        
        # Prevention rate (incidents that didn't require manual intervention)
        prevention_rate = await self._calculate_prevention_rate(start_time, end_time)
        metrics['prevention_rate'] = prevention_rate
        self.prevention_rate_gauge.set(prevention_rate)
        
        # Success rate of automated remediations
        success_rate = await self._calculate_remediation_success_rate(start_time, end_time)
        metrics['remediation_success_rate'] = success_rate
        
        return metrics
    
    async def _calculate_mttr(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate Mean Time to Resolution"""
        query = 'avg(aiops_remediation_duration_seconds)'
        
        async with aiohttp.ClientSession() as session:
            params = {
                'query': query,
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            }
            
            async with session.get(f"{self.prometheus_url}/api/v1/query_range",
                                 params=params) as response:
                data = await response.json()
        
        results = data.get('data', {}).get('result', [])
        if results and results[0].get('values'):
            values = [float(val[1]) for val in results[0]['values']]
            return sum(values) / len(values)
        
        return 300.0  # Default 5 minutes
    
    async def _calculate_automation_rate(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate automation rate (automated vs manual interventions)"""
        auto_query = 'increase(aiops_incidents_handled_total[24h])'
        manual_query = 'increase(manual_incidents_total[24h])'
        
        async with aiohttp.ClientSession() as session:
            auto_result = await self._query_prometheus(session, auto_query)
            manual_result = await self._query_prometheus(session, manual_query)
        
        auto_count = auto_result if auto_result > 0 else 0
        manual_count = manual_result if manual_result > 0 else 0
        total_count = auto_count + manual_count
        
        return (auto_count / total_count) if total_count > 0 else 0.0
    
    async def _calculate_prevention_rate(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate prevention rate (incidents resolved without escalation)"""
        resolved_query = 'increase(aiops_incidents_resolved_total[24h])'
        escalated_query = 'increase(incidents_escalated_total[24h])'
        
        async with aiohttp.ClientSession() as session:
            resolved_count = await self._query_prometheus(session, resolved_query)
            escalated_count = await self._query_prometheus(session, escalated_query)
        
        total_incidents = resolved_count + escalated_count
        return (resolved_count / total_incidents) if total_incidents > 0 else 0.0
    
    async def _calculate_remediation_success_rate(self, start_time: datetime, 
                                                end_time: datetime) -> float:
        """Calculate success rate of automated remediations"""
        success_query = 'increase(aiops_remediations_total{status="completed"}[24h])'
        total_query = 'increase(aiops_remediations_total[24h])'
        
        async with aiohttp.ClientSession() as session:
            success_count = await self._query_prometheus(session, success_query)
            total_count = await self._query_prometheus(session, total_query)
        
        return (success_count / total_count) if total_count > 0 else 0.0
    
    async def _query_prometheus(self, session: aiohttp.ClientSession, query: str) -> float:
        """Execute Prometheus query and return single value"""
        try:
            async with session.get(f"{self.prometheus_url}/api/v1/query",
                                 params={'query': query}) as response:
                data = await response.json()
                
            results = data.get('data', {}).get('result', [])
            if results:
                return float(results[0]['value'][1])
        except Exception as e:
            logger.warning(f"Failed to query Prometheus: {e}")
        
        return 0.0
    
    async def _handle_correlated_incidents(self, incidents: List[IncidentContext]) -> List[RemediationResult]:
        """Handle correlated incidents with coordinated remediation"""
        logger.info(f"Handling {len(incidents)} correlated incidents")
        
        # Prioritize incidents by severity
        incidents.sort(key=lambda x: {'emergency': 3, 'critical': 2, 'warning': 1, 'info': 0}[x.severity.value], reverse=True)
        
        results = []
        
        # Handle highest severity incident first
        primary_incident = incidents[0]
        primary_results = await self.remediation_engine.handle_incident(primary_incident)
        results.extend(primary_results)
        
        # If primary remediation succeeded, check if it resolved other incidents
        if any(r.status == RemediationStatus.COMPLETED for r in primary_results):
            # Wait for metrics to stabilize
            await asyncio.sleep(60)
            
            # Check if correlated incidents are now resolved
            for incident in incidents[1:]:
                if await self._is_incident_resolved(incident):
                    logger.info(f"Incident {incident.incident_id} resolved by correlated remediation")
                    continue
                
                # If not resolved, handle individually
                incident_results = await self.remediation_engine.handle_incident(incident)
                results.extend(incident_results)
        
        return results
    
    async def _is_incident_resolved(self, incident: IncidentContext) -> bool:
        """Check if incident is resolved by querying current metrics"""
        try:
            current_metrics = await self.remediation_engine._collect_incident_metrics(incident)
            
            # Define resolution criteria based on alert type
            if 'CPU' in incident.alert_name:
                cpu = current_metrics.get('cpu_utilization', 0)
                return 60 <= cpu <= 70  # Within acceptable range
                
            elif 'Latency' in incident.alert_name:
                latency = current_metrics.get('scheduling_latency', 0)
                return latency < 100  # Below target
                
            elif 'Availability' in incident.alert_name:
                availability = current_metrics.get('availability', 0)
                return availability > 99.5  # Above warning threshold
                
            elif 'Success' in incident.alert_name:
                success_rate = current_metrics.get('success_rate', 0)
                return success_rate > 98  # Above warning threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking incident resolution: {e}")
            return False

async def main():
    """Main entry point for AIOps service"""
    import argparse
    from aiohttp import web
    
    parser = argparse.ArgumentParser(description='AIOps Automated Remediation Service')
    parser.add_argument('--config', default='/etc/ml-scheduler/aiops_config.yaml',
                       help='AIOps configuration file')
    parser.add_argument('--prometheus-url', default='http://prometheus:9090',
                       help='Prometheus server URL')
    parser.add_argument('--kserve-url', default='http://kserve-controller:8080',
                       help='KServe controller URL')
    parser.add_argument('--redis-url', default='redis://redis:6379',
                       help='Redis server URL')
    parser.add_argument('--port', type=int, default=8080,
                       help='Webhook server port')
    
    args = parser.parse_args()
    
    # Initialize AIOps components
    orchestrator = SelfHealingOrchestrator(
        args.config, args.prometheus_url, args.kserve_url, args.redis_url)
    webhook_handler = AIOpsWebhookHandler(orchestrator)
    metrics_collector = AIOpsMetricsCollector(args.prometheus_url)
    
    async def webhook_endpoint(request):
        """Webhook endpoint for Alertmanager"""
        try:
            payload = await request.json()
            result = await webhook_handler.handle_webhook(payload)
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def health_endpoint(request):
        """Health check endpoint"""
        return web.json_response({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})
    
    async def metrics_endpoint(request):
        """Metrics endpoint for AIOps effectiveness"""
        try:
            metrics = await metrics_collector.calculate_effectiveness_metrics()
            return web.json_response(metrics)
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    # Setup web application
    app = web.Application()
    app.router.add_post('/webhook', webhook_endpoint)
    app.router.add_get('/health', health_endpoint)
    app.router.add_get('/metrics', metrics_endpoint)
    
    # Start server
    logger.info(f"Starting AIOps service on port {args.port}")
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', args.port)
    await site.start()
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(60)
            
            # Periodic effectiveness reporting
            try:
                effectiveness = await metrics_collector.calculate_effectiveness_metrics()
                logger.info(f"AIOps effectiveness: {effectiveness}")
            except Exception as e:
                logger.error(f"Error calculating effectiveness: {e}")
                
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())