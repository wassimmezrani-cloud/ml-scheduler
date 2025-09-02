#!/usr/bin/env python3

import asyncio
import json
import logging
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException

class DeploymentStatus(Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    UPDATING = "updating"
    STOPPING = "stopping"

class ComponentType(Enum):
    ML_MODEL = "ml_model"
    API_SERVICE = "api_service"
    MONITORING = "monitoring"
    SCHEDULER = "scheduler"
    STORAGE = "storage"

@dataclass
class ComponentConfig:
    name: str
    component_type: ComponentType
    image: str
    replicas: int
    ports: List[int]
    environment: Dict[str, str]
    resources: Dict[str, Dict[str, str]]
    volumes: List[Dict[str, Any]]
    health_check: Dict[str, Any]
    dependencies: List[str]

@dataclass
class DeploymentState:
    component: str
    status: DeploymentStatus
    replicas_ready: int
    replicas_desired: int
    last_updated: datetime
    error_message: Optional[str] = None
    deployment_logs: List[str] = None

class DeploymentOrchestrator:
    def __init__(self, config_path: str = "/home/wassim/Desktop/ml scheduler/config/deployment_config.yaml"):
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.k8s_client = self._initialize_k8s_client()
        self.deployment_states: Dict[str, DeploymentState] = {}
        self.components: Dict[str, ComponentConfig] = {}
        
        self.hydatis_config = {
            'cluster_name': 'HYDATIS',
            'master_nodes': ['10.110.190.31', '10.110.190.32', '10.110.190.33'],
            'worker_nodes': ['10.110.190.34', '10.110.190.35', '10.110.190.36'],
            'storage_class': 'longhorn',
            'monitoring_endpoint': 'http://10.110.190.83:9090',
            'mlflow_endpoint': 'http://10.110.190.32:31380'
        }
        
        self._load_component_configurations()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for deployment orchestrator."""
        logger = logging.getLogger('deployment_orchestrator')
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

    def _load_component_configurations(self):
        """Load component configurations from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            for comp_name, comp_config in config_data.get('components', {}).items():
                self.components[comp_name] = ComponentConfig(
                    name=comp_name,
                    component_type=ComponentType(comp_config['type']),
                    image=comp_config['image'],
                    replicas=comp_config.get('replicas', 1),
                    ports=comp_config.get('ports', []),
                    environment=comp_config.get('environment', {}),
                    resources=comp_config.get('resources', {}),
                    volumes=comp_config.get('volumes', []),
                    health_check=comp_config.get('health_check', {}),
                    dependencies=comp_config.get('dependencies', [])
                )
                
        except FileNotFoundError:
            self.logger.warning(f"Configuration file not found: {self.config_path}")
            self._create_default_configuration()
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def _create_default_configuration(self):
        """Create default component configurations."""
        default_components = {
            'xgboost_predictor': ComponentConfig(
                name='xgboost-load-predictor',
                component_type=ComponentType.ML_MODEL,
                image='hydatis/xgboost-predictor:latest',
                replicas=2,
                ports=[8001],
                environment={
                    'MLFLOW_TRACKING_URI': self.hydatis_config['mlflow_endpoint'],
                    'PROMETHEUS_ENDPOINT': self.hydatis_config['monitoring_endpoint'],
                    'MODEL_REGISTRY': 'production'
                },
                resources={
                    'requests': {'cpu': '500m', 'memory': '1Gi'},
                    'limits': {'cpu': '2', 'memory': '4Gi'}
                },
                volumes=[{
                    'name': 'model-storage',
                    'persistentVolumeClaim': {'claimName': 'xgboost-model-pvc'}
                }],
                health_check={
                    'path': '/health',
                    'port': 8001,
                    'initial_delay': 30,
                    'period': 10
                },
                dependencies=[]
            ),
            
            'qlearning_optimizer': ComponentConfig(
                name='qlearning-placement-optimizer',
                component_type=ComponentType.ML_MODEL,
                image='hydatis/qlearning-optimizer:latest',
                replicas=2,
                ports=[8002],
                environment={
                    'MLFLOW_TRACKING_URI': self.hydatis_config['mlflow_endpoint'],
                    'PROMETHEUS_ENDPOINT': self.hydatis_config['monitoring_endpoint'],
                    'PYTORCH_DEVICE': 'cpu'
                },
                resources={
                    'requests': {'cpu': '1', 'memory': '2Gi'},
                    'limits': {'cpu': '4', 'memory': '8Gi'}
                },
                volumes=[{
                    'name': 'model-storage',
                    'persistentVolumeClaim': {'claimName': 'qlearning-model-pvc'}
                }],
                health_check={
                    'path': '/health',
                    'port': 8002,
                    'initial_delay': 45,
                    'period': 15
                },
                dependencies=[]
            ),
            
            'anomaly_detector': ComponentConfig(
                name='isolation-forest-detector',
                component_type=ComponentType.ML_MODEL,
                image='hydatis/anomaly-detector:latest',
                replicas=2,
                ports=[8003],
                environment={
                    'MLFLOW_TRACKING_URI': self.hydatis_config['mlflow_endpoint'],
                    'PROMETHEUS_ENDPOINT': self.hydatis_config['monitoring_endpoint'],
                    'ALERT_MANAGER_URL': 'http://alert-manager-service:8080'
                },
                resources={
                    'requests': {'cpu': '500m', 'memory': '1Gi'},
                    'limits': {'cpu': '2', 'memory': '4Gi'}
                },
                volumes=[{
                    'name': 'model-storage',
                    'persistentVolumeClaim': {'claimName': 'anomaly-model-pvc'}
                }],
                health_check={
                    'path': '/health',
                    'port': 8003,
                    'initial_delay': 30,
                    'period': 10
                },
                dependencies=[]
            ),
            
            'ml_gateway': ComponentConfig(
                name='ml-scheduler-gateway',
                component_type=ComponentType.API_SERVICE,
                image='hydatis/ml-gateway:latest',
                replicas=3,
                ports=[8000],
                environment={
                    'XGBOOST_SERVICE_URL': 'http://xgboost-service:8001',
                    'QLEARNING_SERVICE_URL': 'http://qlearning-service:8002',
                    'ANOMALY_SERVICE_URL': 'http://anomaly-service:8003',
                    'PROMETHEUS_ENDPOINT': self.hydatis_config['monitoring_endpoint']
                },
                resources={
                    'requests': {'cpu': '1', 'memory': '2Gi'},
                    'limits': {'cpu': '4', 'memory': '8Gi'}
                },
                volumes=[],
                health_check={
                    'path': '/health',
                    'port': 8000,
                    'initial_delay': 20,
                    'period': 5
                },
                dependencies=['xgboost_predictor', 'qlearning_optimizer', 'anomaly_detector']
            ),
            
            'scheduler_plugin': ComponentConfig(
                name='hydatis-scheduler-plugin',
                component_type=ComponentType.SCHEDULER,
                image='hydatis/scheduler-plugin:latest',
                replicas=1,
                ports=[9443],
                environment={
                    'ML_GATEWAY_URL': 'http://ml-gateway-service:8000',
                    'PROMETHEUS_ENDPOINT': self.hydatis_config['monitoring_endpoint'],
                    'CLUSTER_NAME': 'HYDATIS'
                },
                resources={
                    'requests': {'cpu': '500m', 'memory': '1Gi'},
                    'limits': {'cpu': '2', 'memory': '4Gi'}
                },
                volumes=[{
                    'name': 'scheduler-config',
                    'configMap': {'name': 'scheduler-plugin-config'}
                }],
                health_check={
                    'path': '/healthz',
                    'port': 9443,
                    'initial_delay': 30,
                    'period': 10
                },
                dependencies=['ml_gateway']
            ),
            
            'monitoring_dashboard': ComponentConfig(
                name='unified-monitoring-dashboard',
                component_type=ComponentType.MONITORING,
                image='hydatis/monitoring-dashboard:latest',
                replicas=1,
                ports=[8080],
                environment={
                    'ML_GATEWAY_URL': 'http://ml-gateway-service:8000',
                    'PROMETHEUS_ENDPOINT': self.hydatis_config['monitoring_endpoint'],
                    'ALERT_MANAGER_URL': 'http://alert-manager-service:8080'
                },
                resources={
                    'requests': {'cpu': '200m', 'memory': '512Mi'},
                    'limits': {'cpu': '1', 'memory': '2Gi'}
                },
                volumes=[],
                health_check={
                    'path': '/health',
                    'port': 8080,
                    'initial_delay': 15,
                    'period': 5
                },
                dependencies=['ml_gateway']
            )
        }
        
        self.components = default_components

    async def deploy_full_stack(self) -> Dict[str, Any]:
        """Deploy complete ML scheduler stack with dependency resolution."""
        self.logger.info("Starting full stack deployment for HYDATIS ML Scheduler")
        
        deployment_order = self._resolve_deployment_order()
        deployment_results = {}
        
        try:
            for component_name in deployment_order:
                self.logger.info(f"Deploying component: {component_name}")
                
                result = await self._deploy_component(component_name)
                deployment_results[component_name] = result
                
                if result['status'] != 'success':
                    self.logger.error(f"Failed to deploy {component_name}: {result['error']}")
                    break
                
                await self._wait_for_component_ready(component_name)
                self.logger.info(f"Component {component_name} is ready")
            
            await self._verify_full_stack_health()
            
            return {
                'status': 'success',
                'deployed_components': list(deployment_results.keys()),
                'deployment_results': deployment_results,
                'cluster_endpoints': self._get_cluster_endpoints()
            }
            
        except Exception as e:
            self.logger.error(f"Full stack deployment failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'partial_results': deployment_results
            }

    def _resolve_deployment_order(self) -> List[str]:
        """Resolve component deployment order based on dependencies."""
        order = []
        visited = set()
        visiting = set()
        
        def visit(component_name: str):
            if component_name in visiting:
                raise ValueError(f"Circular dependency detected involving {component_name}")
            if component_name in visited:
                return
                
            visiting.add(component_name)
            
            component = self.components[component_name]
            for dep in component.dependencies:
                if dep in self.components:
                    visit(dep)
            
            visiting.remove(component_name)
            visited.add(component_name)
            order.append(component_name)
        
        for component_name in self.components:
            visit(component_name)
        
        return order

    async def _deploy_component(self, component_name: str) -> Dict[str, Any]:
        """Deploy individual component to Kubernetes."""
        try:
            component = self.components[component_name]
            
            self.deployment_states[component_name] = DeploymentState(
                component=component_name,
                status=DeploymentStatus.DEPLOYING,
                replicas_ready=0,
                replicas_desired=component.replicas,
                last_updated=datetime.now()
            )
            
            k8s_manifests = self._generate_k8s_manifests(component)
            
            apps_v1 = client.AppsV1Api(self.k8s_client)
            core_v1 = client.CoreV1Api(self.k8s_client)
            
            if k8s_manifests.get('pvc'):
                try:
                    core_v1.create_namespaced_persistent_volume_claim(
                        namespace='ml-scheduler',
                        body=k8s_manifests['pvc']
                    )
                except ApiException as e:
                    if e.status != 409:
                        raise
            
            if k8s_manifests.get('configmap'):
                try:
                    core_v1.create_namespaced_config_map(
                        namespace='ml-scheduler',
                        body=k8s_manifests['configmap']
                    )
                except ApiException as e:
                    if e.status != 409:
                        core_v1.patch_namespaced_config_map(
                            name=k8s_manifests['configmap'].metadata.name,
                            namespace='ml-scheduler',
                            body=k8s_manifests['configmap']
                        )
            
            try:
                apps_v1.create_namespaced_deployment(
                    namespace='ml-scheduler',
                    body=k8s_manifests['deployment']
                )
            except ApiException as e:
                if e.status == 409:
                    apps_v1.patch_namespaced_deployment(
                        name=component.name,
                        namespace='ml-scheduler',
                        body=k8s_manifests['deployment']
                    )
                else:
                    raise
            
            core_v1.create_namespaced_service(
                namespace='ml-scheduler',
                body=k8s_manifests['service']
            )
            
            return {
                'status': 'success',
                'component': component_name,
                'manifests_applied': list(k8s_manifests.keys())
            }
            
        except Exception as e:
            self.deployment_states[component_name].status = DeploymentStatus.FAILED
            self.deployment_states[component_name].error_message = str(e)
            
            return {
                'status': 'error',
                'component': component_name,
                'error': str(e)
            }

    def _generate_k8s_manifests(self, component: ComponentConfig) -> Dict[str, Any]:
        """Generate Kubernetes manifests for component."""
        manifests = {}
        
        labels = {
            'app': component.name,
            'component': component.component_type.value,
            'cluster': 'hydatis'
        }
        
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': component.name,
                'namespace': 'ml-scheduler',
                'labels': labels
            },
            'spec': {
                'replicas': component.replicas,
                'selector': {'matchLabels': labels},
                'template': {
                    'metadata': {'labels': labels},
                    'spec': {
                        'containers': [{
                            'name': component.name,
                            'image': component.image,
                            'ports': [{'containerPort': port} for port in component.ports],
                            'env': [{'name': k, 'value': v} for k, v in component.environment.items()],
                            'resources': component.resources,
                            'volumeMounts': [
                                {
                                    'name': vol['name'],
                                    'mountPath': vol.get('mountPath', f"/data/{vol['name']}")
                                } for vol in component.volumes
                            ]
                        }],
                        'volumes': [
                            {
                                'name': vol['name'],
                                'persistentVolumeClaim': vol.get('persistentVolumeClaim', {})
                            } if 'persistentVolumeClaim' in vol else {
                                'name': vol['name'],
                                'configMap': vol.get('configMap', {})
                            } for vol in component.volumes
                        ],
                        'nodeSelector': {'node-role.kubernetes.io/worker': 'true'}
                    }
                }
            }
        }
        
        if component.health_check:
            health_check = {
                'httpGet': {
                    'path': component.health_check.get('path', '/health'),
                    'port': component.health_check.get('port', component.ports[0])
                },
                'initialDelaySeconds': component.health_check.get('initial_delay', 30),
                'periodSeconds': component.health_check.get('period', 10)
            }
            deployment['spec']['template']['spec']['containers'][0]['livenessProbe'] = health_check
            deployment['spec']['template']['spec']['containers'][0]['readinessProbe'] = health_check
        
        manifests['deployment'] = deployment
        
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{component.name}-service",
                'namespace': 'ml-scheduler',
                'labels': labels
            },
            'spec': {
                'selector': labels,
                'ports': [
                    {
                        'port': port,
                        'targetPort': port,
                        'protocol': 'TCP'
                    } for port in component.ports
                ],
                'type': 'ClusterIP'
            }
        }
        manifests['service'] = service
        
        if component.volumes:
            for vol in component.volumes:
                if 'persistentVolumeClaim' in vol:
                    pvc = {
                        'apiVersion': 'v1',
                        'kind': 'PersistentVolumeClaim',
                        'metadata': {
                            'name': vol['persistentVolumeClaim']['claimName'],
                            'namespace': 'ml-scheduler'
                        },
                        'spec': {
                            'accessModes': ['ReadWriteOnce'],
                            'storageClassName': 'longhorn',
                            'resources': {
                                'requests': {
                                    'storage': vol.get('size', '10Gi')
                                }
                            }
                        }
                    }
                    manifests['pvc'] = pvc
        
        return manifests

    async def _wait_for_component_ready(self, component_name: str, timeout: int = 300):
        """Wait for component to be ready."""
        component = self.components[component_name]
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            try:
                deployment = apps_v1.read_namespaced_deployment(
                    name=component.name,
                    namespace='ml-scheduler'
                )
                
                ready_replicas = deployment.status.ready_replicas or 0
                desired_replicas = deployment.spec.replicas
                
                self.deployment_states[component_name].replicas_ready = ready_replicas
                self.deployment_states[component_name].replicas_desired = desired_replicas
                
                if ready_replicas == desired_replicas:
                    self.deployment_states[component_name].status = DeploymentStatus.RUNNING
                    self.logger.info(f"Component {component_name} is ready ({ready_replicas}/{desired_replicas})")
                    return
                
                self.logger.info(f"Waiting for {component_name}: {ready_replicas}/{desired_replicas} ready")
                await asyncio.sleep(5)
                
            except ApiException as e:
                self.logger.error(f"Error checking {component_name} status: {e}")
                await asyncio.sleep(5)
        
        raise TimeoutError(f"Component {component_name} failed to become ready within {timeout}s")

    async def _verify_full_stack_health(self) -> Dict[str, Any]:
        """Verify health of all deployed components."""
        health_results = {}
        
        for component_name, component in self.components.items():
            if component.health_check and component.ports:
                try:
                    service_url = f"http://{component.name}-service.ml-scheduler.svc.cluster.local:{component.ports[0]}"
                    health_url = f"{service_url}{component.health_check.get('path', '/health')}"
                    
                    response = requests.get(health_url, timeout=10)
                    health_results[component_name] = {
                        'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                        'response_code': response.status_code,
                        'response_time': response.elapsed.total_seconds()
                    }
                    
                except Exception as e:
                    health_results[component_name] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
        
        return health_results

    def _get_cluster_endpoints(self) -> Dict[str, str]:
        """Get accessible endpoints for deployed services."""
        return {
            'ml_gateway': 'http://ml-gateway-service.ml-scheduler.svc.cluster.local:8000',
            'xgboost_predictor': 'http://xgboost-service.ml-scheduler.svc.cluster.local:8001',
            'qlearning_optimizer': 'http://qlearning-service.ml-scheduler.svc.cluster.local:8002',
            'anomaly_detector': 'http://anomaly-service.ml-scheduler.svc.cluster.local:8003',
            'monitoring_dashboard': 'http://monitoring-dashboard-service.ml-scheduler.svc.cluster.local:8080',
            'scheduler_plugin': 'https://hydatis-scheduler-plugin.ml-scheduler.svc.cluster.local:9443'
        }

    async def scale_component(self, component_name: str, replicas: int) -> Dict[str, Any]:
        """Scale component to specified number of replicas."""
        try:
            apps_v1 = client.AppsV1Api(self.k8s_client)
            component = self.components[component_name]
            
            self.deployment_states[component_name].status = DeploymentStatus.UPDATING
            
            apps_v1.patch_namespaced_deployment_scale(
                name=component.name,
                namespace='ml-scheduler',
                body={'spec': {'replicas': replicas}}
            )
            
            await self._wait_for_component_ready(component_name)
            
            return {
                'status': 'success',
                'component': component_name,
                'new_replicas': replicas
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'component': component_name,
                'error': str(e)
            }

    async def update_component_config(self, component_name: str, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update component configuration and redeploy."""
        try:
            component = self.components[component_name]
            
            if 'environment' in config_updates:
                component.environment.update(config_updates['environment'])
            
            if 'resources' in config_updates:
                component.resources.update(config_updates['resources'])
            
            return await self._deploy_component(component_name)
            
        except Exception as e:
            return {
                'status': 'error',
                'component': component_name,
                'error': str(e)
            }

    async def rollback_component(self, component_name: str, revision: Optional[int] = None) -> Dict[str, Any]:
        """Rollback component to previous revision."""
        try:
            apps_v1 = client.AppsV1Api(self.k8s_client)
            component = self.components[component_name]
            
            if revision:
                rollback_cmd = f"kubectl rollout undo deployment/{component.name} --to-revision={revision} -n ml-scheduler"
            else:
                rollback_cmd = f"kubectl rollout undo deployment/{component.name} -n ml-scheduler"
            
            result = subprocess.run(rollback_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                await self._wait_for_component_ready(component_name)
                return {
                    'status': 'success',
                    'component': component_name,
                    'rollback_output': result.stdout
                }
            else:
                return {
                    'status': 'error',
                    'component': component_name,
                    'error': result.stderr
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'component': component_name,
                'error': str(e)
            }

    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status of all components."""
        status_summary = {
            'cluster': self.hydatis_config['cluster_name'],
            'components': {},
            'overall_health': 'unknown',
            'last_updated': datetime.now().isoformat()
        }
        
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        healthy_count = 0
        total_count = len(self.components)
        
        for component_name, component in self.components.items():
            try:
                deployment = apps_v1.read_namespaced_deployment(
                    name=component.name,
                    namespace='ml-scheduler'
                )
                
                ready_replicas = deployment.status.ready_replicas or 0
                desired_replicas = deployment.spec.replicas
                
                component_status = {
                    'status': 'running' if ready_replicas == desired_replicas else 'degraded',
                    'replicas': f"{ready_replicas}/{desired_replicas}",
                    'image': deployment.spec.template.spec.containers[0].image,
                    'created': deployment.metadata.creation_timestamp.isoformat() if deployment.metadata.creation_timestamp else None
                }
                
                if ready_replicas == desired_replicas:
                    healthy_count += 1
                
            except ApiException:
                component_status = {
                    'status': 'not_deployed',
                    'replicas': '0/0',
                    'image': component.image,
                    'created': None
                }
            
            status_summary['components'][component_name] = component_status
        
        if healthy_count == total_count:
            status_summary['overall_health'] = 'healthy'
        elif healthy_count > 0:
            status_summary['overall_health'] = 'degraded'
        else:
            status_summary['overall_health'] = 'critical'
        
        return status_summary

    async def cleanup_deployment(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Clean up deployed components."""
        if component_name:
            components_to_cleanup = [component_name]
        else:
            components_to_cleanup = list(self.components.keys())
        
        cleanup_results = {}
        
        for comp_name in components_to_cleanup:
            try:
                if comp_name in self.deployment_states:
                    self.deployment_states[comp_name].status = DeploymentStatus.STOPPING
                
                component = self.components[comp_name]
                
                cleanup_cmd = f"kubectl delete deployment,service,pvc,configmap -l app={component.name} -n ml-scheduler"
                result = subprocess.run(cleanup_cmd, shell=True, capture_output=True, text=True)
                
                cleanup_results[comp_name] = {
                    'status': 'success' if result.returncode == 0 else 'error',
                    'output': result.stdout if result.returncode == 0 else result.stderr
                }
                
                if comp_name in self.deployment_states:
                    del self.deployment_states[comp_name]
                
            except Exception as e:
                cleanup_results[comp_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return cleanup_results

    async def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        deployment_status = await self.get_deployment_status()
        health_check = await self._verify_full_stack_health()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'cluster_info': self.hydatis_config,
            'deployment_status': deployment_status,
            'health_checks': health_check,
            'component_configurations': {
                name: asdict(config) for name, config in self.components.items()
            },
            'endpoints': self._get_cluster_endpoints(),
            'recommendations': self._generate_recommendations(deployment_status, health_check)
        }
        
        return report

    def _generate_recommendations(self, deployment_status: Dict, health_checks: Dict) -> List[str]:
        """Generate deployment recommendations based on current state."""
        recommendations = []
        
        if deployment_status['overall_health'] != 'healthy':
            recommendations.append("Some components are not healthy - investigate logs and resource usage")
        
        for comp_name, health in health_checks.items():
            if health.get('status') != 'healthy':
                recommendations.append(f"Component {comp_name} health check failed - review configuration")
        
        degraded_components = [
            name for name, status in deployment_status['components'].items()
            if status['status'] == 'degraded'
        ]
        
        if degraded_components:
            recommendations.append(f"Scale up degraded components: {', '.join(degraded_components)}")
        
        return recommendations

    async def create_namespace_if_not_exists(self):
        """Create ml-scheduler namespace if it doesn't exist."""
        core_v1 = client.CoreV1Api(self.k8s_client)
        
        try:
            core_v1.read_namespace('ml-scheduler')
            self.logger.info("Namespace ml-scheduler already exists")
        except ApiException as e:
            if e.status == 404:
                namespace = {
                    'apiVersion': 'v1',
                    'kind': 'Namespace',
                    'metadata': {
                        'name': 'ml-scheduler',
                        'labels': {
                            'name': 'ml-scheduler',
                            'cluster': 'hydatis'
                        }
                    }
                }
                
                core_v1.create_namespace(body=namespace)
                self.logger.info("Created namespace ml-scheduler")
            else:
                raise

async def main():
    """Main deployment orchestration entry point."""
    orchestrator = DeploymentOrchestrator()
    
    try:
        await orchestrator.create_namespace_if_not_exists()
        
        deployment_result = await orchestrator.deploy_full_stack()
        
        if deployment_result['status'] == 'success':
            print("HYDATIS ML Scheduler deployed successfully!")
            print(f"Available endpoints: {deployment_result['cluster_endpoints']}")
        else:
            print(f"Deployment failed: {deployment_result.get('error', 'Unknown error')}")
            
        report = await orchestrator.generate_deployment_report()
        
        with open('/tmp/hydatis_deployment_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("Deployment report saved to /tmp/hydatis_deployment_report.json")
        
    except Exception as e:
        logging.error(f"Deployment orchestration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())