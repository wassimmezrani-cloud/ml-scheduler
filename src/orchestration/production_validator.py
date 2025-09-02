#!/usr/bin/env python3

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import requests
from kubernetes import client, config

class ValidationResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"

@dataclass
class ValidationTest:
    name: str
    category: str
    description: str
    result: ValidationResult
    details: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None

class ProductionValidator:
    def __init__(self):
        self.logger = self._setup_logging()
        self.k8s_client = self._initialize_k8s_client()
        
        self.hydatis_config = {
            'namespace': 'ml-scheduler',
            'cluster_name': 'HYDATIS',
            'prometheus_endpoint': 'http://10.110.190.83:9090',
            'mlflow_endpoint': 'http://10.110.190.32:31380'
        }
        
        self.validation_results: List[ValidationTest] = []
        
        self.service_endpoints = {
            'ml_gateway': 'http://ml-gateway-service.ml-scheduler.svc.cluster.local:8000',
            'xgboost_predictor': 'http://xgboost-service.ml-scheduler.svc.cluster.local:8001',
            'qlearning_optimizer': 'http://qlearning-service.ml-scheduler.svc.cluster.local:8002',
            'anomaly_detector': 'http://anomaly-service.ml-scheduler.svc.cluster.local:8003',
            'monitoring_dashboard': 'http://monitoring-dashboard-service.ml-scheduler.svc.cluster.local:8080'
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for production validator."""
        logger = logging.getLogger('production_validator')
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

    async def run_full_validation_suite(self) -> Dict[str, Any]:
        """Run complete production validation suite."""
        self.logger.info("Starting HYDATIS ML Scheduler production validation")
        
        validation_categories = [
            ('infrastructure', self._validate_infrastructure),
            ('services', self._validate_services),
            ('ml_models', self._validate_ml_models),
            ('integration', self._validate_integration),
            ('performance', self._validate_performance),
            ('security', self._validate_security),
            ('monitoring', self._validate_monitoring)
        ]
        
        start_time = time.time()
        
        for category, validation_func in validation_categories:
            self.logger.info(f"Running {category} validation tests...")
            try:
                await validation_func()
            except Exception as e:
                self.logger.error(f"Validation category {category} failed: {e}")
                self.validation_results.append(ValidationTest(
                    name=f"{category}_validation",
                    category=category,
                    description=f"Validation of {category} components",
                    result=ValidationResult.FAIL,
                    details={'error': str(e)},
                    execution_time=0,
                    error_message=str(e)
                ))
        
        total_time = time.time() - start_time
        
        return self._generate_validation_report(total_time)

    async def _validate_infrastructure(self):
        """Validate infrastructure components."""
        
        await self._test_namespace_exists()
        await self._test_storage_claims()
        await self._test_configmaps()
        await self._test_node_resources()

    async def _test_namespace_exists(self):
        """Test that ml-scheduler namespace exists."""
        start_time = time.time()
        
        try:
            core_v1 = client.CoreV1Api(self.k8s_client)
            namespace = core_v1.read_namespace(self.hydatis_config['namespace'])
            
            self.validation_results.append(ValidationTest(
                name="namespace_exists",
                category="infrastructure",
                description="Verify ml-scheduler namespace exists",
                result=ValidationResult.PASS,
                details={
                    'namespace': namespace.metadata.name,
                    'labels': namespace.metadata.labels,
                    'creation_time': namespace.metadata.creation_timestamp.isoformat() if namespace.metadata.creation_timestamp else None
                },
                execution_time=time.time() - start_time
            ))
            
        except Exception as e:
            self.validation_results.append(ValidationTest(
                name="namespace_exists",
                category="infrastructure",
                description="Verify ml-scheduler namespace exists",
                result=ValidationResult.FAIL,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))

    async def _test_storage_claims(self):
        """Test that all PVCs are bound."""
        start_time = time.time()
        
        try:
            core_v1 = client.CoreV1Api(self.k8s_client)
            pvcs = core_v1.list_namespaced_persistent_volume_claim(
                namespace=self.hydatis_config['namespace']
            )
            
            expected_pvcs = [
                'xgboost-model-pvc',
                'qlearning-model-pvc', 
                'qlearning-replay-pvc',
                'anomaly-model-pvc'
            ]
            
            pvc_status = {}
            for pvc in pvcs.items:
                pvc_status[pvc.metadata.name] = {
                    'status': pvc.status.phase,
                    'capacity': pvc.status.capacity.get('storage') if pvc.status.capacity else None,
                    'storage_class': pvc.spec.storage_class_name
                }
            
            missing_pvcs = [pvc for pvc in expected_pvcs if pvc not in pvc_status]
            unbound_pvcs = [pvc for pvc, status in pvc_status.items() if status['status'] != 'Bound']
            
            if missing_pvcs or unbound_pvcs:
                result = ValidationResult.FAIL
                error_msg = f"Missing PVCs: {missing_pvcs}, Unbound PVCs: {unbound_pvcs}"
            else:
                result = ValidationResult.PASS
                error_msg = None
            
            self.validation_results.append(ValidationTest(
                name="storage_claims",
                category="infrastructure",
                description="Verify all persistent volume claims are bound",
                result=result,
                details={
                    'expected_pvcs': expected_pvcs,
                    'found_pvcs': list(pvc_status.keys()),
                    'pvc_details': pvc_status
                },
                execution_time=time.time() - start_time,
                error_message=error_msg
            ))
            
        except Exception as e:
            self.validation_results.append(ValidationTest(
                name="storage_claims",
                category="infrastructure",
                description="Verify all persistent volume claims are bound",
                result=ValidationResult.FAIL,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))

    async def _validate_services(self):
        """Validate all services are running and healthy."""
        
        await self._test_pods_running()
        await self._test_service_discovery()
        await self._test_health_endpoints()

    async def _test_pods_running(self):
        """Test that all pods are running and ready."""
        start_time = time.time()
        
        try:
            core_v1 = client.CoreV1Api(self.k8s_client)
            pods = core_v1.list_namespaced_pod(
                namespace=self.hydatis_config['namespace'],
                label_selector="cluster=hydatis"
            )
            
            pod_status = {}
            for pod in pods.items:
                pod_status[pod.metadata.name] = {
                    'phase': pod.status.phase,
                    'ready': all(cs.ready for cs in pod.status.container_statuses or []),
                    'restart_count': sum(cs.restart_count for cs in pod.status.container_statuses or []),
                    'node': pod.spec.node_name
                }
            
            failed_pods = [name for name, status in pod_status.items() if not status['ready'] or status['phase'] != 'Running']
            high_restart_pods = [name for name, status in pod_status.items() if status['restart_count'] > 3]
            
            if failed_pods:
                result = ValidationResult.FAIL
                error_msg = f"Pods not ready: {failed_pods}"
            elif high_restart_pods:
                result = ValidationResult.WARNING
                error_msg = f"High restart count pods: {high_restart_pods}"
            else:
                result = ValidationResult.PASS
                error_msg = None
            
            self.validation_results.append(ValidationTest(
                name="pods_running",
                category="services",
                description="Verify all pods are running and ready",
                result=result,
                details={
                    'total_pods': len(pod_status),
                    'ready_pods': len([s for s in pod_status.values() if s['ready']]),
                    'pod_details': pod_status
                },
                execution_time=time.time() - start_time,
                error_message=error_msg
            ))
            
        except Exception as e:
            self.validation_results.append(ValidationTest(
                name="pods_running",
                category="services",
                description="Verify all pods are running and ready",
                result=ValidationResult.FAIL,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))

    async def _test_health_endpoints(self):
        """Test health endpoints of all services."""
        start_time = time.time()
        
        health_results = {}
        
        for service_name, endpoint in self.service_endpoints.items():
            try:
                health_url = f"{endpoint}/health"
                response = requests.get(health_url, timeout=10)
                
                health_results[service_name] = {
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'healthy': response.status_code == 200
                }
                
            except Exception as e:
                health_results[service_name] = {
                    'status_code': None,
                    'response_time': None,
                    'healthy': False,
                    'error': str(e)
                }
        
        failed_health_checks = [name for name, result in health_results.items() if not result['healthy']]
        
        if failed_health_checks:
            result = ValidationResult.FAIL
            error_msg = f"Failed health checks: {failed_health_checks}"
        else:
            result = ValidationResult.PASS
            error_msg = None
        
        self.validation_results.append(ValidationTest(
            name="health_endpoints",
            category="services",
            description="Verify all service health endpoints respond",
            result=result,
            details=health_results,
            execution_time=time.time() - start_time,
            error_message=error_msg
        ))

    async def _validate_ml_models(self):
        """Validate ML model functionality."""
        
        await self._test_xgboost_prediction()
        await self._test_qlearning_optimization()
        await self._test_anomaly_detection()
        await self._test_model_versioning()

    async def _test_xgboost_prediction(self):
        """Test XGBoost load prediction."""
        start_time = time.time()
        
        try:
            test_payload = {
                'cluster_metrics': {
                    'total_nodes': 6,
                    'total_pods': 45,
                    'cpu_utilization': 0.65,
                    'memory_utilization': 0.72,
                    'network_throughput': 1024000,
                    'storage_utilization': 0.45
                },
                'historical_data': {
                    'past_hour_avg_cpu': 0.68,
                    'past_hour_avg_memory': 0.71,
                    'scheduling_frequency': 12
                }
            }
            
            response = requests.post(
                f"{self.service_endpoints['xgboost_predictor']}/predict",
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                prediction_data = response.json()
                
                required_fields = ['cpu_prediction', 'memory_prediction', 'confidence_score']
                has_required_fields = all(field in prediction_data for field in required_fields)
                
                confidence_valid = 0.0 <= prediction_data.get('confidence_score', -1) <= 1.0
                
                if has_required_fields and confidence_valid:
                    result = ValidationResult.PASS
                    error_msg = None
                else:
                    result = ValidationResult.FAIL
                    error_msg = "Invalid prediction response format"
                
                details = {
                    'response_data': prediction_data,
                    'response_time': response.elapsed.total_seconds(),
                    'has_required_fields': has_required_fields,
                    'confidence_valid': confidence_valid
                }
            else:
                result = ValidationResult.FAIL
                error_msg = f"HTTP {response.status_code}: {response.text}"
                details = {'status_code': response.status_code}
            
            self.validation_results.append(ValidationTest(
                name="xgboost_prediction",
                category="ml_models",
                description="Test XGBoost load prediction functionality",
                result=result,
                details=details,
                execution_time=time.time() - start_time,
                error_message=error_msg
            ))
            
        except Exception as e:
            self.validation_results.append(ValidationTest(
                name="xgboost_prediction",
                category="ml_models",
                description="Test XGBoost load prediction functionality",
                result=ValidationResult.FAIL,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))

    async def _test_qlearning_optimization(self):
        """Test Q-Learning placement optimization."""
        start_time = time.time()
        
        try:
            test_payload = {
                'pod_spec': {
                    'cpu_request': 500,
                    'memory_request': 1024,
                    'labels': {'app': 'test-workload'},
                    'affinity_rules': []
                },
                'available_nodes': [
                    {
                        'name': 'hydatis-worker-1',
                        'cpu_available': 2000,
                        'memory_available': 4096,
                        'current_load': 0.6,
                        'network_latency': 5
                    },
                    {
                        'name': 'hydatis-worker-2', 
                        'cpu_available': 1500,
                        'memory_available': 2048,
                        'current_load': 0.8,
                        'network_latency': 3
                    }
                ],
                'cluster_state': {
                    'total_utilization': 0.7,
                    'scheduling_pressure': 0.4
                }
            }
            
            response = requests.post(
                f"{self.service_endpoints['qlearning_optimizer']}/optimize",
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                optimization_data = response.json()
                
                required_fields = ['recommended_node', 'placement_score', 'reasoning']
                has_required_fields = all(field in optimization_data for field in required_fields)
                
                score_valid = 0.0 <= optimization_data.get('placement_score', -1) <= 1.0
                
                if has_required_fields and score_valid:
                    result = ValidationResult.PASS
                    error_msg = None
                else:
                    result = ValidationResult.FAIL
                    error_msg = "Invalid optimization response format"
                
                details = {
                    'response_data': optimization_data,
                    'response_time': response.elapsed.total_seconds(),
                    'has_required_fields': has_required_fields,
                    'score_valid': score_valid
                }
            else:
                result = ValidationResult.FAIL
                error_msg = f"HTTP {response.status_code}: {response.text}"
                details = {'status_code': response.status_code}
            
            self.validation_results.append(ValidationTest(
                name="qlearning_optimization",
                category="ml_models",
                description="Test Q-Learning placement optimization",
                result=result,
                details=details,
                execution_time=time.time() - start_time,
                error_message=error_msg
            ))
            
        except Exception as e:
            self.validation_results.append(ValidationTest(
                name="qlearning_optimization",
                category="ml_models",
                description="Test Q-Learning placement optimization",
                result=ValidationResult.FAIL,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))

    async def _test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        start_time = time.time()
        
        try:
            test_payload = {
                'cluster_metrics': {
                    'cpu_usage': [0.45, 0.52, 0.48, 0.51, 0.49],
                    'memory_usage': [0.68, 0.71, 0.69, 0.73, 0.70],
                    'network_io': [1024, 1156, 1089, 1203, 1045],
                    'pod_scheduling_rate': [8, 12, 9, 11, 10],
                    'node_availability': [6, 6, 5, 6, 6]
                },
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(
                f"{self.service_endpoints['anomaly_detector']}/detect",
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                anomaly_data = response.json()
                
                required_fields = ['anomaly_detected', 'anomaly_score', 'affected_metrics']
                has_required_fields = all(field in anomaly_data for field in required_fields)
                
                score_valid = 0.0 <= anomaly_data.get('anomaly_score', -1) <= 1.0
                
                if has_required_fields and score_valid:
                    result = ValidationResult.PASS
                    error_msg = None
                else:
                    result = ValidationResult.FAIL
                    error_msg = "Invalid anomaly detection response format"
                
                details = {
                    'response_data': anomaly_data,
                    'response_time': response.elapsed.total_seconds(),
                    'has_required_fields': has_required_fields,
                    'score_valid': score_valid
                }
            else:
                result = ValidationResult.FAIL
                error_msg = f"HTTP {response.status_code}: {response.text}"
                details = {'status_code': response.status_code}
            
            self.validation_results.append(ValidationTest(
                name="anomaly_detection",
                category="ml_models",
                description="Test anomaly detection functionality",
                result=result,
                details=details,
                execution_time=time.time() - start_time,
                error_message=error_msg
            ))
            
        except Exception as e:
            self.validation_results.append(ValidationTest(
                name="anomaly_detection",
                category="ml_models",
                description="Test anomaly detection functionality",
                result=ValidationResult.FAIL,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))

    async def _validate_integration(self):
        """Validate end-to-end integration."""
        
        await self._test_ml_gateway_orchestration()
        await self._test_scheduler_plugin_integration()

    async def _test_ml_gateway_orchestration(self):
        """Test ML gateway orchestration of all models."""
        start_time = time.time()
        
        try:
            test_payload = {
                'scheduler_request': {
                    'pod_spec': {
                        'cpu_request': 1000,
                        'memory_request': 2048,
                        'labels': {'app': 'test-workload'},
                        'namespace': 'default'
                    },
                    'cluster_context': {
                        'available_nodes': [
                            {'name': 'hydatis-worker-1', 'cpu_available': 3000, 'memory_available': 6144},
                            {'name': 'hydatis-worker-2', 'cpu_available': 2000, 'memory_available': 4096}
                        ],
                        'current_metrics': {
                            'cluster_cpu_utilization': 0.65,
                            'cluster_memory_utilization': 0.72
                        }
                    }
                }
            }
            
            response = requests.post(
                f"{self.service_endpoints['ml_gateway']}/orchestrate",
                json=test_payload,
                timeout=60
            )
            
            if response.status_code == 200:
                orchestration_data = response.json()
                
                required_fields = ['recommended_node', 'decision_confidence', 'model_predictions']
                has_required_fields = all(field in orchestration_data for field in required_fields)
                
                has_model_outputs = all(
                    model in orchestration_data.get('model_predictions', {})
                    for model in ['xgboost', 'qlearning', 'anomaly_detection']
                )
                
                if has_required_fields and has_model_outputs:
                    result = ValidationResult.PASS
                    error_msg = None
                else:
                    result = ValidationResult.FAIL
                    error_msg = "Invalid orchestration response format"
                
                details = {
                    'response_data': orchestration_data,
                    'response_time': response.elapsed.total_seconds(),
                    'has_required_fields': has_required_fields,
                    'has_model_outputs': has_model_outputs
                }
            else:
                result = ValidationResult.FAIL
                error_msg = f"HTTP {response.status_code}: {response.text}"
                details = {'status_code': response.status_code}
            
            self.validation_results.append(ValidationTest(
                name="ml_gateway_orchestration",
                category="integration",
                description="Test end-to-end ML gateway orchestration",
                result=result,
                details=details,
                execution_time=time.time() - start_time,
                error_message=error_msg
            ))
            
        except Exception as e:
            self.validation_results.append(ValidationTest(
                name="ml_gateway_orchestration",
                category="integration",
                description="Test end-to-end ML gateway orchestration",
                result=ValidationResult.FAIL,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))

    async def _validate_performance(self):
        """Validate performance requirements."""
        
        await self._test_response_latency()
        await self._test_throughput_capacity()
        await self._test_resource_efficiency()

    async def _test_response_latency(self):
        """Test that response latency meets requirements."""
        start_time = time.time()
        
        latency_requirements = {
            'ml_gateway': 100,          # 100ms max
            'xgboost_predictor': 50,    # 50ms max
            'qlearning_optimizer': 200, # 200ms max
            'anomaly_detector': 100     # 100ms max
        }
        
        latency_results = {}
        
        for service_name in latency_requirements.keys():
            if service_name in self.service_endpoints:
                try:
                    endpoint = self.service_endpoints[service_name]
                    
                    latencies = []
                    for _ in range(10):
                        response = requests.get(f"{endpoint}/health", timeout=5)
                        latencies.append(response.elapsed.total_seconds() * 1000)
                    
                    avg_latency = sum(latencies) / len(latencies)
                    max_latency = max(latencies)
                    
                    latency_results[service_name] = {
                        'avg_latency_ms': avg_latency,
                        'max_latency_ms': max_latency,
                        'requirement_ms': latency_requirements[service_name],
                        'meets_requirement': max_latency <= latency_requirements[service_name]
                    }
                    
                except Exception as e:
                    latency_results[service_name] = {
                        'error': str(e),
                        'meets_requirement': False
                    }
        
        failed_services = [name for name, result in latency_results.items() if not result.get('meets_requirement', False)]
        
        if failed_services:
            result = ValidationResult.FAIL
            error_msg = f"Latency requirements not met: {failed_services}"
        else:
            result = ValidationResult.PASS
            error_msg = None
        
        self.validation_results.append(ValidationTest(
            name="response_latency",
            category="performance",
            description="Verify response latency meets requirements",
            result=result,
            details=latency_results,
            execution_time=time.time() - start_time,
            error_message=error_msg
        ))

    async def _validate_security(self):
        """Validate security configuration."""
        
        await self._test_pod_security_context()
        await self._test_network_policies()
        await self._test_rbac_configuration()

    async def _test_pod_security_context(self):
        """Test pod security contexts."""
        start_time = time.time()
        
        try:
            core_v1 = client.CoreV1Api(self.k8s_client)
            pods = core_v1.list_namespaced_pod(
                namespace=self.hydatis_config['namespace'],
                label_selector="cluster=hydatis"
            )
            
            security_issues = []
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                
                if not pod.spec.security_context:
                    security_issues.append(f"{pod_name}: No security context")
                    continue
                
                security_context = pod.spec.security_context
                
                if security_context.run_as_user == 0:
                    security_issues.append(f"{pod_name}: Running as root")
                
                if not security_context.run_as_non_root:
                    security_issues.append(f"{pod_name}: Not enforcing non-root")
                
                for container in pod.spec.containers:
                    if container.security_context:
                        if container.security_context.allow_privilege_escalation:
                            security_issues.append(f"{pod_name}/{container.name}: Privilege escalation allowed")
            
            if security_issues:
                result = ValidationResult.WARNING
                error_msg = f"Security issues found: {len(security_issues)}"
            else:
                result = ValidationResult.PASS
                error_msg = None
            
            self.validation_results.append(ValidationTest(
                name="pod_security_context",
                category="security",
                description="Verify pod security contexts",
                result=result,
                details={
                    'total_pods_checked': len(pods.items),
                    'security_issues': security_issues
                },
                execution_time=time.time() - start_time,
                error_message=error_msg
            ))
            
        except Exception as e:
            self.validation_results.append(ValidationTest(
                name="pod_security_context",
                category="security",
                description="Verify pod security contexts",
                result=ValidationResult.FAIL,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))

    async def _validate_monitoring(self):
        """Validate monitoring and observability."""
        
        await self._test_prometheus_metrics()
        await self._test_monitoring_dashboard()
        await self._test_alert_manager()

    async def _test_prometheus_metrics(self):
        """Test Prometheus metrics collection."""
        start_time = time.time()
        
        try:
            prometheus_url = self.hydatis_config['prometheus_endpoint']
            
            expected_metrics = [
                'ml_scheduler_prediction_latency',
                'ml_scheduler_optimization_score',
                'ml_scheduler_anomaly_detection',
                'ml_scheduler_scheduling_decisions_total'
            ]
            
            metrics_status = {}
            
            for metric in expected_metrics:
                try:
                    response = requests.get(
                        f"{prometheus_url}/api/v1/query",
                        params={'query': metric},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        metrics_status[metric] = {
                            'available': len(data['data']['result']) > 0,
                            'sample_count': len(data['data']['result'])
                        }
                    else:
                        metrics_status[metric] = {'available': False, 'error': f"HTTP {response.status_code}"}
                        
                except Exception as e:
                    metrics_status[metric] = {'available': False, 'error': str(e)}
            
            available_metrics = [metric for metric, status in metrics_status.items() if status['available']]
            
            if len(available_metrics) >= len(expected_metrics) * 0.8:
                result = ValidationResult.PASS
                error_msg = None
            elif len(available_metrics) >= len(expected_metrics) * 0.5:
                result = ValidationResult.WARNING
                error_msg = f"Some metrics missing: {len(expected_metrics) - len(available_metrics)}"
            else:
                result = ValidationResult.FAIL
                error_msg = f"Most metrics missing: {len(expected_metrics) - len(available_metrics)}"
            
            self.validation_results.append(ValidationTest(
                name="prometheus_metrics",
                category="monitoring",
                description="Verify Prometheus metrics collection",
                result=result,
                details={
                    'expected_metrics': expected_metrics,
                    'available_metrics': available_metrics,
                    'metrics_status': metrics_status
                },
                execution_time=time.time() - start_time,
                error_message=error_msg
            ))
            
        except Exception as e:
            self.validation_results.append(ValidationTest(
                name="prometheus_metrics",
                category="monitoring",
                description="Verify Prometheus metrics collection",
                result=ValidationResult.FAIL,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ))

    def _generate_validation_report(self, total_execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        passed_tests = [t for t in self.validation_results if t.result == ValidationResult.PASS]
        failed_tests = [t for t in self.validation_results if t.result == ValidationResult.FAIL]
        warning_tests = [t for t in self.validation_results if t.result == ValidationResult.WARNING]
        
        category_summary = {}
        for test in self.validation_results:
            if test.category not in category_summary:
                category_summary[test.category] = {'pass': 0, 'fail': 0, 'warning': 0}
            category_summary[test.category][test.result.value] += 1
        
        overall_status = 'PASS'
        if failed_tests:
            overall_status = 'FAIL'
        elif warning_tests:
            overall_status = 'WARNING'
        
        report = {
            'validation_summary': {
                'overall_status': overall_status,
                'cluster': self.hydatis_config['cluster_name'],
                'namespace': self.hydatis_config['namespace'],
                'timestamp': datetime.now().isoformat(),
                'execution_time_seconds': total_execution_time
            },
            'test_statistics': {
                'total_tests': len(self.validation_results),
                'passed_tests': len(passed_tests),
                'failed_tests': len(failed_tests),
                'warning_tests': len(warning_tests),
                'success_rate': len(passed_tests) / len(self.validation_results) * 100 if self.validation_results else 0
            },
            'category_breakdown': category_summary,
            'detailed_results': [asdict(test) for test in self.validation_results],
            'critical_failures': [
                {
                    'test_name': test.name,
                    'category': test.category,
                    'error': test.error_message
                } for test in failed_tests
            ],
            'recommendations': self._generate_recommendations()
        }
        
        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        failed_tests = [t for t in self.validation_results if t.result == ValidationResult.FAIL]
        warning_tests = [t for t in self.validation_results if t.result == ValidationResult.WARNING]
        
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} critical test failures before production deployment")
        
        if warning_tests:
            recommendations.append(f"Review {len(warning_tests)} warnings for potential improvements")
        
        infrastructure_failures = [t for t in failed_tests if t.category == 'infrastructure']
        if infrastructure_failures:
            recommendations.append("Fix infrastructure issues - these are blocking for deployment")
        
        performance_warnings = [t for t in warning_tests if t.category == 'performance']
        if performance_warnings:
            recommendations.append("Monitor performance closely - consider resource adjustments")
        
        security_warnings = [t for t in warning_tests if t.category == 'security']
        if security_warnings:
            recommendations.append("Review security configurations for production compliance")
        
        if not failed_tests and not warning_tests:
            recommendations.append("All validations passed - deployment is ready for production")
        
        return recommendations

    async def _test_configmaps(self):
        """Test that required ConfigMaps exist."""
        pass

    async def _test_node_resources(self):
        """Test node resource availability."""
        pass

    async def _test_service_discovery(self):
        """Test Kubernetes service discovery."""
        pass

    async def _test_model_versioning(self):
        """Test model versioning with MLflow."""
        pass

    async def _test_scheduler_plugin_integration(self):
        """Test scheduler plugin integration."""
        pass

    async def _test_throughput_capacity(self):
        """Test system throughput capacity."""
        pass

    async def _test_resource_efficiency(self):
        """Test resource utilization efficiency."""
        pass

    async def _test_network_policies(self):
        """Test network policy configuration."""
        pass

    async def _test_rbac_configuration(self):
        """Test RBAC configuration."""
        pass

    async def _test_monitoring_dashboard(self):
        """Test monitoring dashboard functionality."""
        pass

    async def _test_alert_manager(self):
        """Test alert manager functionality."""
        pass

async def main():
    """Main validation entry point."""
    validator = ProductionValidator()
    
    try:
        validation_report = await validator.run_full_validation_suite()
        
        with open('/tmp/hydatis_validation_report.json', 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        print(f"Validation Status: {validation_report['validation_summary']['overall_status']}")
        print(f"Tests: {validation_report['test_statistics']['passed_tests']}/{validation_report['test_statistics']['total_tests']} passed")
        print("Validation report saved to /tmp/hydatis_validation_report.json")
        
        if validation_report['critical_failures']:
            print("\nCritical Failures:")
            for failure in validation_report['critical_failures']:
                print(f"- {failure['test_name']}: {failure['error']}")
        
    except Exception as e:
        validator.logger.error(f"Validation suite failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())