#!/usr/bin/env python3
"""
Unified ML Scheduler API Gateway for HYDATIS cluster.
Orchestrates XGBoost Load Predictor, Q-Learning Placement Optimizer, and Isolation Forest Anomaly Detector.
"""

import asyncio
import aiohttp
from aiohttp import web, ClientSession
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)


class HYDATISMLSchedulerGateway:
    """Unified API Gateway orchestrating all ML scheduler components."""
    
    def __init__(self):
        self.service_endpoints = {
            'xgboost_load_predictor': {
                'url': 'http://localhost:8080',
                'health_endpoint': '/health',
                'predict_node': '/predict/node',
                'predict_cluster': '/predict/cluster'
            },
            'qlearning_placement': {
                'url': 'http://localhost:8081', 
                'health_endpoint': '/health',
                'optimize_placement': '/optimize/placement',
                'cluster_insights': '/insights/cluster',
                'rebalancing': '/insights/rebalancing'
            },
            'isolation_forest_anomaly': {
                'url': 'http://localhost:8082',
                'health_endpoint': '/health',
                'detect_anomalies': '/detect/anomalies',
                'anomaly_probability': '/predict/anomaly_probability',
                'cluster_status': '/status/cluster',
                'alerts_dashboard': '/alerts/dashboard'
            }
        }
        
        self.gateway_config = {
            'request_timeout_seconds': 30,
            'retry_attempts': 3,
            'circuit_breaker_threshold': 5,
            'health_check_interval': 60,
            'load_balancing': True,
            'caching_enabled': True,
            'cache_ttl_seconds': 30
        }
        
        self.service_health = {
            service: {'status': 'unknown', 'last_check': None, 'consecutive_failures': 0}
            for service in self.service_endpoints.keys()
        }
        
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'service_request_counts': {service: 0 for service in self.service_endpoints.keys()},
            'average_response_times': {service: 0.0 for service in self.service_endpoints.keys()},
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.request_cache = {}
        self.cache_timestamps = {}
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        self._start_health_monitoring()
    
    async def orchestrate_scheduler_decision(self, scheduler_request: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate complete ML scheduler decision using all three models."""
        
        request_start = time.time()
        
        try:
            self.performance_metrics['total_requests'] += 1
            
            pod_spec = scheduler_request.get('pod_spec', {})
            cluster_context = scheduler_request.get('cluster_context', {})
            available_nodes = scheduler_request.get('available_nodes')
            
            orchestration_tasks = []
            
            load_prediction_task = asyncio.create_task(
                self._get_load_predictions(cluster_context)
            )
            orchestration_tasks.append(('load_prediction', load_prediction_task))
            
            placement_optimization_task = asyncio.create_task(
                self._get_placement_optimization(pod_spec, available_nodes)
            )
            orchestration_tasks.append(('placement_optimization', placement_optimization_task))
            
            anomaly_detection_task = asyncio.create_task(
                self._get_anomaly_detection(cluster_context)
            )
            orchestration_tasks.append(('anomaly_detection', anomaly_detection_task))
            
            results = {}
            
            for task_name, task in orchestration_tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=self.gateway_config['request_timeout_seconds'])
                    results[task_name] = result
                except asyncio.TimeoutError:
                    results[task_name] = {'error': 'Service timeout', 'timeout': True}
                except Exception as e:
                    results[task_name] = {'error': str(e), 'service_error': True}
            
            orchestrated_decision = self._synthesize_ml_decision(
                results.get('load_prediction', {}),
                results.get('placement_optimization', {}),
                results.get('anomaly_detection', {}),
                pod_spec
            )
            
            total_latency = (time.time() - request_start) * 1000
            
            orchestrated_decision['orchestration_metrics'] = {
                'total_latency_ms': round(total_latency, 2),
                'services_consulted': len([r for r in results.values() if 'error' not in r]),
                'service_results': {k: 'success' if 'error' not in v else 'failed' for k, v in results.items()},
                'orchestration_timestamp': datetime.now().isoformat()
            }
            
            self.performance_metrics['successful_requests'] += 1
            
            return orchestrated_decision
            
        except Exception as e:
            self.performance_metrics['failed_requests'] += 1
            logger.error(f"Orchestration error: {e}")
            
            return {
                'error': str(e),
                'fallback_decision': self._generate_fallback_decision(scheduler_request),
                'orchestration_metrics': {
                    'total_latency_ms': round((time.time() - request_start) * 1000, 2),
                    'error_mode': True
                },
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_load_predictions(self, cluster_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get load predictions from XGBoost service."""
        
        cache_key = f"load_prediction_{hash(json.dumps(cluster_context, sort_keys=True))}"
        
        if self._check_cache(cache_key):
            self.performance_metrics['cache_hits'] += 1
            return self.request_cache[cache_key]
        
        self.performance_metrics['cache_misses'] += 1
        
        service_url = self.service_endpoints['xgboost_load_predictor']['url']
        endpoint = self.service_endpoints['xgboost_load_predictor']['predict_cluster']
        
        try:
            async with ClientSession() as session:
                async with session.post(
                    f"{service_url}{endpoint}",
                    json=cluster_context,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        self._cache_result(cache_key, result)
                        self.performance_metrics['service_request_counts']['xgboost_load_predictor'] += 1
                        return result
                    else:
                        return {'error': f'Load prediction service error: {response.status}'}
        
        except Exception as e:
            return {'error': f'Load prediction service unavailable: {str(e)}'}
    
    async def _get_placement_optimization(self, pod_spec: Dict[str, Any], 
                                        available_nodes: Optional[List[str]]) -> Dict[str, Any]:
        """Get placement optimization from Q-Learning service."""
        
        service_url = self.service_endpoints['qlearning_placement']['url']
        endpoint = self.service_endpoints['qlearning_placement']['optimize_placement']
        
        try:
            params = {}
            if available_nodes:
                params['available_nodes'] = ','.join(available_nodes)
            
            async with ClientSession() as session:
                async with session.post(
                    f"{service_url}{endpoint}",
                    json=pod_spec,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        self.performance_metrics['service_request_counts']['qlearning_placement'] += 1
                        return result
                    else:
                        return {'error': f'Placement optimization service error: {response.status}'}
        
        except Exception as e:
            return {'error': f'Placement optimization service unavailable: {str(e)}'}
    
    async def _get_anomaly_detection(self, cluster_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get anomaly detection from Isolation Forest service."""
        
        cache_key = f"anomaly_detection_{hash(json.dumps(cluster_context, sort_keys=True))}"
        
        if self._check_cache(cache_key):
            self.performance_metrics['cache_hits'] += 1
            return self.request_cache[cache_key]
        
        self.performance_metrics['cache_misses'] += 1
        
        service_url = self.service_endpoints['isolation_forest_anomaly']['url']
        endpoint = self.service_endpoints['isolation_forest_anomaly']['cluster_status']
        
        try:
            async with ClientSession() as session:
                async with session.get(
                    f"{service_url}{endpoint}",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        self._cache_result(cache_key, result, ttl=60)
                        self.performance_metrics['service_request_counts']['isolation_forest_anomaly'] += 1
                        return result
                    else:
                        return {'error': f'Anomaly detection service error: {response.status}'}
        
        except Exception as e:
            return {'error': f'Anomaly detection service unavailable: {str(e)}'}
    
    def _synthesize_ml_decision(self, 
                               load_prediction: Dict[str, Any],
                               placement_optimization: Dict[str, Any], 
                               anomaly_detection: Dict[str, Any],
                               pod_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final scheduling decision from all ML models."""
        
        synthesis_start = time.time()
        
        decision = {
            'pod_name': pod_spec.get('metadata', {}).get('name', 'unknown'),
            'namespace': pod_spec.get('metadata', {}).get('namespace', 'default'),
            'decision_timestamp': datetime.now().isoformat(),
            'ml_models_consulted': []
        }
        
        if 'error' not in load_prediction:
            decision['ml_models_consulted'].append('xgboost_load_predictor')
            decision['load_prediction'] = {
                'cluster_summary': load_prediction.get('cluster_summary', {}),
                'scheduling_recommendations': load_prediction.get('scheduling_recommendations', {}),
                'capacity_forecast': self._extract_capacity_forecast(load_prediction)
            }
        
        if 'error' not in placement_optimization:
            decision['ml_models_consulted'].append('qlearning_placement')
            decision['placement_optimization'] = {
                'selected_node': placement_optimization.get('selected_node'),
                'placement_reasoning': placement_optimization.get('placement_reasoning', {}),
                'optimization_impact': placement_optimization.get('optimization_impact', {}),
                'cluster_insights': placement_optimization.get('cluster_insights', {})
            }
        
        if 'error' not in anomaly_detection:
            decision['ml_models_consulted'].append('isolation_forest_anomaly')
            decision['anomaly_assessment'] = {
                'cluster_health': anomaly_detection.get('cluster_health_assessment', {}),
                'active_issues': anomaly_detection.get('active_issues', {}),
                'anomaly_risk_level': self._assess_anomaly_risk(anomaly_detection)
            }
        
        final_recommendation = self._generate_final_recommendation(
            decision.get('load_prediction', {}),
            decision.get('placement_optimization', {}),
            decision.get('anomaly_assessment', {}),
            pod_spec
        )
        
        decision['final_recommendation'] = final_recommendation
        
        decision['synthesis_metrics'] = {
            'models_successful': len(decision['ml_models_consulted']),
            'models_failed': 3 - len(decision['ml_models_consulted']),
            'decision_confidence': self._calculate_decision_confidence(decision),
            'synthesis_latency_ms': round((time.time() - synthesis_start) * 1000, 2)
        }
        
        return decision
    
    def _extract_capacity_forecast(self, load_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract capacity forecast from load prediction."""
        
        cluster_summary = load_prediction.get('cluster_summary', {})
        
        return {
            'average_cpu_prediction': cluster_summary.get('avg_cpu_prediction', 0),
            'average_memory_prediction': cluster_summary.get('avg_memory_prediction', 0),
            'best_capacity_node': cluster_summary.get('best_node', 'unknown'),
            'most_loaded_node': cluster_summary.get('most_loaded_node', 'unknown'),
            'cluster_capacity_score': 1 - max(cluster_summary.get('avg_cpu_prediction', 0), 
                                            cluster_summary.get('avg_memory_prediction', 0))
        }
    
    def _assess_anomaly_risk(self, anomaly_detection: Dict[str, Any]) -> str:
        """Assess anomaly risk level for scheduling decisions."""
        
        cluster_health = anomaly_detection.get('cluster_health_assessment', {}).get('overall_health', 'healthy')
        active_issues = anomaly_detection.get('active_issues', {})
        
        critical_alerts = active_issues.get('critical_active', 0)
        high_alerts = active_issues.get('high_active', 0)
        
        if cluster_health == 'critical' or critical_alerts > 0:
            return 'high'
        elif cluster_health == 'degraded' or high_alerts > 2:
            return 'medium'
        elif cluster_health == 'warning' or high_alerts > 0:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_final_recommendation(self, 
                                     load_prediction: Dict[str, Any],
                                     placement_optimization: Dict[str, Any],
                                     anomaly_assessment: Dict[str, Any],
                                     pod_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final scheduling recommendation based on all ML inputs."""
        
        recommendation = {
            'action': 'schedule',
            'selected_node': None,
            'confidence': 'medium',
            'reasoning': [],
            'risk_factors': [],
            'alternative_nodes': []
        }
        
        if placement_optimization and 'selected_node' in placement_optimization:
            primary_node = placement_optimization['selected_node']
            recommendation['selected_node'] = primary_node
            
            placement_quality = placement_optimization.get('placement_reasoning', {}).get('quality_score', 0)
            if placement_quality > 0.8:
                recommendation['confidence'] = 'high'
                recommendation['reasoning'].append(f"Q-Learning optimizer recommends {primary_node} with high confidence")
            elif placement_quality > 0.6:
                recommendation['confidence'] = 'medium'
                recommendation['reasoning'].append(f"Q-Learning optimizer recommends {primary_node} with moderate confidence")
            else:
                recommendation['reasoning'].append(f"Q-Learning recommendation for {primary_node} has low confidence")
        
        if load_prediction and 'capacity_forecast' in load_prediction:
            capacity = load_prediction['capacity_forecast']
            
            if capacity['cluster_capacity_score'] > 0.7:
                recommendation['reasoning'].append("Load predictor indicates good cluster capacity")
            elif capacity['cluster_capacity_score'] > 0.4:
                recommendation['reasoning'].append("Load predictor indicates moderate cluster capacity")
            else:
                recommendation['risk_factors'].append("Load predictor indicates low cluster capacity")
            
            best_capacity_node = capacity.get('best_capacity_node')
            if best_capacity_node and best_capacity_node != recommendation['selected_node']:
                recommendation['alternative_nodes'].append({
                    'node': best_capacity_node,
                    'reason': 'Best capacity according to load predictor'
                })
        
        if anomaly_assessment:
            anomaly_risk = anomaly_assessment.get('anomaly_risk_level', 'minimal')
            
            if anomaly_risk == 'high':
                recommendation['action'] = 'defer'
                recommendation['reasoning'].append("High anomaly risk detected - deferring scheduling")
                recommendation['risk_factors'].append("Critical cluster anomalies detected")
            elif anomaly_risk == 'medium':
                recommendation['risk_factors'].append("Moderate anomaly risk - proceed with caution")
            elif anomaly_risk == 'low':
                recommendation['reasoning'].append("Low anomaly risk - normal scheduling conditions")
        
        resource_requirements = pod_spec.get('resources', {})
        if resource_requirements.get('cpu_request', 0) > 0.5 or resource_requirements.get('memory_request', 0) > 0.5:
            recommendation['risk_factors'].append("High resource requirements - monitor placement impact")
        
        if not recommendation['selected_node']:
            if load_prediction and 'capacity_forecast' in load_prediction:
                recommendation['selected_node'] = load_prediction['capacity_forecast'].get('best_capacity_node', 'worker-1')
                recommendation['reasoning'].append("Fallback to load predictor recommendation")
            else:
                recommendation['selected_node'] = 'worker-1'
                recommendation['reasoning'].append("Fallback to default node selection")
        
        if len(recommendation['risk_factors']) > 2:
            recommendation['confidence'] = 'low'
        elif len(recommendation['risk_factors']) > 0:
            if recommendation['confidence'] == 'high':
                recommendation['confidence'] = 'medium'
        
        return recommendation
    
    def _generate_fallback_decision(self, scheduler_request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback decision when ML services are unavailable."""
        
        pod_spec = scheduler_request.get('pod_spec', {})
        available_nodes = scheduler_request.get('available_nodes', ['worker-1', 'worker-2', 'worker-3'])
        
        cpu_request = pod_spec.get('resources', {}).get('cpu_request', 0.1)
        memory_request = pod_spec.get('resources', {}).get('memory_request', 0.1)
        
        selected_node = available_nodes[0]
        if cpu_request < 0.2 and memory_request < 0.2:
            selected_node = available_nodes[0]
        elif len(available_nodes) > 1:
            selected_node = available_nodes[1]
        
        fallback = {
            'action': 'schedule',
            'selected_node': selected_node,
            'confidence': 'low',
            'reasoning': ['Fallback decision - ML services unavailable'],
            'risk_factors': ['ML-based optimization not available'],
            'fallback_mode': True,
            'fallback_strategy': 'round_robin_capacity_based'
        }
        
        return fallback
    
    def _calculate_decision_confidence(self, decision: Dict[str, Any]) -> float:
        """Calculate overall decision confidence score."""
        
        models_successful = decision.get('synthesis_metrics', {}).get('models_successful', 0)
        base_confidence = models_successful / 3.0
        
        if 'placement_optimization' in decision:
            placement_confidence = decision['placement_optimization'].get('placement_reasoning', {}).get('confidence', 0)
            base_confidence += placement_confidence * 0.3
        
        risk_factors = len(decision.get('final_recommendation', {}).get('risk_factors', []))
        risk_penalty = min(risk_factors * 0.1, 0.3)
        
        final_confidence = max(0.1, min(1.0, base_confidence - risk_penalty))
        
        return final_confidence
    
    def _check_cache(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        
        if cache_key not in self.request_cache:
            return False
        
        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time:
            return False
        
        cache_age = (datetime.now() - cache_time).total_seconds()
        return cache_age < self.gateway_config['cache_ttl_seconds']
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any], ttl: int = None):
        """Cache request result."""
        
        if not self.gateway_config['caching_enabled']:
            return
        
        self.request_cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.now()
        
        if len(self.request_cache) > 1000:
            oldest_keys = sorted(self.cache_timestamps.items(), key=lambda x: x[1])[:100]
            for key, _ in oldest_keys:
                self.request_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
    
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of individual ML service."""
        
        if service_name not in self.service_endpoints:
            return {'error': 'Unknown service'}
        
        service_config = self.service_endpoints[service_name]
        health_url = f"{service_config['url']}{service_config['health_endpoint']}"
        
        try:
            start_time = time.time()
            
            async with ClientSession() as session:
                async with session.get(
                    health_url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    
                    latency = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        health_data = await response.json()
                        
                        self.service_health[service_name] = {
                            'status': 'healthy',
                            'last_check': datetime.now().isoformat(),
                            'consecutive_failures': 0,
                            'response_time_ms': round(latency, 2)
                        }
                        
                        return {
                            'service': service_name,
                            'status': 'healthy',
                            'response_time_ms': round(latency, 2),
                            'service_health': health_data
                        }
                    else:
                        self._record_service_failure(service_name)
                        return {
                            'service': service_name,
                            'status': 'unhealthy',
                            'error': f'HTTP {response.status}',
                            'response_time_ms': round(latency, 2)
                        }
        
        except Exception as e:
            self._record_service_failure(service_name)
            return {
                'service': service_name,
                'status': 'unavailable',
                'error': str(e)
            }
    
    def _record_service_failure(self, service_name: str):
        """Record service failure for circuit breaker logic."""
        
        if service_name in self.service_health:
            self.service_health[service_name]['consecutive_failures'] += 1
            self.service_health[service_name]['status'] = 'unhealthy'
            self.service_health[service_name]['last_check'] = datetime.now().isoformat()
    
    def _start_health_monitoring(self):
        """Start background health monitoring for all services."""
        
        def health_monitor():
            while True:
                try:
                    for service_name in self.service_endpoints.keys():
                        health_result = asyncio.run(self.check_service_health(service_name))
                        
                        if health_result.get('status') == 'healthy':
                            self.service_health[service_name]['consecutive_failures'] = 0
                    
                    time.sleep(self.gateway_config['health_check_interval'])
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(30)
        
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
        
        logger.info("Service health monitoring started")
    
    async def get_comprehensive_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive status from all ML services."""
        
        status_tasks = []
        
        for service_name in self.service_endpoints.keys():
            task = asyncio.create_task(self.check_service_health(service_name))
            status_tasks.append((service_name, task))
        
        service_statuses = {}
        
        for service_name, task in status_tasks:
            try:
                result = await asyncio.wait_for(task, timeout=10)
                service_statuses[service_name] = result
            except asyncio.TimeoutError:
                service_statuses[service_name] = {'service': service_name, 'status': 'timeout'}
            except Exception as e:
                service_statuses[service_name] = {'service': service_name, 'status': 'error', 'error': str(e)}
        
        overall_health = 'healthy'
        healthy_services = len([s for s in service_statuses.values() if s.get('status') == 'healthy'])
        
        if healthy_services == 0:
            overall_health = 'critical'
        elif healthy_services < 2:
            overall_health = 'degraded'
        elif healthy_services < 3:
            overall_health = 'partial'
        
        comprehensive_status = {
            'overall_health': overall_health,
            'services_healthy': healthy_services,
            'total_services': len(self.service_endpoints),
            'service_details': service_statuses,
            'gateway_performance': {
                'total_requests': self.performance_metrics['total_requests'],
                'success_rate': (self.performance_metrics['successful_requests'] / 
                               max(self.performance_metrics['total_requests'], 1)),
                'cache_hit_rate': (self.performance_metrics['cache_hits'] / 
                                 max(self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'], 1)),
                'average_orchestration_latency_ms': self._calculate_average_latency()
            },
            'ml_scheduler_readiness': {
                'load_prediction_ready': service_statuses.get('xgboost_load_predictor', {}).get('status') == 'healthy',
                'placement_optimization_ready': service_statuses.get('qlearning_placement', {}).get('status') == 'healthy',
                'anomaly_detection_ready': service_statuses.get('isolation_forest_anomaly', {}).get('status') == 'healthy',
                'full_ml_pipeline_operational': healthy_services == 3
            },
            'status_timestamp': datetime.now().isoformat()
        }
        
        return comprehensive_status
    
    def _calculate_average_latency(self) -> float:
        """Calculate average orchestration latency."""
        
        if self.performance_metrics['successful_requests'] == 0:
            return 0.0
        
        total_latency = sum(self.performance_metrics['average_response_times'].values())
        return total_latency / len(self.service_endpoints)
    
    def get_gateway_metrics(self) -> Dict[str, Any]:
        """Get comprehensive gateway performance metrics."""
        
        uptime_minutes = 0
        
        metrics = {
            'gateway_uptime_minutes': uptime_minutes,
            'request_statistics': {
                'total_requests': self.performance_metrics['total_requests'],
                'successful_requests': self.performance_metrics['successful_requests'],
                'failed_requests': self.performance_metrics['failed_requests'],
                'success_rate': (self.performance_metrics['successful_requests'] / 
                               max(self.performance_metrics['total_requests'], 1))
            },
            'service_statistics': {
                'service_request_counts': self.performance_metrics['service_request_counts'],
                'service_health_status': {k: v['status'] for k, v in self.service_health.items()},
                'services_operational': len([v for v in self.service_health.values() if v['status'] == 'healthy'])
            },
            'caching_performance': {
                'cache_hits': self.performance_metrics['cache_hits'],
                'cache_misses': self.performance_metrics['cache_misses'],
                'cache_hit_rate': (self.performance_metrics['cache_hits'] / 
                                 max(self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'], 1)),
                'cached_entries': len(self.request_cache)
            },
            'ml_pipeline_effectiveness': {
                'load_prediction_usage_rate': (self.performance_metrics['service_request_counts']['xgboost_load_predictor'] / 
                                             max(self.performance_metrics['total_requests'], 1)),
                'placement_optimization_rate': (self.performance_metrics['service_request_counts']['qlearning_placement'] / 
                                              max(self.performance_metrics['total_requests'], 1)),
                'anomaly_detection_rate': (self.performance_metrics['service_request_counts']['isolation_forest_anomaly'] / 
                                         max(self.performance_metrics['total_requests'], 1))
            },
            'metrics_timestamp': datetime.now().isoformat()
        }
        
        return metrics


app = web.Application()


async def schedule_pod(request):
    """Main pod scheduling endpoint."""
    
    try:
        scheduler_request = await request.json()
        
        gateway = request.app['gateway']
        
        decision = await gateway.orchestrate_scheduler_decision(scheduler_request)
        
        return web.json_response(decision)
        
    except Exception as e:
        return web.json_response({'error': str(e)}, status=400)


async def cluster_status(request):
    """Comprehensive cluster status endpoint."""
    
    try:
        gateway = request.app['gateway']
        
        status = await gateway.get_comprehensive_cluster_status()
        
        return web.json_response(status)
        
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def gateway_metrics(request):
    """Gateway performance metrics endpoint."""
    
    try:
        gateway = request.app['gateway']
        
        metrics = gateway.get_gateway_metrics()
        
        return web.json_response(metrics)
        
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def health_check(request):
    """Gateway health check endpoint."""
    
    try:
        gateway = request.app['gateway']
        
        health_status = await gateway.get_comprehensive_cluster_status()
        
        overall_health = health_status['overall_health']
        status_code = 200 if overall_health in ['healthy', 'partial'] else 503
        
        health_summary = {
            'gateway_status': overall_health,
            'services_operational': health_status['services_healthy'],
            'total_services': health_status['total_services'],
            'ml_pipeline_ready': health_status['ml_scheduler_readiness']['full_ml_pipeline_operational'],
            'health_timestamp': datetime.now().isoformat()
        }
        
        return web.json_response(health_summary, status=status_code)
        
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


def setup_routes(app):
    """Setup API routes."""
    
    app.router.add_post('/schedule', schedule_pod)
    app.router.add_get('/status/cluster', cluster_status)
    app.router.add_get('/metrics/gateway', gateway_metrics)
    app.router.add_get('/health', health_check)


async def init_app():
    """Initialize application."""
    
    app['gateway'] = HYDATISMLSchedulerGateway()
    setup_routes(app)
    
    return app


def main():
    """Main gateway application."""
    
    print("HYDATIS ML Scheduler API Gateway - Week 8")
    print("Orchestrating XGBoost + Q-Learning + Isolation Forest")
    
    gateway = HYDATISMLSchedulerGateway()
    
    print("Service Endpoints:")
    for service, config in gateway.service_endpoints.items():
        print(f"  {service}: {config['url']}")
    
    print("Gateway Configuration:")
    print(f"  Request Timeout: {gateway.gateway_config['request_timeout_seconds']}s")
    print(f"  Retry Attempts: {gateway.gateway_config['retry_attempts']}")
    print(f"  Caching: {'✅ ENABLED' if gateway.gateway_config['caching_enabled'] else '❌ DISABLED'}")
    print(f"  Health Monitoring: {gateway.gateway_config['health_check_interval']}s interval")
    
    print("API Endpoints:")
    print("  POST /schedule - Main pod scheduling endpoint")
    print("  GET /status/cluster - Comprehensive cluster status")
    print("  GET /metrics/gateway - Gateway performance metrics")
    print("  GET /health - Gateway health check")
    
    return gateway


if __name__ == "__main__":
    gateway = main()
    
    if __name__ == "__main__":
        web.run_app(init_app(), host='0.0.0.0', port=8083)