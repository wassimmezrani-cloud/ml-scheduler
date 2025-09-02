import asyncio
import aiohttp
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .redis_cache import get_cache_manager
from .ab_testing import get_ab_testing, ExperimentResult

logger = logging.getLogger(__name__)

@dataclass
class ScoringRequest:
    """Request for combined ML scoring"""
    request_id: str
    pod_spec: Dict[str, Any]
    node_candidates: List[Dict[str, Any]]
    current_metrics: Dict[str, Any]
    user_id: str = "scheduler"
    priority_class: str = "normal"
    deadline_seconds: Optional[int] = None

@dataclass
class NodeScore:
    """Scoring result for a single node"""
    node_name: str
    total_score: float
    component_scores: Dict[str, float]
    confidence: float
    placement_recommendation: bool
    reasoning: List[str]

@dataclass
class CombinedScoringResult:
    """Combined scoring result from all ML models"""
    request_id: str
    node_scores: List[NodeScore]
    recommended_node: Optional[str]
    fallback_required: bool
    processing_time_ms: float
    model_responses: Dict[str, Any]
    anomaly_alerts: List[Dict[str, Any]]
    ab_test_metadata: Dict[str, Any]

class CombinedMLScorer:
    """
    Combined ML scorer that integrates all three models:
    - XGBoost Load Predictor
    - Q-Learning Placement Optimizer  
    - Isolation Forest Anomaly Detector
    """
    
    def __init__(self, 
                 xgboost_url: str = "http://xgboost-load-predictor:8080",
                 qlearning_url: str = "http://qlearning-placement-optimizer:8080", 
                 anomaly_url: str = "http://anomaly-detector:8080",
                 timeout: float = 30.0,
                 max_workers: int = 10):
        """
        Initialize combined scorer
        
        Args:
            xgboost_url: URL for XGBoost load predictor service
            qlearning_url: URL for Q-Learning placement optimizer service
            anomaly_url: URL for anomaly detector service
            timeout: Request timeout in seconds
            max_workers: Maximum worker threads for concurrent requests
        """
        self.xgboost_url = xgboost_url
        self.qlearning_url = qlearning_url
        self.anomaly_url = anomaly_url
        self.timeout = timeout
        
        self.cache_manager = get_cache_manager()
        self.ab_testing = get_ab_testing()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Scoring weights for combining model outputs
        self.scoring_weights = {
            'load_prediction': 0.4,      # 40% weight for load prediction
            'placement_optimization': 0.45,  # 45% weight for placement optimization
            'anomaly_detection': 0.15    # 15% weight for anomaly detection
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'model_timeouts': 0,
            'fallback_triggered': 0,
            'average_latency_ms': 0,
            'model_success_rates': {
                'xgboost': 0.0,
                'qlearning': 0.0,
                'anomaly': 0.0
            }
        }
        self._stats_lock = threading.Lock()
    
    async def _call_model_service(self, session: aiohttp.ClientSession, 
                                 url: str, endpoint: str, 
                                 payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Make async HTTP request to model service
        
        Returns:
            Tuple of (success, response_data)
        """
        try:
            full_url = f"{url}/{endpoint}"
            async with session.post(full_url, json=payload, timeout=self.timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    return True, data
                else:
                    error_text = await response.text()
                    logger.error(f"Model service error {response.status}: {error_text}")
                    return False, {'error': f'HTTP {response.status}: {error_text}'}
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling {url}/{endpoint}")
            with self._stats_lock:
                self.performance_stats['model_timeouts'] += 1
            return False, {'error': 'timeout'}
        except Exception as e:
            logger.error(f"Error calling {url}/{endpoint}: {e}")
            return False, {'error': str(e)}
    
    def _extract_node_features(self, node: Dict[str, Any], 
                              current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for load prediction"""
        node_name = node.get('name', 'unknown')
        
        # Get current node metrics
        node_metrics = current_metrics.get('nodes', {}).get(node_name, {})
        
        return {
            'cpu_allocatable': node.get('allocatable', {}).get('cpu', 0),
            'memory_allocatable': node.get('allocatable', {}).get('memory', 0),
            'cpu_used': node_metrics.get('cpu_usage', 0),
            'memory_used': node_metrics.get('memory_usage', 0),
            'pod_count': node_metrics.get('pod_count', 0),
            'disk_pressure': node.get('conditions', {}).get('DiskPressure', False),
            'memory_pressure': node.get('conditions', {}).get('MemoryPressure', False),
            'pid_pressure': node.get('conditions', {}).get('PIDPressure', False),
            'ready': node.get('conditions', {}).get('Ready', True),
            'node_labels': node.get('labels', {}),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _extract_placement_features(self, pod_spec: Dict[str, Any], 
                                   nodes: List[Dict[str, Any]],
                                   current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for placement optimization"""
        return {
            'pod_spec': {
                'cpu_request': pod_spec.get('resources', {}).get('requests', {}).get('cpu', 0),
                'memory_request': pod_spec.get('resources', {}).get('requests', {}).get('memory', 0),
                'cpu_limit': pod_spec.get('resources', {}).get('limits', {}).get('cpu', 0),
                'memory_limit': pod_spec.get('resources', {}).get('limits', {}).get('memory', 0),
                'priority_class': pod_spec.get('priorityClassName', 'normal'),
                'node_selector': pod_spec.get('nodeSelector', {}),
                'tolerations': pod_spec.get('tolerations', []),
                'affinity': pod_spec.get('affinity', {}),
                'volumes': len(pod_spec.get('volumes', []))
            },
            'cluster_state': {
                'total_nodes': len(nodes),
                'available_nodes': len([n for n in nodes if n.get('conditions', {}).get('Ready', True)]),
                'total_pods': sum(node.get('pod_count', 0) for node in current_metrics.get('nodes', {}).values()),
                'cluster_cpu_usage': current_metrics.get('cluster', {}).get('cpu_usage_percent', 0),
                'cluster_memory_usage': current_metrics.get('cluster', {}).get('memory_usage_percent', 0)
            },
            'nodes': [self._extract_node_features(node, current_metrics) for node in nodes],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _extract_anomaly_features(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for anomaly detection"""
        cluster_metrics = current_metrics.get('cluster', {})
        
        return {
            'cluster_metrics': {
                'cpu_usage_percent': cluster_metrics.get('cpu_usage_percent', 0),
                'memory_usage_percent': cluster_metrics.get('memory_usage_percent', 0),
                'disk_usage_percent': cluster_metrics.get('disk_usage_percent', 0),
                'network_io_bytes': cluster_metrics.get('network_io_bytes_per_sec', 0),
                'disk_io_bytes': cluster_metrics.get('disk_io_bytes_per_sec', 0),
                'pod_creation_rate': cluster_metrics.get('pod_creation_rate_per_min', 0),
                'pod_failure_rate': cluster_metrics.get('pod_failure_rate_per_min', 0),
                'scheduling_latency_p95': cluster_metrics.get('scheduling_latency_p95_ms', 0)
            },
            'node_metrics': current_metrics.get('nodes', {}),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def score_placement(self, request: ScoringRequest) -> CombinedScoringResult:
        """
        Score pod placement using combined ML models
        
        Args:
            request: Scoring request with pod spec and node candidates
            
        Returns:
            Combined scoring result with recommendations
        """
        start_time = time.time()
        
        try:
            with self._stats_lock:
                self.performance_stats['total_requests'] += 1
            
            # Extract features for each model
            load_features = []
            for node in request.node_candidates:
                node_features = self._extract_node_features(node, request.current_metrics)
                load_features.append({
                    'node_name': node.get('name'),
                    'features': node_features
                })
            
            placement_features = self._extract_placement_features(
                request.pod_spec, request.node_candidates, request.current_metrics
            )
            
            anomaly_features = self._extract_anomaly_features(request.current_metrics)
            
            # Get A/B testing assignments for each model
            ab_metadata = {}
            xgb_variant, xgb_exp = self.ab_testing.get_model_variant(
                request.user_id, 'xgboost', {'features': load_features}
            )
            ql_variant, ql_exp = self.ab_testing.get_model_variant(
                request.user_id, 'qlearning', placement_features
            )
            anom_variant, anom_exp = self.ab_testing.get_model_variant(
                request.user_id, 'anomaly', anomaly_features
            )
            
            ab_metadata = {
                'xgboost': {'variant': xgb_variant.variant_id if xgb_variant else None, 'experiment': xgb_exp},
                'qlearning': {'variant': ql_variant.variant_id if ql_variant else None, 'experiment': ql_exp},
                'anomaly': {'variant': anom_variant.variant_id if anom_variant else None, 'experiment': anom_exp}
            }
            
            # Check cache first
            cache_key_data = {
                'pod_spec': request.pod_spec,
                'nodes': [node.get('name') for node in request.node_candidates],
                'metrics_hash': hash(str(request.current_metrics))
            }
            
            cached_result = self.cache_manager.cache.get_prediction('combined_scoring', cache_key_data)
            if cached_result:
                with self._stats_lock:
                    self.performance_stats['cache_hits'] += 1
                
                # Add A/B test metadata to cached result
                cached_result['prediction']['ab_test_metadata'] = ab_metadata
                return CombinedScoringResult(**cached_result['prediction'])
            
            # Make concurrent requests to all model services
            async with aiohttp.ClientSession() as session:
                # Prepare payloads
                xgboost_payload = {'features': load_features}
                qlearning_payload = placement_features
                anomaly_payload = anomaly_features
                
                # Make concurrent requests
                tasks = [
                    self._call_model_service(session, self.xgboost_url, 
                                           "v1/models/xgboost-load-predictor:predict", 
                                           xgboost_payload),
                    self._call_model_service(session, self.qlearning_url,
                                           "v1/models/qlearning-placement-optimizer:optimize",
                                           qlearning_payload),
                    self._call_model_service(session, self.anomaly_url,
                                           "v1/models/anomaly-detector:detect",
                                           anomaly_payload)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                xgb_success, xgb_data = results[0] if not isinstance(results[0], Exception) else (False, {'error': str(results[0])})
                ql_success, ql_data = results[1] if not isinstance(results[1], Exception) else (False, {'error': str(results[1])})
                anom_success, anom_data = results[2] if not isinstance(results[2], Exception) else (False, {'error': str(results[2])})
            
            # Update model success rates
            with self._stats_lock:
                self.performance_stats['model_success_rates']['xgboost'] = xgb_success
                self.performance_stats['model_success_rates']['qlearning'] = ql_success
                self.performance_stats['model_success_rates']['anomaly'] = anom_success
            
            # Record A/B test results
            processing_time = (time.time() - start_time) * 1000
            
            if xgb_exp and xgb_variant:
                self.ab_testing.record_result(ExperimentResult(
                    experiment_id=xgb_exp,
                    variant_id=xgb_variant.variant_id,
                    variant_name=xgb_variant.name,
                    user_id=request.user_id,
                    request_data=xgboost_payload,
                    response_data=xgb_data,
                    timestamp=datetime.utcnow(),
                    latency_ms=processing_time / 3,  # Approximate per-model latency
                    success=xgb_success
                ))
            
            # Combine model outputs into node scores
            node_scores = self._combine_model_scores(
                request.node_candidates,
                xgb_data if xgb_success else {},
                ql_data if ql_success else {},
                anom_data if anom_success else {}
            )
            
            # Determine recommended node
            recommended_node = None
            fallback_required = False
            
            if node_scores:
                # Sort by total score descending
                sorted_scores = sorted(node_scores, key=lambda x: x.total_score, reverse=True)
                best_score = sorted_scores[0]
                
                # Recommend if score is above threshold and no critical anomalies
                if best_score.total_score > 0.7 and best_score.placement_recommendation:
                    recommended_node = best_score.node_name
                else:
                    fallback_required = True
                    logger.warning(f"Low confidence score {best_score.total_score}, triggering fallback")
            else:
                fallback_required = True
                logger.error("No node scores generated, triggering fallback")
            
            # Extract anomaly alerts
            anomaly_alerts = []
            if anom_success and 'anomalies_detected' in anom_data:
                anomaly_alerts = anom_data.get('alerts', [])
            
            # Create combined result
            result = CombinedScoringResult(
                request_id=request.request_id,
                node_scores=node_scores,
                recommended_node=recommended_node,
                fallback_required=fallback_required,
                processing_time_ms=processing_time,
                model_responses={
                    'xgboost': xgb_data,
                    'qlearning': ql_data,
                    'anomaly': anom_data
                },
                anomaly_alerts=anomaly_alerts,
                ab_test_metadata=ab_metadata
            )
            
            # Cache result for future use
            self.cache_manager.cache.set_prediction(
                'combined_scoring', cache_key_data, 
                {'prediction': result.__dict__}, ttl=180  # 3 minutes TTL
            )
            
            # Update performance statistics
            with self._stats_lock:
                self.performance_stats['average_latency_ms'] = (
                    (self.performance_stats['average_latency_ms'] * (self.performance_stats['total_requests'] - 1) + 
                     processing_time) / self.performance_stats['total_requests']
                )
                
                if fallback_required:
                    self.performance_stats['fallback_triggered'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Combined scoring failed for request {request.request_id}: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return CombinedScoringResult(
                request_id=request.request_id,
                node_scores=[],
                recommended_node=None,
                fallback_required=True,
                processing_time_ms=processing_time,
                model_responses={},
                anomaly_alerts=[],
                ab_test_metadata={}
            )
    
    def _combine_model_scores(self, nodes: List[Dict[str, Any]],
                             xgb_response: Dict[str, Any],
                             ql_response: Dict[str, Any],
                             anom_response: Dict[str, Any]) -> List[NodeScore]:
        """
        Combine individual model scores into final node scores
        
        Args:
            nodes: List of node candidates
            xgb_response: XGBoost load prediction response
            ql_response: Q-Learning placement optimization response
            anom_response: Anomaly detection response
            
        Returns:
            List of scored nodes
        """
        node_scores = []
        
        try:
            # Extract predictions from model responses
            load_predictions = {}
            if 'predictions' in xgb_response:
                for pred in xgb_response['predictions']:
                    if 'node_name' in pred:
                        load_predictions[pred['node_name']] = pred
            
            placement_scores = {}
            if 'node_scores' in ql_response:
                for node_name, score_data in ql_response['node_scores'].items():
                    placement_scores[node_name] = score_data
            
            # Check for anomalies that should affect scoring
            anomaly_penalties = {}
            if 'anomalies_detected' in anom_response and anom_response['anomalies_detected']:
                anomalies = anom_response.get('anomaly_details', [])
                for anomaly in anomalies:
                    if anomaly.get('severity') in ['high', 'critical']:
                        # Apply penalty to affected nodes
                        affected_nodes = anomaly.get('affected_nodes', [])
                        for node in affected_nodes:
                            anomaly_penalties[node] = anomaly_penalties.get(node, 0) + 0.3
            
            # Score each node
            for node in nodes:
                node_name = node.get('name', 'unknown')
                
                # Initialize component scores
                load_score = 0.5  # Default neutral score
                placement_score = 0.5
                anomaly_score = 1.0  # Default no anomaly
                
                reasoning = []
                
                # Load prediction score
                if node_name in load_predictions:
                    pred = load_predictions[node_name]
                    cpu_pred = pred.get('cpu_prediction', 50)
                    memory_pred = pred.get('memory_prediction', 50)
                    
                    # Score based on predicted utilization (lower is better, but not too low)
                    cpu_score = max(0, min(1, (85 - cpu_pred) / 50))  # Optimal around 60-70%
                    memory_score = max(0, min(1, (90 - memory_pred) / 60))
                    load_score = (cpu_score + memory_score) / 2
                    
                    reasoning.append(f"Load prediction: CPU {cpu_pred}%, Memory {memory_pred}%")
                
                # Placement optimization score
                if node_name in placement_scores:
                    placement_data = placement_scores[node_name]
                    placement_score = placement_data.get('score', 0.5)
                    reasoning.append(f"Placement score: {placement_score:.3f}")
                
                # Anomaly detection score
                anomaly_penalty = anomaly_penalties.get(node_name, 0)
                anomaly_score = max(0, 1.0 - anomaly_penalty)
                if anomaly_penalty > 0:
                    reasoning.append(f"Anomaly penalty: -{anomaly_penalty:.2f}")
                
                # Calculate weighted total score
                total_score = (
                    load_score * self.scoring_weights['load_prediction'] +
                    placement_score * self.scoring_weights['placement_optimization'] +
                    anomaly_score * self.scoring_weights['anomaly_detection']
                )
                
                # Calculate confidence based on model availability
                confidence_components = []
                if node_name in load_predictions:
                    confidence_components.append(0.4)
                if node_name in placement_scores:
                    confidence_components.append(0.45)
                if anomaly_score > 0.5:
                    confidence_components.append(0.15)
                
                confidence = sum(confidence_components)
                
                # Determine placement recommendation
                placement_recommendation = (
                    total_score > 0.6 and 
                    confidence > 0.7 and
                    anomaly_score > 0.7 and
                    node.get('conditions', {}).get('Ready', True)
                )
                
                node_score = NodeScore(
                    node_name=node_name,
                    total_score=round(total_score, 4),
                    component_scores={
                        'load_prediction': round(load_score, 4),
                        'placement_optimization': round(placement_score, 4),
                        'anomaly_detection': round(anomaly_score, 4)
                    },
                    confidence=round(confidence, 4),
                    placement_recommendation=placement_recommendation,
                    reasoning=reasoning
                )
                
                node_scores.append(node_score)
            
            return node_scores
            
        except Exception as e:
            logger.error(f"Failed to combine model scores: {e}")
            return []
    
    def score_placement_sync(self, request: ScoringRequest) -> CombinedScoringResult:
        """Synchronous wrapper for score_placement"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.score_placement(request))
        finally:
            loop.close()
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all model services"""
        async def check_service_health(session: aiohttp.ClientSession, 
                                     name: str, url: str) -> Dict[str, Any]:
            try:
                async with session.get(f"{url}/health", timeout=5.0) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {'service': name, 'status': 'healthy', 'details': data}
                    else:
                        return {'service': name, 'status': 'unhealthy', 'error': f'HTTP {response.status}'}
            except Exception as e:
                return {'service': name, 'status': 'error', 'error': str(e)}
        
        async def check_all_services():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    check_service_health(session, 'xgboost', self.xgboost_url),
                    check_service_health(session, 'qlearning', self.qlearning_url),
                    check_service_health(session, 'anomaly', self.anomaly_url)
                ]
                return await asyncio.gather(*tasks)
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            service_health = loop.run_until_complete(check_all_services())
            
            overall_healthy = all(s['status'] == 'healthy' for s in service_health)
            
            return {
                'status': 'healthy' if overall_healthy else 'degraded',
                'services': service_health,
                'cache_health': self.cache_manager.cache.health_check(),
                'performance_stats': dict(self.performance_stats),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
        finally:
            loop.close()
    
    def get_scoring_stats(self) -> Dict[str, Any]:
        """Get combined scorer performance statistics"""
        return {
            'performance': dict(self.performance_stats),
            'ab_testing': self.ab_testing.get_framework_stats(),
            'cache': self.cache_manager.cache.get_cache_stats(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def shutdown(self):
        """Shutdown the combined scorer"""
        try:
            self.executor.shutdown(wait=True)
            logger.info("Combined ML scorer shutdown complete")
        except Exception as e:
            logger.error(f"Error during scorer shutdown: {e}")


# Global scorer instance
_combined_scorer_instance = None

def get_combined_scorer() -> CombinedMLScorer:
    """Get global combined ML scorer instance"""
    global _combined_scorer_instance
    if _combined_scorer_instance is None:
        _combined_scorer_instance = CombinedMLScorer()
    return _combined_scorer_instance