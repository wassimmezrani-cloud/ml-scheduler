#!/usr/bin/env python3

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import redis
import mlflow
import mlflow.pytorch
from prometheus_client import Counter, Histogram, Gauge, start_http_server

app = Flask(__name__)

OPTIMIZATION_COUNTER = Counter('qlearning_optimizations_total', 'Total Q-Learning optimizations made')
OPTIMIZATION_LATENCY = Histogram('qlearning_optimization_duration_seconds', 'Q-Learning optimization latency')
MODEL_PERFORMANCE = Gauge('qlearning_model_performance', 'Current Q-Learning model performance score')
CACHE_HITS = Counter('qlearning_cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('qlearning_cache_misses_total', 'Total cache misses')

class QNetworkInference(nn.Module):
    """Q-Network for inference-optimized serving."""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [256, 128, 64]):
        super(QNetworkInference, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state)

class QLearningMLSchedulerServer:
    def __init__(self):
        self.logger = self._setup_logging()
        
        self.model = None
        self.state_preprocessor = None
        self.action_decoder = None
        self.model_version = None
        self.model_metadata = {}
        
        self.redis_client = self._setup_redis()
        
        self.hydatis_config = {
            'mlflow_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://10.110.190.32:31380'),
            'model_name': 'qlearning_placement_optimizer',
            'model_stage': 'Production',
            'cache_ttl': int(os.getenv('CACHE_TTL_SECONDS', '180')),
            'batch_size': int(os.getenv('BATCH_SIZE', '16')),
            'max_optimization_time': float(os.getenv('MAX_OPTIMIZATION_TIME', '0.200')),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.state_config = {
            'cluster_features': [
                'total_utilization', 'scheduling_pressure', 'pending_pods',
                'available_nodes', 'network_congestion', 'storage_pressure'
            ],
            'node_features': [
                'cpu_available', 'memory_available', 'current_load', 
                'network_latency', 'reliability_score', 'pod_count'
            ],
            'pod_features': [
                'cpu_request', 'memory_request', 'priority_class',
                'affinity_weight', 'anti_affinity_penalty'
            ]
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Q-Learning server."""
        logger = logging.getLogger('qlearning_server')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _setup_redis(self):
        """Setup Redis client for caching."""
        try:
            redis_host = os.getenv('REDIS_HOST', 'redis-service.ml-scheduler.svc.cluster.local')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            
            client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0
            )
            
            client.ping()
            self.logger.info(f"Redis connected: {redis_host}:{redis_port}")
            return client
            
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}, caching disabled")
            return None

    def load_model(self) -> bool:
        """Load Q-Learning model from MLflow."""
        try:
            mlflow.set_tracking_uri(self.hydatis_config['mlflow_uri'])
            
            model_name = self.hydatis_config['model_name']
            model_stage = self.hydatis_config['model_stage']
            
            self.logger.info(f"Loading Q-Learning model {model_name} from MLflow...")
            
            model_uri = f"models:/{model_name}/{model_stage}"
            
            run_info = mlflow.get_run(mlflow.get_latest_versions(model_name, [model_stage])[0].run_id)
            
            artifacts_path = f"runs:/{run_info.info.run_id}"
            
            model_path = mlflow.artifacts.download_artifacts(f"{artifacts_path}/model")
            
            state_size = int(run_info.data.params.get('state_size', '23'))
            action_size = int(run_info.data.params.get('action_size', '10'))
            hidden_layers = json.loads(run_info.data.params.get('hidden_layers', '[256, 128, 64]'))
            
            self.model = QNetworkInference(state_size, action_size, hidden_layers)
            self.model.load_state_dict(torch.load(f"{model_path}/model.pth", map_location=self.hydatis_config['device']))
            self.model.eval()
            self.model.to(self.hydatis_config['device'])
            
            self.model_version = run_info.info.run_id[:8]
            self.model_metadata = {
                'run_id': run_info.info.run_id,
                'version': self.model_version,
                'stage': model_stage,
                'state_size': state_size,
                'action_size': action_size,
                'hidden_layers': hidden_layers,
                'device': str(self.hydatis_config['device']),
                'loaded_at': datetime.now().isoformat()
            }
            
            performance_score = float(run_info.data.metrics.get('final_reward', 0.75))
            MODEL_PERFORMANCE.set(performance_score)
            
            self.logger.info(f"Q-Learning model loaded successfully: version {self.model_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Q-Learning model: {e}")
            return False

    def _preprocess_state(self, optimization_request: Dict[str, Any]) -> torch.Tensor:
        """Preprocess optimization request into state tensor."""
        
        pod_spec = optimization_request.get('pod_spec', {})
        available_nodes = optimization_request.get('available_nodes', [])
        cluster_state = optimization_request.get('cluster_state', {})
        
        state_vector = []
        
        state_vector.extend([
            cluster_state.get('total_utilization', 0.65),
            cluster_state.get('scheduling_pressure', 0.4),
            len(available_nodes),
            cluster_state.get('pending_pods', 0),
            cluster_state.get('network_congestion', 0.3),
            cluster_state.get('storage_pressure', 0.2)
        ])
        
        pod_features = [
            pod_spec.get('cpu_request', 1000) / 1000.0,
            pod_spec.get('memory_request', 2048) / 2048.0,
            pod_spec.get('priority_class', 0),
            len(pod_spec.get('affinity_rules', [])) * 0.1,
            len(pod_spec.get('anti_affinity_rules', [])) * 0.1
        ]
        state_vector.extend(pod_features)
        
        max_nodes = 6
        for i in range(max_nodes):
            if i < len(available_nodes):
                node = available_nodes[i]
                node_features = [
                    node.get('cpu_available', 4000) / 4000.0,
                    node.get('memory_available', 8192) / 8192.0,
                    node.get('current_load', 0.5),
                    node.get('network_latency', 5) / 100.0,
                    node.get('reliability_score', 0.95),
                    node.get('pod_count', 10) / 30.0
                ]
            else:
                node_features = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
            
            state_vector.extend(node_features)
        
        while len(state_vector) < 23:
            state_vector.append(0.0)
        
        state_tensor = torch.FloatTensor(state_vector[:23]).unsqueeze(0)
        return state_tensor.to(self.hydatis_config['device'])

    def _decode_action(self, action_values: torch.Tensor, available_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Decode Q-values to placement recommendation."""
        
        action_values_cpu = action_values.cpu().detach().numpy().flatten()
        
        if len(available_nodes) == 0:
            return {
                'recommended_node': 'no_nodes_available',
                'placement_score': 0.0,
                'reasoning': 'No available nodes for placement'
            }
        
        node_scores = action_values_cpu[:len(available_nodes)]
        
        best_node_idx = np.argmax(node_scores)
        best_node = available_nodes[best_node_idx]
        placement_score = float(node_scores[best_node_idx])
        
        normalized_score = 1.0 / (1.0 + np.exp(-placement_score))
        
        ranking = []
        for i, node in enumerate(available_nodes):
            ranking.append({
                'node_name': node.get('name', f'node-{i}'),
                'score': float(node_scores[i] if i < len(node_scores) else 0),
                'normalized_score': float(1.0 / (1.0 + np.exp(-node_scores[i])) if i < len(node_scores) else 0)
            })
        
        ranking.sort(key=lambda x: x['score'], reverse=True)
        
        reasoning_factors = []
        if best_node.get('cpu_available', 0) > 2000:
            reasoning_factors.append("sufficient_cpu_capacity")
        if best_node.get('memory_available', 0) > 4096:
            reasoning_factors.append("sufficient_memory_capacity")
        if best_node.get('current_load', 1.0) < 0.7:
            reasoning_factors.append("low_current_load")
        if best_node.get('network_latency', 100) < 10:
            reasoning_factors.append("low_network_latency")
        
        return {
            'recommended_node': best_node.get('name', f'node-{best_node_idx}'),
            'placement_score': float(normalized_score),
            'raw_q_value': float(placement_score),
            'reasoning': reasoning_factors,
            'node_ranking': ranking,
            'confidence_level': 'high' if normalized_score > 0.8 else 'medium' if normalized_score > 0.6 else 'low'
        }

    def _generate_cache_key(self, optimization_request: Dict[str, Any]) -> str:
        """Generate cache key for optimization request."""
        
        simplified_request = {
            'pod_cpu': optimization_request.get('pod_spec', {}).get('cpu_request', 0),
            'pod_memory': optimization_request.get('pod_spec', {}).get('memory_request', 0),
            'node_count': len(optimization_request.get('available_nodes', [])),
            'cluster_util': round(optimization_request.get('cluster_state', {}).get('total_utilization', 0), 2)
        }
        
        request_string = json.dumps(simplified_request, sort_keys=True)
        
        import hashlib
        return f"qlearning_opt_{hashlib.md5(request_string.encode()).hexdigest()[:12]}"

    def _get_cached_optimization(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get optimization from cache."""
        if not self.redis_client:
            return None
        
        try:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                CACHE_HITS.inc()
                return json.loads(cached_result)
            else:
                CACHE_MISSES.inc()
                return None
                
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {e}")
            CACHE_MISSES.inc()
            return None

    def _cache_optimization(self, cache_key: str, optimization: Dict[str, Any]):
        """Cache optimization result."""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                cache_key, 
                self.hydatis_config['cache_ttl'], 
                json.dumps(optimization)
            )
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")

    def optimize_placement(self, optimization_request: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize pod placement using Q-Learning model."""
        
        if not self.model:
            raise RuntimeError("Q-Learning model not loaded")
        
        start_time = time.time()
        
        try:
            cache_key = self._generate_cache_key(optimization_request)
            
            cached_result = self._get_cached_optimization(cache_key)
            if cached_result:
                cached_result['cached'] = True
                cached_result['cache_key'] = cache_key
                return cached_result
            
            state_tensor = self._preprocess_state(optimization_request)
            
            with torch.no_grad():
                q_values = self.model(state_tensor)
            
            available_nodes = optimization_request.get('available_nodes', [])
            placement_decision = self._decode_action(q_values, available_nodes)
            
            processing_time = (time.time() - start_time) * 1000
            
            optimization_result = {
                'recommended_node': placement_decision['recommended_node'],
                'placement_score': placement_decision['placement_score'],
                'confidence_level': placement_decision['confidence_level'],
                'reasoning': placement_decision['reasoning'],
                'node_ranking': placement_decision['node_ranking'],
                'model_version': self.model_version,
                'optimization_timestamp': datetime.now().isoformat(),
                'processing_time_ms': processing_time,
                'cached': False,
                'alternative_nodes': placement_decision['node_ranking'][1:3] if len(placement_decision['node_ranking']) > 1 else []
            }
            
            self._cache_optimization(cache_key, optimization_result)
            
            OPTIMIZATION_COUNTER.inc()
            OPTIMIZATION_LATENCY.observe(time.time() - start_time)
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Placement optimization failed: {e}")
            raise

    def batch_optimize(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize placement for multiple pods in batch."""
        
        if not self.model:
            raise RuntimeError("Q-Learning model not loaded")
        
        results = []
        
        batch_size = min(len(batch_requests), self.hydatis_config['batch_size'])
        
        for i in range(0, len(batch_requests), batch_size):
            batch = batch_requests[i:i + batch_size]
            
            batch_states = []
            for req in batch:
                state_tensor = self._preprocess_state(req)
                batch_states.append(state_tensor.squeeze(0))
            
            batch_tensor = torch.stack(batch_states).to(self.hydatis_config['device'])
            
            with torch.no_grad():
                batch_q_values = self.model(batch_tensor)
            
            for j, req in enumerate(batch):
                q_values = batch_q_values[j:j+1]
                available_nodes = req.get('available_nodes', [])
                placement_decision = self._decode_action(q_values, available_nodes)
                
                results.append({
                    'batch_index': i + j,
                    'recommended_node': placement_decision['recommended_node'],
                    'placement_score': placement_decision['placement_score'],
                    'confidence_level': placement_decision['confidence_level']
                })
        
        return results

    def simulate_placement_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate placement decision for what-if analysis."""
        
        try:
            pod_spec = scenario.get('pod_spec', {})
            node_configurations = scenario.get('node_configurations', [])
            
            simulation_results = []
            
            for node_config in node_configurations:
                modified_request = {
                    'pod_spec': pod_spec,
                    'available_nodes': [node_config],
                    'cluster_state': scenario.get('cluster_state', {})
                }
                
                placement_result = self.optimize_placement(modified_request)
                
                simulation_results.append({
                    'node_name': node_config.get('name', 'unknown'),
                    'placement_score': placement_result['placement_score'],
                    'expected_performance': self._estimate_performance(pod_spec, node_config)
                })
            
            best_simulation = max(simulation_results, key=lambda x: x['placement_score'])
            
            return {
                'simulation_results': simulation_results,
                'recommended_configuration': best_simulation,
                'scenario_analysis': {
                    'total_configurations': len(node_configurations),
                    'best_score': best_simulation['placement_score'],
                    'performance_variance': np.std([r['placement_score'] for r in simulation_results])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Placement simulation failed: {e}")
            raise

    def _estimate_performance(self, pod_spec: Dict[str, Any], node_config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate performance for pod on specific node."""
        
        cpu_ratio = pod_spec.get('cpu_request', 1000) / node_config.get('cpu_available', 4000)
        memory_ratio = pod_spec.get('memory_request', 2048) / node_config.get('memory_available', 8192)
        
        resource_efficiency = 1.0 - max(cpu_ratio, memory_ratio)
        
        network_performance = 1.0 - (node_config.get('network_latency', 5) / 100.0)
        
        load_impact = 1.0 - node_config.get('current_load', 0.5)
        
        overall_performance = (resource_efficiency + network_performance + load_impact) / 3.0
        
        return {
            'resource_efficiency': float(resource_efficiency),
            'network_performance': float(network_performance),
            'load_impact': float(load_impact),
            'overall_performance': float(overall_performance)
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        if server.model is None:
            return jsonify({'status': 'unhealthy', 'reason': 'model_not_loaded'}), 503
        
        return jsonify({
            'status': 'healthy',
            'model_version': server.model_version,
            'model_metadata': server.model_metadata,
            'cache_available': server.redis_client is not None,
            'device': server.hydatis_config['device'],
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

@app.route('/optimize', methods=['POST'])
def optimize():
    """Main placement optimization endpoint."""
    try:
        start_time = time.time()
        
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        required_fields = ['pod_spec', 'available_nodes']
        for field in required_fields:
            if field not in request.json:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        optimization_result = server.optimize_placement(request.json)
        
        processing_time = (time.time() - start_time) * 1000
        
        if processing_time > server.hydatis_config['max_optimization_time'] * 1000:
            server.logger.warning(f"Optimization exceeded time limit: {processing_time:.1f}ms")
        
        optimization_result['processing_time_ms'] = processing_time
        
        return jsonify(optimization_result), 200
        
    except ValueError as e:
        return jsonify({'error': f'Input validation failed: {str(e)}'}), 400
    except RuntimeError as e:
        return jsonify({'error': f'Service error: {str(e)}'}), 503
    except Exception as e:
        server.logger.error(f"Optimization endpoint failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch_optimize', methods=['POST'])
def batch_optimize():
    """Batch optimization endpoint."""
    try:
        if not request.json or 'batch_requests' not in request.json:
            return jsonify({'error': 'No batch_requests provided'}), 400
        
        batch_requests = request.json['batch_requests']
        
        if len(batch_requests) > 50:
            return jsonify({'error': 'Batch size too large (max 50)'}), 400
        
        start_time = time.time()
        
        batch_results = server.batch_optimize(batch_requests)
        
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'optimizations': batch_results,
            'batch_size': len(batch_requests),
            'processing_time_ms': processing_time,
            'avg_optimization_time_ms': processing_time / len(batch_requests) if batch_requests else 0
        }), 200
        
    except Exception as e:
        server.logger.error(f"Batch optimization failed: {e}")
        return jsonify({'error': 'Batch optimization failed'}), 500

@app.route('/simulate', methods=['POST'])
def simulate():
    """Placement simulation endpoint."""
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        simulation_result = server.simulate_placement_scenario(request.json)
        
        return jsonify(simulation_result), 200
        
    except Exception as e:
        server.logger.error(f"Simulation failed: {e}")
        return jsonify({'error': 'Simulation failed'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get Q-Learning model information."""
    try:
        return jsonify({
            'model_name': server.hydatis_config['model_name'],
            'model_version': server.model_version,
            'model_metadata': server.model_metadata,
            'state_config': server.state_config,
            'cache_enabled': server.redis_client is not None,
            'service_config': {
                'batch_size': server.hydatis_config['batch_size'],
                'cache_ttl': server.hydatis_config['cache_ttl'],
                'max_optimization_time': server.hydatis_config['max_optimization_time'],
                'device': server.hydatis_config['device']
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload_model', methods=['POST'])
def reload_model():
    """Reload Q-Learning model from MLflow."""
    try:
        old_version = server.model_version
        
        success = server.load_model()
        
        if success:
            return jsonify({
                'status': 'success',
                'old_version': old_version,
                'new_version': server.model_version,
                'reload_timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({'error': 'Model reload failed'}), 500
            
    except Exception as e:
        server.logger.error(f"Model reload failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cache_stats', methods=['GET'])
def cache_stats():
    """Get cache statistics."""
    try:
        if not server.redis_client:
            return jsonify({'cache_enabled': False}), 200
        
        info = server.redis_client.info()
        
        return jsonify({
            'cache_enabled': True,
            'cache_stats': {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0)
            },
            'ttl_seconds': server.hydatis_config['cache_ttl']
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest
    return generate_latest()

def initialize_server():
    """Initialize the Q-Learning server."""
    global server
    server = QLearningMLSchedulerServer()
    
    success = server.load_model()
    if not success:
        server.logger.error("Failed to load initial Q-Learning model")
        exit(1)
    
    server.logger.info("Q-Learning ML Scheduler Server initialized successfully")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    initialize_server()
    
    metrics_port = int(os.getenv('METRICS_PORT', '9091'))
    start_http_server(metrics_port)
    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8002'))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    server.logger.info(f"Starting Q-Learning server on {host}:{port}")
    server.logger.info(f"Metrics available on port {metrics_port}")
    
    app.run(host=host, port=port, debug=debug, threaded=True)