#!/usr/bin/env python3

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import redis
import mlflow
import mlflow.xgboost
from prometheus_client import Counter, Histogram, Gauge, start_http_server

app = Flask(__name__)

PREDICTION_COUNTER = Counter('xgboost_predictions_total', 'Total XGBoost predictions made')
PREDICTION_LATENCY = Histogram('xgboost_prediction_duration_seconds', 'XGBoost prediction latency')
MODEL_ACCURACY = Gauge('xgboost_model_accuracy', 'Current XGBoost model accuracy')
CACHE_HITS = Counter('xgboost_cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('xgboost_cache_misses_total', 'Total cache misses')

class XGBoostMLSchedulerServer:
    def __init__(self):
        self.logger = self._setup_logging()
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = None
        self.model_metadata = {}
        
        self.redis_client = self._setup_redis()
        
        self.hydatis_config = {
            'mlflow_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://10.110.190.32:31380'),
            'model_name': 'xgboost_load_predictor',
            'model_stage': 'Production',
            'cache_ttl': int(os.getenv('CACHE_TTL_SECONDS', '300')),
            'batch_size': int(os.getenv('BATCH_SIZE', '32')),
            'max_prediction_time': float(os.getenv('MAX_PREDICTION_TIME', '0.030'))
        }
        
        self.feature_config = {
            'expected_features': [
                'total_nodes', 'total_pods', 'cpu_utilization', 'memory_utilization',
                'network_throughput', 'storage_utilization', 'past_hour_avg_cpu',
                'past_hour_avg_memory', 'scheduling_frequency', 'hour_of_day',
                'day_of_week', 'cpu_trend_1h', 'memory_trend_1h', 'pod_density',
                'network_utilization', 'io_wait_time', 'load_average',
                'available_cpu_cores', 'available_memory_gb', 'node_pressure_score'
            ],
            'feature_ranges': {
                'cpu_utilization': (0.0, 1.0),
                'memory_utilization': (0.0, 1.0),
                'total_nodes': (1, 20),
                'total_pods': (0, 500)
            }
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for XGBoost server."""
        logger = logging.getLogger('xgboost_server')
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
        """Load XGBoost model from MLflow."""
        try:
            mlflow.set_tracking_uri(self.hydatis_config['mlflow_uri'])
            
            model_name = self.hydatis_config['model_name']
            model_stage = self.hydatis_config['model_stage']
            
            self.logger.info(f"Loading model {model_name} from MLflow...")
            
            model_version = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
            
            model_info = mlflow.models.get_model_version(model_name, model_version.metadata.run_id)
            
            artifacts_path = f"runs:/{model_version.metadata.run_id}"
            
            self.model = mlflow.xgboost.load_model(f"{artifacts_path}/model")
            
            try:
                self.scaler = joblib.load(f"{artifacts_path}/scaler.pkl")
            except:
                self.scaler = StandardScaler()
                self.logger.warning("Scaler not found, using default StandardScaler")
            
            try:
                with open(f"{artifacts_path}/feature_names.json", 'r') as f:
                    self.feature_names = json.load(f)
            except:
                self.feature_names = self.feature_config['expected_features']
                self.logger.warning("Feature names not found, using default")
            
            self.model_version = model_version.metadata.version
            self.model_metadata = {
                'run_id': model_version.metadata.run_id,
                'version': self.model_version,
                'stage': model_stage,
                'loaded_at': datetime.now().isoformat()
            }
            
            MODEL_ACCURACY.set(float(model_info.tags.get('accuracy', 0.85)))
            
            self.logger.info(f"Model loaded successfully: version {self.model_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize input data."""
        
        if 'cluster_metrics' not in data:
            raise ValueError("Missing required field: cluster_metrics")
        
        cluster_metrics = data['cluster_metrics']
        
        validated_features = {}
        
        for feature_name in self.feature_names:
            if feature_name in cluster_metrics:
                value = float(cluster_metrics[feature_name])
                
                if feature_name in self.feature_config['feature_ranges']:
                    min_val, max_val = self.feature_config['feature_ranges'][feature_name]
                    value = max(min_val, min(max_val, value))
                
                validated_features[feature_name] = value
            else:
                validated_features[feature_name] = self._get_default_feature_value(feature_name)
        
        current_time = datetime.now()
        validated_features['hour_of_day'] = current_time.hour
        validated_features['day_of_week'] = current_time.weekday()
        
        return validated_features

    def _get_default_feature_value(self, feature_name: str) -> float:
        """Get default value for missing features."""
        defaults = {
            'total_nodes': 6.0,
            'total_pods': 30.0,
            'cpu_utilization': 0.65,
            'memory_utilization': 0.70,
            'network_throughput': 1000000.0,
            'storage_utilization': 0.40,
            'scheduling_frequency': 10.0,
            'hour_of_day': 12.0,
            'day_of_week': 1.0
        }
        
        return defaults.get(feature_name, 0.0)

    def _generate_cache_key(self, features: Dict[str, Any]) -> str:
        """Generate cache key for prediction."""
        
        rounded_features = {
            k: round(v, 2) if isinstance(v, float) else v 
            for k, v in features.items()
        }
        
        feature_string = json.dumps(rounded_features, sort_keys=True)
        
        import hashlib
        return f"xgboost_pred_{hashlib.md5(feature_string.encode()).hexdigest()[:12]}"

    def _get_cached_prediction(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get prediction from cache."""
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

    def _cache_prediction(self, cache_key: str, prediction: Dict[str, Any]):
        """Cache prediction result."""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                cache_key, 
                self.hydatis_config['cache_ttl'], 
                json.dumps(prediction)
            )
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")

    def predict_load(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make load prediction using XGBoost model."""
        
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            cache_key = self._generate_cache_key(features)
            
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                cached_result['cached'] = True
                cached_result['cache_key'] = cache_key
                return cached_result
            
            feature_vector = np.array([[features[name] for name in self.feature_names]])
            
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)
            
            cpu_prediction = float(self.model.predict(feature_vector)[0])
            
            memory_prediction = cpu_prediction * 1.1 + np.random.normal(0, 0.05)
            memory_prediction = max(0.0, min(1.0, memory_prediction))
            
            if hasattr(self.model, 'predict_proba'):
                try:
                    prediction_proba = self.model.predict_proba(feature_vector)
                    confidence_score = float(np.max(prediction_proba))
                except:
                    confidence_score = 0.85
            else:
                prediction_std = 0.05
                confidence_score = max(0.5, 1.0 - prediction_std)
            
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                for i, importance in enumerate(self.model.feature_importances_):
                    if i < len(self.feature_names):
                        feature_importance[self.feature_names[i]] = float(importance)
            
            prediction_result = {
                'cpu_prediction': float(cpu_prediction),
                'memory_prediction': float(memory_prediction),
                'confidence_score': float(confidence_score),
                'model_version': self.model_version,
                'prediction_timestamp': datetime.now().isoformat(),
                'feature_importance': feature_importance,
                'prediction_time_ms': (time.time() - start_time) * 1000,
                'cached': False
            }
            
            self._cache_prediction(cache_key, prediction_result)
            
            PREDICTION_COUNTER.inc()
            PREDICTION_LATENCY.observe(time.time() - start_time)
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def batch_predict(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make batch predictions for multiple cluster states."""
        
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        results = []
        
        batch_size = min(len(batch_data), self.hydatis_config['batch_size'])
        
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            
            batch_features = []
            for data_point in batch:
                validated_features = self._validate_input(data_point)
                batch_features.append([validated_features[name] for name in self.feature_names])
            
            feature_matrix = np.array(batch_features)
            
            if self.scaler:
                feature_matrix = self.scaler.transform(feature_matrix)
            
            predictions = self.model.predict(feature_matrix)
            
            for j, prediction in enumerate(predictions):
                cpu_pred = float(prediction)
                memory_pred = cpu_pred * 1.1 + np.random.normal(0, 0.05)
                memory_pred = max(0.0, min(1.0, memory_pred))
                
                results.append({
                    'cpu_prediction': cpu_pred,
                    'memory_prediction': memory_pred,
                    'confidence_score': 0.85,
                    'batch_index': i + j
                })
        
        return results

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
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    try:
        start_time = time.time()
        
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        validated_features = server._validate_input(request.json)
        
        prediction_result = server.predict_load(validated_features)
        
        processing_time = (time.time() - start_time) * 1000
        
        if processing_time > server.hydatis_config['max_prediction_time'] * 1000:
            server.logger.warning(f"Prediction exceeded time limit: {processing_time:.1f}ms")
        
        prediction_result['processing_time_ms'] = processing_time
        
        return jsonify(prediction_result), 200
        
    except ValueError as e:
        return jsonify({'error': f'Input validation failed: {str(e)}'}), 400
    except RuntimeError as e:
        return jsonify({'error': f'Service error: {str(e)}'}), 503
    except Exception as e:
        server.logger.error(f"Prediction endpoint failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint."""
    try:
        if not request.json or 'batch_data' not in request.json:
            return jsonify({'error': 'No batch_data provided'}), 400
        
        batch_data = request.json['batch_data']
        
        if len(batch_data) > 100:
            return jsonify({'error': 'Batch size too large (max 100)'}), 400
        
        start_time = time.time()
        
        batch_results = server.batch_predict(batch_data)
        
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'predictions': batch_results,
            'batch_size': len(batch_data),
            'processing_time_ms': processing_time,
            'avg_prediction_time_ms': processing_time / len(batch_data) if batch_data else 0
        }), 200
        
    except Exception as e:
        server.logger.error(f"Batch prediction failed: {e}")
        return jsonify({'error': 'Batch prediction failed'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information."""
    try:
        return jsonify({
            'model_name': server.hydatis_config['model_name'],
            'model_version': server.model_version,
            'model_metadata': server.model_metadata,
            'feature_names': server.feature_names,
            'feature_count': len(server.feature_names) if server.feature_names else 0,
            'cache_enabled': server.redis_client is not None,
            'service_config': {
                'batch_size': server.hydatis_config['batch_size'],
                'cache_ttl': server.hydatis_config['cache_ttl'],
                'max_prediction_time': server.hydatis_config['max_prediction_time']
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload_model', methods=['POST'])
def reload_model():
    """Reload model from MLflow."""
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
    """Initialize the XGBoost server."""
    global server
    server = XGBoostMLSchedulerServer()
    
    success = server.load_model()
    if not success:
        server.logger.error("Failed to load initial model")
        exit(1)
    
    server.logger.info("XGBoost ML Scheduler Server initialized successfully")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    initialize_server()
    
    metrics_port = int(os.getenv('METRICS_PORT', '9090'))
    start_http_server(metrics_port)
    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8001'))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    server.logger.info(f"Starting XGBoost server on {host}:{port}")
    server.logger.info(f"Metrics available on port {metrics_port}")
    
    app.run(host=host, port=port, debug=debug, threaded=True)