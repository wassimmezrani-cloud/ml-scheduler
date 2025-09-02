#!/usr/bin/env python3

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import redis
import mlflow
import mlflow.sklearn
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import requests

app = Flask(__name__)

DETECTION_COUNTER = Counter('anomaly_detections_total', 'Total anomaly detections made')
DETECTION_LATENCY = Histogram('anomaly_detection_duration_seconds', 'Anomaly detection latency')
ANOMALY_RATE = Gauge('anomaly_detection_rate', 'Current anomaly detection rate')
CACHE_HITS = Counter('anomaly_cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('anomaly_cache_misses_total', 'Total cache misses')
ALERT_COUNTER = Counter('anomaly_alerts_sent_total', 'Total anomaly alerts sent')

class AnomalyDetectionMLSchedulerServer:
    def __init__(self):
        self.logger = self._setup_logging()
        
        self.ensemble_models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_version = None
        self.model_metadata = {}
        
        self.redis_client = self._setup_redis()
        
        self.hydatis_config = {
            'mlflow_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://10.110.190.32:31380'),
            'model_name': 'isolation_forest_anomaly_detector',
            'model_stage': 'Production',
            'cache_ttl': int(os.getenv('CACHE_TTL_SECONDS', '120')),
            'batch_size': int(os.getenv('BATCH_SIZE', '64')),
            'max_detection_time': float(os.getenv('MAX_DETECTION_TIME', '0.100')),
            'alert_manager_url': os.getenv('ALERT_MANAGER_URL', 'http://alert-manager-service:8080')
        }
        
        self.detection_config = {
            'anomaly_threshold': float(os.getenv('ANOMALY_THRESHOLD', '0.1')),
            'ensemble_voting_threshold': 0.6,
            'severity_thresholds': {
                'critical': 0.9,
                'warning': 0.7,
                'info': 0.5
            },
            'alert_cooldown_minutes': 15,
            'max_alerts_per_hour': 10
        }
        
        self.feature_categories = {
            'resource_metrics': [
                'cpu_usage', 'memory_usage', 'network_io', 'storage_io',
                'cpu_pressure', 'memory_pressure', 'io_pressure'
            ],
            'cluster_metrics': [
                'pod_scheduling_rate', 'node_availability', 'cluster_utilization',
                'pending_pods', 'failed_pods', 'evicted_pods'
            ],
            'temporal_metrics': [
                'hour_of_day', 'day_of_week', 'time_since_last_anomaly',
                'recent_scheduling_frequency', 'load_trend'
            ]
        }
        
        self.alert_history = []

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for anomaly detection server."""
        logger = logging.getLogger('anomaly_server')
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

    def load_models(self) -> bool:
        """Load ensemble anomaly detection models from MLflow."""
        try:
            mlflow.set_tracking_uri(self.hydatis_config['mlflow_uri'])
            
            model_name = self.hydatis_config['model_name']
            model_stage = self.hydatis_config['model_stage']
            
            self.logger.info(f"Loading ensemble anomaly models {model_name} from MLflow...")
            
            latest_version = mlflow.get_latest_versions(model_name, [model_stage])[0]
            run_info = mlflow.get_run(latest_version.run_id)
            
            artifacts_path = f"runs:/{run_info.info.run_id}"
            
            ensemble_types = ['resource_anomaly', 'cluster_anomaly', 'temporal_anomaly']
            
            for ensemble_type in ensemble_types:
                try:
                    model_path = f"{artifacts_path}/{ensemble_type}_model.pkl"
                    scaler_path = f"{artifacts_path}/{ensemble_type}_scaler.pkl"
                    
                    model = joblib.load(mlflow.artifacts.download_artifacts(model_path))
                    scaler = joblib.load(mlflow.artifacts.download_artifacts(scaler_path))
                    
                    self.ensemble_models[ensemble_type] = model
                    self.scalers[ensemble_type] = scaler
                    
                    self.logger.info(f"Loaded {ensemble_type} model successfully")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load {ensemble_type} model: {e}")
                    
                    self.ensemble_models[ensemble_type] = IsolationForest(
                        n_estimators=100,
                        contamination=0.1,
                        random_state=42
                    )
                    self.scalers[ensemble_type] = StandardScaler()
            
            try:
                feature_names_path = f"{artifacts_path}/feature_names.json"
                with open(mlflow.artifacts.download_artifacts(feature_names_path), 'r') as f:
                    self.feature_names = json.load(f)
            except:
                self.feature_names = (
                    self.feature_categories['resource_metrics'] +
                    self.feature_categories['cluster_metrics'] +
                    self.feature_categories['temporal_metrics']
                )
            
            self.model_version = latest_version.version
            self.model_metadata = {
                'run_id': run_info.info.run_id,
                'version': self.model_version,
                'stage': model_stage,
                'ensemble_count': len(self.ensemble_models),
                'loaded_at': datetime.now().isoformat()
            }
            
            precision_score = float(run_info.data.metrics.get('precision', 0.94))
            ANOMALY_RATE.set(precision_score)
            
            self.logger.info(f"Ensemble anomaly models loaded: version {self.model_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load anomaly models: {e}")
            return False

    def _validate_metrics_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize metrics input."""
        
        if 'cluster_metrics' not in data:
            raise ValueError("Missing required field: cluster_metrics")
        
        cluster_metrics = data['cluster_metrics']
        
        validated_metrics = {}
        
        for category, feature_list in self.feature_categories.items():
            for feature_name in feature_list:
                if feature_name in cluster_metrics:
                    if isinstance(cluster_metrics[feature_name], list):
                        validated_metrics[feature_name] = np.mean(cluster_metrics[feature_name])
                    else:
                        validated_metrics[feature_name] = float(cluster_metrics[feature_name])
                else:
                    validated_metrics[feature_name] = self._get_default_metric_value(feature_name)
        
        current_time = datetime.now()
        validated_metrics['hour_of_day'] = current_time.hour
        validated_metrics['day_of_week'] = current_time.weekday()
        
        last_anomaly_time = self._get_last_anomaly_time()
        if last_anomaly_time:
            time_since_anomaly = (current_time - last_anomaly_time).total_seconds() / 3600.0
        else:
            time_since_anomaly = 24.0
        
        validated_metrics['time_since_last_anomaly'] = time_since_anomaly
        
        return validated_metrics

    def _get_default_metric_value(self, metric_name: str) -> float:
        """Get default value for missing metrics."""
        defaults = {
            'cpu_usage': 0.65,
            'memory_usage': 0.70,
            'network_io': 1000000.0,
            'storage_io': 100000.0,
            'pod_scheduling_rate': 10.0,
            'node_availability': 6.0,
            'cluster_utilization': 0.65,
            'pending_pods': 0.0,
            'failed_pods': 0.0,
            'hour_of_day': 12.0,
            'day_of_week': 1.0
        }
        
        return defaults.get(metric_name, 0.0)

    def _get_last_anomaly_time(self) -> Optional[datetime]:
        """Get timestamp of last detected anomaly."""
        if self.alert_history:
            return max(alert['timestamp'] for alert in self.alert_history)
        return None

    def detect_anomalies(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies using ensemble models."""
        
        if not self.ensemble_models:
            raise RuntimeError("Anomaly detection models not loaded")
        
        start_time = time.time()
        
        try:
            cache_key = self._generate_cache_key(metrics)
            
            cached_result = self._get_cached_detection(cache_key)
            if cached_result:
                cached_result['cached'] = True
                return cached_result
            
            validated_metrics = self._validate_metrics_input({'cluster_metrics': metrics})
            
            ensemble_predictions = {}
            anomaly_scores = {}
            
            for ensemble_type, model in self.ensemble_models.items():
                try:
                    feature_subset = self._select_features_for_ensemble(validated_metrics, ensemble_type)
                    feature_vector = np.array([list(feature_subset.values())]).reshape(1, -1)
                    
                    scaler = self.scalers.get(ensemble_type)
                    if scaler:
                        feature_vector = scaler.transform(feature_vector)
                    
                    anomaly_score = model.decision_function(feature_vector)[0]
                    is_anomaly = model.predict(feature_vector)[0] == -1
                    
                    ensemble_predictions[ensemble_type] = {
                        'is_anomaly': bool(is_anomaly),
                        'anomaly_score': float(anomaly_score),
                        'confidence': float(abs(anomaly_score))
                    }
                    
                    anomaly_scores[ensemble_type] = float(anomaly_score)
                    
                except Exception as e:
                    self.logger.warning(f"Ensemble {ensemble_type} detection failed: {e}")
                    ensemble_predictions[ensemble_type] = {
                        'is_anomaly': False,
                        'anomaly_score': 0.0,
                        'confidence': 0.0,
                        'error': str(e)
                    }
            
            overall_anomaly_detected = self._aggregate_ensemble_decisions(ensemble_predictions)
            
            overall_anomaly_score = np.mean(list(anomaly_scores.values())) if anomaly_scores else 0.0
            
            severity = self._calculate_severity(overall_anomaly_score)
            
            affected_metrics = self._identify_affected_metrics(validated_metrics, ensemble_predictions)
            
            processing_time = (time.time() - start_time) * 1000
            
            detection_result = {
                'anomaly_detected': overall_anomaly_detected,
                'anomaly_score': float(overall_anomaly_score),
                'severity': severity,
                'affected_metrics': affected_metrics,
                'ensemble_predictions': ensemble_predictions,
                'model_version': self.model_version,
                'detection_timestamp': datetime.now().isoformat(),
                'processing_time_ms': processing_time,
                'cached': False
            }
            
            if overall_anomaly_detected and severity in ['critical', 'warning']:
                asyncio.create_task(self._send_anomaly_alert(detection_result))
            
            self._cache_detection(cache_key, detection_result)
            
            DETECTION_COUNTER.inc()
            DETECTION_LATENCY.observe(time.time() - start_time)
            
            return detection_result
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            raise

    def _select_features_for_ensemble(self, metrics: Dict[str, Any], ensemble_type: str) -> Dict[str, float]:
        """Select appropriate features for specific ensemble model."""
        
        if ensemble_type == 'resource_anomaly':
            feature_subset = {k: v for k, v in metrics.items() if k in self.feature_categories['resource_metrics']}
        elif ensemble_type == 'cluster_anomaly':
            feature_subset = {k: v for k, v in metrics.items() if k in self.feature_categories['cluster_metrics']}
        elif ensemble_type == 'temporal_anomaly':
            feature_subset = {k: v for k, v in metrics.items() if k in self.feature_categories['temporal_metrics']}
        else:
            feature_subset = metrics
        
        return feature_subset

    def _aggregate_ensemble_decisions(self, ensemble_predictions: Dict[str, Dict[str, Any]]) -> bool:
        """Aggregate ensemble model decisions using voting."""
        
        anomaly_votes = 0
        total_votes = 0
        
        for ensemble_type, prediction in ensemble_predictions.items():
            if 'error' not in prediction:
                total_votes += 1
                if prediction['is_anomaly']:
                    anomaly_votes += 1
        
        if total_votes == 0:
            return False
        
        voting_ratio = anomaly_votes / total_votes
        
        return voting_ratio >= self.detection_config['ensemble_voting_threshold']

    def _calculate_severity(self, anomaly_score: float) -> str:
        """Calculate anomaly severity based on score."""
        
        abs_score = abs(anomaly_score)
        
        if abs_score >= self.detection_config['severity_thresholds']['critical']:
            return 'critical'
        elif abs_score >= self.detection_config['severity_thresholds']['warning']:
            return 'warning'
        elif abs_score >= self.detection_config['severity_thresholds']['info']:
            return 'info'
        else:
            return 'normal'

    def _identify_affected_metrics(self, metrics: Dict[str, Any], ensemble_predictions: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify which metrics are most affected by anomaly."""
        
        affected_metrics = []
        
        for ensemble_type, prediction in ensemble_predictions.items():
            if prediction.get('is_anomaly', False):
                if ensemble_type == 'resource_anomaly':
                    resource_values = {k: v for k, v in metrics.items() if k in self.feature_categories['resource_metrics']}
                    high_values = [k for k, v in resource_values.items() if v > 0.8]
                    affected_metrics.extend(high_values)
                
                elif ensemble_type == 'cluster_anomaly':
                    cluster_values = {k: v for k, v in metrics.items() if k in self.feature_categories['cluster_metrics']}
                    anomalous_values = [k for k, v in cluster_values.items() if v > 50 or v < 1]
                    affected_metrics.extend(anomalous_values)
                
                elif ensemble_type == 'temporal_anomaly':
                    affected_metrics.append('temporal_pattern_deviation')
        
        return list(set(affected_metrics))

    async def _send_anomaly_alert(self, detection_result: Dict[str, Any]):
        """Send anomaly alert to alert manager."""
        
        try:
            if not self._should_send_alert(detection_result):
                return
            
            alert_payload = {
                'alert_type': 'ml_scheduler_anomaly',
                'severity': detection_result['severity'],
                'message': f"Anomaly detected in HYDATIS cluster (score: {detection_result['anomaly_score']:.3f})",
                'details': {
                    'anomaly_score': detection_result['anomaly_score'],
                    'affected_metrics': detection_result['affected_metrics'],
                    'detection_timestamp': detection_result['detection_timestamp'],
                    'model_version': detection_result['model_version']
                },
                'source': 'isolation_forest_detector',
                'cluster': 'HYDATIS'
            }
            
            response = requests.post(
                f"{self.hydatis_config['alert_manager_url']}/alerts",
                json=alert_payload,
                timeout=5
            )
            
            if response.status_code == 200:
                self.alert_history.append({
                    'timestamp': datetime.now(),
                    'severity': detection_result['severity'],
                    'score': detection_result['anomaly_score']
                })
                
                ALERT_COUNTER.inc()
                self.logger.info(f"Anomaly alert sent: {detection_result['severity']}")
            else:
                self.logger.error(f"Alert sending failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send anomaly alert: {e}")

    def _should_send_alert(self, detection_result: Dict[str, Any]) -> bool:
        """Check if alert should be sent based on cooldown and rate limiting."""
        
        current_time = datetime.now()
        
        cooldown_minutes = self.detection_config['alert_cooldown_minutes']
        recent_alerts = [
            alert for alert in self.alert_history
            if (current_time - alert['timestamp']).total_seconds() < cooldown_minutes * 60
        ]
        
        if recent_alerts:
            return False
        
        last_hour = current_time - timedelta(hours=1)
        hourly_alerts = [
            alert for alert in self.alert_history
            if alert['timestamp'] > last_hour
        ]
        
        if len(hourly_alerts) >= self.detection_config['max_alerts_per_hour']:
            return False
        
        return detection_result['severity'] in ['critical', 'warning']

    def _generate_cache_key(self, metrics: Dict[str, Any]) -> str:
        """Generate cache key for anomaly detection."""
        
        rounded_metrics = {
            k: round(v, 2) if isinstance(v, (int, float)) else v 
            for k, v in metrics.items()
        }
        
        metrics_string = json.dumps(rounded_metrics, sort_keys=True)
        
        import hashlib
        return f"anomaly_det_{hashlib.md5(metrics_string.encode()).hexdigest()[:12]}"

    def _get_cached_detection(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get detection from cache."""
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

    def _cache_detection(self, cache_key: str, detection: Dict[str, Any]):
        """Cache detection result."""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                cache_key, 
                self.hydatis_config['cache_ttl'], 
                json.dumps(detection, default=str)
            )
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")

    def batch_detect(self, batch_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch anomaly detection for multiple metric sets."""
        
        if not self.ensemble_models:
            raise RuntimeError("Anomaly detection models not loaded")
        
        results = []
        
        batch_size = min(len(batch_metrics), self.hydatis_config['batch_size'])
        
        for i in range(0, len(batch_metrics), batch_size):
            batch = batch_metrics[i:i + batch_size]
            
            for j, metrics in enumerate(batch):
                try:
                    detection_result = self.detect_anomalies(metrics)
                    detection_result['batch_index'] = i + j
                    results.append(detection_result)
                    
                except Exception as e:
                    results.append({
                        'batch_index': i + j,
                        'anomaly_detected': False,
                        'error': str(e)
                    })
        
        return results

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        if not server.ensemble_models:
            return jsonify({'status': 'unhealthy', 'reason': 'models_not_loaded'}), 503
        
        return jsonify({
            'status': 'healthy',
            'model_version': server.model_version,
            'model_metadata': server.model_metadata,
            'ensemble_models': list(server.ensemble_models.keys()),
            'cache_available': server.redis_client is not None,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

@app.route('/detect', methods=['POST'])
def detect():
    """Main anomaly detection endpoint."""
    try:
        start_time = time.time()
        
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'cluster_metrics' not in request.json:
            return jsonify({'error': 'Missing cluster_metrics field'}), 400
        
        detection_result = server.detect_anomalies(request.json['cluster_metrics'])
        
        processing_time = (time.time() - start_time) * 1000
        
        if processing_time > server.hydatis_config['max_detection_time'] * 1000:
            server.logger.warning(f"Detection exceeded time limit: {processing_time:.1f}ms")
        
        detection_result['processing_time_ms'] = processing_time
        
        return jsonify(detection_result), 200
        
    except ValueError as e:
        return jsonify({'error': f'Input validation failed: {str(e)}'}), 400
    except RuntimeError as e:
        return jsonify({'error': f'Service error: {str(e)}'}), 503
    except Exception as e:
        server.logger.error(f"Detection endpoint failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch_detect', methods=['POST'])
def batch_detect():
    """Batch anomaly detection endpoint."""
    try:
        if not request.json or 'batch_metrics' not in request.json:
            return jsonify({'error': 'No batch_metrics provided'}), 400
        
        batch_metrics = request.json['batch_metrics']
        
        if len(batch_metrics) > 100:
            return jsonify({'error': 'Batch size too large (max 100)'}), 400
        
        start_time = time.time()
        
        batch_results = server.batch_detect(batch_metrics)
        
        processing_time = (time.time() - start_time) * 1000
        
        anomalies_detected = len([r for r in batch_results if r.get('anomaly_detected', False)])
        
        return jsonify({
            'detections': batch_results,
            'batch_size': len(batch_metrics),
            'anomalies_detected': anomalies_detected,
            'anomaly_rate': anomalies_detected / len(batch_metrics) if batch_metrics else 0,
            'processing_time_ms': processing_time,
            'avg_detection_time_ms': processing_time / len(batch_metrics) if batch_metrics else 0
        }), 200
        
    except Exception as e:
        server.logger.error(f"Batch detection failed: {e}")
        return jsonify({'error': 'Batch detection failed'}), 500

@app.route('/alert_history', methods=['GET'])
def alert_history():
    """Get recent alert history."""
    try:
        hours_back = int(request.args.get('hours', 24))
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_alerts = [
            {
                'timestamp': alert['timestamp'].isoformat(),
                'severity': alert['severity'],
                'score': alert['score']
            }
            for alert in server.alert_history
            if alert['timestamp'] > cutoff_time
        ]
        
        return jsonify({
            'alert_history': recent_alerts,
            'total_alerts': len(recent_alerts),
            'hours_requested': hours_back,
            'alert_rate_per_hour': len(recent_alerts) / hours_back if hours_back > 0 else 0
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get anomaly detection model information."""
    try:
        return jsonify({
            'model_name': server.hydatis_config['model_name'],
            'model_version': server.model_version,
            'model_metadata': server.model_metadata,
            'ensemble_models': list(server.ensemble_models.keys()),
            'feature_categories': server.feature_categories,
            'detection_config': server.detection_config,
            'cache_enabled': server.redis_client is not None
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload_models', methods=['POST'])
def reload_models():
    """Reload anomaly detection models from MLflow."""
    try:
        old_version = server.model_version
        
        success = server.load_models()
        
        if success:
            return jsonify({
                'status': 'success',
                'old_version': old_version,
                'new_version': server.model_version,
                'ensemble_count': len(server.ensemble_models),
                'reload_timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({'error': 'Models reload failed'}), 500
            
    except Exception as e:
        server.logger.error(f"Models reload failed: {e}")
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
                'keyspace_misses': info.get('keyspace_misses', 0)
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
    """Initialize the anomaly detection server."""
    global server
    server = AnomalyDetectionMLSchedulerServer()
    
    success = server.load_models()
    if not success:
        server.logger.error("Failed to load initial anomaly detection models")
        exit(1)
    
    server.logger.info("Anomaly Detection ML Scheduler Server initialized successfully")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    initialize_server()
    
    metrics_port = int(os.getenv('METRICS_PORT', '9092'))
    start_http_server(metrics_port)
    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8003'))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    server.logger.info(f"Starting Anomaly Detection server on {host}:{port}")
    server.logger.info(f"Metrics available on port {metrics_port}")
    
    app.run(host=host, port=port, debug=debug, threaded=True)