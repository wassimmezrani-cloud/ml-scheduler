#!/usr/bin/env python3
"""
XGBoost model serving endpoint for HYDATIS ML Scheduler.
Provides real-time load prediction API for scheduler plugin integration.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
from pathlib import Path
from flask import Flask, request, jsonify
import joblib
import time

from .model import HYDATISXGBoostPredictor

logger = logging.getLogger(__name__)


class XGBoostServingEngine:
    """Real-time serving engine for XGBoost load prediction models."""
    
    def __init__(self, model_dir: str = "/data/ml_scheduler_longhorn/models/xgboost"):
        self.model_dir = Path(model_dir)
        self.predictor = HYDATISXGBoostPredictor()
        self.model_loaded = False
        self.last_model_check = None
        self.model_check_interval = 300  # 5 minutes
        
        # Performance tracking
        self.prediction_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        
        # Load models on initialization
        self._load_latest_models()
    
    def _load_latest_models(self) -> bool:
        """Load the latest trained models."""
        
        try:
            success = self.predictor.load_models(str(self.model_dir))
            
            if success:
                self.model_loaded = True
                self.last_model_check = datetime.now()
                logger.info("XGBoost models loaded successfully")
            else:
                logger.warning("Failed to load XGBoost models")
                
            return success
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _check_model_updates(self):
        """Check for model updates and reload if necessary."""
        
        if (self.last_model_check is None or 
            (datetime.now() - self.last_model_check).seconds > self.model_check_interval):
            
            logger.info("Checking for model updates...")
            self._load_latest_models()
    
    def predict_node_load(self, node_features: Dict[str, Any], 
                         horizon_minutes: int = 5) -> Dict[str, Any]:
        """Predict load for a specific node."""
        
        start_time = time.time()
        
        try:
            # Check for model updates
            self._check_model_updates()
            
            if not self.model_loaded:
                raise ValueError("XGBoost models not loaded")
            
            # Convert features to DataFrame
            if isinstance(node_features, dict):
                features_df = pd.DataFrame([node_features])
            else:
                features_df = node_features
            
            # Make predictions
            predictions = self.predictor.predict_load(features_df, horizon_minutes)
            
            # Extract single node prediction
            result = {
                'node_id': node_features.get('instance', 'unknown'),
                'prediction_timestamp': predictions['timestamp'].isoformat(),
                'horizon_minutes': horizon_minutes,
                'cpu_prediction': {
                    'value': float(predictions['cpu_prediction'][0]),
                    'confidence': 'high' if self.prediction_count > 100 else 'medium'
                },
                'memory_prediction': {
                    'value': float(predictions['memory_prediction'][0]),
                    'confidence': 'high' if self.prediction_count > 100 else 'medium'
                },
                'capacity_forecast': {
                    'cpu_capacity_remaining': 1 - float(predictions['cpu_prediction'][0]),
                    'memory_capacity_remaining': 1 - float(predictions['memory_prediction'][0]),
                    'overall_capacity_score': (
                        (1 - float(predictions['cpu_prediction'][0])) * 0.6 +
                        (1 - float(predictions['memory_prediction'][0])) * 0.4
                    )
                }
            }
            
            # Update performance tracking
            latency = (time.time() - start_time) * 1000  # Convert to ms
            self.prediction_count += 1
            self.total_latency += latency
            
            result['serving_metrics'] = {
                'prediction_latency_ms': round(latency, 2),
                'average_latency_ms': round(self.total_latency / self.prediction_count, 2),
                'total_predictions': self.prediction_count
            }
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction error: {e}")
            
            return {
                'error': str(e),
                'node_id': node_features.get('instance', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'error_count': self.error_count
            }
    
    def predict_cluster_load(self, cluster_features: List[Dict[str, Any]],
                           horizon_minutes: int = 5) -> Dict[str, Any]:
        """Predict load for entire cluster (all nodes)."""
        
        start_time = time.time()
        
        try:
            if not cluster_features:
                raise ValueError("No cluster features provided")
            
            # Process all nodes
            node_predictions = []
            
            for node_features in cluster_features:
                node_pred = self.predict_node_load(node_features, horizon_minutes)
                if 'error' not in node_pred:
                    node_predictions.append(node_pred)
            
            if not node_predictions:
                raise ValueError("No successful predictions for cluster")
            
            # Aggregate cluster-level insights
            cpu_predictions = [pred['cpu_prediction']['value'] for pred in node_predictions]
            memory_predictions = [pred['memory_prediction']['value'] for pred in node_predictions]
            capacity_scores = [pred['capacity_forecast']['overall_capacity_score'] for pred in node_predictions]
            
            cluster_result = {
                'cluster_id': 'HYDATIS-6node',
                'prediction_timestamp': datetime.now().isoformat(),
                'horizon_minutes': horizon_minutes,
                'cluster_summary': {
                    'total_nodes': len(node_predictions),
                    'avg_cpu_prediction': np.mean(cpu_predictions),
                    'avg_memory_prediction': np.mean(memory_predictions),
                    'avg_capacity_score': np.mean(capacity_scores),
                    'best_node': max(node_predictions, key=lambda x: x['capacity_forecast']['overall_capacity_score'])['node_id'],
                    'most_loaded_node': max(node_predictions, key=lambda x: x['cpu_prediction']['value'])['node_id']
                },
                'node_predictions': node_predictions,
                'scheduling_recommendations': self._generate_scheduling_recommendations(node_predictions),
                'cluster_latency_ms': round((time.time() - start_time) * 1000, 2)
            }
            
            return cluster_result
            
        except Exception as e:
            logger.error(f"Cluster prediction error: {e}")
            return {
                'error': str(e),
                'cluster_id': 'HYDATIS-6node',
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_scheduling_recommendations(self, node_predictions: List[Dict]) -> Dict[str, Any]:
        """Generate scheduling recommendations based on predictions."""
        
        # Sort nodes by capacity score
        sorted_nodes = sorted(node_predictions, 
                            key=lambda x: x['capacity_forecast']['overall_capacity_score'], 
                            reverse=True)
        
        recommendations = {
            'preferred_nodes': [node['node_id'] for node in sorted_nodes[:3]],
            'avoid_nodes': [node['node_id'] for node in sorted_nodes[-2:]],
            'load_balancing_needed': max(pred['cpu_prediction']['value'] for pred in node_predictions) - 
                                   min(pred['cpu_prediction']['value'] for pred in node_predictions) > 0.3,
            'cluster_pressure': np.mean([pred['cpu_prediction']['value'] for pred in node_predictions]) > 0.8
        }
        
        return recommendations
    
    def get_serving_health(self) -> Dict[str, Any]:
        """Get serving engine health and performance metrics."""
        
        avg_latency = self.total_latency / self.prediction_count if self.prediction_count > 0 else 0
        error_rate = self.error_count / self.prediction_count if self.prediction_count > 0 else 0
        
        health = {
            'status': 'healthy' if self.model_loaded and error_rate < 0.1 else 'degraded',
            'model_loaded': self.model_loaded,
            'last_model_check': self.last_model_check.isoformat() if self.last_model_check else None,
            'performance_metrics': {
                'total_predictions': self.prediction_count,
                'average_latency_ms': round(avg_latency, 2),
                'error_rate': round(error_rate, 4),
                'latency_target_met': avg_latency < 100  # <100ms target
            },
            'model_info': {
                'cpu_model_available': self.predictor.cpu_model is not None,
                'memory_model_available': self.predictor.memory_model is not None,
                'feature_count': len(self.predictor.feature_names)
            }
        }
        
        return health


# Flask API for serving
app = Flask(__name__)
serving_engine = XGBoostServingEngine()


@app.route('/predict/node', methods=['POST'])
def predict_node():
    """API endpoint for single node load prediction."""
    
    try:
        features = request.json
        horizon = request.args.get('horizon_minutes', 5, type=int)
        
        result = serving_engine.predict_node_load(features, horizon)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/cluster', methods=['POST'])
def predict_cluster():
    """API endpoint for cluster-wide load prediction."""
    
    try:
        cluster_features = request.json
        horizon = request.args.get('horizon_minutes', 5, type=int)
        
        result = serving_engine.predict_cluster_load(cluster_features, horizon)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    
    health = serving_engine.get_serving_health()
    status_code = 200 if health['status'] == 'healthy' else 503
    
    return jsonify(health), status_code


@app.route('/reload', methods=['POST'])
def reload_models():
    """Manually reload models."""
    
    success = serving_engine._load_latest_models()
    return jsonify({
        'reloaded': success,
        'timestamp': datetime.now().isoformat()
    })


def main():
    """Main serving application."""
    
    print("HYDATIS XGBoost Load Predictor Serving Engine")
    print(f"Model Directory: {serving_engine.model_dir}")
    print(f"Models Loaded: {serving_engine.model_loaded}")
    print("API Endpoints:")
    print("  POST /predict/node - Single node prediction")
    print("  POST /predict/cluster - Cluster-wide prediction")
    print("  GET /health - Health check")
    print("  POST /reload - Reload models")
    
    # In production, this would run with a WSGI server
    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == "__main__":
    main()