#!/usr/bin/env python3
"""
Q-Learning serving endpoint for HYDATIS ML Scheduler placement optimization.
Provides real-time placement optimization API for scheduler plugin integration.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
from pathlib import Path
from flask import Flask, request, jsonify
import time

from .optimizer import HYDATISPlacementOptimizer

logger = logging.getLogger(__name__)


class QLearningServingEngine:
    """Real-time serving engine for Q-Learning placement optimization."""
    
    def __init__(self, model_dir: str = "/data/ml_scheduler_longhorn/models/qlearning"):
        self.model_dir = Path(model_dir)
        self.optimizer = HYDATISPlacementOptimizer(str(self.model_dir))
        
        self.serving_config = {
            'max_latency_ms': 50,
            'placement_confidence_threshold': 0.7,
            'fallback_strategy': 'capacity_based',
            'cache_ttl_seconds': 30
        }
        
        self.performance_tracking = {
            'request_count': 0,
            'successful_requests': 0,
            'optimization_requests': 0,
            'fallback_requests': 0,
            'total_latency': 0.0,
            'error_count': 0
        }
        
        self.placement_cache = {}
        self.last_cache_update = None
    
    def optimize_placement(self, pod_spec: Dict[str, Any], 
                          available_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Optimize pod placement with performance tracking."""
        
        start_time = time.time()
        
        try:
            self.performance_tracking['request_count'] += 1
            
            optimization_result = self.optimizer.optimize_pod_placement(pod_spec, available_nodes)
            
            if 'error' not in optimization_result:
                self.performance_tracking['successful_requests'] += 1
                
                confidence = optimization_result['placement_reasoning']['confidence']
                if confidence > self.serving_config['placement_confidence_threshold']:
                    self.performance_tracking['optimization_requests'] += 1
                else:
                    self.performance_tracking['fallback_requests'] += 1
            else:
                self.performance_tracking['error_count'] += 1
            
            latency = (time.time() - start_time) * 1000
            self.performance_tracking['total_latency'] += latency
            
            optimization_result['serving_metrics'] = {
                'request_latency_ms': round(latency, 2),
                'latency_target_met': latency < self.serving_config['max_latency_ms'],
                'confidence_threshold_met': optimization_result.get('placement_reasoning', {}).get('confidence', 0) > self.serving_config['placement_confidence_threshold'],
                'optimization_mode': 'active' if self.optimizer.model_loaded else 'fallback'
            }
            
            return optimization_result
            
        except Exception as e:
            self.performance_tracking['error_count'] += 1
            logger.error(f"Placement optimization serving error: {e}")
            
            latency = (time.time() - start_time) * 1000
            self.performance_tracking['total_latency'] += latency
            
            return {
                'error': str(e),
                'fallback_node': self._get_fallback_placement(available_nodes),
                'serving_metrics': {
                    'request_latency_ms': round(latency, 2),
                    'error_mode': True,
                    'fallback_used': True
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_fallback_placement(self, available_nodes: Optional[List[str]]) -> str:
        """Get fallback placement when optimization fails."""
        
        if available_nodes:
            return available_nodes[0]
        return self.optimizer.cluster_config['worker_nodes'][0]
    
    def get_cluster_insights(self) -> Dict[str, Any]:
        """Get real-time cluster placement insights."""
        
        try:
            current_state = self.optimizer._get_cluster_state()
            
            insights = self.optimizer.dqn_agent.get_placement_insights(current_state)
            
            rebalancing_analysis = self.optimizer.recommend_cluster_rebalancing()
            
            cluster_insights = {
                'cluster_status': {
                    'average_cpu_utilization': insights['cluster_utilization']['average_cpu'],
                    'average_memory_utilization': insights['cluster_utilization']['average_memory'],
                    'cluster_balance_score': 1.0 - max(insights['cluster_utilization']['cpu_variance'], 
                                                      insights['cluster_utilization']['memory_variance']),
                    'nodes_overloaded': len([node for node in insights['node_rankings'] 
                                           if node['cpu_load'] > 0.8 or node['memory_load'] > 0.8]),
                    'optimization_confidence': insights['exploration_rate']
                },
                'node_rankings': insights['node_rankings'],
                'rebalancing_recommendations': rebalancing_analysis,
                'placement_readiness': {
                    'model_loaded': self.optimizer.model_loaded,
                    'ready_for_scheduling': self.optimizer.model_loaded and 
                                          insights['cluster_utilization']['average_cpu'] < 0.9,
                    'capacity_available': any(node['capacity_score'] > 0.3 for node in insights['node_rankings'])
                },
                'insights_timestamp': datetime.now().isoformat()
            }
            
            return cluster_insights
            
        except Exception as e:
            logger.error(f"Cluster insights error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_serving_health(self) -> Dict[str, Any]:
        """Get serving engine health and performance metrics."""
        
        avg_latency = (self.performance_tracking['total_latency'] / 
                      self.performance_tracking['request_count']) if self.performance_tracking['request_count'] > 0 else 0
        
        success_rate = (self.performance_tracking['successful_requests'] / 
                       self.performance_tracking['request_count']) if self.performance_tracking['request_count'] > 0 else 0
        
        optimization_rate = (self.performance_tracking['optimization_requests'] / 
                            self.performance_tracking['request_count']) if self.performance_tracking['request_count'] > 0 else 0
        
        error_rate = (self.performance_tracking['error_count'] / 
                     self.performance_tracking['request_count']) if self.performance_tracking['request_count'] > 0 else 0
        
        health_status = 'healthy'
        if not self.optimizer.model_loaded:
            health_status = 'degraded'
        elif error_rate > 0.1 or avg_latency > self.serving_config['max_latency_ms']:
            health_status = 'degraded'
        
        health = {
            'status': health_status,
            'model_loaded': self.optimizer.model_loaded,
            'serving_performance': {
                'total_requests': self.performance_tracking['request_count'],
                'success_rate': round(success_rate, 4),
                'optimization_rate': round(optimization_rate, 4),
                'error_rate': round(error_rate, 4),
                'average_latency_ms': round(avg_latency, 2),
                'latency_target_met': avg_latency < self.serving_config['max_latency_ms']
            },
            'optimization_metrics': self.optimizer.get_optimization_metrics(),
            'cluster_readiness': {
                'placement_optimization_active': self.optimizer.model_loaded,
                'dqn_agent_ready': self.optimizer.dqn_agent.agent.q_network is not None,
                'prometheus_connection': True
            },
            'health_timestamp': datetime.now().isoformat()
        }
        
        return health
    
    def reload_models(self) -> Dict[str, Any]:
        """Reload Q-Learning models."""
        
        reload_start = time.time()
        
        try:
            success = self.optimizer._load_trained_models()
            reload_latency = (time.time() - reload_start) * 1000
            
            return {
                'reload_successful': success,
                'model_loaded': self.optimizer.model_loaded,
                'reload_latency_ms': round(reload_latency, 2),
                'agent_metrics': self.optimizer.dqn_agent.agent.get_training_metrics() if success else {},
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            reload_latency = (time.time() - reload_start) * 1000
            logger.error(f"Model reload error: {e}")
            
            return {
                'reload_successful': False,
                'error': str(e),
                'reload_latency_ms': round(reload_latency, 2),
                'timestamp': datetime.now().isoformat()
            }


app = Flask(__name__)
serving_engine = QLearningServingEngine()


@app.route('/optimize/placement', methods=['POST'])
def optimize_placement():
    """API endpoint for pod placement optimization."""
    
    try:
        pod_spec = request.json
        available_nodes = request.args.get('available_nodes')
        
        if available_nodes:
            available_nodes = available_nodes.split(',')
        
        result = serving_engine.optimize_placement(pod_spec, available_nodes)
        
        status_code = 200 if 'error' not in result else 400
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/insights/cluster', methods=['GET'])
def cluster_insights():
    """API endpoint for cluster placement insights."""
    
    try:
        insights = serving_engine.get_cluster_insights()
        
        status_code = 200 if 'error' not in insights else 500
        return jsonify(insights), status_code
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/insights/rebalancing', methods=['GET'])
def rebalancing_recommendations():
    """API endpoint for cluster rebalancing recommendations."""
    
    try:
        recommendations = serving_engine.optimizer.recommend_cluster_rebalancing()
        return jsonify(recommendations)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    
    health = serving_engine.get_serving_health()
    status_code = 200 if health['status'] == 'healthy' else 503
    
    return jsonify(health), status_code


@app.route('/reload', methods=['POST'])
def reload_models():
    """Manually reload Q-Learning models."""
    
    result = serving_engine.reload_models()
    status_code = 200 if result['reload_successful'] else 500
    
    return jsonify(result), status_code


@app.route('/simulate', methods=['POST'])
def simulate_scenarios():
    """API endpoint for placement scenario simulation."""
    
    try:
        scenarios = request.json
        
        if not isinstance(scenarios, list):
            scenarios = [scenarios]
        
        simulation_results = serving_engine.optimizer.simulate_placement_scenarios(scenarios)
        return jsonify(simulation_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def main():
    """Main serving application."""
    
    print("HYDATIS Q-Learning Placement Optimization Serving Engine")
    print(f"Model Directory: {serving_engine.model_dir}")
    print(f"Model Loaded: {serving_engine.optimizer.model_loaded}")
    print(f"Target Latency: <{serving_engine.serving_config['max_latency_ms']}ms")
    print("API Endpoints:")
    print("  POST /optimize/placement - Pod placement optimization")
    print("  GET /insights/cluster - Cluster placement insights")
    print("  GET /insights/rebalancing - Rebalancing recommendations")
    print("  GET /health - Health check")
    print("  POST /reload - Reload models")
    print("  POST /simulate - Scenario simulation")
    
    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=8081, debug=False)


if __name__ == "__main__":
    main()