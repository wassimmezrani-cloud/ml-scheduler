#!/usr/bin/env python3
"""
Q-Learning placement optimization logic for HYDATIS ML Scheduler.
Integrates DQN agent with Kubernetes API for intelligent pod placement.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path
import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from .agent import HYDATISPlacementDQN
from .environment import HYDATISClusterEnvironment
from ...monitoring.prometheus_collector import HYDATISPrometheusCollector

logger = logging.getLogger(__name__)


class HYDATISPlacementOptimizer:
    """Production placement optimizer using Q-Learning for HYDATIS cluster."""
    
    def __init__(self, model_dir: str = "/data/ml_scheduler_longhorn/models/qlearning"):
        self.model_dir = Path(model_dir)
        
        self.cluster_config = {
            'nodes': 6,
            'masters': 3,
            'workers': 3,
            'cpu_cores_per_node': 8,
            'memory_gb_per_node': 16,
            'worker_nodes': ['worker-1', 'worker-2', 'worker-3'],
            'master_nodes': ['master-1', 'master-2', 'master-3']
        }
        
        self.dqn_agent = HYDATISPlacementDQN(self.cluster_config)
        self.prometheus_collector = HYDATISPrometheusCollector()
        
        self.optimization_config = {
            'prediction_horizon_minutes': 5,
            'placement_confidence_threshold': 0.7,
            'load_balance_weight': 0.4,
            'capacity_weight': 0.6,
            'resource_buffer': 0.1,
            'rebalancing_threshold': 0.3
        }
        
        self.placement_history = []
        self.performance_tracking = {
            'total_placements': 0,
            'successful_placements': 0,
            'optimization_decisions': 0,
            'load_balance_improvements': [],
            'placement_latencies': []
        }
        
        self.model_loaded = False
        self._load_trained_models()
    
    def _load_trained_models(self) -> bool:
        """Load trained Q-Learning models."""
        
        try:
            success = self.dqn_agent.agent.load_agent(
                str(self.model_dir), 
                experiment_name="hydatis_placement_optimizer"
            )
            
            if success:
                self.model_loaded = True
                logger.info("Q-Learning placement models loaded successfully")
            else:
                logger.warning("Failed to load Q-Learning models")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def optimize_pod_placement(self, 
                              pod_spec: Dict[str, Any],
                              available_nodes: List[str] = None) -> Dict[str, Any]:
        """Optimize pod placement using Q-Learning agent."""
        
        start_time = time.time()
        
        try:
            if not self.model_loaded:
                raise ValueError("Q-Learning models not loaded")
            
            current_state = self._get_cluster_state()
            
            if available_nodes is None:
                available_nodes = self.cluster_config['worker_nodes']
            
            action, placement_decision = self.dqn_agent.select_placement_node(
                current_state, 
                pod_spec.get('resources', {}),
                training=False
            )
            
            selected_node = placement_decision['selected_node']
            
            if selected_node not in available_nodes:
                fallback_action = self._select_fallback_node(current_state, available_nodes)
                selected_node = available_nodes[fallback_action]
                placement_decision['fallback_used'] = True
            else:
                placement_decision['fallback_used'] = False
            
            load_balance_analysis = self._analyze_load_balance_impact(
                current_state, action, pod_spec
            )
            
            capacity_analysis = self._analyze_capacity_impact(
                current_state, action, pod_spec
            )
            
            optimization_result = {
                'selected_node': selected_node,
                'node_index': action,
                'pod_name': pod_spec.get('metadata', {}).get('name', 'unknown'),
                'namespace': pod_spec.get('metadata', {}).get('namespace', 'default'),
                'placement_reasoning': {
                    'q_value': placement_decision['q_values'][action],
                    'confidence': placement_decision['decision_confidence'],
                    'quality_score': placement_decision['placement_quality']['overall_quality']
                },
                'resource_analysis': {
                    'cpu_request': pod_spec.get('resources', {}).get('cpu_request', 0),
                    'memory_request': pod_spec.get('resources', {}).get('memory_request', 0),
                    'node_cpu_after_placement': current_state[action] + pod_spec.get('resources', {}).get('cpu_request', 0),
                    'node_memory_after_placement': current_state[6 + action] + pod_spec.get('resources', {}).get('memory_request', 0)
                },
                'optimization_impact': {
                    'load_balance': load_balance_analysis,
                    'capacity_utilization': capacity_analysis,
                    'expected_improvement': self._calculate_expected_improvement(placement_decision)
                },
                'cluster_insights': self.dqn_agent.get_placement_insights(current_state),
                'optimization_latency_ms': round((time.time() - start_time) * 1000, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            self._update_placement_tracking(optimization_result)
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Placement optimization error: {e}")
            
            return {
                'error': str(e),
                'pod_name': pod_spec.get('metadata', {}).get('name', 'unknown'),
                'fallback_node': self._get_fallback_node(available_nodes or self.cluster_config['worker_nodes']),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_cluster_state(self) -> np.ndarray:
        """Get current cluster state for decision making."""
        
        try:
            metrics = self.prometheus_collector.collect_cluster_metrics()
            
            node_cpu_loads = []
            node_memory_loads = []
            
            for node in self.cluster_config['worker_nodes']:
                node_metrics = metrics.get('node_metrics', {}).get(node, {})
                
                cpu_load = node_metrics.get('cpu_utilization', 0.1)
                memory_load = node_metrics.get('memory_utilization', 0.4)
                
                node_cpu_loads.append(cpu_load)
                node_memory_loads.append(memory_load)
            
            pod_queue_size = metrics.get('cluster_metrics', {}).get('pending_pods', 0)
            cluster_pressure = np.mean(node_cpu_loads + node_memory_loads)
            time_factor = self._get_time_factor()
            
            state = np.array(node_cpu_loads + node_memory_loads + [pod_queue_size, cluster_pressure, time_factor])
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting cluster state: {e}")
            
            default_state = np.array([0.1] * 6 + [0.4] * 3 + [0, 0.25, 0.5])
            return default_state
    
    def _get_time_factor(self) -> float:
        """Get time-based factor for scheduling decisions."""
        current_hour = datetime.now().hour
        
        if 8 <= current_hour <= 18:
            return 0.8
        elif 19 <= current_hour <= 23 or 6 <= current_hour <= 7:
            return 0.5
        else:
            return 0.2
    
    def _select_fallback_node(self, state: np.ndarray, available_nodes: List[str]) -> int:
        """Select fallback node when DQN choice is unavailable."""
        
        node_indices = [self.cluster_config['worker_nodes'].index(node) 
                       for node in available_nodes 
                       if node in self.cluster_config['worker_nodes']]
        
        if not node_indices:
            return 0
        
        node_scores = []
        for idx in node_indices:
            cpu_load = state[idx]
            memory_load = state[6 + idx]
            capacity_score = (1 - cpu_load) * 0.6 + (1 - memory_load) * 0.4
            node_scores.append(capacity_score)
        
        best_idx = node_indices[np.argmax(node_scores)]
        return available_nodes.index(self.cluster_config['worker_nodes'][best_idx])
    
    def _get_fallback_node(self, available_nodes: List[str]) -> str:
        """Get fallback node for error cases."""
        return available_nodes[0] if available_nodes else self.cluster_config['worker_nodes'][0]
    
    def _analyze_load_balance_impact(self, state: np.ndarray, action: int, 
                                   pod_spec: Dict[str, Any]) -> Dict[str, float]:
        """Analyze load balancing impact of placement decision."""
        
        current_cpu_loads = state[:3]
        current_memory_loads = state[6:9]
        
        cpu_request = pod_spec.get('resources', {}).get('cpu_request', 0)
        memory_request = pod_spec.get('resources', {}).get('memory_request', 0)
        
        new_cpu_loads = current_cpu_loads.copy()
        new_memory_loads = current_memory_loads.copy()
        new_cpu_loads[action] += cpu_request
        new_memory_loads[action] += memory_request
        
        current_cpu_variance = np.var(current_cpu_loads)
        new_cpu_variance = np.var(new_cpu_loads)
        cpu_balance_improvement = (current_cpu_variance - new_cpu_variance) / (current_cpu_variance + 1e-8)
        
        current_memory_variance = np.var(current_memory_loads)
        new_memory_variance = np.var(new_memory_loads)
        memory_balance_improvement = (current_memory_variance - new_memory_variance) / (current_memory_variance + 1e-8)
        
        return {
            'cpu_balance_improvement': cpu_balance_improvement,
            'memory_balance_improvement': memory_balance_improvement,
            'overall_balance_improvement': cpu_balance_improvement * 0.6 + memory_balance_improvement * 0.4,
            'cpu_variance_before': current_cpu_variance,
            'cpu_variance_after': new_cpu_variance,
            'memory_variance_before': current_memory_variance,
            'memory_variance_after': new_memory_variance
        }
    
    def _analyze_capacity_impact(self, state: np.ndarray, action: int,
                               pod_spec: Dict[str, Any]) -> Dict[str, float]:
        """Analyze capacity utilization impact."""
        
        node_cpu_load = state[action]
        node_memory_load = state[6 + action]
        
        cpu_request = pod_spec.get('resources', {}).get('cpu_request', 0)
        memory_request = pod_spec.get('resources', {}).get('memory_request', 0)
        
        cpu_utilization_after = node_cpu_load + cpu_request
        memory_utilization_after = node_memory_load + memory_request
        
        cpu_capacity_remaining = 1.0 - cpu_utilization_after
        memory_capacity_remaining = 1.0 - memory_utilization_after
        
        resource_efficiency = min(cpu_utilization_after / 0.8, memory_utilization_after / 0.8)
        
        return {
            'cpu_utilization_after': cpu_utilization_after,
            'memory_utilization_after': memory_utilization_after,
            'cpu_capacity_remaining': cpu_capacity_remaining,
            'memory_capacity_remaining': memory_capacity_remaining,
            'resource_efficiency': resource_efficiency,
            'safe_capacity_buffer': cpu_capacity_remaining > self.optimization_config['resource_buffer'] and 
                                   memory_capacity_remaining > self.optimization_config['resource_buffer']
        }
    
    def _calculate_expected_improvement(self, placement_decision: Dict[str, Any]) -> float:
        """Calculate expected improvement from this placement."""
        
        quality_score = placement_decision['placement_quality']['overall_quality']
        confidence = placement_decision['decision_confidence']
        
        expected_improvement = quality_score * min(confidence, 1.0)
        
        return expected_improvement
    
    def _update_placement_tracking(self, optimization_result: Dict[str, Any]):
        """Update placement performance tracking."""
        
        self.performance_tracking['total_placements'] += 1
        
        if 'error' not in optimization_result:
            self.performance_tracking['successful_placements'] += 1
            
            if optimization_result['placement_reasoning']['confidence'] > self.optimization_config['placement_confidence_threshold']:
                self.performance_tracking['optimization_decisions'] += 1
            
            load_balance_improvement = optimization_result['optimization_impact']['load_balance']['overall_balance_improvement']
            self.performance_tracking['load_balance_improvements'].append(load_balance_improvement)
            
            latency = optimization_result['optimization_latency_ms']
            self.performance_tracking['placement_latencies'].append(latency)
        
        self.placement_history.append({
            'timestamp': optimization_result['timestamp'],
            'result': optimization_result
        })
        
        if len(self.placement_history) > 1000:
            self.placement_history = self.placement_history[-1000:]
    
    def recommend_cluster_rebalancing(self) -> Dict[str, Any]:
        """Recommend cluster rebalancing actions."""
        
        try:
            current_state = self._get_cluster_state()
            
            cpu_loads = current_state[:3]
            memory_loads = current_state[6:9]
            
            cpu_imbalance = np.max(cpu_loads) - np.min(cpu_loads)
            memory_imbalance = np.max(memory_loads) - np.min(memory_loads)
            
            rebalancing_needed = (cpu_imbalance > self.optimization_config['rebalancing_threshold'] or
                                memory_imbalance > self.optimization_config['rebalancing_threshold'])
            
            recommendations = []
            
            if rebalancing_needed:
                overloaded_nodes = []
                underloaded_nodes = []
                
                avg_cpu = np.mean(cpu_loads)
                avg_memory = np.mean(memory_loads)
                
                for i, node in enumerate(self.cluster_config['worker_nodes']):
                    if cpu_loads[i] > avg_cpu + 0.2 or memory_loads[i] > avg_memory + 0.2:
                        overloaded_nodes.append({
                            'node': node,
                            'cpu_load': cpu_loads[i],
                            'memory_load': memory_loads[i],
                            'overload_severity': max(cpu_loads[i] - avg_cpu, memory_loads[i] - avg_memory)
                        })
                    elif cpu_loads[i] < avg_cpu - 0.2 and memory_loads[i] < avg_memory - 0.2:
                        underloaded_nodes.append({
                            'node': node,
                            'cpu_load': cpu_loads[i],
                            'memory_load': memory_loads[i],
                            'available_capacity': min(1 - cpu_loads[i], 1 - memory_loads[i])
                        })
                
                recommendations = self._generate_rebalancing_actions(overloaded_nodes, underloaded_nodes)
            
            rebalancing_analysis = {
                'rebalancing_needed': rebalancing_needed,
                'cluster_balance_score': 1.0 - max(cpu_imbalance, memory_imbalance),
                'cpu_load_variance': np.var(cpu_loads),
                'memory_load_variance': np.var(memory_loads),
                'overloaded_nodes': len([i for i, load in enumerate(cpu_loads) if load > 0.8]),
                'underutilized_nodes': len([i for i, load in enumerate(cpu_loads) if load < 0.3]),
                'recommendations': recommendations,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return rebalancing_analysis
            
        except Exception as e:
            logger.error(f"Rebalancing analysis error: {e}")
            return {
                'error': str(e),
                'rebalancing_needed': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_rebalancing_actions(self, overloaded_nodes: List[Dict], 
                                    underloaded_nodes: List[Dict]) -> List[Dict[str, Any]]:
        """Generate specific rebalancing action recommendations."""
        
        actions = []
        
        for overloaded in overloaded_nodes:
            for underloaded in underloaded_nodes:
                if underloaded['available_capacity'] > 0.2:
                    actions.append({
                        'action_type': 'migrate_pods',
                        'source_node': overloaded['node'],
                        'target_node': underloaded['node'],
                        'priority': overloaded['overload_severity'],
                        'expected_benefit': min(overloaded['overload_severity'], underloaded['available_capacity'])
                    })
        
        actions.sort(key=lambda x: x['priority'], reverse=True)
        
        return actions[:5]
    
    def _get_cluster_state(self) -> np.ndarray:
        """Get current cluster state from Prometheus metrics."""
        
        try:
            metrics = self.prometheus_collector.collect_cluster_metrics()
            
            node_cpu_loads = []
            node_memory_loads = []
            
            for node in self.cluster_config['worker_nodes']:
                node_metrics = metrics.get('node_metrics', {}).get(node, {})
                
                cpu_load = node_metrics.get('cpu_utilization', 0.1)
                memory_load = node_metrics.get('memory_utilization', 0.4)
                
                node_cpu_loads.append(cpu_load)
                node_memory_loads.append(memory_load)
            
            pod_queue_size = metrics.get('cluster_metrics', {}).get('pending_pods', 0)
            cluster_pressure = np.mean(node_cpu_loads + node_memory_loads)
            time_factor = self._get_time_factor()
            
            state = np.array(node_cpu_loads + node_memory_loads + [pod_queue_size, cluster_pressure, time_factor])
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting cluster state: {e}")
            
            default_state = np.array([0.1] * 3 + [0.4] * 3 + [0, 0.25, 0.5])
            return default_state
    
    def _get_time_factor(self) -> float:
        """Calculate time-based scheduling factor."""
        current_hour = datetime.now().hour
        
        if 8 <= current_hour <= 18:
            return 0.8
        elif 19 <= current_hour <= 23 or 6 <= current_hour <= 7:
            return 0.5
        else:
            return 0.2
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization performance metrics."""
        
        success_rate = (self.performance_tracking['successful_placements'] / 
                       self.performance_tracking['total_placements']) if self.performance_tracking['total_placements'] > 0 else 0
        
        optimization_rate = (self.performance_tracking['optimization_decisions'] / 
                           self.performance_tracking['total_placements']) if self.performance_tracking['total_placements'] > 0 else 0
        
        avg_latency = (np.mean(self.performance_tracking['placement_latencies']) 
                      if self.performance_tracking['placement_latencies'] else 0)
        
        avg_load_improvement = (np.mean(self.performance_tracking['load_balance_improvements']) 
                               if self.performance_tracking['load_balance_improvements'] else 0)
        
        metrics = {
            'optimization_status': 'active' if self.model_loaded else 'inactive',
            'performance_summary': {
                'total_placements': self.performance_tracking['total_placements'],
                'success_rate': success_rate,
                'optimization_rate': optimization_rate,
                'average_latency_ms': round(avg_latency, 2),
                'average_load_balance_improvement': round(avg_load_improvement, 4)
            },
            'agent_status': self.dqn_agent.agent.get_training_metrics(),
            'cluster_config': self.cluster_config,
            'optimization_config': self.optimization_config,
            'recent_placements': len([p for p in self.placement_history 
                                    if datetime.fromisoformat(p['timestamp']) > datetime.now() - timedelta(hours=1)]),
            'metrics_timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def simulate_placement_scenarios(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate different placement scenarios for analysis."""
        
        simulation_results = []
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"Simulating scenario {i + 1}/{len(scenarios)}")
            
            scenario_state = scenario.get('cluster_state')
            if scenario_state is None:
                scenario_state = self._get_cluster_state()
            
            pod_spec = scenario['pod_spec']
            
            optimization_result = self.optimize_pod_placement(pod_spec)
            
            simulation_results.append({
                'scenario_id': i + 1,
                'scenario_description': scenario.get('description', f'Scenario {i + 1}'),
                'placement_result': optimization_result,
                'scenario_inputs': scenario
            })
        
        simulation_summary = {
            'total_scenarios': len(scenarios),
            'successful_optimizations': len([r for r in simulation_results if 'error' not in r['placement_result']]),
            'average_quality_score': np.mean([
                r['placement_result'].get('placement_reasoning', {}).get('quality_score', 0)
                for r in simulation_results if 'error' not in r['placement_result']
            ]),
            'simulation_results': simulation_results,
            'simulation_timestamp': datetime.now().isoformat()
        }
        
        return simulation_summary


def main():
    """Main placement optimizer demonstration."""
    
    print("HYDATIS Q-Learning Placement Optimizer - Week 6")
    print("Target: +34% improvement over random placement")
    
    optimizer = HYDATISPlacementOptimizer()
    
    print(f"Model Loaded: {optimizer.model_loaded}")
    print(f"Cluster Configuration: {optimizer.cluster_config['nodes']} nodes")
    print(f"Worker Nodes: {optimizer.cluster_config['worker_nodes']}")
    
    sample_pod = {
        'metadata': {'name': 'test-pod', 'namespace': 'default'},
        'resources': {'cpu_request': 0.2, 'memory_request': 0.3}
    }
    
    placement_result = optimizer.optimize_pod_placement(sample_pod)
    
    print(f"Sample Placement Decision:")
    print(f"  Selected Node: {placement_result.get('selected_node', 'error')}")
    
    if 'placement_reasoning' in placement_result:
        print(f"  Quality Score: {placement_result['placement_reasoning']['quality_score']:.3f}")
        print(f"  Confidence: {placement_result['placement_reasoning']['confidence']:.3f}")
    
    metrics = optimizer.get_optimization_metrics()
    print(f"Optimization Status: {metrics['optimization_status']}")
    
    return optimizer


if __name__ == "__main__":
    optimizer = main()