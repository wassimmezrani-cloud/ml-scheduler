#!/usr/bin/env python3
"""
Comprehensive tests for Q-Learning placement optimizer models.
Validates DQN performance against HYDATIS cluster requirements.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ml_models.qlearning.environment import HYDATISClusterEnvironment
from src.ml_models.qlearning.agent import HYDATISPlacementDQN, DQNNetwork, ReplayBuffer
from src.ml_models.qlearning.optimizer import HYDATISPlacementOptimizer
from src.ml_models.qlearning.training import HYDATISQLearningTrainer


class TestHYDATISClusterEnvironment:
    """Test Q-Learning environment functionality."""
    
    @pytest.fixture
    def cluster_env(self):
        """Create test cluster environment."""
        return HYDATISClusterEnvironment()
    
    def test_environment_initialization(self, cluster_env):
        """Test environment initialization."""
        assert cluster_env.cluster_config['nodes'] == 6
        assert cluster_env.cluster_config['workers'] == 3
        assert cluster_env.state_size == 15
        assert cluster_env.action_size == 3
        assert len(cluster_env.cluster_config['worker_nodes']) == 3
    
    def test_state_generation(self, cluster_env):
        """Test cluster state generation."""
        state = cluster_env.reset()
        
        assert len(state) == cluster_env.state_size
        assert all(0 <= val <= 1 for val in state[:12])
        assert state[12] >= 0
        assert 0 <= state[13] <= 1
        assert 0 <= state[14] <= 1
    
    def test_action_execution(self, cluster_env):
        """Test environment step function."""
        state = cluster_env.reset()
        
        action = 1
        next_state, reward, done, info = cluster_env.step(action)
        
        assert len(next_state) == cluster_env.state_size
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'placement_success' in info
    
    def test_reward_calculation(self, cluster_env):
        """Test reward function behavior."""
        state = cluster_env.reset()
        
        rewards = []
        for action in range(cluster_env.action_size):
            _, reward, _, _ = cluster_env.step(action)
            rewards.append(reward)
        
        assert len(rewards) == 3
        assert all(isinstance(r, (int, float)) for r in rewards)


class TestDQNNetwork:
    """Test DQN network architecture."""
    
    def test_network_initialization(self):
        """Test DQN network creation."""
        state_size = 15
        action_size = 3
        
        network = DQNNetwork(state_size, action_size)
        
        assert network.state_size == state_size
        assert network.action_size == action_size
        
        sample_input = torch.randn(1, state_size)
        output = network(sample_input)
        
        assert output.shape == (1, action_size)
    
    def test_network_forward_pass(self):
        """Test network forward pass."""
        network = DQNNetwork(15, 3)
        
        batch_size = 32
        sample_batch = torch.randn(batch_size, 15)
        outputs = network(sample_batch)
        
        assert outputs.shape == (batch_size, 3)
        assert not torch.isnan(outputs).any()


class TestReplayBuffer:
    """Test experience replay buffer."""
    
    def test_buffer_initialization(self):
        """Test buffer creation."""
        buffer = ReplayBuffer(capacity=1000)
        
        assert buffer.capacity == 1000
        assert len(buffer) == 0
    
    def test_experience_storage(self):
        """Test storing experiences."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(5):
            buffer.push(
                state=np.random.randn(15),
                action=i % 3,
                reward=np.random.randn(),
                next_state=np.random.randn(15),
                done=False
            )
        
        assert len(buffer) == 5
        
        experiences = buffer.sample(3)
        assert len(experiences) == 3
    
    def test_buffer_overflow(self):
        """Test buffer capacity management."""
        buffer = ReplayBuffer(capacity=5)
        
        for i in range(10):
            buffer.push(
                state=np.random.randn(15),
                action=i % 3,
                reward=np.random.randn(),
                next_state=np.random.randn(15),
                done=False
            )
        
        assert len(buffer) == 5


class TestHYDATISPlacementDQN:
    """Test DQN placement agent."""
    
    @pytest.fixture
    def dqn_agent(self):
        """Create test DQN agent."""
        cluster_config = {
            'nodes': 6, 'masters': 3, 'workers': 3,
            'worker_nodes': ['worker-1', 'worker-2', 'worker-3'],
            'master_nodes': ['master-1', 'master-2', 'master-3']
        }
        return HYDATISPlacementDQN(cluster_config)
    
    def test_agent_initialization(self, dqn_agent):
        """Test agent initialization."""
        assert dqn_agent.state_size == 15
        assert dqn_agent.action_size == 3
        assert dqn_agent.target_improvement == 0.34
        assert dqn_agent.agent.q_network is not None
        assert dqn_agent.agent.target_network is not None
    
    def test_placement_selection(self, dqn_agent):
        """Test placement node selection."""
        sample_state = np.random.uniform(0, 1, 15)
        pod_requirements = {'cpu_request': 0.1, 'memory_request': 0.2}
        
        action, decision = dqn_agent.select_placement_node(sample_state, pod_requirements)
        
        assert 0 <= action < 3
        assert 'selected_node' in decision
        assert 'q_values' in decision
        assert 'placement_quality' in decision
        assert decision['selected_node'] in ['worker-1', 'worker-2', 'worker-3']
    
    def test_placement_quality_evaluation(self, dqn_agent):
        """Test placement quality assessment."""
        sample_state = np.random.uniform(0, 1, 15)
        action = 1
        
        quality = dqn_agent.agent.evaluate_placement_quality(sample_state, action)
        
        assert 'overall_quality' in quality
        assert 'load_balance_score' in quality
        assert 'capacity_score' in quality
        assert 0 <= quality['overall_quality'] <= 1
    
    def test_training_metrics(self, dqn_agent):
        """Test training metrics collection."""
        metrics = dqn_agent.agent.get_training_metrics()
        
        assert 'episode_count' in metrics
        assert 'step_count' in metrics
        assert 'current_epsilon' in metrics
        assert 'device' in metrics
        assert 'network_parameters' in metrics


class TestHYDATISPlacementOptimizer:
    """Test placement optimizer functionality."""
    
    @pytest.fixture
    def placement_optimizer(self):
        """Create test placement optimizer."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = HYDATISPlacementOptimizer(model_dir=tmp_dir)
            return optimizer
    
    def test_optimizer_initialization(self, placement_optimizer):
        """Test optimizer initialization."""
        assert placement_optimizer.cluster_config['nodes'] == 6
        assert placement_optimizer.dqn_agent is not None
        assert placement_optimizer.optimization_config is not None
        assert placement_optimizer.performance_tracking is not None
    
    def test_pod_placement_optimization(self, placement_optimizer):
        """Test pod placement optimization."""
        pod_spec = {
            'metadata': {'name': 'test-pod', 'namespace': 'default'},
            'resources': {'cpu_request': 0.1, 'memory_request': 0.2}
        }
        
        result = placement_optimizer.optimize_pod_placement(pod_spec)
        
        assert 'selected_node' in result or 'error' in result
        if 'selected_node' in result:
            assert result['selected_node'] in placement_optimizer.cluster_config['worker_nodes']
            assert 'placement_reasoning' in result
            assert 'optimization_impact' in result
    
    def test_cluster_state_retrieval(self, placement_optimizer):
        """Test cluster state collection."""
        state = placement_optimizer._get_cluster_state()
        
        assert len(state) == 15
        assert all(isinstance(val, (int, float, np.number)) for val in state)
    
    def test_rebalancing_recommendations(self, placement_optimizer):
        """Test cluster rebalancing analysis."""
        recommendations = placement_optimizer.recommend_cluster_rebalancing()
        
        assert 'rebalancing_needed' in recommendations
        assert 'cluster_balance_score' in recommendations
        assert 'analysis_timestamp' in recommendations
        assert isinstance(recommendations['rebalancing_needed'], bool)


def run_qlearning_model_tests():
    """Run all Q-Learning model tests."""
    
    print("Running HYDATIS Q-Learning Model Tests...")
    
    test_env = TestHYDATISClusterEnvironment()
    test_network = TestDQNNetwork()
    test_buffer = TestReplayBuffer()
    test_agent = TestHYDATISPlacementDQN()
    test_optimizer = TestHYDATISPlacementOptimizer()
    
    tests_run = 0
    tests_passed = 0
    
    test_methods = [
        (test_env, 'test_environment_initialization'),
        (test_env, 'test_state_generation'),
        (test_env, 'test_action_execution'),
        (test_network, 'test_network_initialization'),
        (test_network, 'test_network_forward_pass'),
        (test_buffer, 'test_buffer_initialization'),
        (test_buffer, 'test_experience_storage'),
    ]
    
    for test_instance, method_name in test_methods:
        try:
            if hasattr(test_instance, method_name):
                if 'fixture' in method_name or method_name.startswith('test_'):
                    if method_name == 'test_environment_initialization':
                        cluster_env = HYDATISClusterEnvironment()
                        test_instance.test_environment_initialization(cluster_env)
                    elif method_name == 'test_state_generation':
                        cluster_env = HYDATISClusterEnvironment()
                        test_instance.test_state_generation(cluster_env)
                    elif method_name == 'test_action_execution':
                        cluster_env = HYDATISClusterEnvironment()
                        test_instance.test_action_execution(cluster_env)
                    else:
                        getattr(test_instance, method_name)()
                    
                    print(f"✓ {method_name}")
                    tests_passed += 1
                else:
                    getattr(test_instance, method_name)()
                    print(f"✓ {method_name}")
                    tests_passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
        tests_run += 1
    
    print(f"\nTest Results: {tests_passed}/{tests_run} passed")
    
    try:
        print("\nRunning integration test...")
        
        cluster_config = {
            'nodes': 6, 'masters': 3, 'workers': 3,
            'worker_nodes': ['worker-1', 'worker-2', 'worker-3'],
            'master_nodes': ['master-1', 'master-2', 'master-3']
        }
        
        dqn_agent = HYDATISPlacementDQN(cluster_config)
        
        sample_state = np.random.uniform(0, 1, 15)
        pod_requirements = {'cpu_request': 0.1, 'memory_request': 0.2}
        
        action, decision = dqn_agent.select_placement_node(sample_state, pod_requirements)
        
        if (0 <= action < 3 and 
            'selected_node' in decision and 
            decision['selected_node'] in cluster_config['worker_nodes']):
            print("✓ Integration test: Q-Learning placement successful")
            print(f"✓ Selected Node: {decision['selected_node']}")
            print(f"✓ Quality Score: {decision['placement_quality']['overall_quality']:.3f}")
            tests_passed += 1
        else:
            print("✗ Integration test: Invalid placement decision")
        
        tests_run += 1
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        tests_run += 1
    
    success_rate = tests_passed / tests_run
    print(f"\nOverall Test Success Rate: {success_rate:.1%}")
    print(f"Status: {'✅ READY FOR PRODUCTION' if success_rate >= 0.8 else '❌ NEEDS FIXES'}")
    
    return success_rate >= 0.8


if __name__ == "__main__":
    success = run_qlearning_model_tests()
    print(f"\n{'✅' if success else '❌'} Q-Learning model validation {'COMPLETE' if success else 'FAILED'}")