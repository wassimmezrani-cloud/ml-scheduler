#!/usr/bin/env python3
"""
Kubernetes cluster simulation environment for Q-Learning placement optimization.
Simulates HYDATIS 6-node cluster for reinforcement learning training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import gym
from gym import spaces
import random

logger = logging.getLogger(__name__)


class HYDATISClusterEnvironment(gym.Env):
    """Gym environment simulating HYDATIS cluster for Q-Learning placement optimization."""
    
    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        super(HYDATISClusterEnvironment, self).__init__()
        
        # HYDATIS cluster configuration
        self.cluster_config = {
            'nodes': 6,
            'masters': 3,
            'workers': 3,
            'cpu_cores_per_node': 8,
            'memory_gb_per_node': 16,
            'worker_nodes': ['worker-1', 'worker-2', 'worker-3'],
            'master_nodes': ['master-1', 'master-2', 'master-3']
        }
        
        # State space: node resource utilization + workload characteristics
        # [node1_cpu, node1_mem, node2_cpu, node2_mem, ..., pod_cpu_req, pod_mem_req, pod_type]
        self.state_size = (self.cluster_config['nodes'] * 2) + 3  # 6 nodes * 2 resources + pod features
        self.action_size = self.cluster_config['workers']  # Choose among worker nodes
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.state_size,), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(self.action_size)
        
        # Environment state
        self.node_states = np.zeros((self.cluster_config['nodes'], 2))  # [cpu, memory] per node
        self.current_pod = None
        self.step_count = 0
        self.episode_length = 100
        
        # Historical data for realistic simulation
        self.historical_data = historical_data
        self.current_time_idx = 0
        
        # Performance tracking
        self.placement_history = []
        self.reward_history = []
        
        # Initialize cluster state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        
        if self.historical_data is not None and len(self.historical_data) > 0:
            # Use historical data for realistic initialization
            sample_data = self.historical_data.sample(min(6, len(self.historical_data)))
            
            for i, (_, row) in enumerate(sample_data.iterrows()):
                if i < self.cluster_config['nodes']:
                    self.node_states[i, 0] = row.get('cpu_utilization', np.random.uniform(0.08, 0.13))
                    self.node_states[i, 1] = row.get('memory_utilization', np.random.uniform(0.36, 0.43))
        else:
            # Random initialization within HYDATIS expected ranges
            for i in range(self.cluster_config['nodes']):
                self.node_states[i, 0] = np.random.uniform(0.08, 0.13)  # CPU: 8-13%
                self.node_states[i, 1] = np.random.uniform(0.36, 0.43)  # Memory: 36-43%
        
        # Generate initial pod to place
        self.current_pod = self._generate_pod()
        
        self.step_count = 0
        self.current_time_idx = 0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute placement action and return results."""
        
        # Validate action
        if action < 0 or action >= self.action_size:
            return self._get_state(), -1.0, True, {'error': 'Invalid action'}
        
        # Calculate reward for this placement
        reward = self._calculate_placement_reward(action)
        
        # Apply placement to environment
        self._apply_placement(action)
        
        # Update environment state
        self._update_environment()
        
        # Generate next pod
        self.current_pod = self._generate_pod()
        
        # Check episode completion
        self.step_count += 1
        done = self.step_count >= self.episode_length
        
        # Info for debugging
        info = {
            'node_utilization': self.node_states.tolist(),
            'pod_placed': self.current_pod,
            'action_taken': action,
            'step': self.step_count,
            'placement_efficiency': reward
        }
        
        # Track placement history
        self.placement_history.append({
            'step': self.step_count,
            'action': action,
            'reward': reward,
            'pod': self.current_pod.copy(),
            'node_state_before': self.node_states.copy()
        })
        
        self.reward_history.append(reward)
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current environment state."""
        
        # Flatten node states
        node_state_flat = self.node_states.flatten()
        
        # Current pod requirements
        pod_state = [
            self.current_pod['cpu_requirement'],
            self.current_pod['memory_requirement'],
            self.current_pod['workload_type_encoded']
        ]
        
        # Combine into full state vector
        full_state = np.concatenate([node_state_flat, pod_state])
        
        return full_state.astype(np.float32)
    
    def _generate_pod(self) -> Dict[str, Any]:
        """Generate a pod with realistic resource requirements."""
        
        # Pod workload types with different resource patterns
        workload_patterns = {
            'ml_training': {'cpu': (0.4, 0.8), 'memory': (0.5, 0.9), 'type_code': 0.8},
            'web_service': {'cpu': (0.1, 0.3), 'memory': (0.2, 0.4), 'type_code': 0.4},
            'database': {'cpu': (0.2, 0.5), 'memory': (0.6, 0.9), 'type_code': 0.6},
            'batch_job': {'cpu': (0.3, 0.7), 'memory': (0.3, 0.6), 'type_code': 0.2},
            'monitoring': {'cpu': (0.05, 0.15), 'memory': (0.1, 0.3), 'type_code': 0.1}
        }
        
        # Select workload type
        workload_type = random.choice(list(workload_patterns.keys()))
        pattern = workload_patterns[workload_type]
        
        pod = {
            'id': f"pod_{self.step_count}_{random.randint(1000, 9999)}",
            'workload_type': workload_type,
            'workload_type_encoded': pattern['type_code'],
            'cpu_requirement': np.random.uniform(*pattern['cpu']),
            'memory_requirement': np.random.uniform(*pattern['memory']),
            'priority': random.choice([0.3, 0.5, 0.7, 0.9]),  # Pod priority
            'generation_time': datetime.now()
        }
        
        return pod
    
    def _calculate_placement_reward(self, action: int) -> float:
        """Calculate reward for placing current pod on selected node."""
        
        node_idx = action  # Action directly maps to worker node index
        
        # Get target node state
        node_cpu = self.node_states[node_idx, 0]
        node_memory = self.node_states[node_idx, 1]
        
        # Pod requirements
        pod_cpu_req = self.current_pod['cpu_requirement']
        pod_memory_req = self.current_pod['memory_requirement']
        
        # Base reward components
        
        # 1. Resource availability reward (higher = better)
        cpu_availability = 1.0 - node_cpu
        memory_availability = 1.0 - node_memory
        
        # 2. Capacity fitness (can the node handle the pod?)
        cpu_fits = 1.0 if (node_cpu + pod_cpu_req) <= 0.9 else 0.0
        memory_fits = 1.0 if (node_memory + pod_memory_req) <= 0.9 else 0.0
        capacity_fitness = (cpu_fits + memory_fits) / 2
        
        # 3. Load balancing reward (prefer less utilized nodes)
        cluster_cpu_avg = np.mean(self.node_states[:, 0])
        cluster_memory_avg = np.mean(self.node_states[:, 1])
        
        cpu_balance_bonus = max(0, cluster_cpu_avg - node_cpu) * 2
        memory_balance_bonus = max(0, cluster_memory_avg - node_memory) * 2
        
        # 4. Workload-specific preferences
        workload_bonus = 0.0
        if self.current_pod['workload_type'] == 'ml_training':
            # ML workloads prefer high-capacity nodes
            workload_bonus = (cpu_availability + memory_availability) * 0.3
        elif self.current_pod['workload_type'] == 'web_service':
            # Web services prefer balanced load
            workload_bonus = (1 - abs(node_cpu - node_memory)) * 0.2
        
        # 5. Penalty for overloading
        overload_penalty = 0.0
        if (node_cpu + pod_cpu_req) > 0.95:
            overload_penalty = -2.0
        if (node_memory + pod_memory_req) > 0.95:
            overload_penalty -= 2.0
        
        # Combine reward components
        reward = (
            capacity_fitness * 3.0 +          # Must fit (critical)
            cpu_availability * 1.5 +          # CPU availability
            memory_availability * 1.5 +       # Memory availability
            cpu_balance_bonus * 1.0 +         # Load balancing
            memory_balance_bonus * 1.0 +      # Load balancing
            workload_bonus +                  # Workload preferences
            overload_penalty                  # Overload penalty
        )
        
        # Normalize reward to [-1, 1] range
        reward = np.tanh(reward / 10.0)
        
        return reward
    
    def _apply_placement(self, action: int):
        """Apply the placement decision to the environment."""
        
        node_idx = action
        
        # Update node resource utilization
        self.node_states[node_idx, 0] += self.current_pod['cpu_requirement']
        self.node_states[node_idx, 1] += self.current_pod['memory_requirement']
        
        # Ensure utilization doesn't exceed 100%
        self.node_states[node_idx, 0] = min(self.node_states[node_idx, 0], 1.0)
        self.node_states[node_idx, 1] = min(self.node_states[node_idx, 1], 1.0)
    
    def _update_environment(self):
        """Update environment state to simulate time progression."""
        
        # Simulate natural resource decay (pods completing/scaling down)
        decay_rate = 0.02  # 2% per step
        
        for i in range(self.cluster_config['nodes']):
            # Natural decay with some randomness
            self.node_states[i, 0] = max(0.05, self.node_states[i, 0] - np.random.exponential(decay_rate))
            self.node_states[i, 1] = max(0.05, self.node_states[i, 1] - np.random.exponential(decay_rate))
        
        # Add some background load variation
        for i in range(self.cluster_config['nodes']):
            # Small random fluctuations
            self.node_states[i, 0] += np.random.normal(0, 0.005)
            self.node_states[i, 1] += np.random.normal(0, 0.005)
            
            # Keep within bounds
            self.node_states[i, 0] = np.clip(self.node_states[i, 0], 0.05, 1.0)
            self.node_states[i, 1] = np.clip(self.node_states[i, 1], 0.05, 1.0)
    
    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get current cluster performance metrics."""
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cluster_utilization': {
                'avg_cpu': float(np.mean(self.node_states[:, 0])),
                'avg_memory': float(np.mean(self.node_states[:, 1])),
                'max_cpu': float(np.max(self.node_states[:, 0])),
                'max_memory': float(np.max(self.node_states[:, 1])),
                'cpu_std': float(np.std(self.node_states[:, 0])),
                'memory_std': float(np.std(self.node_states[:, 1]))
            },
            'load_balancing': {
                'cpu_balance_score': 1 - float(np.std(self.node_states[:, 0])),
                'memory_balance_score': 1 - float(np.std(self.node_states[:, 1])),
                'overall_balance': 1 - float(np.mean([np.std(self.node_states[:, 0]), np.std(self.node_states[:, 1])]))
            },
            'placement_efficiency': {
                'total_placements': len(self.placement_history),
                'avg_reward': float(np.mean(self.reward_history)) if self.reward_history else 0.0,
                'successful_placements': len([r for r in self.reward_history if r > 0]),
                'placement_success_rate': len([r for r in self.reward_history if r > 0]) / len(self.reward_history) if self.reward_history else 0
            }
        }
        
        return metrics
    
    def render(self, mode='human'):
        """Render current cluster state."""
        
        if mode == 'human':
            print(f"\n=== HYDATIS Cluster State (Step {self.step_count}) ===")
            print("Node Utilization:")
            
            for i, node_type in enumerate(['worker-1', 'worker-2', 'worker-3', 'master-1', 'master-2', 'master-3']):
                cpu_pct = self.node_states[i, 0] * 100
                mem_pct = self.node_states[i, 1] * 100
                load_indicator = "🔴" if cpu_pct > 80 or mem_pct > 80 else "🟡" if cpu_pct > 60 or mem_pct > 60 else "🟢"
                print(f"  {node_type}: CPU {cpu_pct:5.1f}% | MEM {mem_pct:5.1f}% {load_indicator}")
            
            if self.current_pod:
                print(f"Current Pod: {self.current_pod['workload_type']} (CPU: {self.current_pod['cpu_requirement']:.2f}, MEM: {self.current_pod['memory_requirement']:.2f})")
            
            avg_reward = np.mean(self.reward_history) if self.reward_history else 0
            print(f"Average Reward: {avg_reward:.3f}")
        
        elif mode == 'metrics':
            return self.get_cluster_metrics()


class QLearningTrainingEnvironment:
    """Training environment wrapper for Q-Learning agent development."""
    
    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        self.env = HYDATISClusterEnvironment(historical_data)
        self.training_episodes = 1000
        self.evaluation_episodes = 100
        
    def generate_training_episodes(self, num_episodes: int = None) -> List[Dict]:
        """Generate training episodes for Q-Learning."""
        
        episodes = num_episodes or self.training_episodes
        logger.info(f"Generating {episodes} training episodes...")
        
        episode_data = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_rewards = []
            episode_transitions = []
            
            done = False
            while not done:
                # Random action for data generation (will be replaced by learned policy)
                action = self.env.action_space.sample()
                
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                transition = {
                    'state': state.copy(),
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done,
                    'info': info
                }
                
                episode_transitions.append(transition)
                episode_rewards.append(reward)
                
                state = next_state
            
            episode_summary = {
                'episode': episode,
                'total_reward': sum(episode_rewards),
                'avg_reward': np.mean(episode_rewards),
                'steps': len(episode_transitions),
                'transitions': episode_transitions,
                'final_metrics': self.env.get_cluster_metrics()
            }
            
            episode_data.append(episode_summary)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean([ep['avg_reward'] for ep in episode_data[-100:]])
                logger.info(f"Episode {episode + 1}: Average reward = {avg_reward:.3f}")
        
        return episode_data
    
    def evaluate_random_baseline(self) -> Dict[str, float]:
        """Evaluate random placement strategy as baseline."""
        
        logger.info("Evaluating random placement baseline...")
        
        evaluation_rewards = []
        placement_success_rates = []
        
        for episode in range(self.evaluation_episodes):
            state = self.env.reset()
            episode_rewards = []
            
            done = False
            while not done:
                action = self.env.action_space.sample()  # Random action
                next_state, reward, done, info = self.env.step(action)
                episode_rewards.append(reward)
                state = next_state
            
            evaluation_rewards.extend(episode_rewards)
            
            # Calculate placement success rate for this episode
            success_rate = len([r for r in episode_rewards if r > 0]) / len(episode_rewards)
            placement_success_rates.append(success_rate)
        
        baseline_metrics = {
            'avg_reward': float(np.mean(evaluation_rewards)),
            'std_reward': float(np.std(evaluation_rewards)),
            'avg_placement_success_rate': float(np.mean(placement_success_rates)),
            'total_episodes': self.evaluation_episodes,
            'total_placements': len(evaluation_rewards)
        }
        
        logger.info(f"Random baseline: {baseline_metrics['avg_reward']:.3f} avg reward, {baseline_metrics['avg_placement_success_rate']:.1%} success rate")
        
        return baseline_metrics
    
    def calculate_improvement_potential(self, baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate potential improvement over random baseline."""
        
        # Target: +34% improvement over random placement
        target_improvement = 0.34
        baseline_reward = baseline_metrics['avg_reward']
        target_reward = baseline_reward * (1 + target_improvement)
        
        improvement_analysis = {
            'baseline_reward': baseline_reward,
            'target_improvement_pct': target_improvement,
            'target_reward': target_reward,
            'improvement_gap': target_reward - baseline_reward,
            'baseline_success_rate': baseline_metrics['avg_placement_success_rate'],
            'target_success_rate': min(0.95, baseline_metrics['avg_placement_success_rate'] * (1 + target_improvement))
        }
        
        return improvement_analysis


def main():
    """Test HYDATIS cluster environment."""
    
    # Initialize environment
    training_env = QLearningTrainingEnvironment()
    
    print("HYDATIS Q-Learning Environment - Week 6")
    print(f"Cluster: {training_env.env.cluster_config['nodes']} nodes")
    print(f"State space: {training_env.env.observation_space.shape}")
    print(f"Action space: {training_env.env.action_space.n} actions")
    
    # Test environment
    state = training_env.env.reset()
    print(f"\nInitial state shape: {state.shape}")
    
    # Test single step
    action = training_env.env.action_space.sample()
    next_state, reward, done, info = training_env.env.step(action)
    
    print(f"Test step: action={action}, reward={reward:.3f}, done={done}")
    
    # Evaluate random baseline
    baseline = training_env.evaluate_random_baseline()
    improvement_potential = training_env.calculate_improvement_potential(baseline)
    
    print(f"\nRandom Baseline Results:")
    print(f"- Average reward: {baseline['avg_reward']:.3f}")
    print(f"- Success rate: {baseline['avg_placement_success_rate']:.1%}")
    
    print(f"\nImprovement Targets:")
    print(f"- Target improvement: +34%")
    print(f"- Target reward: {improvement_potential['target_reward']:.3f}")
    print(f"- Target success rate: {improvement_potential['target_success_rate']:.1%}")
    
    print("\n✓ HYDATIS Q-Learning environment ready for agent training")
    
    return training_env

if __name__ == "__main__":
    env = main()