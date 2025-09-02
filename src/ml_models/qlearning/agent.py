#!/usr/bin/env python3
"""
Deep Q-Network (DQN) agent for HYDATIS ML Scheduler placement optimization.
Implements PyTorch-based reinforcement learning for pod placement decisions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """Deep Q-Network for pod placement decision making."""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [256, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class HYDATISDQNAgent:
    """DQN Agent for HYDATIS cluster pod placement optimization."""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 50000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"DQN Agent using device: {self.device}")
        
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.memory = ReplayBuffer(memory_size)
        
        self.update_target_network()
        
        self.step_count = 0
        self.episode_count = 0
        
        self.performance_metrics = {
            'total_rewards': [],
            'average_rewards': [],
            'loss_history': [],
            'epsilon_history': [],
            'placement_success_rate': []
        }
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        
        if training and random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self) -> Optional[float]:
        """Train the model on a batch of experiences."""
        
        if len(self.memory) < self.batch_size:
            return None
        
        experiences = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        self.step_count += 1
        
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def evaluate_placement_quality(self, state: np.ndarray, action: int) -> Dict[str, float]:
        """Evaluate the quality of a placement decision."""
        
        node_loads = state[:6]
        node_memory = state[6:12]
        
        selected_node_cpu = node_loads[action]
        selected_node_memory = node_memory[action]
        
        avg_cpu_load = np.mean(node_loads)
        avg_memory_load = np.mean(node_memory)
        
        load_balance_score = 1.0 - abs(selected_node_cpu - avg_cpu_load)
        memory_balance_score = 1.0 - abs(selected_node_memory - avg_memory_load)
        
        capacity_score = (1.0 - selected_node_cpu) * 0.6 + (1.0 - selected_node_memory) * 0.4
        
        overall_quality = (load_balance_score * 0.3 + 
                          memory_balance_score * 0.3 + 
                          capacity_score * 0.4)
        
        return {
            'load_balance_score': load_balance_score,
            'memory_balance_score': memory_balance_score,
            'capacity_score': capacity_score,
            'overall_quality': overall_quality,
            'selected_node_cpu': selected_node_cpu,
            'selected_node_memory': selected_node_memory
        }
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().numpy().flatten()
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics."""
        
        avg_reward = np.mean(self.performance_metrics['total_rewards'][-100:]) if self.performance_metrics['total_rewards'] else 0
        avg_loss = np.mean(self.performance_metrics['loss_history'][-100:]) if self.performance_metrics['loss_history'] else 0
        
        return {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'current_epsilon': self.epsilon,
            'average_reward_last_100': avg_reward,
            'average_loss_last_100': avg_loss,
            'memory_size': len(self.memory),
            'total_episodes_trained': len(self.performance_metrics['total_rewards']),
            'exploration_rate': self.epsilon,
            'network_parameters': sum(p.numel() for p in self.q_network.parameters()),
            'device': str(self.device)
        }
    
    def save_agent(self, model_dir: str, experiment_name: str = "hydatis_qlearning") -> Dict[str, str]:
        """Save agent models and metadata."""
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files_saved = {}
        
        q_network_path = model_path / f"{experiment_name}_q_network_{timestamp}.pth"
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }
        }, q_network_path)
        files_saved['q_network'] = str(q_network_path)
        
        target_network_path = model_path / f"{experiment_name}_target_network_{timestamp}.pth"
        torch.save(self.target_network.state_dict(), target_network_path)
        files_saved['target_network'] = str(target_network_path)
        
        metrics_path = model_path / f"{experiment_name}_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'performance_metrics': self.performance_metrics,
                'training_metrics': self.get_training_metrics(),
                'agent_config': {
                    'state_size': self.state_size,
                    'action_size': self.action_size,
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'epsilon_start': self.epsilon_start,
                    'epsilon_end': self.epsilon_end,
                    'epsilon_decay': self.epsilon_decay,
                    'batch_size': self.batch_size,
                    'target_update_freq': self.target_update_freq
                },
                'save_timestamp': datetime.now().isoformat()
            }, f, indent=2)
        files_saved['metrics'] = str(metrics_path)
        
        logger.info(f"DQN Agent saved to {model_dir}")
        return files_saved
    
    def load_agent(self, model_dir: str, experiment_name: str = "hydatis_qlearning") -> bool:
        """Load agent from saved models."""
        
        model_path = Path(model_dir)
        
        try:
            q_network_files = list(model_path.glob(f"{experiment_name}_q_network_*.pth"))
            if not q_network_files:
                logger.error(f"No Q-network model found in {model_dir}")
                return False
            
            latest_q_file = max(q_network_files, key=lambda x: x.stat().st_mtime)
            
            checkpoint = torch.load(latest_q_file, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            hyperparams = checkpoint.get('hyperparameters', {})
            self.epsilon = hyperparams.get('epsilon', self.epsilon_end)
            
            target_network_files = list(model_path.glob(f"{experiment_name}_target_network_*.pth"))
            if target_network_files:
                latest_target_file = max(target_network_files, key=lambda x: x.stat().st_mtime)
                self.target_network.load_state_dict(torch.load(latest_target_file, map_location=self.device))
            else:
                self.update_target_network()
            
            logger.info(f"DQN Agent loaded from {latest_q_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading agent: {e}")
            return False


class HYDATISPlacementDQN:
    """Main DQN placement optimizer for HYDATIS cluster."""
    
    def __init__(self, cluster_config: Dict[str, Any]):
        self.cluster_config = cluster_config
        
        self.state_size = (cluster_config['nodes'] * 2) + 3
        self.action_size = cluster_config['workers']
        
        self.agent = HYDATISDQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=1e-4,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )
        
        self.training_history = {
            'episode_rewards': [],
            'placement_improvements': [],
            'baseline_comparisons': []
        }
        
        self.target_improvement = 0.34
        
    def select_placement_node(self, cluster_state: np.ndarray, 
                             pod_requirements: Dict[str, float],
                             training: bool = False) -> Tuple[int, Dict[str, Any]]:
        """Select optimal node for pod placement."""
        
        action = self.agent.act(cluster_state, training=training)
        
        placement_analysis = self.agent.evaluate_placement_quality(cluster_state, action)
        
        placement_decision = {
            'selected_node_index': action,
            'selected_node': self.cluster_config['worker_nodes'][action],
            'q_values': self.agent.get_q_values(cluster_state).tolist(),
            'placement_quality': placement_analysis,
            'exploration_factor': self.agent.epsilon,
            'decision_confidence': max(self.agent.get_q_values(cluster_state)) - np.mean(self.agent.get_q_values(cluster_state)),
            'pod_requirements': pod_requirements
        }
        
        return action, placement_decision
    
    def calculate_placement_improvement(self, dqn_rewards: List[float], 
                                      random_rewards: List[float]) -> float:
        """Calculate placement improvement over random baseline."""
        
        if not dqn_rewards or not random_rewards:
            return 0.0
        
        dqn_avg = np.mean(dqn_rewards)
        random_avg = np.mean(random_rewards)
        
        if random_avg == 0:
            return 1.0 if dqn_avg > 0 else 0.0
        
        improvement = (dqn_avg - random_avg) / abs(random_avg)
        return improvement
    
    def train_episode(self, environment) -> Dict[str, Any]:
        """Train agent for one episode."""
        
        state = environment.reset()
        total_reward = 0
        episode_steps = 0
        episode_losses = []
        
        done = False
        while not done and episode_steps < 200:
            action = self.agent.act(state, training=True)
            
            next_state, reward, done, info = environment.step(action)
            
            self.agent.remember(state, action, reward, next_state, done)
            
            if len(self.agent.memory) > self.agent.batch_size:
                loss = self.agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            episode_steps += 1
        
        self.agent.episode_count += 1
        self.agent.performance_metrics['total_rewards'].append(total_reward)
        
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            self.agent.performance_metrics['loss_history'].append(avg_loss)
        
        self.agent.performance_metrics['epsilon_history'].append(self.agent.epsilon)
        
        episode_metrics = {
            'episode': self.agent.episode_count,
            'total_reward': total_reward,
            'episode_steps': episode_steps,
            'average_loss': np.mean(episode_losses) if episode_losses else 0,
            'epsilon': self.agent.epsilon,
            'memory_usage': len(self.agent.memory),
            'placement_decisions': info.get('placement_decisions', 0)
        }
        
        return episode_metrics
    
    def evaluate_performance(self, environment, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate agent performance against random baseline."""
        
        dqn_rewards = []
        random_rewards = []
        
        for episode in range(num_episodes):
            state = environment.reset()
            total_dqn_reward = 0
            total_random_reward = 0
            
            done = False
            steps = 0
            
            while not done and steps < 200:
                dqn_action = self.agent.act(state, training=False)
                random_action = np.random.randint(0, self.action_size)
                
                next_state, dqn_reward, done, _ = environment.step(dqn_action)
                environment_copy = environment.copy() if hasattr(environment, 'copy') else environment
                _, random_reward, _, _ = environment_copy.step(random_action) if hasattr(environment, 'copy') else (None, 0, False, {})
                
                total_dqn_reward += dqn_reward
                total_random_reward += random_reward if random_reward else 0
                
                state = next_state
                steps += 1
            
            dqn_rewards.append(total_dqn_reward)
            random_rewards.append(total_random_reward)
        
        improvement = self.calculate_placement_improvement(dqn_rewards, random_rewards)
        
        evaluation_results = {
            'episodes_evaluated': num_episodes,
            'dqn_average_reward': np.mean(dqn_rewards),
            'random_average_reward': np.mean(random_rewards),
            'improvement_over_random': improvement,
            'target_achievement': improvement >= self.target_improvement,
            'target_improvement': self.target_improvement,
            'evaluation_timestamp': datetime.now().isoformat(),
            'performance_stability': np.std(dqn_rewards) / (np.mean(dqn_rewards) + 1e-8)
        }
        
        return evaluation_results
    
    def get_placement_insights(self, state: np.ndarray) -> Dict[str, Any]:
        """Get detailed insights about placement decision."""
        
        q_values = self.agent.get_q_values(state)
        best_action = np.argmax(q_values)
        
        node_loads = state[:6]
        node_memory = state[6:12]
        
        insights = {
            'recommended_node': self.cluster_config['worker_nodes'][best_action],
            'confidence_score': (max(q_values) - np.mean(q_values)) / (np.std(q_values) + 1e-8),
            'cluster_utilization': {
                'average_cpu': np.mean(node_loads),
                'average_memory': np.mean(node_memory),
                'cpu_variance': np.var(node_loads),
                'memory_variance': np.var(node_memory)
            },
            'node_rankings': [
                {
                    'node': self.cluster_config['worker_nodes'][i],
                    'q_value': float(q_values[i]),
                    'cpu_load': float(node_loads[i]),
                    'memory_load': float(node_memory[i]),
                    'capacity_score': (1 - node_loads[i]) * 0.6 + (1 - node_memory[i]) * 0.4
                }
                for i in range(self.action_size)
            ],
            'exploration_rate': self.agent.epsilon
        }
        
        insights['node_rankings'].sort(key=lambda x: x['q_value'], reverse=True)
        
        return insights


def main():
    """Main DQN agent demonstration."""
    
    print("HYDATIS DQN Placement Agent - Week 6")
    print("Target: +34% improvement over random placement")
    
    cluster_config = {
        'nodes': 6,
        'masters': 3,
        'workers': 3,
        'cpu_cores_per_node': 8,
        'memory_gb_per_node': 16,
        'worker_nodes': ['worker-1', 'worker-2', 'worker-3'],
        'master_nodes': ['master-1', 'master-2', 'master-3']
    }
    
    dqn_placement = HYDATISPlacementDQN(cluster_config)
    
    print(f"State Size: {dqn_placement.state_size}")
    print(f"Action Size: {dqn_placement.action_size}")
    print(f"Device: {dqn_placement.agent.device}")
    print(f"Network Parameters: {sum(p.numel() for p in dqn_placement.agent.q_network.parameters()):,}")
    
    sample_state = np.random.uniform(0, 1, dqn_placement.state_size)
    sample_pod = {'cpu_request': 0.1, 'memory_request': 0.2}
    
    action, decision = dqn_placement.select_placement_node(sample_state, sample_pod)
    
    print(f"Sample Decision: Node {decision['selected_node']}")
    print(f"Placement Quality: {decision['placement_quality']['overall_quality']:.3f}")
    
    return dqn_placement


if __name__ == "__main__":
    agent = main()