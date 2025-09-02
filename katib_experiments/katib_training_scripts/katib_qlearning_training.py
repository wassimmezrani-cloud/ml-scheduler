#!/usr/bin/env python3
"""
Katib Q-Learning Training Script for HYDATIS ML Scheduler
Optimizes Q-Learning hyperparameters for placement optimization.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import mlflow
import mlflow.pytorch
from collections import deque, namedtuple
import random
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class HYDATISQNetwork(nn.Module):
    """Q-Network optimized for HYDATIS 6-node cluster scheduling."""
    
    def __init__(self, state_size, action_size, hidden_size=128, dropout_rate=0.1):
        super(HYDATISQNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Initialize weights for better convergence
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state):
        return self.network(state)


class HYDATISSchedulingEnvironment:
    """HYDATIS cluster environment simulation for Q-Learning optimization."""
    
    def __init__(self, reward_weights):
        self.num_nodes = 6  # HYDATIS cluster configuration
        self.node_capacity = {'cpu': 8, 'memory': 16}  # Per node: 8 CPU cores, 16GB RAM
        
        # Current HYDATIS cluster state
        self.baseline_utilization = {'cpu': 0.85, 'memory': 0.4}
        self.target_utilization = {'cpu': 0.65, 'memory': 0.5}
        
        # Business-aligned reward function weights (from Katib)
        self.reward_weights = reward_weights
        
        # Cluster state
        self.reset()
    
    def reset(self):
        """Reset environment to initial HYDATIS cluster state."""
        
        # Initialize node loads based on current HYDATIS patterns
        self.node_cpu_loads = np.random.uniform(0.7, 0.9, self.num_nodes)  # High utilization
        self.node_memory_loads = np.random.uniform(0.3, 0.5, self.num_nodes)  # Moderate memory
        
        # Scheduling queue and metrics
        self.pending_pods = []
        self.scheduled_pods = 0
        self.failed_placements = 0
        self.step_count = 0
        self.episode_reward = 0
        
        # Business metrics tracking
        self.availability_violations = 0
        self.cpu_target_violations = 0
        
        return self.get_state()
    
    def get_state(self):
        """Get current cluster state representation."""
        
        # State vector: [node_loads, cluster_metrics, queue_state, business_metrics]
        state = np.concatenate([
            self.node_cpu_loads,  # 6 node CPU loads
            self.node_memory_loads,  # 6 node memory loads
            [np.mean(self.node_cpu_loads)],  # Average cluster CPU
            [np.std(self.node_cpu_loads)],   # CPU load distribution
            [np.mean(self.node_memory_loads)],  # Average cluster memory
            [len(self.pending_pods) / 20.0],  # Normalized queue length
            [self.scheduled_pods / 100.0],    # Normalized scheduled count
            [max(0, 1.0 - self.failed_placements / max(1, self.scheduled_pods))]  # Success rate
        ])
        
        return state
    
    def step(self, action):
        """Execute scheduling action and return reward."""
        
        # Action: select node (0-5 for 6 HYDATIS nodes)
        selected_node = action % self.num_nodes
        
        # Generate pod with realistic HYDATIS workload characteristics
        pod_cpu_req = np.random.gamma(2, 0.15)  # 0.3 CPU average
        pod_memory_req = np.random.gamma(2, 0.5)  # 1GB average
        
        # Simulate placement attempt
        placement_successful = self.attempt_placement(selected_node, pod_cpu_req, pod_memory_req)
        
        # Calculate business-aligned reward
        reward = self.calculate_business_reward(selected_node, pod_cpu_req, pod_memory_req, placement_successful)
        
        # Update environment state
        if placement_successful:
            self.node_cpu_loads[selected_node] += pod_cpu_req
            self.node_memory_loads[selected_node] += pod_memory_req
            self.scheduled_pods += 1
        else:
            self.failed_placements += 1
            
        # Clip loads to realistic ranges
        self.node_cpu_loads = np.clip(self.node_cpu_loads, 0, 1)
        self.node_memory_loads = np.clip(self.node_memory_loads, 0, 1)
        
        self.step_count += 1
        self.episode_reward += reward
        
        # Episode termination conditions
        done = (
            self.step_count >= 200 or  # Max episode length
            np.any(self.node_cpu_loads > 0.95) or  # CPU overload
            np.any(self.node_memory_loads > 0.95) or  # Memory overload
            self.failed_placements > 10  # Too many failures
        )
        
        return self.get_state(), reward, done
    
    def attempt_placement(self, node_id, cpu_req, memory_req):
        """Simulate pod placement attempt on HYDATIS cluster."""
        
        # Check node capacity
        cpu_available = 1.0 - self.node_cpu_loads[node_id]
        memory_available = 1.0 - self.node_memory_loads[node_id]
        
        # Placement success based on available capacity
        cpu_fits = cpu_req <= cpu_available
        memory_fits = memory_req <= memory_available
        
        # Add some randomness for realistic scheduling
        placement_probability = 0.95 if (cpu_fits and memory_fits) else 0.1
        
        return np.random.random() < placement_probability
    
    def calculate_business_reward(self, node_id, cpu_req, memory_req, placement_successful):
        """Calculate reward aligned with HYDATIS business objectives."""
        
        if not placement_successful:
            return -1.0  # Heavy penalty for failed placements
        
        # Get current and predicted node state
        current_cpu = self.node_cpu_loads[node_id]
        current_memory = self.node_memory_loads[node_id]
        predicted_cpu = current_cpu + cpu_req
        predicted_memory = current_memory + memory_req
        
        # Reward components aligned with HYDATIS business targets
        
        # 1. CPU utilization optimization (target: 65%)
        cpu_target = 0.65
        cpu_efficiency_reward = 1.0 - abs(predicted_cpu - cpu_target)
        
        # 2. Load balancing across 6 HYDATIS nodes
        cluster_balance_reward = 1.0 - np.std(self.node_cpu_loads)
        
        # 3. Availability protection (target: 99.7%)
        availability_penalty = 0.0
        if predicted_cpu > 0.9 or predicted_memory > 0.9:
            availability_penalty = -2.0  # Severe penalty for overload risk
        elif predicted_cpu > 0.8:
            availability_penalty = -0.5  # Moderate penalty for high utilization
        
        # 4. Scheduling efficiency bonus
        scheduling_efficiency_bonus = 0.0
        if predicted_cpu >= 0.5 and predicted_cpu <= 0.7:  # Sweet spot for efficiency
            scheduling_efficiency_bonus = 0.3
        
        # 5. Business SLA alignment bonus
        sla_bonus = 0.0
        if predicted_cpu <= cpu_target and predicted_memory <= 0.6:
            sla_bonus = 0.2  # Bonus for meeting business targets
        
        # Weighted reward calculation using Katib-optimized weights
        total_reward = (
            cpu_efficiency_reward * self.reward_weights['cpu_efficiency'] +
            cluster_balance_reward * self.reward_weights['load_balance'] +
            availability_penalty +  # Always apply availability penalty
            scheduling_efficiency_bonus * 0.1 +
            sla_bonus * self.reward_weights['availability']
        )
        
        # Track business metric violations
        if predicted_cpu > cpu_target:
            self.cpu_target_violations += 1
        
        if predicted_cpu > 0.9:
            self.availability_violations += 1
        
        return total_reward


class HYDATISQLearningTrainer:
    """Q-Learning trainer for HYDATIS cluster placement optimization."""
    
    def __init__(self, prometheus_url, mlflow_uri):
        self.prometheus_url = prometheus_url
        self.mlflow_uri = mlflow_uri
        
        # HYDATIS business targets
        self.business_targets = {
            'improvement_vs_random': 0.34,  # 34% improvement target
            'cpu_utilization_target': 0.65,
            'availability_target': 0.997,
            'monthly_savings_target': 30000
        }
    
    def train_agent(self, hyperparams, reward_weights):
        """Train Q-Learning agent with Katib hyperparameters."""
        
        logger.info("🧠 Training Q-Learning agent for HYDATIS placement optimization")
        
        # Initialize environment with reward weights
        env = HYDATISSchedulingEnvironment(reward_weights)
        
        state_size = len(env.get_state())
        action_size = env.num_nodes
        
        # Initialize Q-Network with Katib hyperparameters
        q_network = HYDATISQNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hyperparams['hidden_size']
        )
        
        target_network = HYDATISQNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hyperparams['hidden_size']
        )
        
        # Initialize target network with same weights
        target_network.load_state_dict(q_network.state_dict())
        
        optimizer = optim.Adam(q_network.parameters(), lr=hyperparams['learning_rate'])
        criterion = nn.MSELoss()
        
        # Experience replay buffer
        memory = deque(maxlen=hyperparams['memory_size'])
        
        # Training parameters
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = hyperparams['epsilon_decay']
        gamma = hyperparams['gamma']
        batch_size = hyperparams['batch_size']
        target_update_frequency = hyperparams['target_update_frequency']
        
        # Training metrics
        episode_rewards = []
        episode_business_scores = []
        convergence_metrics = []
        
        # Training loop
        num_episodes = 1000
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(200):  # Max steps per episode
                
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, action_size - 1)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        q_values = q_network(state_tensor)
                        action = q_values.argmax().item()
                
                # Take action in environment
                next_state, reward, done = env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Store experience
                memory.append(Experience(state, action, reward, next_state, done))
                state = next_state
                
                # Train network
                if len(memory) >= batch_size:
                    batch = random.sample(memory, batch_size)
                    self.train_network(q_network, target_network, batch, optimizer, criterion, gamma)
                
                # Update target network
                if episode_steps % target_update_frequency == 0:
                    target_network.load_state_dict(q_network.state_dict())
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Calculate business performance score
            business_score = self.calculate_episode_business_score(env, episode_reward)
            episode_business_scores.append(business_score)
            
            # Update epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # Track convergence
            if episode >= 100:
                recent_rewards = episode_rewards[-100:]
                reward_stability = 1.0 - (np.std(recent_rewards) / max(1, np.mean(recent_rewards)))
                convergence_metrics.append(reward_stability)
            
            # Periodic logging
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_business_score = np.mean(episode_business_scores[-100:])
                
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.4f}, Business Score = {avg_business_score:.4f}, Epsilon = {epsilon:.4f}")
        
        # Calculate final performance metrics
        final_avg_reward = np.mean(episode_rewards[-100:])
        final_business_score = np.mean(episode_business_scores[-100:])
        reward_convergence = np.mean(convergence_metrics[-50:]) if convergence_metrics else 0
        
        # Calculate improvement vs random baseline
        random_baseline_reward = -0.3  # Random placement typically performs poorly
        improvement_vs_random = (final_avg_reward - random_baseline_reward) / abs(random_baseline_reward)
        
        return {
            'q_network': q_network,
            'final_average_reward': final_avg_reward,
            'business_score': final_business_score,
            'improvement_vs_random': improvement_vs_random,
            'reward_convergence_rate': reward_convergence,
            'training_episodes': num_episodes,
            'cpu_violations': env.cpu_target_violations,
            'availability_violations': env.availability_violations
        }
    
    def train_network(self, q_network, target_network, batch, optimizer, criterion, gamma):
        """Train Q-Network with experience replay batch."""
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.BoolTensor(dones)
        
        # Current Q-values
        current_q_values = q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        # Next Q-values from target network
        next_q_values = target_network(next_states_tensor).max(1)[0].detach()
        target_q_values = rewards_tensor + (gamma * next_q_values * ~dones_tensor)
        
        # Compute loss
        loss = criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)
        
        optimizer.step()
    
    def calculate_episode_business_score(self, env, episode_reward):
        """Calculate business-aligned performance score."""
        
        # CPU utilization achievement
        avg_cpu = np.mean(env.node_cpu_loads)
        cpu_target_achievement = max(0, 1.0 - abs(avg_cpu - 0.65) / 0.65)
        
        # Availability protection (penalize violations)
        availability_protection = 1.0 - (env.availability_violations / max(1, env.step_count))
        
        # Scheduling efficiency
        scheduling_efficiency = env.scheduled_pods / max(1, env.scheduled_pods + env.failed_placements)
        
        # Load balancing
        load_balance_score = 1.0 - np.std(env.node_cpu_loads)
        
        # Combined business score
        business_score = (
            cpu_target_achievement * 0.4 +
            availability_protection * 0.3 +
            scheduling_efficiency * 0.2 +
            load_balance_score * 0.1
        )
        
        return business_score


def main():
    """Main Katib Q-Learning training function."""
    
    parser = argparse.ArgumentParser(description='HYDATIS Q-Learning Katib Training')
    
    # Katib hyperparameters
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--gamma', type=float, required=True)
    parser.add_argument('--epsilon_decay', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--memory_size', type=int, required=True)
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--target_update_frequency', type=int, required=True)
    
    # Reward function weights (Katib optimization targets)
    parser.add_argument('--reward_efficiency_weight', type=float, required=True)
    parser.add_argument('--reward_balance_weight', type=float, required=True)
    parser.add_argument('--reward_availability_weight', type=float, required=True)
    
    args = parser.parse_args()
    
    # Setup environment
    prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus-server.monitoring:9090')
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server.kubeflow:5000')
    katib_experiment = os.getenv('KATIB_EXPERIMENT_NAME', 'hydatis-qlearning-hpo')
    
    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(f"katib-{katib_experiment}")
    
    # Extract hyperparameters
    hyperparams = {
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'epsilon_decay': args.epsilon_decay,
        'batch_size': args.batch_size,
        'memory_size': args.memory_size,
        'hidden_size': args.hidden_size,
        'target_update_frequency': args.target_update_frequency
    }
    
    # Extract reward weights
    reward_weights = {
        'cpu_efficiency': args.reward_efficiency_weight,
        'load_balance': args.reward_balance_weight,
        'availability': args.reward_availability_weight
    }
    
    # Normalize reward weights
    total_weight = sum(reward_weights.values())
    reward_weights = {k: v/total_weight for k, v in reward_weights.items()}
    
    # Initialize trainer
    trainer = HYDATISQLearningTrainer(prometheus_url, mlflow_uri)
    
    # Train agent with MLflow tracking
    with mlflow.start_run(run_name=f"katib_qlearning_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        
        # Log hyperparameters and configuration
        mlflow.log_params(hyperparams)
        mlflow.log_params({f"reward_weight_{k}": v for k, v in reward_weights.items()})
        mlflow.log_param("optimization_source", "katib_hpo")
        mlflow.log_param("cluster_target", "hydatis_34_percent_improvement")
        mlflow.log_param("business_objective", "65_cpu_997_availability")
        
        # Train agent
        training_results = trainer.train_agent(hyperparams, reward_weights)
        
        # Log training metrics
        mlflow.log_metric("final_average_reward", training_results['final_average_reward'])
        mlflow.log_metric("business_score", training_results['business_score'])
        mlflow.log_metric("improvement_vs_random", training_results['improvement_vs_random'])
        mlflow.log_metric("reward_convergence_rate", training_results['reward_convergence_rate'])
        mlflow.log_metric("cpu_violations", training_results['cpu_violations'])
        mlflow.log_metric("availability_violations", training_results['availability_violations'])
        
        # Target achievement validation
        target_achieved = training_results['improvement_vs_random'] >= 0.34
        mlflow.log_metric("target_achieved", int(target_achieved))
        
        # Log trained model
        mlflow.pytorch.log_model(
            pytorch_model=training_results['q_network'],
            artifact_path="model",
            registered_model_name="hydatis-katib-qlearning-optimizer"
        )
        
        # Calculate Katib objective metrics
        
        # Primary objective: Placement improvement score
        placement_improvement_score = min(1.0, training_results['improvement_vs_random'] / 0.34)
        
        # Additional metrics for Katib multi-objective optimization
        reward_convergence_rate = training_results['reward_convergence_rate']
        
        # Policy stability (low violation rate indicates stable policy)
        total_violations = training_results['cpu_violations'] + training_results['availability_violations']
        policy_stability_score = max(0, 1.0 - (total_violations / 1000))  # Normalize by episode steps
        
        # Business impact score (alignment with HYDATIS targets)
        cpu_target_alignment = max(0, 1.0 - (training_results['cpu_violations'] / 1000))
        availability_alignment = max(0, 1.0 - (training_results['availability_violations'] / 1000))
        business_impact_score = (cpu_target_alignment + availability_alignment) / 2
        
        # Cluster efficiency improvement (vs baseline)
        baseline_efficiency = 0.72  # Current HYDATIS efficiency
        projected_efficiency = baseline_efficiency * (1 + training_results['improvement_vs_random'])
        cluster_efficiency_improvement = min(1.0, projected_efficiency / 0.91)  # Target 91% efficiency
        
        # Katib metrics output
        katib_metrics = {
            'placement_improvement_score': placement_improvement_score,
            'reward_convergence_rate': reward_convergence_rate,
            'policy_stability_score': policy_stability_score,
            'business_impact_score': business_impact_score,
            'cluster_efficiency_improvement': cluster_efficiency_improvement
        }
        
        # Print Katib metrics (captured by Katib)
        print("📊 Katib Q-Learning Optimization Results:")
        for metric_name, value in katib_metrics.items():
            print(f"   {metric_name}={value:.6f}")
        
        # Save detailed results
        detailed_results = {
            'hyperparameters': hyperparams,
            'reward_weights': reward_weights,
            'katib_metrics': katib_metrics,
            'training_results': {
                'final_average_reward': training_results['final_average_reward'],
                'improvement_vs_random': training_results['improvement_vs_random'],
                'business_score': training_results['business_score'],
                'convergence_rate': training_results['reward_convergence_rate']
            },
            'business_projections': {
                'cpu_target_alignment': cpu_target_alignment,
                'availability_alignment': availability_alignment,
                'projected_efficiency': projected_efficiency,
                'target_achievement': target_achieved
            },
            'mlflow_run_id': run.info.run_id
        }
        
        with open('/app/output/katib_qlearning_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"💾 Katib Q-Learning results saved: Improvement = {training_results['improvement_vs_random']:.2%}")
        
        return placement_improvement_score


if __name__ == "__main__":
    improvement_score = main()
    print(f"🎯 Final Placement Improvement Score: {improvement_score:.6f}")