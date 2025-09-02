#!/usr/bin/env python3
"""
Q-Learning training pipeline for HYDATIS ML Scheduler placement optimization.
Implements comprehensive DQN training with environment simulation and validation.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from .environment import HYDATISClusterEnvironment
from .agent import HYDATISPlacementDQN
from ...mlflow_configs.experiment_config import HYDATISMLflowManager

logger = logging.getLogger(__name__)


class HYDATISQLearningTrainer:
    """Production Q-Learning trainer for HYDATIS cluster placement optimization."""
    
    def __init__(self, mlflow_manager: HYDATISMLflowManager):
        self.mlflow_manager = mlflow_manager
        
        self.cluster_config = {
            'nodes': 6,
            'masters': 3,
            'workers': 3,
            'cpu_cores_per_node': 8,
            'memory_gb_per_node': 16,
            'worker_nodes': ['worker-1', 'worker-2', 'worker-3'],
            'master_nodes': ['master-1', 'master-2', 'master-3']
        }
        
        self.training_config = {
            'max_episodes': 2000,
            'evaluation_freq': 100,
            'save_freq': 500,
            'target_improvement': 0.34,
            'convergence_threshold': 0.01,
            'convergence_window': 100,
            'early_stopping_patience': 300
        }
        
        self.environment = HYDATISClusterEnvironment()
        self.dqn_agent = HYDATISPlacementDQN(self.cluster_config)
        
        self.training_metrics = {
            'episode_rewards': [],
            'evaluation_results': [],
            'improvement_over_time': [],
            'convergence_metrics': [],
            'training_timeline': []
        }
    
    def train_placement_optimizer(self, historical_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive DQN training for placement optimization."""
        
        logger.info("Starting HYDATIS Q-Learning placement optimization training...")
        
        self.mlflow_manager.setup_mlflow_environment()
        mlflow.set_experiment(self.mlflow_manager.experiments['qlearning_placement']['name'])
        
        with mlflow.start_run(run_name=f"hydatis_dqn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            mlflow.log_params({
                'cluster_nodes': self.cluster_config['nodes'],
                'max_episodes': self.training_config['max_episodes'],
                'target_improvement': self.training_config['target_improvement'],
                'learning_rate': self.dqn_agent.agent.learning_rate,
                'gamma': self.dqn_agent.agent.gamma,
                'epsilon_decay': self.dqn_agent.agent.epsilon_decay,
                'batch_size': self.dqn_agent.agent.batch_size
            })
            
            if historical_data_path and Path(historical_data_path).exists():
                historical_data = pd.read_csv(historical_data_path)
                self.environment = HYDATISClusterEnvironment(historical_data)
                logger.info(f"Loaded historical data: {len(historical_data)} samples")
            
            training_start = datetime.now()
            best_improvement = 0.0
            episodes_without_improvement = 0
            
            progress_bar = tqdm(range(self.training_config['max_episodes']), 
                              desc="Training DQN Agent")
            
            for episode in progress_bar:
                episode_metrics = self.dqn_agent.train_episode(self.environment)
                
                self.training_metrics['episode_rewards'].append(episode_metrics['total_reward'])
                
                progress_bar.set_postfix({
                    'Reward': f"{episode_metrics['total_reward']:.2f}",
                    'Epsilon': f"{episode_metrics['epsilon']:.3f}",
                    'Memory': f"{episode_metrics['memory_usage']}"
                })
                
                if (episode + 1) % self.training_config['evaluation_freq'] == 0:
                    evaluation_results = self._evaluate_performance(episode + 1)
                    self.training_metrics['evaluation_results'].append(evaluation_results)
                    
                    current_improvement = evaluation_results['improvement_over_random']
                    
                    if current_improvement > best_improvement:
                        best_improvement = current_improvement
                        episodes_without_improvement = 0
                        
                        mlflow.log_metric("best_improvement", best_improvement, step=episode)
                    else:
                        episodes_without_improvement += self.training_config['evaluation_freq']
                    
                    mlflow.log_metrics({
                        'episode_reward': episode_metrics['total_reward'],
                        'improvement_over_random': current_improvement,
                        'epsilon': episode_metrics['epsilon'],
                        'average_loss': episode_metrics['average_loss'],
                        'memory_usage': episode_metrics['memory_usage']
                    }, step=episode)
                    
                    logger.info(f"Episode {episode + 1}: Improvement {current_improvement:.3f} "
                               f"(Target: {self.training_config['target_improvement']:.3f})")
                    
                    if current_improvement >= self.training_config['target_improvement']:
                        logger.info(f"Target improvement achieved at episode {episode + 1}")
                        break
                
                if (episode + 1) % self.training_config['save_freq'] == 0:
                    self._save_training_checkpoint(episode + 1)
                
                if episodes_without_improvement >= self.training_config['early_stopping_patience']:
                    logger.info(f"Early stopping at episode {episode + 1} (no improvement for {episodes_without_improvement} episodes)")
                    break
            
            training_duration = (datetime.now() - training_start).total_seconds()
            
            final_evaluation = self._evaluate_performance(episode + 1, comprehensive=True)
            
            training_summary = {
                'training_completed': datetime.now().isoformat(),
                'total_episodes': episode + 1,
                'training_duration_minutes': training_duration / 60,
                'best_improvement': best_improvement,
                'target_achieved': best_improvement >= self.training_config['target_improvement'],
                'final_evaluation': final_evaluation,
                'convergence_analysis': self._analyze_convergence(),
                'agent_metrics': self.dqn_agent.agent.get_training_metrics()
            }
            
            mlflow.log_params(training_summary)
            
            model_files = self._save_final_models()
            
            for model_type, file_path in model_files.items():
                mlflow.log_artifact(file_path, f"models/{model_type}")
            
            logger.info("Q-Learning training completed")
            logger.info(f"Best improvement: {best_improvement:.3f} (Target: {self.training_config['target_improvement']:.3f})")
            logger.info(f"Status: {'✅ TARGET ACHIEVED' if training_summary['target_achieved'] else '❌ TARGET MISSED'}")
            
            return training_summary
    
    def _evaluate_performance(self, episode: int, comprehensive: bool = False) -> Dict[str, Any]:
        """Evaluate DQN performance against baselines."""
        
        num_eval_episodes = 20 if comprehensive else 10
        
        evaluation_results = self.dqn_agent.evaluate_performance(
            self.environment, 
            num_episodes=num_eval_episodes
        )
        
        evaluation_results['episode'] = episode
        evaluation_results['comprehensive'] = comprehensive
        
        return evaluation_results
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze training convergence patterns."""
        
        if len(self.training_metrics['episode_rewards']) < self.training_config['convergence_window']:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        recent_rewards = self.training_metrics['episode_rewards'][-self.training_config['convergence_window']:]
        
        reward_variance = np.var(recent_rewards)
        reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        
        converged = (reward_variance < self.training_config['convergence_threshold'] and 
                    abs(reward_trend) < 0.001)
        
        convergence_analysis = {
            'converged': converged,
            'reward_variance': reward_variance,
            'reward_trend': reward_trend,
            'convergence_threshold': self.training_config['convergence_threshold'],
            'analysis_window': self.training_config['convergence_window'],
            'average_reward_last_100': np.mean(recent_rewards)
        }
        
        return convergence_analysis
    
    def _save_training_checkpoint(self, episode: int):
        """Save training checkpoint."""
        
        checkpoint_dir = Path("/data/ml_scheduler_longhorn/models/qlearning/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'episode': episode,
            'agent_state': self.dqn_agent.agent.get_training_metrics(),
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = checkpoint_dir / f"training_checkpoint_episode_{episode}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
    
    def _save_final_models(self) -> Dict[str, str]:
        """Save final trained models."""
        
        model_dir = "/data/ml_scheduler_longhorn/models/qlearning"
        
        saved_files = self.dqn_agent.agent.save_agent(
            model_dir, 
            experiment_name="hydatis_placement_optimizer"
        )
        
        training_metadata_path = Path(model_dir) / f"training_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(training_metadata_path, 'w') as f:
            json.dump({
                'training_config': self.training_config,
                'cluster_config': self.cluster_config,
                'training_metrics': self.training_metrics,
                'final_performance': self.training_metrics['evaluation_results'][-1] if self.training_metrics['evaluation_results'] else {}
            }, f, indent=2, default=str)
        
        saved_files['training_metadata'] = str(training_metadata_path)
        
        return saved_files
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        
        if not self.training_metrics['evaluation_results']:
            return {'error': 'No evaluation results available'}
        
        latest_eval = self.training_metrics['evaluation_results'][-1]
        
        report = {
            'training_summary': {
                'total_episodes': len(self.training_metrics['episode_rewards']),
                'final_improvement': latest_eval['improvement_over_random'],
                'target_achieved': latest_eval['target_achievement'],
                'convergence_status': self._analyze_convergence()['converged']
            },
            'performance_analysis': {
                'best_episode_reward': max(self.training_metrics['episode_rewards']) if self.training_metrics['episode_rewards'] else 0,
                'average_reward_final_100': np.mean(self.training_metrics['episode_rewards'][-100:]) if len(self.training_metrics['episode_rewards']) >= 100 else 0,
                'improvement_progression': [eval_result['improvement_over_random'] for eval_result in self.training_metrics['evaluation_results']],
                'stability_score': 1.0 - latest_eval.get('performance_stability', 1.0)
            },
            'agent_insights': self.dqn_agent.agent.get_training_metrics(),
            'recommendations': self._generate_recommendations(latest_eval),
            'report_timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate training recommendations based on results."""
        
        recommendations = []
        
        improvement = evaluation_results['improvement_over_random']
        target = self.training_config['target_improvement']
        
        if improvement < target:
            gap = target - improvement
            if gap > 0.2:
                recommendations.append("Consider increasing training episodes or adjusting reward function")
                recommendations.append("Evaluate hyperparameter tuning for learning rate and network architecture")
            elif gap > 0.1:
                recommendations.append("Fine-tune epsilon decay schedule for better exploration-exploitation balance")
            else:
                recommendations.append("Minor adjustments to training duration may achieve target")
        else:
            recommendations.append("Target improvement achieved - model ready for production deployment")
            recommendations.append("Consider implementing continuous learning pipeline for production updates")
        
        stability = evaluation_results.get('performance_stability', 1.0)
        if stability > 0.3:
            recommendations.append("High performance variance detected - consider stabilizing training process")
        
        convergence_status = self._analyze_convergence()
        if not convergence_status['converged']:
            recommendations.append("Training may benefit from additional episodes to reach convergence")
        
        return recommendations


class QLearningProductionPipeline:
    """Complete Q-Learning production training pipeline."""
    
    def __init__(self):
        self.mlflow_manager = HYDATISMLflowManager()
        self.trainer = HYDATISQLearningTrainer(self.mlflow_manager)
        
        self.pipeline_config = {
            'training_phases': ['exploration', 'exploitation', 'fine_tuning'],
            'phase_episodes': [800, 800, 400],
            'phase_learning_rates': [1e-4, 5e-5, 1e-5],
            'validation_splits': 5
        }
    
    def run_production_training(self, 
                               historical_data_path: Optional[str] = None,
                               output_dir: str = "/data/ml_scheduler_longhorn/models/qlearning") -> Dict[str, Any]:
        """Run complete production training pipeline."""
        
        logger.info("Starting HYDATIS Q-Learning production training pipeline...")
        
        pipeline_start = datetime.now()
        
        training_results = self.trainer.train_placement_optimizer(historical_data_path)
        
        validation_results = self._run_cross_validation()
        
        production_readiness = self._assess_production_readiness(training_results, validation_results)
        
        training_report = self.trainer.generate_training_report()
        
        pipeline_summary = {
            'pipeline_completed': datetime.now().isoformat(),
            'pipeline_duration_minutes': (datetime.now() - pipeline_start).total_seconds() / 60,
            'training_results': training_results,
            'validation_results': validation_results,
            'production_readiness': production_readiness,
            'training_report': training_report,
            'model_artifacts': output_dir,
            'ready_for_deployment': production_readiness['deployment_ready']
        }
        
        logger.info("Q-Learning production training completed")
        logger.info(f"Final improvement: {training_results['final_evaluation']['improvement_over_random']:.3f}")
        logger.info(f"Production ready: {'✅ YES' if production_readiness['deployment_ready'] else '❌ NO'}")
        
        return pipeline_summary
    
    def _run_cross_validation(self) -> Dict[str, Any]:
        """Run cross-validation to assess model robustness."""
        
        logger.info("Running cross-validation for Q-Learning model...")
        
        cv_results = []
        
        for fold in range(self.pipeline_config['validation_splits']):
            logger.info(f"Cross-validation fold {fold + 1}/{self.pipeline_config['validation_splits']}")
            
            fold_environment = HYDATISClusterEnvironment()
            fold_agent = HYDATISPlacementDQN(self.cluster_config)
            
            fold_rewards = []
            for episode in range(200):
                episode_metrics = fold_agent.train_episode(fold_environment)
                fold_rewards.append(episode_metrics['total_reward'])
            
            fold_evaluation = fold_agent.evaluate_performance(fold_environment, num_episodes=20)
            
            cv_results.append({
                'fold': fold + 1,
                'average_reward': np.mean(fold_rewards),
                'improvement_over_random': fold_evaluation['improvement_over_random'],
                'performance_stability': fold_evaluation['performance_stability']
            })
        
        cv_summary = {
            'cross_validation_folds': len(cv_results),
            'mean_improvement': np.mean([result['improvement_over_random'] for result in cv_results]),
            'std_improvement': np.std([result['improvement_over_random'] for result in cv_results]),
            'mean_stability': np.mean([result['performance_stability'] for result in cv_results]),
            'cv_results': cv_results,
            'model_robustness': 'high' if np.std([result['improvement_over_random'] for result in cv_results]) < 0.05 else 'medium'
        }
        
        return cv_summary
    
    def _assess_production_readiness(self, training_results: Dict[str, Any], 
                                   validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if model is ready for production deployment."""
        
        readiness_criteria = {
            'target_improvement_achieved': training_results['target_achieved'],
            'cross_validation_stable': validation_results['std_improvement'] < 0.1,
            'convergence_achieved': training_results.get('convergence_analysis', {}).get('converged', False),
            'performance_consistent': validation_results['mean_stability'] < 0.3,
            'minimum_episodes_completed': training_results['total_episodes'] >= 1000
        }
        
        readiness_score = sum(readiness_criteria.values()) / len(readiness_criteria)
        deployment_ready = readiness_score >= 0.8
        
        assessment = {
            'deployment_ready': deployment_ready,
            'readiness_score': readiness_score,
            'criteria_met': readiness_criteria,
            'recommendations': self._generate_deployment_recommendations(readiness_criteria, training_results),
            'risk_assessment': 'low' if readiness_score >= 0.9 else 'medium' if readiness_score >= 0.7 else 'high'
        }
        
        return assessment
    
    def _generate_deployment_recommendations(self, criteria: Dict[str, bool], 
                                           training_results: Dict[str, Any]) -> List[str]:
        """Generate deployment recommendations."""
        
        recommendations = []
        
        if not criteria['target_improvement_achieved']:
            recommendations.append("Extend training or adjust reward function to achieve target improvement")
        
        if not criteria['cross_validation_stable']:
            recommendations.append("Improve model stability through regularization or architecture changes")
        
        if not criteria['convergence_achieved']:
            recommendations.append("Allow more training episodes for proper convergence")
        
        if not criteria['performance_consistent']:
            recommendations.append("Reduce performance variance through hyperparameter tuning")
        
        if criteria['target_improvement_achieved'] and criteria['cross_validation_stable']:
            recommendations.append("Model ready for staging environment deployment")
            recommendations.append("Implement A/B testing with current scheduler for production validation")
        
        return recommendations


def main():
    """Main Q-Learning training execution."""
    
    print("HYDATIS Q-Learning Training Pipeline - Week 6")
    print("Target: +34% improvement over random placement")
    print("Training DQN agent for optimal pod placement...")
    
    pipeline = QLearningProductionPipeline()
    
    print("Training Configuration:")
    print(f"  Max Episodes: {pipeline.training_config['max_episodes']}")
    print(f"  Target Improvement: {pipeline.training_config['target_improvement']:.1%}")
    print(f"  Cluster Nodes: {pipeline.cluster_config['nodes']}")
    print(f"  Worker Nodes: {len(pipeline.cluster_config['worker_nodes'])}")
    
    results = pipeline.run_production_training()
    
    print(f"Training Status: {'✅ SUCCESS' if results['ready_for_deployment'] else '❌ NEEDS WORK'}")
    print(f"Final Improvement: {results['training_results']['final_evaluation']['improvement_over_random']:.3f}")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()