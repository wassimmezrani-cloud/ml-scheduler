#!/usr/bin/env python3
"""
MLflow experiment configuration for HYDATIS ML Scheduler.
Sets up experiment tracking, model registry, and automated logging.
"""

import mlflow
import mlflow.xgboost
import mlflow.sklearn
import mlflow.pytorch
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


class HYDATISMLflowManager:
    """MLflow experiment management for HYDATIS ML scheduler project."""
    
    def __init__(self, tracking_uri: str = "http://10.110.190.32:31380"):
        self.tracking_uri = tracking_uri
        self.experiments = {
            'xgboost_load_prediction': {
                'name': 'hydatis-xgboost-load-prediction',
                'description': 'XGBoost models for CPU and Memory load prediction (Target: 89% CPU, 86% Memory accuracy)',
                'tags': {'team': 'hydatis-mlops', 'model_type': 'regression', 'week': '5'}
            },
            'qlearning_placement': {
                'name': 'hydatis-qlearning-placement',
                'description': 'Q-Learning reinforcement learning for optimal pod placement (Target: +34% improvement)',
                'tags': {'team': 'hydatis-mlops', 'model_type': 'reinforcement_learning', 'week': '6'}
            },
            'isolation_forest_anomaly': {
                'name': 'hydatis-isolation-forest-anomaly',
                'description': 'Isolation Forest for node anomaly detection (Target: 94% precision, ≤8% false positives)',
                'tags': {'team': 'hydatis-mlops', 'model_type': 'anomaly_detection', 'week': '7'}
            }
        }
        
        self.model_registry_prefix = "hydatis-ml-scheduler"
        
    def setup_mlflow_environment(self):
        """Initialize MLflow environment and experiments."""
        
        logger.info(f"Setting up MLflow environment: {self.tracking_uri}")
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create experiments
        for exp_key, exp_config in self.experiments.items():
            try:
                experiment_id = mlflow.create_experiment(
                    name=exp_config['name'],
                    tags=exp_config['tags']
                )
                logger.info(f"Created experiment: {exp_config['name']} (ID: {experiment_id})")
                
            except mlflow.exceptions.MlflowException as e:
                if "already exists" in str(e):
                    experiment = mlflow.get_experiment_by_name(exp_config['name'])
                    logger.info(f"Using existing experiment: {exp_config['name']} (ID: {experiment.experiment_id})")
                else:
                    logger.error(f"Failed to create experiment {exp_config['name']}: {e}")
    
    def log_xgboost_experiment(self, model_name: str, xgb_model, 
                             features: List[str], metrics: Dict[str, float],
                             dataset_path: str, hyperparams: Dict) -> str:
        """Log XGBoost experiment with comprehensive tracking."""
        
        mlflow.set_experiment(self.experiments['xgboost_load_prediction']['name'])
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            # Log hyperparameters
            mlflow.log_params(hyperparams)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("cluster", "HYDATIS-6node")
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_param("feature_count", len(features))
            
            # Log performance metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log target achievement
            if model_name == 'cpu_predictor':
                target_accuracy = 0.89
                mlflow.log_metric("target_accuracy", target_accuracy)
                mlflow.log_metric("target_achieved", int(metrics.get('accuracy', 0) >= target_accuracy))
                
            elif model_name == 'memory_predictor':
                target_accuracy = 0.86
                mlflow.log_metric("target_accuracy", target_accuracy)
                mlflow.log_metric("target_achieved", int(metrics.get('accuracy', 0) >= target_accuracy))
            
            # Log model
            mlflow.xgboost.log_model(
                xgb_model=xgb_model,
                artifact_path="model",
                registered_model_name=f"{self.model_registry_prefix}-{model_name}"
            )
            
            # Log feature importance
            if hasattr(xgb_model, 'get_score'):
                importance = xgb_model.get_score(importance_type='weight')
                mlflow.log_dict(importance, "feature_importance.json")
            
            # Log feature list
            mlflow.log_dict({"features": features}, "feature_names.json")
            
            run_id = run.info.run_id
            logger.info(f"XGBoost experiment logged: {run_id}")
            
            return run_id
    
    def log_model_comparison(self, model_results: Dict[str, Dict]) -> str:
        """Log model comparison across different configurations."""
        
        mlflow.set_experiment(self.experiments['xgboost_load_prediction']['name'])
        
        with mlflow.start_run(run_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            # Log comparison metrics
            for model_name, results in model_results.items():
                for metric_name, value in results.get('metrics', {}).items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", value)
            
            # Log best model identification
            best_cpu_model = max(model_results.items(), 
                               key=lambda x: x[1].get('metrics', {}).get('accuracy', 0))
            
            mlflow.log_param("best_cpu_model", best_cpu_model[0])
            mlflow.log_metric("best_cpu_accuracy", best_cpu_model[1].get('metrics', {}).get('accuracy', 0))
            
            # Log summary
            summary = {
                'comparison_timestamp': datetime.now().isoformat(),
                'models_compared': list(model_results.keys()),
                'best_model': best_cpu_model[0],
                'target_achievements': {}
            }
            
            for model_name, results in model_results.items():
                accuracy = results.get('metrics', {}).get('accuracy', 0)
                target = 0.89 if 'cpu' in model_name else 0.86
                summary['target_achievements'][model_name] = accuracy >= target
            
            mlflow.log_dict(summary, "comparison_summary.json")
            
            return run.info.run_id
    
    def register_production_model(self, run_id: str, model_name: str, 
                                stage: str = "Staging") -> str:
        """Register model for production deployment."""
        
        model_uri = f"runs:/{run_id}/model"
        registered_name = f"{self.model_registry_prefix}-{model_name}"
        
        # Register model version
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_name
        )
        
        # Transition to specified stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=registered_name,
            version=model_version.version,
            stage=stage
        )
        
        logger.info(f"Model registered: {registered_name} v{model_version.version} -> {stage}")
        
        return f"{registered_name}:{model_version.version}"


def setup_hydatis_mlflow():
    """Setup MLflow for HYDATIS ML scheduler project."""
    
    manager = HYDATISMLflowManager()
    manager.setup_mlflow_environment()
    
    print("✓ HYDATIS MLflow Environment Setup Complete")
    print(f"✓ Tracking URI: {manager.tracking_uri}")
    print(f"✓ Experiments created: {len(manager.experiments)}")
    print("✓ Ready for Week 5 XGBoost training with 30+ experiments")
    
    return manager


if __name__ == "__main__":
    manager = setup_hydatis_mlflow()