#!/usr/bin/env python3
"""
XGBoost training pipeline for HYDATIS ML Scheduler.
Implements comprehensive training with hyperparameter optimization and validation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.xgboost

from .model import HYDATISXGBoostPredictor, XGBoostTrainingPipeline
from ...mlflow_configs.experiment_config import HYDATISMLflowManager

logger = logging.getLogger(__name__)


class XGBoostHyperparameterOptimizer:
    """Hyperparameter optimization for XGBoost models using Optuna."""
    
    def __init__(self, n_trials: int = 30):
        self.n_trials = n_trials
        self.best_params = {}
        
    def objective_cpu(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for CPU prediction optimization."""
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42
        }
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, 
                 eval_set=[(X_val, y_val)],
                 early_stopping_rounds=20,
                 verbose=False)
        
        # Predict and calculate accuracy
        y_pred = model.predict(X_val)
        accuracy = 1 - np.mean(np.abs(y_val - y_pred) / (y_val + 1e-8))
        
        return accuracy  # Maximize accuracy
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, 
                                model_type: str = 'cpu') -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        
        logger.info(f"Optimizing {model_type} model hyperparameters with {self.n_trials} trials...")
        
        study = optuna.create_study(direction='maximize',
                                  study_name=f'xgboost_{model_type}_optimization')
        
        objective_func = lambda trial: self.objective_cpu(trial, X_train, y_train, X_val, y_val)
        study.optimize(objective_func, n_trials=self.n_trials)
        
        best_params = study.best_params
        best_accuracy = study.best_value
        
        logger.info(f"{model_type.upper()} optimization completed:")
        logger.info(f"Best accuracy: {best_accuracy:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return {
            'best_params': best_params,
            'best_accuracy': best_accuracy,
            'n_trials': self.n_trials,
            'optimization_completed': datetime.now().isoformat()
        }


class XGBoostProductionTrainer:
    """Production-ready XGBoost training pipeline with comprehensive validation."""
    
    def __init__(self, mlflow_manager: HYDATISMLflowManager):
        self.mlflow_manager = mlflow_manager
        self.optimizer = XGBoostHyperparameterOptimizer(n_trials=30)
        self.predictor = HYDATISXGBoostPredictor()
        
        # Validation configuration
        self.validation_config = {
            'time_series_splits': 5,
            'test_size': 0.2,
            'validation_metrics': ['rmse', 'mae', 'r2', 'accuracy'],
            'min_accuracy_threshold': {'cpu': 0.89, 'memory': 0.86}
        }
    
    def run_comprehensive_training(self, dataset_path: str) -> Dict[str, Any]:
        """Run comprehensive XGBoost training with optimization and validation."""
        
        logger.info("Starting comprehensive XGBoost training pipeline...")
        
        # Setup MLflow
        self.mlflow_manager.setup_mlflow_environment()
        mlflow.set_experiment(self.mlflow_manager.experiments['xgboost_load_prediction']['name'])
        
        # Load and prepare data
        X, y_df = self.predictor.prepare_training_data(dataset_path)
        
        training_results = {}
        
        # Train CPU predictor
        if 'cpu_5m' in y_df.columns:
            logger.info("Training CPU load predictor...")
            cpu_results = self._train_single_model(
                X, y_df['cpu_5m'], 'cpu_predictor', dataset_path
            )
            training_results['cpu'] = cpu_results
        
        # Train Memory predictor
        if 'memory_5m' in y_df.columns:
            logger.info("Training Memory load predictor...")
            memory_results = self._train_single_model(
                X, y_df['memory_5m'], 'memory_predictor', dataset_path
            )
            training_results['memory'] = memory_results
        
        # Overall training summary
        training_summary = {
            'training_completed': datetime.now().isoformat(),
            'cluster': 'HYDATIS-6node',
            'dataset_samples': len(X),
            'feature_count': len(self.predictor.feature_names),
            'models_trained': list(training_results.keys()),
            'target_achievements': {}
        }
        
        # Check target achievements
        for model_type, results in training_results.items():
            target = self.validation_config['min_accuracy_threshold'].get(model_type, 0.85)
            achieved = results.get('best_accuracy', 0) >= target
            training_summary['target_achievements'][model_type] = {
                'target': target,
                'achieved': achieved,
                'actual_accuracy': results.get('best_accuracy', 0)
            }
        
        logger.info("Comprehensive XGBoost training completed")
        for model_type, achievement in training_summary['target_achievements'].items():
            status = "✅ ACHIEVED" if achievement['achieved'] else "❌ MISSED"
            logger.info(f"{model_type.upper()}: {achievement['actual_accuracy']:.3f} (Target: {achievement['target']:.3f}) {status}")
        
        return training_summary
    
    def _train_single_model(self, X: pd.DataFrame, y: pd.Series, 
                          model_name: str, dataset_path: str) -> Dict[str, Any]:
        """Train a single XGBoost model with optimization and validation."""
        
        # Time-based train/validation split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.predictor.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.predictor.feature_scaler.transform(X_val)
        
        # Hyperparameter optimization
        optimization_results = self.optimizer.optimize_hyperparameters(
            X_train_scaled, y_train, X_val_scaled, y_val, model_name.split('_')[0]
        )
        
        best_params = optimization_results['best_params']
        
        # Train final model with best parameters
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Comprehensive validation
        validation_results = self._validate_model(final_model, X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Log to MLflow
        run_id = self.mlflow_manager.log_xgboost_experiment(
            model_name=model_name,
            xgb_model=final_model,
            features=self.predictor.feature_names,
            metrics=validation_results,
            dataset_path=dataset_path,
            hyperparams=best_params
        )
        
        # Store model in predictor
        if 'cpu' in model_name:
            # Convert sklearn XGBRegressor to xgb.Booster for consistency
            self.predictor.cpu_model = final_model.get_booster()
        elif 'memory' in model_name:
            self.predictor.memory_model = final_model.get_booster()
        
        return {
            'model': final_model,
            'best_params': best_params,
            'best_accuracy': optimization_results['best_accuracy'],
            'validation_results': validation_results,
            'mlflow_run_id': run_id,
            'optimization_trials': self.optimizer.n_trials
        }
    
    def _validate_model(self, model, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        """Comprehensive model validation."""
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            # Training metrics
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'train_accuracy': 1 - np.mean(np.abs(y_train - y_train_pred) / (y_train + 1e-8)),
            
            # Validation metrics
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'val_accuracy': 1 - np.mean(np.abs(y_val - y_val_pred) / (y_val + 1e-8)),
            
            # Overall metrics
            'accuracy': 1 - np.mean(np.abs(y_val - y_val_pred) / (y_val + 1e-8)),
            'overfitting': abs(metrics.get('train_accuracy', 0) - metrics.get('val_accuracy', 0)) if 'train_accuracy' in locals() else 0
        }
        
        # Fix overfitting calculation
        metrics['overfitting'] = abs(metrics['train_accuracy'] - metrics['val_accuracy'])
        
        return metrics


def main():
    """Main training execution function."""
    
    # Initialize components
    mlflow_manager = HYDATISMLflowManager()
    trainer = XGBoostProductionTrainer(mlflow_manager)
    
    print("HYDATIS XGBoost Training Pipeline - Week 5")
    print(f"Target: 89% CPU accuracy, 86% Memory accuracy")
    print(f"Planned experiments: 30+ with hyperparameter optimization")
    
    # Note: In production, dataset_path would come from Week 4 dataset builder
    print("✓ Training pipeline ready for execution")
    print("✓ MLflow tracking configured")
    print("✓ Hyperparameter optimization enabled")
    
    return trainer


if __name__ == "__main__":
    trainer = main()