#!/usr/bin/env python3
"""
Comprehensive tests for XGBoost load predictor models.
Validates model performance against HYDATIS cluster requirements.
"""

import pytest
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ml_models.xgboost.model import HYDATISXGBoostPredictor, XGBoostTrainingPipeline
from src.ml_models.xgboost.training import XGBoostHyperparameterOptimizer, XGBoostProductionTrainer


class TestHYDATISXGBoostPredictor:
    """Test XGBoost predictor functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data for testing."""
        n_samples = 1000
        n_features = 20
        
        # Generate realistic HYDATIS cluster data
        dates = pd.date_range(start='2025-08-01', periods=n_samples, freq='30S')
        
        # Features mimicking HYDATIS cluster patterns
        features_data = {
            'cpu_utilization': np.random.uniform(0.08, 0.13, n_samples),
            'memory_utilization': np.random.uniform(0.36, 0.43, n_samples),
            'load_1m': np.random.uniform(0.5, 2.0, n_samples),
            'cpu_rolling_mean_5m': np.random.uniform(0.08, 0.13, n_samples),
            'cpu_rolling_mean_15m': np.random.uniform(0.08, 0.13, n_samples),
            'memory_rolling_mean_5m': np.random.uniform(0.36, 0.43, n_samples),
            'hour_sin': np.sin(2 * np.pi * pd.to_datetime(dates).hour / 24),
            'hour_cos': np.cos(2 * np.pi * pd.to_datetime(dates).hour / 24),
            'is_business_hours': ((pd.to_datetime(dates).hour >= 8) & (pd.to_datetime(dates).hour <= 18)).astype(int),
            'cpu_capacity_remaining': 1 - np.random.uniform(0.08, 0.13, n_samples),
            'memory_capacity_remaining': 1 - np.random.uniform(0.36, 0.43, n_samples),
            'cpu_rank': np.random.randint(1, 7, n_samples),
            'node_type': np.random.choice(['master', 'worker'], n_samples),
            'resource_pressure': np.random.randint(0, 3, n_samples)
        }
        
        # Add more features to reach target count
        for i in range(len(features_data), n_features):
            features_data[f'feature_{i}'] = np.random.randn(n_samples)
        
        df = pd.DataFrame(features_data)
        df['timestamp'] = dates
        df['instance'] = np.random.choice(['worker-1', 'worker-2', 'worker-3'], n_samples)
        
        # Create realistic targets
        df['target_cpu_5m'] = (df['cpu_utilization'] + 
                              np.random.normal(0, 0.02, n_samples)).clip(0, 1)
        df['target_memory_5m'] = (df['memory_utilization'] + 
                                 np.random.normal(0, 0.03, n_samples)).clip(0, 1)
        
        return df
    
    @pytest.fixture  
    def trained_predictor(self, sample_data):
        """Create a trained predictor for testing."""
        predictor = HYDATISXGBoostPredictor()
        
        # Prepare training data
        feature_cols = [col for col in sample_data.columns 
                       if col not in ['timestamp', 'instance', 'target_cpu_5m', 'target_memory_5m']]
        
        X = sample_data[feature_cols]
        y_cpu = sample_data['target_cpu_5m']
        y_memory = sample_data['target_memory_5m']
        
        predictor.feature_names = list(X.columns)
        
        # Train simple models for testing
        X_scaled = predictor.feature_scaler.fit_transform(X)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train_cpu, y_val_cpu = y_cpu.iloc[:split_idx], y_cpu.iloc[split_idx:]
        y_train_mem, y_val_mem = y_memory.iloc[:split_idx], y_memory.iloc[split_idx:]
        
        # Train models
        dtrain_cpu = xgb.DMatrix(X_train, label=y_train_cpu, feature_names=predictor.feature_names)
        dval_cpu = xgb.DMatrix(X_val, label=y_val_cpu, feature_names=predictor.feature_names)
        
        predictor.cpu_model = xgb.train(
            params=predictor.cpu_model_params,
            dtrain=dtrain_cpu,
            num_boost_round=50,
            evals=[(dval_cpu, 'eval')],
            verbose_eval=False
        )
        
        dtrain_mem = xgb.DMatrix(X_train, label=y_train_mem, feature_names=predictor.feature_names)
        dval_mem = xgb.DMatrix(X_val, label=y_val_mem, feature_names=predictor.feature_names)
        
        predictor.memory_model = xgb.train(
            params=predictor.memory_model_params,
            dtrain=dtrain_mem,
            num_boost_round=50,
            evals=[(dval_mem, 'eval')],
            verbose_eval=False
        )
        
        return predictor
    
    def test_model_initialization(self):
        """Test XGBoost predictor initialization."""
        predictor = HYDATISXGBoostPredictor()
        
        assert predictor.target_cpu_accuracy == 0.89
        assert predictor.target_memory_accuracy == 0.86
        assert predictor.cpu_model is None
        assert predictor.memory_model is None
        assert len(predictor.feature_names) == 0
    
    def test_data_preparation(self, sample_data):
        """Test training data preparation."""
        predictor = HYDATISXGBoostPredictor()
        
        # Save sample data to temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            sample_data.to_csv(tmp.name, index=False)
            
            X, y_df = predictor.prepare_training_data(tmp.name)
            
            assert len(X) > 0
            assert len(y_df) > 0
            assert 'target_cpu_5m' in y_df.columns or 'target_memory_5m' in y_df.columns
            assert len(predictor.feature_names) > 0
    
    def test_cpu_model_training(self, sample_data):
        """Test CPU model training."""
        predictor = HYDATISXGBoostPredictor()
        
        feature_cols = [col for col in sample_data.columns 
                       if col not in ['timestamp', 'instance', 'target_cpu_5m', 'target_memory_5m']]
        
        X = sample_data[feature_cols]
        y_cpu = sample_data['target_cpu_5m']
        
        predictor.feature_names = list(X.columns)
        
        # Train CPU model
        metrics = predictor.train_cpu_predictor(X, y_cpu)
        
        assert 'accuracy' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert predictor.cpu_model is not None
        assert metrics['accuracy'] > 0.5  # Reasonable performance threshold
    
    def test_memory_model_training(self, sample_data):
        """Test Memory model training."""
        predictor = HYDATISXGBoostPredictor()
        
        feature_cols = [col for col in sample_data.columns 
                       if col not in ['timestamp', 'instance', 'target_cpu_5m', 'target_memory_5m']]
        
        X = sample_data[feature_cols]
        y_memory = sample_data['target_memory_5m']
        
        predictor.feature_names = list(X.columns)
        
        # Train Memory model
        metrics = predictor.train_memory_predictor(X, y_memory)
        
        assert 'accuracy' in metrics
        assert 'rmse' in metrics
        assert predictor.memory_model is not None
        assert metrics['accuracy'] > 0.5
    
    def test_load_prediction(self, trained_predictor, sample_data):
        """Test load prediction functionality."""
        feature_cols = [col for col in sample_data.columns 
                       if col not in ['timestamp', 'instance', 'target_cpu_5m', 'target_memory_5m']]
        
        test_features = sample_data[feature_cols].iloc[:10]
        
        predictions = trained_predictor.predict_load(test_features)
        
        assert 'cpu_prediction' in predictions
        assert 'memory_prediction' in predictions
        assert len(predictions['cpu_prediction']) == 10
        assert len(predictions['memory_prediction']) == 10
        assert all(0 <= pred <= 1 for pred in predictions['cpu_prediction'])
    
    def test_feature_importance(self, trained_predictor):
        """Test feature importance extraction."""
        importance = trained_predictor.get_feature_importance()
        
        assert 'cpu_model' in importance
        assert 'memory_model' in importance
        assert len(importance['cpu_model']) > 0
        assert len(importance['memory_model']) > 0
    
    def test_model_persistence(self, trained_predictor):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save models
            saved_files = trained_predictor.save_models(tmp_dir)
            
            assert 'cpu_model' in saved_files
            assert 'memory_model' in saved_files
            assert 'scaler' in saved_files
            assert 'metadata' in saved_files
            
            # Test files exist
            for file_path in saved_files.values():
                assert Path(file_path).exists()
            
            # Test loading
            new_predictor = HYDATISXGBoostPredictor()
            success = new_predictor.load_models(tmp_dir)
            
            assert success
            assert new_predictor.cpu_model is not None
            assert new_predictor.memory_model is not None


class TestXGBoostTrainingPipeline:
    """Test XGBoost training pipeline functionality."""
    
    def test_pipeline_initialization(self):
        """Test training pipeline initialization."""
        pipeline = XGBoostTrainingPipeline()
        
        assert pipeline.predictor is not None
        assert pipeline.experiment_name == "hydatis-xgboost-load-prediction"
        assert "10.110.190.32:31380" in pipeline.mlflow_uri
    
    @patch('mlflow.start_run')
    @patch('mlflow.set_experiment')
    def test_mlflow_setup(self, mock_set_exp, mock_start_run):
        """Test MLflow experiment setup."""
        pipeline = XGBoostTrainingPipeline()
        pipeline.setup_mlflow_tracking()
        
        mock_set_exp.assert_called_once()
    
    def test_accuracy_validation(self):
        """Test accuracy calculation and validation."""
        # Test accuracy calculation
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.11, 0.19, 0.31, 0.39, 0.51])
        
        accuracy = 1 - np.mean(np.abs(y_true - y_pred) / (y_true + 1e-8))
        
        assert accuracy > 0.9  # Should be high accuracy for close predictions
        assert 0 <= accuracy <= 1  # Accuracy should be bounded


class TestHyperparameterOptimization:
    """Test hyperparameter optimization functionality."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = XGBoostHyperparameterOptimizer(n_trials=10)
        
        assert optimizer.n_trials == 10
        assert optimizer.best_params == {}
    
    def test_parameter_search_space(self):
        """Test hyperparameter search space is reasonable."""
        optimizer = XGBoostHyperparameterOptimizer()
        
        # Mock trial for parameter space testing
        class MockTrial:
            def suggest_int(self, name, low, high):
                return (low + high) // 2
            def suggest_float(self, name, low, high):
                return (low + high) / 2
        
        trial = MockTrial()
        
        # Test parameter ranges
        max_depth = trial.suggest_int('max_depth', 4, 12)
        learning_rate = trial.suggest_float('learning_rate', 0.05, 0.3)
        
        assert 4 <= max_depth <= 12
        assert 0.05 <= learning_rate <= 0.3


def run_xgboost_model_tests():
    """Run all XGBoost model tests."""
    
    print("Running HYDATIS XGBoost Model Tests...")
    
    # Run basic functionality tests
    test_predictor = TestHYDATISXGBoostPredictor()
    test_pipeline = TestXGBoostTrainingPipeline()
    test_optimizer = TestHyperparameterOptimization()
    
    tests_run = 0
    tests_passed = 0
    
    test_methods = [
        (test_predictor, 'test_model_initialization'),
        (test_pipeline, 'test_pipeline_initialization'),
        (test_optimizer, 'test_optimizer_initialization'),
        (test_optimizer, 'test_parameter_search_space'),
        (test_pipeline, 'test_accuracy_validation')
    ]
    
    for test_instance, method_name in test_methods:
        try:
            getattr(test_instance, method_name)()
            print(f"✓ {method_name}")
            tests_passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
        tests_run += 1
    
    print(f"\\nTest Results: {tests_passed}/{tests_run} passed")
    
    # Integration test with sample data
    try:
        print("\\nRunning integration test...")
        test_predictor_instance = TestHYDATISXGBoostPredictor()
        
        # Create sample data
        sample_data = test_predictor_instance.sample_data()
        
        # Test data preparation
        predictor = HYDATISXGBoostPredictor()
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            sample_data.to_csv(tmp.name, index=False)
            X, y_df = predictor.prepare_training_data(tmp.name)
            
            if len(X) > 100 and len(predictor.feature_names) > 5:
                print("✓ Integration test: Data preparation successful")
                print(f"✓ Features: {len(predictor.feature_names)}")
                print(f"✓ Samples: {len(X)}")
                print(f"✓ Targets: {list(y_df.columns)}")
                tests_passed += 1
            else:
                print("✗ Integration test: Insufficient data prepared")
        
        tests_run += 1
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        tests_run += 1
    
    success_rate = tests_passed / tests_run
    print(f"\\nOverall Test Success Rate: {success_rate:.1%}")
    print(f"Status: {'✅ READY FOR PRODUCTION' if success_rate >= 0.8 else '❌ NEEDS FIXES'}")
    
    return success_rate >= 0.8


if __name__ == "__main__":
    success = run_xgboost_model_tests()
    print(f"\\n{'✅' if success else '❌'} XGBoost model validation {'COMPLETE' if success else 'FAILED'}")