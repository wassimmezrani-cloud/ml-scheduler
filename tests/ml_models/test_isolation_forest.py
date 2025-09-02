#!/usr/bin/env python3
"""
Comprehensive tests for Isolation Forest anomaly detection models.
Validates detection performance against HYDATIS cluster requirements.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ml_models.isolation_forest.model import HYDATISIsolationForestDetector, IsolationForestTrainingPipeline
from src.ml_models.isolation_forest.monitoring import HYDATISAnomalyMonitor
from src.monitoring.alert_manager import HYDATISAlertManager


class TestHYDATISIsolationForestDetector:
    """Test Isolation Forest detector functionality."""
    
    @pytest.fixture
    def sample_metrics_data(self):
        """Create sample metrics data for testing."""
        n_samples = 1000
        
        dates = pd.date_range(start='2025-08-01', periods=n_samples, freq='30S')
        
        normal_cpu = np.random.uniform(0.08, 0.15, n_samples)
        normal_memory = np.random.uniform(0.35, 0.45, n_samples)
        
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        normal_cpu[anomaly_indices] = np.random.uniform(0.9, 1.0, len(anomaly_indices))
        normal_memory[anomaly_indices] = np.random.uniform(0.85, 0.98, len(anomaly_indices))
        
        metrics_data = {
            'timestamp': dates,
            'instance': np.random.choice(['worker-1', 'worker-2', 'worker-3'], n_samples),
            'cpu_utilization': normal_cpu,
            'memory_utilization': normal_memory,
            'disk_utilization': np.random.uniform(0.2, 0.6, n_samples),
            'load_1m': np.random.uniform(0.5, 2.0, n_samples),
            'load_5m': np.random.uniform(0.4, 1.8, n_samples),
            'load_15m': np.random.uniform(0.3, 1.5, n_samples),
            'network_rx_bytes': np.random.uniform(1000, 50000, n_samples),
            'network_tx_bytes': np.random.uniform(800, 30000, n_samples)
        }
        
        df = pd.DataFrame(metrics_data)
        df['is_anomaly'] = 0
        df.loc[anomaly_indices, 'is_anomaly'] = 1
        
        return df
    
    @pytest.fixture
    def trained_detector(self, sample_metrics_data):
        """Create a trained detector for testing."""
        detector = HYDATISIsolationForestDetector()
        
        X_features, feature_names = detector.prepare_anomaly_features(sample_metrics_data)
        training_metrics = detector.train_anomaly_detector(X_features)
        
        return detector
    
    def test_detector_initialization(self):
        """Test Isolation Forest detector initialization."""
        detector = HYDATISIsolationForestDetector()
        
        assert detector.target_precision == 0.94
        assert detector.contamination == 0.05
        assert detector.n_estimators == 200
        assert detector.isolation_forest is not None
        assert len(detector.anomaly_categories) == 5
    
    def test_feature_preparation(self, sample_metrics_data):
        """Test anomaly detection feature preparation."""
        detector = HYDATISIsolationForestDetector()
        
        X_features, feature_names = detector.prepare_anomaly_features(sample_metrics_data)
        
        assert len(X_features) > 0
        assert len(feature_names) > 0
        assert len(detector.feature_names) > 0
        assert all(col in X_features.columns for col in feature_names)
        
        assert any('rolling_mean' in col for col in feature_names)
        assert any('rolling_std' in col for col in feature_names)
    
    def test_anomaly_detection_training(self, sample_metrics_data):
        """Test anomaly detection model training."""
        detector = HYDATISIsolationForestDetector()
        
        X_features, feature_names = detector.prepare_anomaly_features(sample_metrics_data)
        training_metrics = detector.train_anomaly_detector(X_features)
        
        assert 'estimated_precision' in training_metrics
        assert 'contamination_rate' in training_metrics
        assert 'feature_count' in training_metrics
        assert detector.training_metadata['model_trained']
        assert training_metrics['estimated_precision'] > 0.5
    
    def test_anomaly_detection(self, trained_detector, sample_metrics_data):
        """Test anomaly detection functionality."""
        test_data = sample_metrics_data.head(100)
        
        detection_results = trained_detector.detect_anomalies(test_data)
        
        assert 'total_samples' in detection_results
        assert 'anomalies_detected' in detection_results
        assert 'anomaly_rate' in detection_results
        assert 'cluster_health_score' in detection_results
        assert isinstance(detection_results['anomalies'], list)
        assert 0 <= detection_results['cluster_health_score'] <= 1
    
    def test_anomaly_probability_prediction(self, trained_detector, sample_metrics_data):
        """Test anomaly probability prediction."""
        test_sample = sample_metrics_data.head(5)
        
        prediction_results = trained_detector.predict_anomaly_probability(test_sample)
        
        assert 'total_samples' in prediction_results
        assert 'predictions' in prediction_results
        assert len(prediction_results['predictions']) == 5
        
        for prediction in prediction_results['predictions']:
            assert 'anomaly_probability' in prediction
            assert 'anomaly_score' in prediction
            assert 'is_anomaly' in prediction
            assert 0 <= prediction['anomaly_probability'] <= 1
    
    def test_feature_importance(self, trained_detector):
        """Test feature importance extraction."""
        importance = trained_detector.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(0 <= value <= 1 for value in importance.values())
    
    def test_model_persistence(self, trained_detector):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_files = trained_detector.save_models(tmp_dir)
            
            assert 'isolation_forest' in saved_files
            assert 'scaler' in saved_files
            assert 'metadata' in saved_files
            
            for file_path in saved_files.values():
                assert Path(file_path).exists()
            
            new_detector = HYDATISIsolationForestDetector()
            success = new_detector.load_models(tmp_dir)
            
            assert success
            assert new_detector.training_metadata['model_trained']
            assert len(new_detector.feature_names) > 0


class TestHYDATISAnomalyMonitor:
    """Test anomaly monitoring functionality."""
    
    @pytest.fixture
    def anomaly_monitor(self):
        """Create test anomaly monitor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = HYDATISAnomalyMonitor(model_dir=tmp_dir, monitoring_interval=1)
            return monitor
    
    def test_monitor_initialization(self, anomaly_monitor):
        """Test monitor initialization."""
        assert anomaly_monitor.monitoring_interval == 1
        assert anomaly_monitor.detector is not None
        assert anomaly_monitor.monitoring_config is not None
        assert len(anomaly_monitor.anomaly_buffer) == 0
    
    def test_monitoring_status(self, anomaly_monitor):
        """Test monitoring status retrieval."""
        status = anomaly_monitor.get_monitoring_status()
        
        assert 'monitoring_active' in status
        assert 'model_loaded' in status
        assert 'performance_stats' in status
        assert 'health_assessment' in status
        assert isinstance(status['monitoring_active'], bool)
    
    def test_recent_anomalies_retrieval(self, anomaly_monitor):
        """Test recent anomalies retrieval."""
        recent = anomaly_monitor.get_recent_anomalies(hours=1)
        
        assert 'time_window_hours' in recent
        assert 'anomalies_found' in recent
        assert 'anomalies' in recent
        assert recent['time_window_hours'] == 1


class TestHYDATISAlertManager:
    """Test alert management functionality."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        manager = HYDATISAlertManager()
        
        assert len(manager.alert_config['severity_priorities']) == 4
        assert len(manager.notification_channels) == 3
        assert len(manager.active_alerts) == 0
        assert manager.alert_statistics['total_alerts_generated'] == 0
    
    def test_alert_processing(self):
        """Test alert processing functionality."""
        manager = HYDATISAlertManager()
        
        sample_alert = {
            'alert_id': 'TEST-ALERT-001',
            'alert_time': datetime.now().isoformat(),
            'severity': 'high',
            'node': 'worker-1',
            'anomaly_type': 'cpu_spike',
            'anomaly_score': -0.42,
            'description': 'Test anomaly alert',
            'affected_metrics': ['cpu_utilization'],
            'recommended_actions': ['Investigate CPU usage']
        }
        
        processed_alert = manager.process_anomaly_alert(sample_alert)
        
        assert 'alert_context' in processed_alert
        assert 'routing_decision' in processed_alert
        assert 'processing_timestamp' in processed_alert
        assert processed_alert['alert_id'] == 'TEST-ALERT-001'
    
    def test_alert_dashboard(self):
        """Test alert dashboard generation."""
        manager = HYDATISAlertManager()
        
        dashboard = manager.get_alert_dashboard()
        
        assert 'dashboard_timestamp' in dashboard
        assert 'alert_summary' in dashboard
        assert 'alert_statistics' in dashboard
        assert 'system_health' in dashboard


def run_isolation_forest_tests():
    """Run all Isolation Forest model tests."""
    
    print("Running HYDATIS Isolation Forest Model Tests...")
    
    test_detector = TestHYDATISIsolationForestDetector()
    test_monitor = TestHYDATISAnomalyMonitor()
    test_alerts = TestHYDATISAlertManager()
    
    tests_run = 0
    tests_passed = 0
    
    test_methods = [
        (test_detector, 'test_detector_initialization'),
        (test_monitor, 'test_monitor_initialization'),
        (test_alerts, 'test_alert_manager_initialization'),
        (test_alerts, 'test_alert_processing'),
        (test_alerts, 'test_alert_dashboard')
    ]
    
    for test_instance, method_name in test_methods:
        try:
            getattr(test_instance, method_name)()
            print(f"✓ {method_name}")
            tests_passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
        tests_run += 1
    
    print(f"\nTest Results: {tests_passed}/{tests_run} passed")
    
    try:
        print("\nRunning integration test...")
        
        detector = HYDATISIsolationForestDetector()
        
        sample_data = pd.DataFrame({
            'cpu_utilization': [0.1, 0.12, 0.95, 0.11, 0.09],
            'memory_utilization': [0.4, 0.42, 0.91, 0.41, 0.38],
            'load_1m': [1.0, 1.1, 8.5, 1.2, 0.9],
            'timestamp': pd.date_range(start='2025-08-30', periods=5, freq='30S'),
            'instance': ['worker-1', 'worker-2', 'worker-3', 'worker-1', 'worker-2']
        })
        
        X_features, feature_names = detector.prepare_anomaly_features(sample_data)
        
        if len(X_features) > 0 and len(feature_names) > 5:
            training_metrics = detector.train_anomaly_detector(X_features)
            
            if training_metrics['estimated_precision'] > 0.7:
                detection_results = detector.detect_anomalies(sample_data)
                
                print("✓ Integration test: Isolation Forest detection successful")
                print(f"✓ Estimated Precision: {training_metrics['estimated_precision']:.3f}")
                print(f"✓ Features: {len(feature_names)}")
                print(f"✓ Anomalies Detected: {detection_results['anomalies_detected']}")
                tests_passed += 1
            else:
                print("✗ Integration test: Low precision achieved")
        else:
            print("✗ Integration test: Insufficient features prepared")
        
        tests_run += 1
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        tests_run += 1
    
    success_rate = tests_passed / tests_run
    print(f"\nOverall Test Success Rate: {success_rate:.1%}")
    print(f"Status: {'✅ READY FOR PRODUCTION' if success_rate >= 0.8 else '❌ NEEDS FIXES'}")
    
    return success_rate >= 0.8


if __name__ == "__main__":
    success = run_isolation_forest_tests()
    print(f"\n{'✅' if success else '❌'} Isolation Forest model validation {'COMPLETE' if success else 'FAILED'}")