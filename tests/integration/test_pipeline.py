#!/usr/bin/env python3
"""
Integration tests for ML Scheduler pipeline components.
Tests end-to-end workflow from data collection to scheduling decisions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from data_collection.prometheus_collector import PrometheusCollector
from feature_engineering.temporal_features import TemporalFeatureEngineer
from ml_models.xgboost.model import HYDATISXGBoostPredictor
from ml_models.qlearning.agent import HYDATISDQNAgent
from ml_models.isolation_forest.model import HYDATISIsolationForest


class TestMLSchedulerPipeline:
    """Test complete ML scheduler pipeline integration."""
    
    @pytest.fixture
    def sample_cluster_data(self):
        """Create sample cluster data for testing."""
        
        # Generate 7 days of sample metrics
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='5min'
        )
        
        nodes = ['worker-1', 'worker-2', 'worker-3']
        data = []
        
        for timestamp in timestamps:
            for node in nodes:
                # Simulate realistic cluster metrics
                base_cpu = 0.3 + 0.2 * np.sin(timestamp.hour * np.pi / 12)
                base_memory = 0.4 + 0.1 * np.cos(timestamp.hour * np.pi / 24)
                
                record = {
                    'timestamp': timestamp,
                    'node': node,
                    'cpu_usage': max(0.05, min(0.95, base_cpu + np.random.normal(0, 0.1))),
                    'memory_usage': max(0.1, min(0.9, base_memory + np.random.normal(0, 0.05))),
                    'load_1m': max(0.1, 4.0 + np.random.normal(0, 1.0)),
                    'network_rx_bytes': max(0, 1000 + np.random.normal(0, 200)),
                    'network_tx_bytes': max(0, 800 + np.random.normal(0, 150)),
                    'disk_read_bytes': max(0, 50 + np.random.normal(0, 10)),
                    'disk_write_bytes': max(0, 30 + np.random.normal(0, 8))
                }
                data.append(record)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineering instance."""
        return TemporalFeatureEngineer()
    
    def test_data_collection_pipeline(self, sample_cluster_data):
        """Test data collection and preprocessing pipeline."""
        
        # Test data quality
        assert not sample_cluster_data.empty
        assert len(sample_cluster_data) > 1000  # 7 days * 3 nodes * 288 samples/day
        assert 'timestamp' in sample_cluster_data.columns
        assert 'node' in sample_cluster_data.columns
        
        # Test data completeness
        required_metrics = ['cpu_usage', 'memory_usage', 'load_1m']
        for metric in required_metrics:
            assert metric in sample_cluster_data.columns
            assert sample_cluster_data[metric].notna().sum() > 0
        
        # Test data ranges
        assert 0 <= sample_cluster_data['cpu_usage'].max() <= 1.0
        assert 0 <= sample_cluster_data['memory_usage'].max() <= 1.0
        
        print("✅ Data collection pipeline test passed")
    
    def test_feature_engineering_pipeline(self, sample_cluster_data, feature_engineer):
        """Test feature engineering transforms data correctly."""
        
        # Apply feature engineering
        features_df = feature_engineer.create_temporal_features(sample_cluster_data)
        
        # Test feature creation
        assert not features_df.empty
        assert 'hour' in features_df.columns
        assert 'day_of_week' in features_df.columns
        
        # Test rolling window features
        rolling_features = [col for col in features_df.columns if 'rolling' in col]
        assert len(rolling_features) > 0
        
        # Test feature data types
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        assert len(numeric_features) >= len(sample_cluster_data.select_dtypes(include=[np.number]).columns)
        
        print("✅ Feature engineering pipeline test passed")
    
    def test_xgboost_prediction_pipeline(self, sample_cluster_data):
        """Test XGBoost load prediction pipeline."""
        
        predictor = HYDATISXGBoostPredictor()
        
        # Create mock training targets
        df_with_targets = sample_cluster_data.copy()
        df_with_targets['target_cpu_5m'] = df_with_targets['cpu_usage'].shift(-1)
        df_with_targets['target_memory_5m'] = df_with_targets['memory_usage'].shift(-1)
        df_with_targets = df_with_targets.dropna()
        
        # Test data preparation
        X, y = predictor.prepare_training_data_from_df(df_with_targets)
        assert not X.empty
        assert not y.empty
        assert len(predictor.feature_names) > 0
        
        print("✅ XGBoost prediction pipeline test passed")
    
    def test_qlearning_agent_pipeline(self):
        """Test Q-Learning agent initialization and decision making."""
        
        cluster_config = {
            'nodes': 6, 'workers': 3, 'masters': 3,
            'worker_nodes': ['worker-1', 'worker-2', 'worker-3']
        }
        
        from ml_models.qlearning.agent import HYDATISPlacementDQN
        dqn_placement = HYDATISPlacementDQN(cluster_config)
        
        # Test agent initialization
        assert dqn_placement.state_size > 0
        assert dqn_placement.action_size == 3
        assert dqn_placement.target_improvement == 0.34
        
        # Test placement decision
        sample_state = np.random.uniform(0.1, 0.8, dqn_placement.state_size)
        sample_pod = {'cpu_request': 0.1, 'memory_request': 0.2}
        
        action, decision = dqn_placement.select_placement_node(sample_state, sample_pod)
        
        assert 0 <= action < dqn_placement.action_size
        assert 'selected_node' in decision
        assert 'placement_quality' in decision
        assert 'q_values' in decision
        
        print("✅ Q-Learning agent pipeline test passed")
    
    def test_anomaly_detection_pipeline(self, sample_cluster_data):
        """Test anomaly detection pipeline."""
        
        detector = HYDATISIsolationForest()
        
        # Prepare test data
        feature_cols = ['cpu_usage', 'memory_usage', 'load_1m', 'network_rx_bytes']
        X = sample_cluster_data[feature_cols].dropna()
        
        # Test anomaly detection
        detector.fit(X)
        predictions = detector.predict(X)
        scores = detector.decision_function(X)
        
        assert len(predictions) == len(X)
        assert len(scores) == len(X)
        assert predictions.dtype == bool
        assert isinstance(scores, np.ndarray)
        
        # Test anomaly rate is reasonable
        anomaly_rate = predictions.sum() / len(predictions)
        assert 0.01 <= anomaly_rate <= 0.20  # Between 1% and 20%
        
        print("✅ Anomaly detection pipeline test passed")
    
    def test_end_to_end_scheduling_pipeline(self, sample_cluster_data):
        """Test complete end-to-end scheduling pipeline."""
        
        print("🚀 Testing end-to-end ML scheduling pipeline...")
        
        # Step 1: Data preprocessing
        feature_cols = ['cpu_usage', 'memory_usage', 'load_1m']
        cluster_features = sample_cluster_data[['node'] + feature_cols].dropna()
        
        # Step 2: Create current cluster state
        current_state = cluster_features.groupby('node')[feature_cols].mean()
        state_vector = current_state.values.flatten()
        
        # Step 3: Anomaly detection
        detector = HYDATISIsolationForest()
        detector.fit(current_state)
        node_health = detector.predict(current_state)
        
        # Step 4: Q-Learning placement decision
        cluster_config = {
            'nodes': 6, 'workers': 3, 'masters': 3,
            'worker_nodes': list(current_state.index)
        }
        
        from ml_models.qlearning.agent import HYDATISPlacementDQN
        dqn_agent = HYDATISPlacementDQN(cluster_config)
        
        # Extend state vector to match agent requirements
        extended_state = np.concatenate([state_vector, [0.5, 0.3, 0.2]])  # Add pod requirements
        extended_state = extended_state[:dqn_agent.state_size]  # Truncate if too long
        
        pod_requirements = {'cpu_request': 0.2, 'memory_request': 0.3}
        action, decision = dqn_agent.select_placement_node(extended_state, pod_requirements)
        
        # Step 5: Validate complete pipeline
        assert 0 <= action < len(cluster_config['worker_nodes'])
        assert 'placement_quality' in decision
        assert len(node_health) == len(current_state)
        
        # Step 6: Generate scheduling report
        scheduling_report = {
            'timestamp': datetime.now().isoformat(),
            'cluster_nodes': len(current_state),
            'healthy_nodes': (~node_health).sum(),
            'anomalous_nodes': node_health.sum(),
            'recommended_node': decision['selected_node'],
            'placement_confidence': decision['decision_confidence'],
            'placement_quality': decision['placement_quality']['overall_quality'],
            'pod_requirements': pod_requirements
        }
        
        print(f"📊 End-to-End Pipeline Results:")
        print(f"   Nodes analyzed: {scheduling_report['cluster_nodes']}")
        print(f"   Healthy nodes: {scheduling_report['healthy_nodes']}")
        print(f"   Recommended node: {scheduling_report['recommended_node']}")
        print(f"   Placement quality: {scheduling_report['placement_quality']:.3f}")
        print(f"   Decision confidence: {scheduling_report['placement_confidence']:.3f}")
        
        assert scheduling_report['placement_quality'] > 0.3  # Reasonable quality
        assert scheduling_report['healthy_nodes'] > 0  # At least one healthy node
        
        print("✅ End-to-end scheduling pipeline test passed")
        
        return scheduling_report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])