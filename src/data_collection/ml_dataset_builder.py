#!/usr/bin/env python3
"""
ML dataset builder for HYDATIS scheduler training.
Creates production-ready datasets for XGBoost, Q-Learning, and Isolation Forest models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path
import joblib

from .prometheus_collector import PrometheusCollector
from ..feature_engineering.temporal_features import TemporalFeatureEngineer
from ..feature_engineering.node_features import NodeFeatureEngineer
from ..feature_engineering.workload_features import WorkloadFeatureEngineer
from ..feature_engineering.feature_store import AdvancedFeatureSelector

logger = logging.getLogger(__name__)


class MLDatasetBuilder:
    """Builds ML-ready datasets for HYDATIS scheduler training."""
    
    def __init__(self, prometheus_url: str = "http://10.110.190.83:9090"):
        self.collector = PrometheusCollector(prometheus_url)
        self.temporal_engineer = TemporalFeatureEngineer()
        self.node_engineer = NodeFeatureEngineer()
        self.workload_engineer = WorkloadFeatureEngineer()
        self.feature_selector = AdvancedFeatureSelector()
        
        self.output_dir = Path("/data/ml_scheduler_longhorn/ml_datasets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_training_data(self, days_back: int = 30) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Collect comprehensive training data from HYDATIS cluster."""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        logger.info(f"Collecting {days_back} days of HYDATIS training data...")
        
        # Collect all metric types
        node_metrics = self.collector.collect_node_metrics(start_time, end_time)
        scheduler_metrics = self.collector.collect_scheduler_metrics(start_time, end_time)
        pod_metrics = self.collector.collect_pod_metrics(start_time, end_time)
        
        return {
            'node': node_metrics,
            'scheduler': scheduler_metrics,
            'pod': pod_metrics
        }
    
    def build_xgboost_dataset(self, metrics_data: Dict) -> Tuple[pd.DataFrame, List[str]]:
        """Build dataset specifically for XGBoost load prediction."""
        
        logger.info("Building XGBoost load prediction dataset...")
        
        # Generate comprehensive features
        node_features = self.node_engineer.process_complete_node_features(metrics_data['node'])
        temporal_features = self.temporal_engineer.process_node_temporal_features(metrics_data['node'])
        
        if node_features.empty:
            logger.error("No node features generated for XGBoost")
            return pd.DataFrame(), []
        
        # Merge features
        if not temporal_features.empty:
            # Select key temporal features to avoid feature explosion
            temporal_subset = [col for col in temporal_features.columns 
                             if any(keyword in col for keyword in ['rolling_mean', 'trend', 'sin', 'cos'])][:20]
            
            combined_df = node_features.merge(
                temporal_features[['timestamp', 'instance'] + temporal_subset],
                on=['timestamp', 'instance'], how='left'
            )
        else:
            combined_df = node_features
        
        # Create prediction targets
        combined_df = combined_df.sort_values(['instance', 'timestamp'])
        combined_df['target_cpu_5m'] = combined_df.groupby('instance')['cpu_utilization'].shift(-10)  # 5min ahead
        combined_df['target_cpu_15m'] = combined_df.groupby('instance')['cpu_utilization'].shift(-30)  # 15min ahead
        
        if 'memory_utilization' in combined_df.columns:
            combined_df['target_memory_5m'] = combined_df.groupby('instance')['memory_utilization'].shift(-10)
            combined_df['target_memory_15m'] = combined_df.groupby('instance')['memory_utilization'].shift(-30)
        
        # Select optimal features for XGBoost
        feature_cols = [col for col in combined_df.columns 
                       if col not in ['timestamp', 'instance', 'value'] and not col.startswith('target_')]
        
        X = combined_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y_cpu = combined_df['target_cpu_5m'].fillna(combined_df['cpu_utilization'])
        
        if len(X) > 1000 and len(X.columns) > 10:
            # Apply feature selection
            selected_features_dict, _ = self.feature_selector.select_features_multimethod(X, y_cpu)
            optimal_features = self.feature_selector.create_consensus_feature_set(selected_features_dict)
        else:
            optimal_features = list(X.columns[:30])
        
        # Final XGBoost dataset
        xgboost_df = combined_df[['timestamp', 'instance'] + optimal_features + 
                               ['target_cpu_5m', 'target_cpu_15m'] +
                               (['target_memory_5m', 'target_memory_15m'] if 'memory_utilization' in combined_df.columns else [])
                              ].dropna()
        
        logger.info(f"XGBoost dataset: {len(xgboost_df)} samples, {len(optimal_features)} features")
        
        return xgboost_df, optimal_features
    
    def build_qlearning_dataset(self, metrics_data: Dict) -> pd.DataFrame:
        """Build dataset for Q-Learning placement optimization."""
        
        logger.info("Building Q-Learning placement dataset...")
        
        # Node capacity features for state representation
        node_features = self.node_engineer.process_complete_node_features(metrics_data['node'])
        
        if node_features.empty:
            logger.error("No node features for Q-Learning")
            return pd.DataFrame()
        
        # Create state-action-reward structure
        qlearning_df = node_features[['timestamp', 'instance', 'cpu_utilization', 'memory_utilization',
                                    'overall_capacity_score', 'cpu_rank', 'memory_rank']].copy()
        
        # State: Current cluster resource state (normalized)
        state_features = ['cpu_utilization', 'memory_utilization', 'overall_capacity_score']
        for feature in state_features:
            if feature in qlearning_df.columns:
                qlearning_df[f'{feature}_normalized'] = (
                    qlearning_df[feature] - qlearning_df[feature].min()
                ) / (qlearning_df[feature].max() - qlearning_df[feature].min() + 1e-8)
        
        # Action: Node selection (encoded)
        nodes = qlearning_df['instance'].unique()
        node_mapping = {node: idx for idx, node in enumerate(nodes)}
        qlearning_df['action'] = qlearning_df['instance'].map(node_mapping)
        
        # Reward: Inverse of resource utilization (higher capacity = higher reward)
        qlearning_df['reward'] = (
            (1 - qlearning_df['cpu_utilization']) * 0.6 +
            (1 - qlearning_df.get('memory_utilization', 0.4)) * 0.4
        )
        
        # Next state for Q-Learning updates
        qlearning_df = qlearning_df.sort_values(['timestamp'])
        for feature in state_features:
            if f'{feature}_normalized' in qlearning_df.columns:
                qlearning_df[f'next_{feature}_normalized'] = qlearning_df[f'{feature}_normalized'].shift(-1)
        
        qlearning_df['done'] = qlearning_df['next_cpu_utilization_normalized'].isna().astype(int)
        
        logger.info(f"Q-Learning dataset: {len(qlearning_df)} transitions, {len(nodes)} actions")
        
        return qlearning_df.dropna()
    
    def build_anomaly_dataset(self, metrics_data: Dict) -> pd.DataFrame:
        """Build dataset for Isolation Forest anomaly detection."""
        
        logger.info("Building Isolation Forest anomaly dataset...")
        
        # Comprehensive features for anomaly detection
        node_features = self.node_engineer.process_complete_node_features(metrics_data['node'])
        
        if node_features.empty:
            logger.error("No features for anomaly detection")
            return pd.DataFrame()
        
        # Select features most relevant for anomaly detection
        anomaly_features = [
            'cpu_utilization', 'memory_utilization', 'load_1m',
            'cpu_capacity_remaining', 'memory_capacity_remaining',
            'resource_pressure', 'performance_consistency', 'reliability_score'
        ]
        
        available_features = [f for f in anomaly_features if f in node_features.columns]
        
        anomaly_df = node_features[['timestamp', 'instance'] + available_features].copy()
        
        # Create anomaly labels for training (semi-supervised approach)
        # Normal behavior: within expected ranges
        # Anomalous: extreme resource usage or instability
        
        conditions = []
        
        # CPU anomalies: outside normal HYDATIS range (8-13%)
        if 'cpu_utilization' in anomaly_df.columns:
            cpu_anomalies = (anomaly_df['cpu_utilization'] < 0.05) | (anomaly_df['cpu_utilization'] > 0.95)
            conditions.append(cpu_anomalies)
        
        # Memory anomalies: outside normal range (36-43%)
        if 'memory_utilization' in anomaly_df.columns:
            memory_anomalies = (anomaly_df['memory_utilization'] < 0.20) | (anomaly_df['memory_utilization'] > 0.90)
            conditions.append(memory_anomalies)
        
        # Performance anomalies: low reliability or high resource pressure
        if 'reliability_score' in anomaly_df.columns:
            performance_anomalies = anomaly_df['reliability_score'] < 0.7
            conditions.append(performance_anomalies)
        
        # Combine all anomaly conditions
        if conditions:
            anomaly_df['is_anomaly'] = np.logical_or.reduce(conditions).astype(int)
        else:
            # Fallback: statistical outliers
            anomaly_df['is_anomaly'] = 0
            for feature in available_features:
                if anomaly_df[feature].dtype in ['float64', 'int64']:
                    q1, q3 = anomaly_df[feature].quantile([0.01, 0.99])
                    outliers = (anomaly_df[feature] < q1) | (anomaly_df[feature] > q3)
                    anomaly_df.loc[outliers, 'is_anomaly'] = 1
        
        normal_count = (anomaly_df['is_anomaly'] == 0).sum()
        anomaly_count = (anomaly_df['is_anomaly'] == 1).sum()
        
        logger.info(f"Anomaly dataset: {normal_count} normal, {anomaly_count} anomalies ({anomaly_count/(normal_count+anomaly_count)*100:.1f}%)")
        
        return anomaly_df
    
    def save_ml_datasets(self, xgboost_df: pd.DataFrame, qlearning_df: pd.DataFrame, 
                        anomaly_df: pd.DataFrame, optimal_features: List[str]) -> Dict[str, str]:
        """Save all ML datasets to Longhorn storage."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}
        
        # Save XGBoost dataset
        if not xgboost_df.empty:
            xgboost_path = self.output_dir / f"xgboost_load_prediction_{timestamp}.parquet"
            xgboost_df.to_parquet(xgboost_path)
            saved_files['xgboost'] = str(xgboost_path)
            logger.info(f"XGBoost dataset saved: {xgboost_path}")
        
        # Save Q-Learning dataset
        if not qlearning_df.empty:
            qlearning_path = self.output_dir / f"qlearning_placement_{timestamp}.parquet"
            qlearning_df.to_parquet(qlearning_path)
            saved_files['qlearning'] = str(qlearning_path)
            logger.info(f"Q-Learning dataset saved: {qlearning_path}")
        
        # Save Isolation Forest dataset
        if not anomaly_df.empty:
            anomaly_path = self.output_dir / f"isolation_forest_anomaly_{timestamp}.parquet"
            anomaly_df.to_parquet(anomaly_path)
            saved_files['anomaly'] = str(anomaly_path)
            logger.info(f"Isolation Forest dataset saved: {anomaly_path}")
        
        # Save feature metadata
        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'cluster': 'HYDATIS-6node',
            'optimal_features': optimal_features,
            'datasets': saved_files,
            'ready_for_training': True
        }
        
        metadata_path = self.output_dir / f"dataset_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        saved_files['metadata'] = str(metadata_path)
        
        return saved_files
    
    def build_complete_ml_pipeline(self, days_back: int = 30) -> Dict[str, str]:
        """Build complete ML training pipeline for all three models."""
        
        logger.info("Building complete ML training datasets for HYDATIS cluster...")
        
        # Step 1: Collect training data
        metrics_data = self.collect_training_data(days_back)
        
        if not any(metrics_data.values()):
            logger.error("No training data collected")
            return {}
        
        # Step 2: Build model-specific datasets
        xgboost_df, optimal_features = self.build_xgboost_dataset(metrics_data)
        qlearning_df = self.build_qlearning_dataset(metrics_data)
        anomaly_df = self.build_anomaly_dataset(metrics_data)
        
        # Step 3: Save datasets
        saved_files = self.save_ml_datasets(xgboost_df, qlearning_df, anomaly_df, optimal_features)
        
        # Step 4: Generate summary
        summary = {
            'pipeline_completed': datetime.now().isoformat(),
            'datasets_created': len(saved_files),
            'total_features': len(optimal_features),
            'data_period_days': days_back,
            'ready_for_week5': True
        }
        
        summary_path = self.output_dir / f"ml_pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        saved_files['summary'] = str(summary_path)
        
        logger.info(f"Complete ML pipeline built: {len(saved_files)} files created")
        
        return saved_files


def main():
    """Build ML datasets for HYDATIS cluster."""
    builder = MLDatasetBuilder()
    
    # Build complete training pipeline
    result_files = builder.build_complete_ml_pipeline(days_back=30)
    
    print("✓ HYDATIS ML Dataset Builder Complete")
    print(f"✓ Files created: {len(result_files)}")
    print(f"✓ Storage location: /data/ml_scheduler_longhorn/ml_datasets/")
    print("✓ Ready for Week 5: ML Model Development")
    
    return builder


if __name__ == "__main__":
    builder = main()