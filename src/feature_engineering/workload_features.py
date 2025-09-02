#!/usr/bin/env python3
"""
Workload characterization features for HYDATIS ML scheduler.
Creates pod and workload-specific features for intelligent placement decisions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class WorkloadFeatureEngineer:
    """Creates workload and pod characterization features for ML scheduler."""
    
    def __init__(self):
        self.resource_categories = {
            'cpu_intensive': {'cpu_threshold': 0.5, 'memory_threshold': 0.3},
            'memory_intensive': {'cpu_threshold': 0.2, 'memory_threshold': 0.6},
            'balanced': {'cpu_threshold': 0.4, 'memory_threshold': 0.4},
            'lightweight': {'cpu_threshold': 0.1, 'memory_threshold': 0.1}
        }
        
        self.workload_patterns = {
            'batch_jobs': ['job', 'batch', 'etl', 'processing'],
            'web_services': ['web', 'api', 'service', 'frontend'],
            'databases': ['db', 'database', 'mysql', 'postgres', 'redis'],
            'ml_workloads': ['ml', 'jupyter', 'model', 'training', 'inference'],
            'monitoring': ['prometheus', 'grafana', 'alert', 'monitor']
        }
    
    def extract_workload_metadata(self, pod_data: pd.DataFrame) -> pd.DataFrame:
        """Extract workload metadata from pod names and labels."""
        
        df = pod_data.copy()
        
        # Extract workload type from pod names
        df['workload_type'] = 'unknown'
        
        for workload_type, keywords in self.workload_patterns.items():
            for keyword in keywords:
                mask = df['pod'].str.contains(keyword, case=False, na=False)
                df.loc[mask, 'workload_type'] = workload_type
        
        # Extract namespace-based features
        if 'namespace' in df.columns:
            df['is_system_namespace'] = df['namespace'].isin([
                'kube-system', 'kube-public', 'monitoring', 'longhorn-system'
            ]).astype(int)
            
            df['is_mlops_namespace'] = df['namespace'].isin([
                'hydatis-mlops', 'kubeflow', 'ml-scheduler'
            ]).astype(int)
            
            df['is_user_namespace'] = (~df['is_system_namespace'].astype(bool) & 
                                     ~df['is_mlops_namespace'].astype(bool)).astype(int)
        
        # Extract deployment patterns
        df['is_stateful'] = df['pod'].str.contains('sts|stateful', case=False, na=False).astype(int)
        df['is_daemon'] = df['pod'].str.contains('daemon|ds', case=False, na=False).astype(int)
        df['is_job'] = df['pod'].str.contains('job|batch', case=False, na=False).astype(int)
        
        return df
    
    def create_resource_profile_features(self, pod_data: pd.DataFrame) -> pd.DataFrame:
        """Create resource usage profile features for pods."""
        
        df = self.extract_workload_metadata(pod_data)
        
        # Resource intensity classification
        if 'value' in df.columns:  # Assuming 'value' is CPU usage
            df['cpu_usage'] = df['value']
            
            # CPU intensity categories
            df['cpu_category'] = pd.cut(
                df['cpu_usage'],
                bins=[0, 0.1, 0.3, 0.6, 1.0],
                labels=['low', 'medium', 'high', 'critical'],
                include_lowest=True
            )
        
        # Add memory data if available
        if 'memory_usage' in df.columns:
            df['memory_category'] = pd.cut(
                df['memory_usage'],
                bins=[0, 0.2, 0.5, 0.8, 1.0],
                labels=['low', 'medium', 'high', 'critical'],
                include_lowest=True
            )
            
            # Combined resource profile
            df['resource_profile'] = df['cpu_category'].astype(str) + '_cpu_' + df['memory_category'].astype(str) + '_mem'
        
        # Resource efficiency metrics
        if 'cpu_usage' in df.columns:
            # CPU efficiency over time
            df['cpu_efficiency'] = df.groupby('pod')['cpu_usage'].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean()
            )
            
            # Resource utilization trend
            df['cpu_trend'] = df.groupby('pod')['cpu_usage'].transform(
                lambda x: x.rolling(window=5, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
                )
            )
        
        return df
    
    def create_placement_affinity_features(self, pod_data: pd.DataFrame, 
                                         node_data: pd.DataFrame) -> pd.DataFrame:
        """Create features indicating pod-node affinity patterns."""
        
        df = pod_data.copy()
        
        # Calculate historical placement success for pod types
        if 'node' in df.columns and 'workload_type' in df.columns:
            # Workload-node affinity
            affinity_stats = df.groupby(['workload_type', 'node']).agg({
                'cpu_usage': ['mean', 'std', 'count'],
                'pod': 'nunique'
            }).round(3)
            
            # Flatten column names
            affinity_stats.columns = ['_'.join(col).strip() for col in affinity_stats.columns]
            affinity_stats = affinity_stats.reset_index()
            
            # Calculate affinity scores
            for workload in df['workload_type'].unique():
                workload_data = affinity_stats[affinity_stats['workload_type'] == workload]
                
                if len(workload_data) > 0:
                    # Best performing node for this workload type
                    best_node = workload_data.loc[workload_data['cpu_usage_mean'].idxmin(), 'node']
                    
                    # Add affinity features
                    mask = (df['workload_type'] == workload) & (df['node'] == best_node)
                    df.loc[mask, 'optimal_node_placement'] = 1
                    df.loc[~mask & (df['workload_type'] == workload), 'optimal_node_placement'] = 0
        
        # Node capacity alignment
        if 'node' in df.columns:
            # Merge with current node capacity
            node_capacity = node_data.groupby('instance').agg({
                'cpu_capacity_remaining': 'mean',
                'memory_capacity_remaining': 'mean',
                'overall_capacity_score': 'mean'
            }).reset_index()
            
            df = df.merge(
                node_capacity.rename(columns={'instance': 'node'}),
                on='node',
                how='left'
            )
            
            # Placement suitability score
            df['placement_suitability'] = (
                df.get('cpu_capacity_remaining', 0.5) * 0.4 +
                df.get('memory_capacity_remaining', 0.5) * 0.4 +
                df.get('overall_capacity_score', 0.5) * 0.2
            )
        
        return df
    
    def create_workload_performance_features(self, pod_data: pd.DataFrame) -> pd.DataFrame:
        """Create workload performance and health features."""
        
        df = pod_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Performance metrics per pod
        pod_performance = df.groupby('pod').agg({
            'cpu_usage': ['mean', 'std', 'max', 'min'],
            'timestamp': ['min', 'max', 'count']
        }).round(4)
        
        # Flatten columns
        pod_performance.columns = ['_'.join(col).strip() for col in pod_performance.columns]
        pod_performance = pod_performance.reset_index()
        
        # Calculate performance features
        pod_performance['runtime_hours'] = (
            pd.to_datetime(pod_performance['timestamp_max']) - 
            pd.to_datetime(pod_performance['timestamp_min'])
        ).dt.total_seconds() / 3600
        
        pod_performance['cpu_stability'] = 1 - (pod_performance['cpu_usage_std'] / pod_performance['cpu_usage_mean'])
        pod_performance['cpu_peak_ratio'] = pod_performance['cpu_usage_max'] / pod_performance['cpu_usage_mean']
        
        # Performance categories
        pod_performance['performance_category'] = pd.cut(
            pod_performance['cpu_stability'],
            bins=[0, 0.7, 0.85, 1.0],
            labels=['unstable', 'moderate', 'stable'],
            include_lowest=True
        )
        
        # Merge back to main dataframe
        df = df.merge(pod_performance[['pod', 'cpu_stability', 'cpu_peak_ratio', 'performance_category']], 
                     on='pod', how='left')
        
        return df
    
    def create_scheduling_decision_features(self, pod_data: pd.DataFrame,
                                          scheduler_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for scheduling decision optimization."""
        
        df = pod_data.copy()
        
        # Pod resource requirements (normalized)
        if 'cpu_usage' in df.columns:
            df['cpu_requirement_normalized'] = df['cpu_usage'] / df['cpu_usage'].max()
            
            # Resource request vs usage efficiency
            df['cpu_efficiency_score'] = np.where(
                df['cpu_usage'] > 0,
                np.minimum(df['cpu_usage'] * 2, 1.0),  # Favor higher utilization
                0.1
            )
        
        # Scheduling difficulty score
        df['scheduling_difficulty'] = 0.0
        
        # Higher difficulty for resource-intensive pods
        if 'cpu_usage' in df.columns:
            df['scheduling_difficulty'] += df['cpu_usage'] * 0.4
        
        if 'memory_usage' in df.columns:
            df['scheduling_difficulty'] += df['memory_usage'] * 0.3
        
        # Higher difficulty for stateful workloads
        df['scheduling_difficulty'] += df.get('is_stateful', 0) * 0.2
        
        # Lower difficulty for system pods
        df['scheduling_difficulty'] -= df.get('is_system_namespace', 0) * 0.1
        
        df['scheduling_difficulty'] = np.clip(df['scheduling_difficulty'], 0, 1)
        
        # Priority scoring for ML scheduler
        df['ml_priority_score'] = (
            df.get('cpu_efficiency_score', 0.5) * 0.3 +
            df.get('placement_suitability', 0.5) * 0.4 +
            (1 - df['scheduling_difficulty']) * 0.3
        )
        
        # Placement recommendation categories
        df['placement_recommendation'] = pd.cut(
            df['ml_priority_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['avoid', 'consider', 'preferred'],
            include_lowest=True
        )
        
        return df
    
    def process_complete_workload_features(self, pod_metrics: Dict[str, pd.DataFrame],
                                         node_metrics: Dict[str, pd.DataFrame],
                                         scheduler_metrics: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate complete workload feature set for ML training."""
        
        logger.info("Processing complete workload characterization features...")
        
        # Start with pod CPU data as base
        if 'pod_cpu_usage' not in pod_metrics or pod_metrics['pod_cpu_usage'].empty:
            logger.warning("No pod CPU data available")
            return pd.DataFrame()
        
        base_df = pod_metrics['pod_cpu_usage'].copy()
        
        # Add memory data if available
        if 'pod_memory_usage' in pod_metrics and not pod_metrics['pod_memory_usage'].empty:
            memory_df = pod_metrics['pod_memory_usage'][['timestamp', 'pod', 'value']].copy()
            memory_df = memory_df.rename(columns={'value': 'memory_usage'})
            base_df = base_df.merge(memory_df, on=['timestamp', 'pod'], how='left')
        
        # Step 1: Resource profile features
        workload_features = self.create_resource_profile_features(base_df)
        
        # Step 2: Performance features
        workload_features = self.create_workload_performance_features(workload_features)
        
        # Step 3: Placement affinity (need node capacity data)
        if 'cpu_usage' in node_metrics:
            node_capacity_df = self._prepare_node_capacity_data(node_metrics)
            workload_features = self.create_placement_affinity_features(workload_features, node_capacity_df)
        
        # Step 4: Scheduling decision features
        if scheduler_metrics:
            workload_features = self.create_scheduling_decision_features(workload_features, scheduler_metrics)
        
        # Feature summary
        feature_cols = [col for col in workload_features.columns 
                       if col not in ['timestamp', 'pod', 'node', 'value']]
        
        logger.info(f"Generated {len(feature_cols)} workload characterization features")
        
        return workload_features
    
    def _prepare_node_capacity_data(self, node_metrics: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare node capacity data for workload-node matching."""
        
        if 'cpu_usage' not in node_metrics:
            return pd.DataFrame()
        
        node_df = node_metrics['cpu_usage'].copy()
        node_df['cpu_capacity_remaining'] = 1.0 - node_df['value']
        
        if 'memory_usage' in node_metrics:
            memory_df = node_metrics['memory_usage'][['timestamp', 'instance', 'value']].copy()
            memory_df = memory_df.rename(columns={'value': 'memory_usage'})
            node_df = node_df.merge(memory_df, on=['timestamp', 'instance'], how='left')
            node_df['memory_capacity_remaining'] = 1.0 - node_df['memory_usage']
        else:
            node_df['memory_capacity_remaining'] = 0.6  # Default assumption
        
        node_df['overall_capacity_score'] = (
            node_df['cpu_capacity_remaining'] * 0.6 +
            node_df['memory_capacity_remaining'] * 0.4
        )
        
        return node_df
    
    def create_pod_lifecycle_features(self, pod_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on pod lifecycle and behavior patterns."""
        
        df = pod_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Pod age and lifecycle
        pod_first_seen = df.groupby('pod')['timestamp'].min()
        pod_last_seen = df.groupby('pod')['timestamp'].max()
        
        df = df.merge(
            pod_first_seen.rename('first_seen').reset_index(),
            on='pod', how='left'
        )
        df = df.merge(
            pod_last_seen.rename('last_seen').reset_index(), 
            on='pod', how='left'
        )
        
        # Calculate lifecycle features
        df['pod_age_hours'] = (df['timestamp'] - df['first_seen']).dt.total_seconds() / 3600
        df['pod_lifetime_hours'] = (df['last_seen'] - df['first_seen']).dt.total_seconds() / 3600
        
        # Lifecycle stage
        df['lifecycle_stage'] = pd.cut(
            df['pod_age_hours'] / df['pod_lifetime_hours'],
            bins=[0, 0.1, 0.5, 0.9, 1.0],
            labels=['startup', 'early', 'mature', 'ending'],
            include_lowest=True
        )
        
        # Stability indicators
        df['is_short_lived'] = (df['pod_lifetime_hours'] < 1).astype(int)
        df['is_long_running'] = (df['pod_lifetime_hours'] > 24).astype(int)
        
        return df
    
    def create_workload_interaction_features(self, pod_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on workload interactions and co-location."""
        
        df = pod_data.copy()
        
        # Co-location analysis
        if 'node' in df.columns and 'workload_type' in df.columns:
            # Count of different workload types per node per timestamp
            colocated_counts = df.groupby(['timestamp', 'node', 'workload_type']).size().reset_index(name='count')
            colocated_pivot = colocated_counts.pivot_table(
                values='count', 
                index=['timestamp', 'node'], 
                columns='workload_type', 
                fill_value=0
            ).reset_index()
            
            # Merge back co-location information
            df = df.merge(colocated_pivot, on=['timestamp', 'node'], how='left', suffixes=('', '_colocated'))
            
            # Co-location diversity score
            workload_cols = [col for col in colocated_pivot.columns if col.endswith('_colocated')]
            df['colocated_workload_diversity'] = df[workload_cols].apply(
                lambda row: len([x for x in row if x > 0]), axis=1
            )
            
            # Resource contention indicators
            df['node_workload_density'] = df[workload_cols].sum(axis=1)
            df['resource_contention_risk'] = np.where(
                df['node_workload_density'] > 5, 1, 0
            )
        
        # Scheduling preferences based on workload type
        workload_preferences = {
            'ml_workloads': {'prefers_high_memory': 1, 'prefers_high_cpu': 1, 'isolation_preferred': 1},
            'databases': {'prefers_high_memory': 1, 'prefers_low_latency': 1, 'isolation_preferred': 1},
            'web_services': {'prefers_balanced': 1, 'scaling_friendly': 1, 'isolation_preferred': 0},
            'batch_jobs': {'prefers_high_cpu': 1, 'time_flexible': 1, 'isolation_preferred': 0},
            'monitoring': {'prefers_system_nodes': 1, 'always_available': 1, 'isolation_preferred': 0}
        }
        
        # Apply workload preferences
        for pref_key in ['prefers_high_memory', 'prefers_high_cpu', 'isolation_preferred']:
            df[pref_key] = 0
            
            for workload_type, preferences in workload_preferences.items():
                mask = df['workload_type'] == workload_type
                df.loc[mask, pref_key] = preferences.get(pref_key, 0)
        
        return df
    
    def generate_ml_training_targets(self, pod_data: pd.DataFrame, 
                                   prediction_horizon_minutes: int = 5) -> pd.DataFrame:
        """Generate target variables for ML training."""
        
        df = pod_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort for proper target generation
        df = df.sort_values(['pod', 'timestamp'])
        
        # Future resource usage targets (for load prediction)
        df['target_cpu_next'] = df.groupby('pod')['cpu_usage'].shift(-1)
        
        if 'memory_usage' in df.columns:
            df['target_memory_next'] = df.groupby('pod')['memory_usage'].shift(-1)
        
        # Binary placement success target (for placement optimization)
        df['placement_success'] = 1  # Default: assume current placements are successful
        
        # Mark potential placement failures based on resource pressure
        high_cpu_pressure = df['cpu_usage'] > 0.9
        high_memory_pressure = df.get('memory_usage', 0.4) > 0.9
        
        df.loc[high_cpu_pressure | high_memory_pressure, 'placement_success'] = 0
        
        # Anomaly detection target (for anomaly detection model)
        # Mark outliers in resource usage as anomalies
        df['is_anomaly'] = 0
        
        for pod in df['pod'].unique():
            pod_mask = df['pod'] == pod
            pod_cpu = df.loc[pod_mask, 'cpu_usage']
            
            if len(pod_cpu) > 10:  # Need sufficient data
                q1, q3 = pod_cpu.quantile([0.25, 0.75])
                iqr = q3 - q1
                outlier_mask = (pod_cpu < (q1 - 1.5 * iqr)) | (pod_cpu > (q3 + 1.5 * iqr))
                df.loc[pod_mask & outlier_mask, 'is_anomaly'] = 1
        
        return df


def main():
    """Test workload feature engineering pipeline."""
    engineer = WorkloadFeatureEngineer()
    
    # Create sample HYDATIS workload data
    dates = pd.date_range(start='2025-08-23', periods=500, freq='30S')
    
    sample_pods = [
        'ml-scheduler-notebook-0', 'prometheus-server-0', 'grafana-0',
        'jupyter-ml-dev-1', 'mlflow-server-0', 'batch-training-job-1'
    ]
    
    sample_pod_data = pd.DataFrame({
        'timestamp': np.random.choice(dates, 1000),
        'pod': np.random.choice(sample_pods, 1000),
        'node': np.random.choice(['worker-1', 'worker-2', 'worker-3'], 1000),
        'namespace': np.random.choice(['hydatis-mlops', 'monitoring', 'kubeflow'], 1000),
        'cpu_usage': np.random.uniform(0.05, 0.8, 1000),
        'memory_usage': np.random.uniform(0.1, 0.7, 1000)
    })
    
    # Test feature generation
    workload_features = engineer.process_complete_workload_features(
        {'pod_cpu_usage': sample_pod_data},
        {},  # Node metrics would be provided in real usage
        {}   # Scheduler metrics would be provided in real usage
    )
    
    if not workload_features.empty:
        feature_count = len([col for col in workload_features.columns 
                           if col not in ['timestamp', 'pod', 'node', 'value']])
        print(f"✓ Workload characterization features created: {feature_count}")
        print(f"✓ Sample workloads analyzed: {workload_features['pod'].nunique()}")
        print(f"✓ Workload types identified: {workload_features['workload_type'].nunique()}")
        
        # Show workload distribution
        workload_dist = workload_features['workload_type'].value_counts()
        print(f"✓ Workload distribution: {dict(workload_dist)}")
    
    return engineer


if __name__ == "__main__":
    engineer = main()