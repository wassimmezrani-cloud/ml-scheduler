#!/usr/bin/env python3
"""
Node characterization features for HYDATIS ML scheduler.
Creates node-specific features for optimal pod placement decisions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NodeFeatureEngineer:
    """Creates node characterization features for ML scheduler."""
    
    def __init__(self):
        self.hydatis_node_specs = {
            'masters': ['10.110.190.32', '10.110.190.33', '10.110.190.34'],
            'workers': ['10.110.190.35', '10.110.190.36', '10.110.190.37'],
            'cpu_cores': 8,
            'memory_gb': 16
        }
    
    def create_node_capacity_features(self, metrics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create node capacity and utilization features."""
        
        # Start with CPU data as base
        if 'cpu_usage' not in metrics_data or metrics_data['cpu_usage'].empty:
            logger.warning("No CPU data available for node capacity features")
            return pd.DataFrame()
        
        base_df = metrics_data['cpu_usage'].copy()
        base_df['cpu_utilization'] = base_df['value']
        
        # Add memory utilization
        if 'memory_usage' in metrics_data and not metrics_data['memory_usage'].empty:
            memory_df = metrics_data['memory_usage'][['timestamp', 'instance', 'value']].copy()
            memory_df = memory_df.rename(columns={'value': 'memory_utilization'})
            base_df = base_df.merge(memory_df, on=['timestamp', 'instance'], how='left')
        
        # Calculate capacity features
        base_df['cpu_capacity_remaining'] = 1.0 - base_df['cpu_utilization']
        base_df['memory_capacity_remaining'] = 1.0 - base_df.get('memory_utilization', 0.4)
        
        # Combined capacity score
        base_df['overall_capacity_score'] = (
            base_df['cpu_capacity_remaining'] * 0.6 + 
            base_df['memory_capacity_remaining'] * 0.4
        )
        
        # Resource pressure indicators
        base_df['cpu_pressure'] = np.where(base_df['cpu_utilization'] > 0.8, 1, 0)
        base_df['memory_pressure'] = np.where(base_df.get('memory_utilization', 0.4) > 0.8, 1, 0)
        base_df['resource_pressure'] = base_df['cpu_pressure'] + base_df['memory_pressure']
        
        return base_df
    
    def create_node_performance_features(self, metrics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create node performance and efficiency features."""
        
        base_df = self.create_node_capacity_features(metrics_data)
        if base_df.empty:
            return pd.DataFrame()
        
        # Add load metrics if available
        if 'load_1m' in metrics_data and not metrics_data['load_1m'].empty:
            load_df = metrics_data['load_1m'][['timestamp', 'instance', 'value']].copy()
            load_df = load_df.rename(columns={'value': 'load_1m'})
            base_df = base_df.merge(load_df, on=['timestamp', 'instance'], how='left')
            
            # Load efficiency
            base_df['load_cpu_efficiency'] = base_df['load_1m'] / (base_df['cpu_utilization'] + 1e-8)
            base_df['load_per_core'] = base_df['load_1m'] / self.hydatis_node_specs['cpu_cores']
        
        # Network performance features
        if 'network_rx_bytes' in metrics_data and not metrics_data['network_rx_bytes'].empty:
            net_rx = metrics_data['network_rx_bytes'][['timestamp', 'instance', 'value']].copy()
            net_rx = net_rx.rename(columns={'value': 'network_rx_rate'})
            base_df = base_df.merge(net_rx, on=['timestamp', 'instance'], how='left')
        
        if 'network_tx_bytes' in metrics_data and not metrics_data['network_tx_bytes'].empty:
            net_tx = metrics_data['network_tx_bytes'][['timestamp', 'instance', 'value']].copy()
            net_tx = net_tx.rename(columns={'value': 'network_tx_rate'})
            base_df = base_df.merge(net_tx, on=['timestamp', 'instance'], how='left')
            
            # Network utilization
            if 'network_rx_rate' in base_df.columns:
                base_df['network_total_rate'] = base_df['network_rx_rate'] + base_df['network_tx_rate']
        
        # Disk I/O features
        if 'disk_read_bytes' in metrics_data and not metrics_data['disk_read_bytes'].empty:
            disk_read = metrics_data['disk_read_bytes'][['timestamp', 'instance', 'value']].copy()
            disk_read = disk_read.rename(columns={'value': 'disk_read_rate'})
            base_df = base_df.merge(disk_read, on=['timestamp', 'instance'], how='left')
        
        if 'disk_write_bytes' in metrics_data and not metrics_data['disk_write_bytes'].empty:
            disk_write = metrics_data['disk_write_bytes'][['timestamp', 'instance', 'value']].copy()
            disk_write = disk_write.rename(columns={'value': 'disk_write_rate'})
            base_df = base_df.merge(disk_write, on=['timestamp', 'instance'], how='left')
            
            # Disk I/O patterns
            if 'disk_read_rate' in base_df.columns:
                base_df['disk_total_io'] = base_df['disk_read_rate'] + base_df['disk_write_rate']
                base_df['disk_read_write_ratio'] = base_df['disk_read_rate'] / (base_df['disk_write_rate'] + 1e-8)
        
        return base_df
    
    def create_node_stability_features(self, node_data: pd.DataFrame, 
                                     window_hours: int = 24) -> pd.DataFrame:
        """Create node stability and reliability features."""
        
        df = node_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate stability metrics per node
        stability_features = []
        
        for node in df['instance'].unique():
            node_df = df[df['instance'] == node].sort_values('timestamp')
            
            # Rolling stability metrics
            node_df['cpu_stability'] = 1 - node_df['cpu_utilization'].rolling(f'{window_hours}h').std()
            node_df['memory_stability'] = 1 - node_df.get('memory_utilization', 0.4).rolling(f'{window_hours}h').std()
            
            # Performance consistency
            node_df['performance_consistency'] = 1 - (
                node_df['cpu_utilization'].rolling(f'{window_hours}h').std() + 
                node_df.get('memory_utilization', 0.4).rolling(f'{window_hours}h').std()
            ) / 2
            
            # Outlier detection (values beyond 2 std devs)
            cpu_mean = node_df['cpu_utilization'].rolling(f'{window_hours}h').mean()
            cpu_std = node_df['cpu_utilization'].rolling(f'{window_hours}h').std()
            node_df['cpu_outlier'] = np.abs(node_df['cpu_utilization'] - cpu_mean) > (2 * cpu_std)
            
            # Reliability score (1 - outlier rate)
            node_df['reliability_score'] = 1 - node_df['cpu_outlier'].rolling(f'{window_hours}h').mean()
            
            stability_features.append(node_df)
        
        return pd.concat(stability_features, ignore_index=True)
    
    def create_node_ranking_features(self, node_data: pd.DataFrame) -> pd.DataFrame:
        """Create relative node ranking features for placement decisions."""
        
        df = node_data.copy()
        
        # Group by timestamp for cross-node comparisons
        grouped = df.groupby('timestamp')
        
        # Ranking features (1 = best, 6 = worst for HYDATIS 6-node cluster)
        df['cpu_rank'] = grouped['cpu_utilization'].rank(ascending=True)  # Lower usage = better rank
        df['memory_rank'] = grouped.get('memory_utilization', df['cpu_utilization']).rank(ascending=True)
        df['capacity_rank'] = grouped['overall_capacity_score'].rank(ascending=False)  # Higher capacity = better rank
        
        # Percentile positions
        df['cpu_percentile'] = grouped['cpu_utilization'].rank(pct=True)
        df['memory_percentile'] = grouped.get('memory_utilization', df['cpu_utilization']).rank(pct=True)
        
        # Best/worst node indicators
        df['is_least_utilized'] = (df['cpu_rank'] == 1).astype(int)
        df['is_most_utilized'] = (df['cpu_rank'] == df.groupby('timestamp')['cpu_rank'].transform('max')).astype(int)
        
        # Node type classification
        df['node_type'] = df['instance'].apply(
            lambda x: 'master' if any(master in str(x) for master in self.hydatis_node_specs['masters']) else 'worker'
        )
        
        # Load balancing opportunity score
        df['load_balance_opportunity'] = grouped['cpu_utilization'].transform('std')  # Higher std = more opportunity
        
        return df
    
    def create_node_specialization_features(self, node_data: pd.DataFrame, 
                                          pod_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create features indicating node specialization and workload patterns."""
        
        df = node_data.copy()
        
        # Resource utilization patterns
        df['is_cpu_intensive_node'] = (df['cpu_utilization'] > df['cpu_utilization'].median()).astype(int)
        df['is_memory_intensive_node'] = (df.get('memory_utilization', 0.4) > df.get('memory_utilization', 0.4).median()).astype(int)
        
        # Historical usage patterns (last 7 days)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent_cutoff = df['timestamp'].max() - timedelta(days=7)
        recent_data = df[df['timestamp'] >= recent_cutoff]
        
        # Node specialization scores
        for node in df['instance'].unique():
            node_recent = recent_data[recent_data['instance'] == node]
            
            if len(node_recent) > 0:
                # Historical average utilization
                avg_cpu = node_recent['cpu_utilization'].mean()
                avg_memory = node_recent.get('memory_utilization', 0.4).mean()
                
                # Update specialization based on historical patterns
                df.loc[df['instance'] == node, 'historical_cpu_avg'] = avg_cpu
                df.loc[df['instance'] == node, 'historical_memory_avg'] = avg_memory
                
                # Specialization indicators
                cpu_dominant = avg_cpu > avg_memory
                df.loc[df['instance'] == node, 'cpu_dominant_workload'] = int(cpu_dominant)
                df.loc[df['instance'] == node, 'balanced_workload'] = int(abs(avg_cpu - avg_memory) < 0.1)
        
        # Node health and success indicators
        df['resource_balance_score'] = 1 - abs(df['cpu_utilization'] - df.get('memory_utilization', 0.4))
        df['optimal_utilization_zone'] = ((df['cpu_utilization'] > 0.4) & (df['cpu_utilization'] < 0.8)).astype(int)
        
        return df
    
    def process_complete_node_features(self, metrics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate complete node feature set for ML training."""
        
        logger.info("Processing complete node characterization features...")
        
        # Step 1: Basic capacity features
        node_features = self.create_node_capacity_features(metrics_data)
        if node_features.empty:
            return pd.DataFrame()
        
        # Step 2: Performance features
        node_features = self.create_node_performance_features(metrics_data)
        
        # Step 3: Stability features
        node_features = self.create_node_stability_features(node_features)
        
        # Step 4: Ranking features
        node_features = self.create_node_ranking_features(node_features)
        
        # Step 5: Specialization features
        node_features = self.create_node_specialization_features(node_features)
        
        # Feature summary
        feature_cols = [col for col in node_features.columns 
                       if col not in ['timestamp', 'instance', 'value']]
        
        logger.info(f"Generated {len(feature_cols)} node characterization features")
        logger.info(f"Feature categories: capacity, performance, stability, ranking, specialization")
        
        # Add feature metadata
        node_features['feature_generation_time'] = datetime.now()
        node_features['total_features'] = len(feature_cols)
        
        return node_features


def main():
    """Test node feature engineering pipeline."""
    engineer = NodeFeatureEngineer()
    
    # Create sample HYDATIS node data
    dates = pd.date_range(start='2025-08-23', periods=1000, freq='30S')
    
    sample_data = {
        'cpu_usage': pd.DataFrame({
            'timestamp': dates,
            'instance': np.random.choice(engineer.hydatis_node_specs['workers'], len(dates)),
            'value': np.random.uniform(0.08, 0.13, len(dates))
        }),
        'memory_usage': pd.DataFrame({
            'timestamp': dates,
            'instance': np.random.choice(engineer.hydatis_node_specs['workers'], len(dates)),
            'value': np.random.uniform(0.36, 0.43, len(dates))
        }),
        'load_1m': pd.DataFrame({
            'timestamp': dates,
            'instance': np.random.choice(engineer.hydatis_node_specs['workers'], len(dates)),
            'value': np.random.uniform(0.5, 2.0, len(dates))
        })
    }
    
    # Test feature generation
    node_features = engineer.process_complete_node_features(sample_data)
    
    if not node_features.empty:
        feature_count = node_features['total_features'].iloc[0]
        print(f"✓ Node characterization features created: {feature_count}")
        print(f"✓ Sample data points: {len(node_features)}")
        print(f"✓ HYDATIS nodes analyzed: {node_features['instance'].nunique()}")
        
        # Show key features
        key_features = [col for col in node_features.columns if any(keyword in col for keyword in 
                       ['capacity', 'rank', 'score', 'efficiency', 'stability'])][:10]
        print(f"✓ Key features: {key_features}")
    
    return engineer


if __name__ == "__main__":
    engineer = main()