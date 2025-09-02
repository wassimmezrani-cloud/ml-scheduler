#!/usr/bin/env python3
"""
Data processor for ML scheduler training data.
Processes raw Prometheus metrics into ML-ready format with feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLSchedulerDataProcessor:
    """Processes raw metrics data into ML training format."""
    
    def __init__(self, data_dir: str = "/data/ml_scheduler_longhorn"):
        self.data_dir = Path(data_dir)
        self.processed_data_dir = self.data_dir / "processed"
        self.processed_data_dir.mkdir(exist_ok=True, parents=True)
    
    def load_raw_data(self, data_type: str) -> Dict[str, pd.DataFrame]:
        """Load raw CSV data files by type (node, scheduler, pod)."""
        pattern = f"{data_type}_*.csv"
        data_files = list(self.data_dir.glob(pattern))
        
        if not data_files:
            logger.warning(f"No {data_type} data files found")
            return {}
        
        datasets = {}
        for file_path in data_files:
            metric_name = file_path.stem.split('_', 2)[1]
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                datasets[metric_name] = df
                logger.info(f"Loaded {len(df)} records for {metric_name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return datasets
    
    def process_node_metrics(self, node_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Process node metrics into consolidated training dataset."""
        
        if not node_data:
            logger.error("No node data to process")
            return pd.DataFrame()
        
        # Start with CPU usage as base
        if 'cpu_usage' not in node_data:
            logger.error("CPU usage data not found")
            return pd.DataFrame()
        
        base_df = node_data['cpu_usage'].copy()
        
        # Group by node and timestamp, aggregate CPU cores
        base_df = base_df.groupby(['timestamp', 'instance']).agg({
            'value': 'mean'
        }).reset_index()
        base_df.rename(columns={'value': 'cpu_usage_rate', 'instance': 'node'}, inplace=True)
        
        # Add other metrics
        for metric_name, df in node_data.items():
            if metric_name == 'cpu_usage':
                continue
                
            # Process each metric appropriately
            if metric_name in ['memory_usage', 'memory_available', 'memory_total']:
                metric_df = df.groupby(['timestamp', 'instance']).agg({
                    'value': 'mean'
                }).reset_index()
            elif metric_name in ['load_1m', 'load_5m', 'load_15m']:
                metric_df = df.groupby(['timestamp', 'instance']).agg({
                    'value': 'mean'
                }).reset_index()
            else:
                # For network and disk metrics, sum across interfaces/devices
                metric_df = df.groupby(['timestamp', 'instance']).agg({
                    'value': 'sum'
                }).reset_index()
            
            metric_df.rename(columns={'value': metric_name, 'instance': 'node'}, inplace=True)
            
            # Merge with base dataframe
            base_df = pd.merge(base_df, metric_df, on=['timestamp', 'node'], how='left')
        
        # Add temporal features
        base_df = self._add_temporal_features(base_df)
        
        # Add rolling window features
        base_df = self._add_rolling_features(base_df)
        
        return base_df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features for ML models."""
        df = df.copy()
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features for trend analysis."""
        df = df.copy()
        df = df.sort_values(['node', 'timestamp'])
        
        # Rolling windows: 1h, 6h, 24h (2, 12, 48 periods of 30s)
        windows = [2, 12, 48]
        metrics = ['cpu_usage_rate', 'memory_usage', 'load_1m']
        
        for metric in metrics:
            if metric in df.columns:
                for window in windows:
                    window_hours = window * 0.5  # Convert 30s periods to hours
                    
                    # Rolling mean
                    df[f'{metric}_mean_{window_hours}h'] = df.groupby('node')[metric].rolling(
                        window=window, min_periods=1
                    ).mean().reset_index(0, drop=True)
                    
                    # Rolling std
                    df[f'{metric}_std_{window_hours}h'] = df.groupby('node')[metric].rolling(
                        window=window, min_periods=1
                    ).std().reset_index(0, drop=True)
                    
                    # Rolling max
                    df[f'{metric}_max_{window_hours}h'] = df.groupby('node')[metric].rolling(
                        window=window, min_periods=1
                    ).max().reset_index(0, drop=True)
        
        return df
    
    def create_training_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for ML training (future values to predict)."""
        df = df.copy()
        df = df.sort_values(['node', 'timestamp'])
        
        # Prediction horizons: 1h, 2h, 4h (2, 4, 8 periods ahead)
        horizons = [2, 4, 8]
        target_metrics = ['cpu_usage_rate', 'memory_usage', 'load_1m']
        
        for metric in target_metrics:
            if metric in df.columns:
                for horizon in horizons:
                    horizon_hours = horizon * 0.5
                    target_col = f'{metric}_target_{horizon_hours}h'
                    
                    df[target_col] = df.groupby('node')[metric].shift(-horizon)
        
        # Remove rows without targets (last N rows per node)
        df = df.dropna(subset=[col for col in df.columns if 'target_' in col])
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to CSV and Parquet formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as CSV
        csv_path = self.processed_data_dir / f"{filename}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")
        
        # Save as Parquet for efficient ML loading
        parquet_path = self.processed_data_dir / f"{filename}_{timestamp}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved Parquet: {parquet_path}")
        
        return csv_path, parquet_path
    
    def process_full_dataset(self, days_back: int = 30) -> str:
        """Process complete dataset for ML training."""
        
        logger.info(f"Processing {days_back} days of historical data")
        
        # Load raw data
        node_data = self.load_raw_data("node")
        scheduler_data = self.load_raw_data("scheduler")
        pod_data = self.load_raw_data("pod")
        
        if not node_data:
            logger.error("No node data available for processing")
            return ""
        
        # Process node metrics
        processed_df = self.process_node_metrics(node_data)
        
        if processed_df.empty:
            logger.error("Failed to process node metrics")
            return ""
        
        # Add training targets
        processed_df = self.create_training_targets(processed_df)
        
        # Data quality validation
        initial_rows = len(processed_df)
        processed_df = processed_df.dropna()
        final_rows = len(processed_df)
        
        data_quality = (final_rows / initial_rows) * 100 if initial_rows > 0 else 0
        logger.info(f"Data quality: {data_quality:.1f}% ({final_rows}/{initial_rows} rows)")
        
        if data_quality < 95:
            logger.warning(f"Data quality below target (95%): {data_quality:.1f}%")
        
        # Save processed dataset
        csv_path, parquet_path = self.save_processed_data(processed_df, "ml_scheduler_training_data")
        
        # Generate data summary
        self._generate_data_summary(processed_df)
        
        return str(parquet_path)
    
    def _generate_data_summary(self, df: pd.DataFrame):
        """Generate data summary report."""
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'nodes': df['node'].unique().tolist(),
            'features': len(df.columns),
            'data_quality_metrics': {
                'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'node_coverage': len(df['node'].unique()),
                'temporal_coverage_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            }
        }
        
        summary_path = self.processed_data_dir / f"data_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Data summary saved: {summary_path}")
        logger.info(f"Summary: {summary['total_records']} records, {summary['features']} features, {summary['data_quality_metrics']['node_coverage']} nodes")


if __name__ == "__main__":
    processor = MLSchedulerDataProcessor()
    dataset_path = processor.process_full_dataset()
    logger.info(f"ML training dataset ready: {dataset_path}")