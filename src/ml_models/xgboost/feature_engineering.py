#!/usr/bin/env python3
"""
Temporal feature engineering for ML scheduler.
Creates time-based features from HYDATIS cluster metrics for improved scheduling predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """Creates temporal features for ML scheduler training."""
    
    def __init__(self):
        self.rolling_windows = [5, 15, 30, 60]  # minutes
        self.seasonal_patterns = ['hour', 'day_of_week', 'day_of_month']
    
    def create_rolling_features(self, df: pd.DataFrame, 
                              value_col: str = 'value',
                              time_col: str = 'timestamp') -> pd.DataFrame:
        """Create rolling window features for trend analysis."""
        
        # Ensure datetime index
        df = df.copy()
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col).sort_index()
        
        result_df = df.copy()
        
        # Rolling statistics for different windows
        for window_min in self.rolling_windows:
            window_freq = f'{window_min}min'
            
            # Rolling mean (trend)
            result_df[f'{value_col}_rolling_mean_{window_min}m'] = (
                df[value_col].rolling(window_freq).mean()
            )
            
            # Rolling std (volatility)
            result_df[f'{value_col}_rolling_std_{window_min}m'] = (
                df[value_col].rolling(window_freq).std()
            )
            
            # Rolling min/max (range)
            result_df[f'{value_col}_rolling_min_{window_min}m'] = (
                df[value_col].rolling(window_freq).min()
            )
            result_df[f'{value_col}_rolling_max_{window_min}m'] = (
                df[value_col].rolling(window_freq).max()
            )
            
            # Rate of change
            result_df[f'{value_col}_rate_change_{window_min}m'] = (
                df[value_col].pct_change(periods=window_min)
            )
        
        return result_df.reset_index()
    
    def create_seasonal_features(self, df: pd.DataFrame, 
                               time_col: str = 'timestamp') -> pd.DataFrame:
        """Create seasonal and cyclical features."""
        
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Hour of day (0-23)
        df['hour'] = df[time_col].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (0-6)
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Day of month (1-31)
        df['day_of_month'] = df[time_col].dt.day
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Business hours indicator
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Peak time categories for HYDATIS cluster
        df['peak_category'] = pd.cut(
            df['hour'], 
            bins=[0, 6, 9, 12, 14, 18, 22, 24],
            labels=['night', 'early_morning', 'morning_peak', 'midday', 'afternoon', 'evening', 'late_night'],
            include_lowest=True
        )
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                          value_col: str = 'value',
                          lags: List[int] = [1, 5, 15, 30]) -> pd.DataFrame:
        """Create lagged features for sequence learning."""
        
        df = df.copy()
        
        for lag in lags:
            # Simple lag
            df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
            
            # Lag difference
            df[f'{value_col}_lag_diff_{lag}'] = df[value_col] - df[value_col].shift(lag)
            
            # Lag ratio
            df[f'{value_col}_lag_ratio_{lag}'] = df[value_col] / (df[value_col].shift(lag) + 1e-8)
        
        return df
    
    def create_trend_features(self, df: pd.DataFrame, 
                            value_col: str = 'value') -> pd.DataFrame:
        """Create trend and momentum features."""
        
        df = df.copy()
        
        # Short-term trend (5-minute slope)
        df[f'{value_col}_trend_5m'] = df[value_col].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=False
        )
        
        # Medium-term trend (15-minute slope) 
        df[f'{value_col}_trend_15m'] = df[value_col].rolling(30).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=False
        )
        
        # Momentum indicators
        df[f'{value_col}_momentum_short'] = df[value_col] - df[value_col].rolling(5).mean()
        df[f'{value_col}_momentum_medium'] = df[value_col] - df[value_col].rolling(15).mean()
        
        # Volatility features
        df[f'{value_col}_volatility_5m'] = df[value_col].rolling(10).std()
        df[f'{value_col}_volatility_15m'] = df[value_col].rolling(30).std()
        
        return df
    
    def create_cross_metric_features(self, metrics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features combining multiple metrics for richer patterns."""
        
        # Start with the most complete metric as base
        base_metrics = ['cpu_usage', 'memory_usage', 'load_1m']
        base_df = None
        
        for metric in base_metrics:
            if metric in metrics_data and not metrics_data[metric].empty:
                base_df = metrics_data[metric].copy()
                break
        
        if base_df is None:
            logger.warning("No suitable base metric found for cross-metric features")
            return pd.DataFrame()
        
        # Merge other metrics by timestamp and instance
        for metric_name, metric_df in metrics_data.items():
            if metric_name not in base_metrics and not metric_df.empty:
                base_df = base_df.merge(
                    metric_df[['timestamp', 'instance', 'value']].rename(columns={'value': metric_name}),
                    on=['timestamp', 'instance'],
                    how='left'
                )
        
        # Create cross-metric ratios and interactions
        if 'cpu_usage' in base_df.columns and 'memory_usage' in base_df.columns:
            base_df['cpu_memory_ratio'] = base_df['cpu_usage'] / (base_df['memory_usage'] + 1e-8)
            base_df['resource_pressure'] = base_df['cpu_usage'] * base_df['memory_usage']
        
        if 'load_1m' in base_df.columns and 'cpu_usage' in base_df.columns:
            base_df['load_cpu_efficiency'] = base_df['load_1m'] / (base_df['cpu_usage'] + 1e-8)
        
        return base_df
    
    def process_node_temporal_features(self, node_metrics: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Complete temporal feature processing for node metrics."""
        
        logger.info("Processing temporal features for node metrics...")
        
        # Process each metric type
        processed_metrics = {}
        
        for metric_name, metric_df in node_metrics.items():
            if metric_df.empty:
                continue
                
            logger.info(f"Processing {metric_name} temporal features...")
            
            # Apply all temporal transformations
            df_with_features = metric_df.copy()
            df_with_features = self.create_rolling_features(df_with_features)
            df_with_features = self.create_seasonal_features(df_with_features)
            df_with_features = self.create_lag_features(df_with_features)
            df_with_features = self.create_trend_features(df_with_features)
            
            processed_metrics[metric_name] = df_with_features
        
        # Create cross-metric features
        combined_df = self.create_cross_metric_features(processed_metrics)
        
        # Add feature metadata
        feature_count = len([col for col in combined_df.columns if col not in ['timestamp', 'instance', 'value']])
        logger.info(f"Generated {feature_count} temporal features")
        
        return combined_df


def main():
    """Test temporal feature engineering pipeline."""
    engineer = TemporalFeatureEngineer()
    
    # Create sample data for testing
    dates = pd.date_range(start='2025-08-01', end='2025-08-30', freq='30S')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'instance': 'worker-1',
        'value': np.random.uniform(0.08, 0.13, len(dates))
    })
    
    # Test feature creation
    features_df = engineer.create_rolling_features(sample_data)
    features_df = engineer.create_seasonal_features(features_df)
    
    print(f"Sample temporal features created: {len(features_df.columns)} columns")
    print(f"Sample data points: {len(features_df)}")
    
    return engineer


if __name__ == "__main__":
    engineer = main()