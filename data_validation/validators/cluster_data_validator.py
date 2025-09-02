#!/usr/bin/env python3
"""
HYDATIS cluster data validation for ML scheduler pipeline.
Validates collected metrics against expected cluster behavior and ML requirements.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HYDATISClusterValidator:
    """Validates data collected from HYDATIS 6-node cluster."""
    
    def __init__(self):
        self.cluster_specs = {
            'total_nodes': 6,
            'master_nodes': 3,
            'worker_nodes': 3,
            'cpu_cores_per_node': 8,
            'memory_gb_per_node': 16,
            'expected_cpu_range': (0.08, 0.13),  # 8-13% from cluster audit
            'expected_memory_range': (0.36, 0.43)  # 36-43% from cluster audit
        }
    
    def validate_node_metrics(self, node_data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Validate node metrics against HYDATIS cluster expectations."""
        validations = {}
        
        # CPU validation
        if 'cpu_usage' in node_data and not node_data['cpu_usage'].empty:
            cpu_df = node_data['cpu_usage']
            
            # Check node count
            unique_nodes = cpu_df['instance'].nunique()
            validations['correct_node_count'] = unique_nodes == self.cluster_specs['total_nodes']
            
            # Check CPU ranges per node
            cpu_summary = cpu_df.groupby('instance')['value'].mean()
            expected_min, expected_max = self.cluster_specs['expected_cpu_range']
            
            nodes_in_range = ((cpu_summary >= expected_min) & (cpu_summary <= expected_max)).sum()
            validations['cpu_utilization_realistic'] = nodes_in_range >= 3  # At least 3 workers in range
            
            # Check for data completeness
            validations['cpu_data_complete'] = cpu_df['value'].notna().mean() > 0.95
            
        else:
            validations['correct_node_count'] = False
            validations['cpu_utilization_realistic'] = False
            validations['cpu_data_complete'] = False
        
        # Memory validation
        if 'memory_usage' in node_data and not node_data['memory_usage'].empty:
            memory_df = node_data['memory_usage']
            
            memory_summary = memory_df.groupby('instance')['value'].mean()
            expected_min, expected_max = self.cluster_specs['expected_memory_range']
            
            nodes_in_range = ((memory_summary >= expected_min) & (memory_summary <= expected_max)).sum()
            validations['memory_utilization_realistic'] = nodes_in_range >= 3
            validations['memory_data_complete'] = memory_df['value'].notna().mean() > 0.95
            
        else:
            validations['memory_utilization_realistic'] = False
            validations['memory_data_complete'] = False
        
        return validations
    
    def validate_temporal_coverage(self, data: pd.DataFrame, expected_days: int = 30) -> Dict[str, bool]:
        """Validate temporal data coverage for ML training requirements."""
        validations = {}
        
        if data.empty or 'timestamp' not in data.columns:
            return {'sufficient_temporal_coverage': False, 'consistent_collection_interval': False}
        
        # Check time span coverage
        data_span = (data['timestamp'].max() - data['timestamp'].min()).days
        validations['sufficient_temporal_coverage'] = data_span >= expected_days * 0.9  # Allow 10% tolerance
        
        # Check collection consistency (should be ~30s intervals)
        time_diffs = data['timestamp'].diff().dt.total_seconds()
        median_interval = time_diffs.median()
        validations['consistent_collection_interval'] = abs(median_interval - 30) < 10  # ±10s tolerance
        
        # Check for significant gaps
        max_gap_hours = time_diffs.max() / 3600
        validations['no_significant_gaps'] = max_gap_hours < 2  # No gaps > 2 hours
        
        return validations
    
    def validate_ml_training_readiness(self, processed_data: pd.DataFrame) -> Dict[str, bool]:
        """Validate data is ready for ML model training."""
        validations = {}
        
        if processed_data.empty:
            return {'sufficient_training_samples': False, 'feature_completeness': False}
        
        # Minimum samples for ML training (30 days * 24 hours * 2 samples/min = ~86k)
        min_samples = 50000  # Conservative minimum
        validations['sufficient_training_samples'] = len(processed_data) >= min_samples
        
        # Feature completeness check
        required_features = [
            'cpu_usage', 'memory_usage', 'load_1m', 'timestamp',
            'scheduling_duration', 'pending_pods'
        ]
        
        available_features = set(processed_data.columns)
        missing_features = set(required_features) - available_features
        validations['feature_completeness'] = len(missing_features) == 0
        
        # Data quality for training
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        null_percentage = processed_data[numeric_cols].isnull().mean().mean()
        validations['training_data_quality'] = null_percentage < 0.05  # <5% missing
        
        # Target variable availability (for supervised learning)
        if 'target_cpu_next' in processed_data.columns:
            validations['target_variable_available'] = processed_data['target_cpu_next'].notna().mean() > 0.95
        else:
            validations['target_variable_available'] = False
        
        return validations
    
    def generate_validation_report(self, 
                                 node_data: Dict[str, pd.DataFrame],
                                 processed_data: pd.DataFrame) -> Dict:
        """Generate comprehensive validation report for Week 2."""
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'cluster': 'HYDATIS-6node',
            'validation_results': {}
        }
        
        # Run all validations
        node_validations = self.validate_node_metrics(node_data)
        temporal_validations = self.validate_temporal_coverage(processed_data)
        ml_validations = self.validate_ml_training_readiness(processed_data)
        
        # Combine results
        all_validations = {**node_validations, **temporal_validations, **ml_validations}
        report['validation_results'] = all_validations
        
        # Calculate overall score
        passed_checks = sum(1 for result in all_validations.values() if result)
        total_checks = len(all_validations)
        overall_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        report['overall_score'] = round(overall_score, 2)
        report['passed_checks'] = passed_checks
        report['total_checks'] = total_checks
        report['week2_target_met'] = overall_score >= 95.0
        
        # Detailed breakdown
        report['validation_breakdown'] = {
            'node_metrics': node_validations,
            'temporal_coverage': temporal_validations, 
            'ml_readiness': ml_validations
        }
        
        return report


def main():
    """Standalone validation execution."""
    validator = HYDATISClusterValidator()
    
    # This would typically be called from the main collection pipeline
    print("HYDATIS Cluster Data Validator initialized")
    print(f"Expected cluster specs: {validator.cluster_specs}")
    
    return validator


if __name__ == "__main__":
    validator = main()