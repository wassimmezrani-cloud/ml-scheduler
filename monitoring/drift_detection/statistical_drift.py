#!/usr/bin/env python3
"""
Statistical Drift Detection for HYDATIS ML Scheduler.
Monitors feature distribution changes using KS test, PSI, and Jensen-Shannon divergence.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


class HYDATISStatisticalDriftDetector:
    """Statistical drift detection for ML scheduler features."""
    
    def __init__(self, drift_threshold: float = 0.05, psi_threshold: float = 0.25):
        self.drift_threshold = drift_threshold
        self.psi_threshold = psi_threshold
        self.reference_distributions = {}
        self.feature_stats = {}
        
    def fit_reference(self, reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit reference distributions from historical data."""
        
        logger.info(f"Fitting reference distributions from {len(reference_data)} samples")
        
        numeric_features = reference_data.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            feature_data = reference_data[feature].dropna()
            
            self.reference_distributions[feature] = {
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'median': feature_data.median(),
                'q25': feature_data.quantile(0.25),
                'q75': feature_data.quantile(0.75),
                'min': feature_data.min(),
                'max': feature_data.max(),
                'distribution': feature_data.values,
                'bins': np.histogram(feature_data, bins=20)[1]
            }
            
            # Calculate PSI bins for this feature
            hist, bins = np.histogram(feature_data, bins=10)
            self.reference_distributions[feature]['psi_bins'] = bins
            self.reference_distributions[feature]['psi_expected'] = hist / len(feature_data)
        
        self.feature_stats['reference_timestamp'] = datetime.now()
        self.feature_stats['reference_samples'] = len(reference_data)
        
        return {
            'features_fitted': len(numeric_features),
            'reference_samples': len(reference_data),
            'timestamp': self.feature_stats['reference_timestamp']
        }
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect statistical drift between reference and current data."""
        
        if not self.reference_distributions:
            raise ValueError("Must fit reference distributions first")
        
        logger.info(f"Detecting drift in {len(current_data)} current samples")
        
        drift_results = {
            'timestamp': datetime.now(),
            'overall_drift_detected': False,
            'feature_drift': {},
            'summary_stats': {}
        }
        
        numeric_features = current_data.select_dtypes(include=[np.number]).columns
        drifted_features = []
        
        for feature in numeric_features:
            if feature not in self.reference_distributions:
                continue
                
            current_feature = current_data[feature].dropna()
            reference_feature = self.reference_distributions[feature]['distribution']
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(reference_feature, current_feature)
            
            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(
                current_feature,
                self.reference_distributions[feature]['psi_bins'],
                self.reference_distributions[feature]['psi_expected']
            )
            
            # Jensen-Shannon divergence
            js_divergence = self._calculate_js_divergence(reference_feature, current_feature)
            
            # Drift detection
            drift_detected = (
                ks_pvalue < self.drift_threshold or 
                psi_score > self.psi_threshold or
                js_divergence > 0.5
            )
            
            if drift_detected:
                drifted_features.append(feature)
            
            drift_results['feature_drift'][feature] = {
                'drift_detected': drift_detected,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'psi_score': psi_score,
                'js_divergence': js_divergence,
                'current_mean': current_feature.mean(),
                'reference_mean': self.reference_distributions[feature]['mean'],
                'mean_shift': abs(current_feature.mean() - self.reference_distributions[feature]['mean'])
            }
        
        # Overall drift assessment
        drift_results['overall_drift_detected'] = len(drifted_features) > 0
        drift_results['summary_stats'] = {
            'total_features': len(numeric_features),
            'drifted_features': len(drifted_features),
            'drift_percentage': len(drifted_features) / len(numeric_features),
            'drifted_feature_names': drifted_features
        }
        
        logger.info(f"Drift detection completed: {len(drifted_features)}/{len(numeric_features)} features drifted")
        
        return drift_results
    
    def _calculate_psi(self, current_data: pd.Series, bins: np.ndarray, expected_freq: np.ndarray) -> float:
        """Calculate Population Stability Index."""
        
        # Calculate current frequency distribution
        actual_freq, _ = np.histogram(current_data, bins=bins)
        actual_freq = actual_freq / len(current_data)
        
        # Avoid division by zero
        expected_freq = np.where(expected_freq == 0, 1e-10, expected_freq)
        actual_freq = np.where(actual_freq == 0, 1e-10, actual_freq)
        
        # Calculate PSI
        psi = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
        
        return psi
    
    def _calculate_js_divergence(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between distributions."""
        
        # Create histograms with same bins
        min_val = min(reference_data.min(), current_data.min())
        max_val = max(reference_data.max(), current_data.max())
        bins = np.linspace(min_val, max_val, 50)
        
        ref_hist, _ = np.histogram(reference_data, bins=bins, density=True)
        curr_hist, _ = np.histogram(current_data, bins=bins, density=True)
        
        # Normalize to probabilities
        ref_prob = ref_hist / ref_hist.sum()
        curr_prob = curr_hist / curr_hist.sum()
        
        # Calculate JS divergence
        js_div = jensenshannon(ref_prob, curr_prob) ** 2
        
        return js_div
    
    def get_drift_alert(self, drift_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate drift alert for monitoring systems."""
        
        if not drift_results['overall_drift_detected']:
            return None
        
        drifted_features = drift_results['summary_stats']['drifted_feature_names']
        drift_percentage = drift_results['summary_stats']['drift_percentage']
        
        # High priority if >30% features drifted
        priority = "HIGH" if drift_percentage > 0.3 else "MEDIUM"
        
        alert = {
            'alert_type': 'STATISTICAL_DRIFT',
            'priority': priority,
            'timestamp': drift_results['timestamp'],
            'message': f"Statistical drift detected in {len(drifted_features)} features ({drift_percentage:.1%})",
            'affected_features': drifted_features,
            'drift_percentage': drift_percentage,
            'recommended_actions': [
                'Investigate feature distribution changes',
                'Retrain ML models if drift persists',
                'Check data collection pipeline health'
            ],
            'prometheus_metrics': {
                'hydatis_drift_features_total': len(drifted_features),
                'hydatis_drift_percentage': drift_percentage,
                'hydatis_drift_detected': 1
            }
        }
        
        return alert


def main():
    """Main statistical drift detection function."""
    
    print("HYDATIS Statistical Drift Detector")
    print("Monitoring feature distribution changes...")
    
    detector = HYDATISStatisticalDriftDetector()
    
    # Example usage with sample data
    reference_df = pd.DataFrame({
        'cpu_usage': np.random.normal(0.3, 0.1, 1000),
        'memory_usage': np.random.normal(0.4, 0.15, 1000),
        'load_1m': np.random.normal(2.0, 0.5, 1000)
    })
    
    detector.fit_reference(reference_df)
    
    # Simulate drifted data
    current_df = pd.DataFrame({
        'cpu_usage': np.random.normal(0.5, 0.2, 500),  # Mean shift
        'memory_usage': np.random.normal(0.4, 0.15, 500),  # No drift
        'load_1m': np.random.normal(3.5, 1.0, 500)  # Mean and variance shift
    })
    
    drift_results = detector.detect_drift(current_df)
    alert = detector.get_drift_alert(drift_results)
    
    if alert:
        print(f"🚨 {alert['priority']} PRIORITY ALERT: {alert['message']}")
        print(f"   Affected features: {alert['affected_features']}")
    else:
        print("✅ No statistical drift detected")
    
    return detector


if __name__ == "__main__":
    detector = main()