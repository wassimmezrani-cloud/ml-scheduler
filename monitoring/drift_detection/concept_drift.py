#!/usr/bin/env python3
"""
Concept Drift Detection for HYDATIS ML Scheduler.
Monitors changes in cluster behavior patterns and workload characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import warnings

logger = logging.getLogger(__name__)


class HYDATISConceptDriftDetector:
    """Concept drift detection for cluster behavior and workload patterns."""
    
    def __init__(self, drift_threshold: float = 0.15, window_size: int = 1000):
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.reference_model = None
        self.reference_accuracy = None
        self.drift_detection_history = []
        
    def fit_reference_concept(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit reference concept from historical cluster behavior."""
        
        logger.info(f"Learning reference concept from {len(historical_data)} samples")
        
        # Prepare features and targets for concept learning
        features = self._extract_behavior_features(historical_data)
        targets = self._extract_behavior_targets(historical_data)
        
        # Train reference model on historical behavior patterns
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Use Random Forest as reference concept learner
        self.reference_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.reference_model.fit(X_train, y_train)
        
        # Evaluate reference performance
        y_pred = self.reference_model.predict(X_test)
        self.reference_accuracy = accuracy_score(y_test, y_pred)
        reference_f1 = f1_score(y_test, y_pred, average='weighted')
        
        fit_results = {
            'timestamp': datetime.now(),
            'samples_used': len(historical_data),
            'features_extracted': len(features.columns),
            'reference_accuracy': self.reference_accuracy,
            'reference_f1_score': reference_f1,
            'behavior_classes': len(np.unique(targets))
        }
        
        logger.info(f"Reference concept learned: accuracy={self.reference_accuracy:.3f}, f1={reference_f1:.3f}")
        
        return fit_results
    
    def detect_concept_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect concept drift in current cluster behavior."""
        
        if self.reference_model is None:
            raise ValueError("Must fit reference concept first")
        
        logger.info(f"Detecting concept drift in {len(current_data)} samples")
        
        # Extract current behavior patterns
        current_features = self._extract_behavior_features(current_data)
        current_targets = self._extract_behavior_targets(current_data)
        
        # Test reference model on current data
        current_predictions = self.reference_model.predict(current_features)
        current_accuracy = accuracy_score(current_targets, current_predictions)
        current_f1 = f1_score(current_targets, current_predictions, average='weighted')
        
        # Calculate accuracy degradation
        accuracy_drop = self.reference_accuracy - current_accuracy
        drift_detected = accuracy_drop > self.drift_threshold
        
        # Analyze behavior pattern changes
        behavior_analysis = self._analyze_behavior_changes(
            current_features, current_targets, current_predictions
        )
        
        drift_results = {
            'timestamp': datetime.now(),
            'concept_drift_detected': drift_detected,
            'performance_metrics': {
                'reference_accuracy': self.reference_accuracy,
                'current_accuracy': current_accuracy,
                'accuracy_drop': accuracy_drop,
                'current_f1_score': current_f1,
                'threshold': self.drift_threshold
            },
            'behavior_analysis': behavior_analysis,
            'drift_severity': self._assess_drift_severity(accuracy_drop)
        }
        
        # Store in history
        self.drift_detection_history.append(drift_results)
        
        logger.info(f"Concept drift analysis: accuracy drop={accuracy_drop:.3f}, drift={drift_detected}")
        
        return drift_results
    
    def _extract_behavior_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract cluster behavior features for concept learning."""
        
        # Temporal features
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            data['is_weekend'] = data['day_of_week'].isin([5, 6])
        else:
            # Generate synthetic temporal features for testing
            data['hour'] = np.random.randint(0, 24, len(data))
            data['day_of_week'] = np.random.randint(0, 7, len(data))
            data['is_weekend'] = data['day_of_week'].isin([5, 6])
        
        # Resource utilization patterns
        resource_features = []
        for metric in ['cpu_usage', 'memory_usage', 'load_1m']:
            if metric in data.columns:
                resource_features.append(metric)
                # Add derived features
                data[f'{metric}_rolling_mean'] = data[metric].rolling(window=5, min_periods=1).mean()
                data[f'{metric}_rolling_std'] = data[metric].rolling(window=5, min_periods=1).std().fillna(0)
                resource_features.extend([f'{metric}_rolling_mean', f'{metric}_rolling_std'])
        
        # Workload intensity features
        if 'network_rx' in data.columns and 'network_tx' in data.columns:
            data['network_total'] = data['network_rx'] + data['network_tx']
            data['network_ratio'] = data['network_rx'] / (data['network_tx'] + 1e-6)
            resource_features.extend(['network_total', 'network_ratio'])
        
        if 'disk_io_read' in data.columns and 'disk_io_write' in data.columns:
            data['disk_io_total'] = data['disk_io_read'] + data['disk_io_write']
            data['disk_io_ratio'] = data['disk_io_read'] / (data['disk_io_write'] + 1e-6)
            resource_features.extend(['disk_io_total', 'disk_io_ratio'])
        
        # Select final feature set
        feature_columns = ['hour', 'day_of_week', 'is_weekend'] + resource_features
        available_features = [col for col in feature_columns if col in data.columns]
        
        return data[available_features].fillna(0)
    
    def _extract_behavior_targets(self, data: pd.DataFrame) -> pd.Series:
        """Extract behavior targets for concept learning."""
        
        # Create workload pattern classes based on resource usage
        if 'cpu_usage' in data.columns and 'memory_usage' in data.columns:
            cpu = data['cpu_usage']
            memory = data['memory_usage']
            
            # Define workload patterns
            conditions = [
                (cpu < 0.3) & (memory < 0.4),  # Low utilization
                (cpu >= 0.3) & (cpu < 0.6) & (memory >= 0.4) & (memory < 0.7),  # Medium utilization
                (cpu >= 0.6) & (memory >= 0.7),  # High utilization  
                (cpu >= 0.6) & (memory < 0.4),  # CPU intensive
                (cpu < 0.3) & (memory >= 0.7),  # Memory intensive
            ]
            
            choices = ['low_load', 'medium_load', 'high_load', 'cpu_intensive', 'memory_intensive']
            
            behavior_class = np.select(conditions, choices, default='unknown')
            
        else:
            # Fallback: random behavior classes for testing
            behavior_class = np.random.choice(
                ['low_load', 'medium_load', 'high_load'], 
                size=len(data)
            )
        
        return pd.Series(behavior_class, index=data.index)
    
    def _analyze_behavior_changes(self, 
                                features: pd.DataFrame,
                                true_targets: pd.Series,
                                predicted_targets: pd.Series) -> Dict[str, Any]:
        """Analyze specific behavior pattern changes."""
        
        # Class distribution changes
        true_dist = true_targets.value_counts(normalize=True).to_dict()
        pred_dist = pd.Series(predicted_targets).value_counts(normalize=True).to_dict()
        
        # Find most changed behavior patterns
        distribution_changes = {}
        for behavior_class in set(list(true_dist.keys()) + list(pred_dist.keys())):
            true_freq = true_dist.get(behavior_class, 0)
            pred_freq = pred_dist.get(behavior_class, 0)
            distribution_changes[behavior_class] = abs(true_freq - pred_freq)
        
        # Feature importance changes
        feature_importance_changes = {}
        if hasattr(self.reference_model, 'feature_importances_'):
            for i, feature in enumerate(features.columns):
                feature_importance_changes[feature] = self.reference_model.feature_importances_[i]
        
        analysis = {
            'behavior_class_changes': distribution_changes,
            'most_changed_behavior': max(distribution_changes, key=distribution_changes.get),
            'max_distribution_change': max(distribution_changes.values()),
            'feature_importance_shifts': feature_importance_changes,
            'prediction_confidence': self._calculate_prediction_confidence(predicted_targets)
        }
        
        return analysis
    
    def _assess_drift_severity(self, accuracy_drop: float) -> str:
        """Assess the severity of concept drift."""
        
        if accuracy_drop > 0.3:
            return "CRITICAL"
        elif accuracy_drop > 0.2:
            return "HIGH"
        elif accuracy_drop > 0.1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_prediction_confidence(self, predictions: pd.Series) -> float:
        """Calculate prediction confidence based on class distribution."""
        
        class_counts = pd.Series(predictions).value_counts()
        total_predictions = len(predictions)
        
        # Shannon entropy as confidence measure
        probabilities = class_counts / total_predictions
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Normalize to 0-1 scale (higher = more confident/diverse)
        max_entropy = np.log2(len(class_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return 1 - normalized_entropy  # Invert so higher = more confident
    
    def get_concept_drift_alert(self, drift_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate concept drift alert for monitoring systems."""
        
        if not drift_results['concept_drift_detected']:
            return None
        
        performance = drift_results['performance_metrics']
        behavior_analysis = drift_results['behavior_analysis']
        severity = drift_results['drift_severity']
        
        alert = {
            'alert_type': 'CONCEPT_DRIFT',
            'priority': severity,
            'timestamp': drift_results['timestamp'],
            'message': f"Concept drift detected: {performance['accuracy_drop']:.1%} accuracy drop",
            'accuracy_degradation': performance['accuracy_drop'],
            'changed_behaviors': [behavior_analysis['most_changed_behavior']],
            'severity': severity,
            'recommended_actions': self._get_recommended_actions(severity),
            'prometheus_metrics': {
                'hydatis_concept_drift_detected': 1,
                'hydatis_concept_accuracy_drop': performance['accuracy_drop'],
                'hydatis_concept_drift_severity': self._severity_to_numeric(severity)
            }
        }
        
        return alert
    
    def _get_recommended_actions(self, severity: str) -> List[str]:
        """Get recommended actions based on drift severity."""
        
        base_actions = [
            'Investigate workload pattern changes',
            'Review recent cluster configuration changes',
            'Check for new application deployments'
        ]
        
        if severity in ['HIGH', 'CRITICAL']:
            base_actions.extend([
                'Schedule urgent model retraining',
                'Consider fallback to standard scheduler',
                'Escalate to SRE team'
            ])
        elif severity == 'MEDIUM':
            base_actions.extend([
                'Schedule model retraining within 24h',
                'Increase monitoring frequency'
            ])
        
        return base_actions
    
    def _severity_to_numeric(self, severity: str) -> int:
        """Convert severity string to numeric for Prometheus."""
        return {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}.get(severity, 0)


def main():
    """Main concept drift detection function."""
    
    print("HYDATIS Concept Drift Detector")
    print("Monitoring cluster behavior pattern changes...")
    
    detector = HYDATISConceptDriftDetector()
    
    # Generate sample historical data for concept learning
    historical_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=2000, freq='5min'),
        'cpu_usage': np.random.normal(0.3, 0.1, 2000),
        'memory_usage': np.random.normal(0.4, 0.15, 2000),
        'load_1m': np.random.normal(2.0, 0.5, 2000),
        'network_rx': np.random.normal(1000, 200, 2000),
        'network_tx': np.random.normal(800, 150, 2000),
        'disk_io_read': np.random.normal(50, 10, 2000),
        'disk_io_write': np.random.normal(30, 8, 2000)
    })
    
    # Fit reference concept
    fit_results = detector.fit_reference_concept(historical_data)
    print(f"✅ Reference concept learned: {fit_results['reference_accuracy']:.1%} accuracy")
    
    # Generate current data with concept shift
    current_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-08-01', periods=500, freq='5min'),
        'cpu_usage': np.random.normal(0.5, 0.2, 500),  # Pattern shift
        'memory_usage': np.random.normal(0.6, 0.1, 500),  # Different usage pattern
        'load_1m': np.random.normal(3.5, 0.8, 500),
        'network_rx': np.random.normal(2000, 400, 500),  # Increased network activity
        'network_tx': np.random.normal(1600, 300, 500),
        'disk_io_read': np.random.normal(80, 15, 500),
        'disk_io_write': np.random.normal(50, 12, 500)
    })
    
    # Detect concept drift
    drift_results = detector.detect_concept_drift(current_data)
    alert = detector.get_concept_drift_alert(drift_results)
    
    if alert:
        print(f"🚨 {alert['priority']} PRIORITY ALERT: {alert['message']}")
        print(f"   Accuracy degradation: {alert['accuracy_degradation']:.1%}")
        print(f"   Changed behaviors: {alert['changed_behaviors']}")
    else:
        print("✅ No concept drift detected")
    
    return detector


if __name__ == "__main__":
    detector = main()