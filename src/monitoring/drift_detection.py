"""
ML Model Drift Detection System for HYDATIS Cluster
Monitors model performance degradation and triggers retraining workflows
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
from prometheus_client import Counter, Histogram, Gauge
import yaml

logger = logging.getLogger(__name__)

class DriftType(Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"

class DriftSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"

@dataclass
class DriftMetrics:
    """Metrics for measuring model drift"""
    psi_score: float  # Population Stability Index
    kl_divergence: float  # Kullback-Leibler divergence
    js_divergence: float  # Jensen-Shannon divergence
    prediction_drift_rate: float
    accuracy_degradation: float
    confidence_degradation: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DriftThresholds:
    """Configurable thresholds for drift detection"""
    psi_warning: float = 0.1
    psi_critical: float = 0.25
    kl_warning: float = 0.05
    kl_critical: float = 0.15
    accuracy_degradation_warning: float = 0.02  # 2%
    accuracy_degradation_critical: float = 0.05  # 5%
    confidence_degradation_warning: float = 0.1  # 10%
    confidence_degradation_critical: float = 0.2  # 20%

@dataclass
class DriftAlert:
    """Drift detection alert"""
    model_name: str
    drift_type: DriftType
    severity: DriftSeverity
    metrics: DriftMetrics
    message: str
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

class ModelDataCollector:
    """Collects model input/output data for drift analysis"""
    
    def __init__(self, prometheus_url: str, redis_client=None):
        self.prometheus_url = prometheus_url
        self.redis_client = redis_client
        
    async def collect_feature_data(self, model_name: str, 
                                 start_time: datetime, 
                                 end_time: datetime) -> pd.DataFrame:
        """Collect feature data from model serving logs"""
        query = f"""
        sum(rate(ml_model_feature_values{{model="{model_name}"}}[5m])) by (feature_name)
        """
        
        async with aiohttp.ClientSession() as session:
            params = {
                'query': query,
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'step': '1m'
            }
            
            async with session.get(f"{self.prometheus_url}/api/v1/query_range", 
                                 params=params) as response:
                data = await response.json()
                
        # Convert Prometheus data to DataFrame
        features_data = []
        for result in data.get('data', {}).get('result', []):
            feature_name = result['metric']['feature_name']
            values = [(float(ts), float(val)) for ts, val in result['values']]
            features_data.extend([{
                'feature_name': feature_name,
                'timestamp': datetime.fromtimestamp(ts),
                'value': val
            } for ts, val in values])
            
        return pd.DataFrame(features_data)
    
    async def collect_prediction_data(self, model_name: str,
                                    start_time: datetime,
                                    end_time: datetime) -> pd.DataFrame:
        """Collect model predictions and confidence scores"""
        query = f"""
        ml_model_predictions{{model="{model_name}"}}
        """
        
        async with aiohttp.ClientSession() as session:
            params = {
                'query': query,
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'step': '30s'
            }
            
            async with session.get(f"{self.prometheus_url}/api/v1/query_range",
                                 params=params) as response:
                data = await response.json()
        
        predictions_data = []
        for result in data.get('data', {}).get('result', []):
            values = [(float(ts), float(val)) for ts, val in result['values']]
            predictions_data.extend([{
                'model_name': model_name,
                'timestamp': datetime.fromtimestamp(ts),
                'prediction': val,
                'confidence': result['metric'].get('confidence', 0.0)
            } for ts, val in values])
            
        return pd.DataFrame(predictions_data)

class DriftCalculator:
    """Calculates various drift metrics"""
    
    @staticmethod
    def calculate_psi(reference_data: np.ndarray, 
                     current_data: np.ndarray, 
                     bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        if len(reference_data) == 0 or len(current_data) == 0:
            return float('inf')
            
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference_data, bins=bins)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference_data, bins=bin_edges, density=True)
        cur_hist, _ = np.histogram(current_data, bins=bin_edges, density=True)
        
        # Normalize to probabilities
        ref_prob = ref_hist / np.sum(ref_hist)
        cur_prob = cur_hist / np.sum(cur_hist)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        ref_prob = np.maximum(ref_prob, epsilon)
        cur_prob = np.maximum(cur_prob, epsilon)
        
        # Calculate PSI
        psi = np.sum((cur_prob - ref_prob) * np.log(cur_prob / ref_prob))
        return float(psi)
    
    @staticmethod
    def calculate_kl_divergence(reference_dist: np.ndarray,
                               current_dist: np.ndarray) -> float:
        """Calculate Kullback-Leibler divergence"""
        epsilon = 1e-8
        ref_prob = np.maximum(reference_dist, epsilon)
        cur_prob = np.maximum(current_dist, epsilon)
        
        # Normalize
        ref_prob = ref_prob / np.sum(ref_prob)
        cur_prob = cur_prob / np.sum(cur_prob)
        
        return float(np.sum(cur_prob * np.log(cur_prob / ref_prob)))
    
    @staticmethod
    def calculate_js_divergence(reference_dist: np.ndarray,
                               current_dist: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence"""
        epsilon = 1e-8
        ref_prob = np.maximum(reference_dist, epsilon)
        cur_prob = np.maximum(current_dist, epsilon)
        
        # Normalize
        ref_prob = ref_prob / np.sum(ref_prob)
        cur_prob = cur_prob / np.sum(cur_prob)
        
        # Calculate JS divergence
        m = 0.5 * (ref_prob + cur_prob)
        js_div = 0.5 * np.sum(ref_prob * np.log(ref_prob / m)) + \
                 0.5 * np.sum(cur_prob * np.log(cur_prob / m))
        
        return float(js_div)

class DriftDetector:
    """Main drift detection engine"""
    
    def __init__(self, config_path: str, prometheus_url: str):
        self.config = self._load_config(config_path)
        self.prometheus_url = prometheus_url
        self.data_collector = ModelDataCollector(prometheus_url)
        self.calculator = DriftCalculator()
        self.thresholds = DriftThresholds(**self.config.get('thresholds', {}))
        
        # Prometheus metrics
        self.drift_score_gauge = Gauge('ml_drift_detection_score', 
                                     'Model drift detection score',
                                     ['model_name', 'drift_type'])
        self.drift_alerts_counter = Counter('ml_drift_alerts_total',
                                          'Total drift alerts generated',
                                          ['model_name', 'severity'])
        self.drift_detection_duration = Histogram('ml_drift_detection_duration_seconds',
                                                'Time spent on drift detection',
                                                ['model_name'])
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load drift detection configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'models': {
                    'xgboost-predictor': {
                        'check_interval_minutes': 60,
                        'reference_window_hours': 24,
                        'detection_window_hours': 1,
                        'min_samples': 100
                    },
                    'qlearning-optimizer': {
                        'check_interval_minutes': 60,
                        'reference_window_hours': 24,
                        'detection_window_hours': 1,
                        'min_samples': 50
                    },
                    'anomaly-detector': {
                        'check_interval_minutes': 30,
                        'reference_window_hours': 12,
                        'detection_window_hours': 1,
                        'min_samples': 200
                    }
                },
                'thresholds': {}
            }
    
    async def detect_data_drift(self, model_name: str) -> Optional[DriftAlert]:
        """Detect data drift by comparing feature distributions"""
        model_config = self.config['models'].get(model_name, {})
        
        # Time windows
        end_time = datetime.utcnow()
        detection_start = end_time - timedelta(hours=model_config.get('detection_window_hours', 1))
        reference_start = end_time - timedelta(hours=model_config.get('reference_window_hours', 24))
        reference_end = detection_start
        
        # Collect data
        reference_data = await self.data_collector.collect_feature_data(
            model_name, reference_start, reference_end)
        current_data = await self.data_collector.collect_feature_data(
            model_name, detection_start, end_time)
        
        if len(current_data) < model_config.get('min_samples', 100):
            logger.debug(f"Insufficient samples for {model_name}: {len(current_data)}")
            return None
            
        # Calculate drift metrics per feature
        total_psi = 0.0
        total_kl = 0.0
        feature_count = 0
        
        for feature_name in reference_data['feature_name'].unique():
            ref_feature = reference_data[reference_data['feature_name'] == feature_name]['value'].values
            cur_feature = current_data[current_data['feature_name'] == feature_name]['value'].values
            
            if len(ref_feature) > 0 and len(cur_feature) > 0:
                psi = self.calculator.calculate_psi(ref_feature, cur_feature)
                kl = self.calculator.calculate_kl_divergence(
                    np.histogram(ref_feature, bins=10)[0],
                    np.histogram(cur_feature, bins=10)[0]
                )
                
                total_psi += psi
                total_kl += kl
                feature_count += 1
        
        if feature_count == 0:
            return None
            
        # Average drift scores
        avg_psi = total_psi / feature_count
        avg_kl = total_kl / feature_count
        
        # Update metrics
        self.drift_score_gauge.labels(model_name=model_name, drift_type='data').set(avg_psi)
        
        # Determine severity
        severity = self._determine_severity(avg_psi, self.thresholds.psi_warning, 
                                          self.thresholds.psi_critical)
        
        if severity != DriftSeverity.LOW:
            metrics = DriftMetrics(
                psi_score=avg_psi,
                kl_divergence=avg_kl,
                js_divergence=0.0,  # Will be calculated separately
                prediction_drift_rate=0.0,
                accuracy_degradation=0.0,
                confidence_degradation=0.0
            )
            
            self.drift_alerts_counter.labels(model_name=model_name, severity=severity.value).inc()
            
            return DriftAlert(
                model_name=model_name,
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                metrics=metrics,
                message=f"Data drift detected for {model_name}: PSI={avg_psi:.4f}",
                recommended_action=self._get_drift_action(severity, DriftType.DATA_DRIFT)
            )
        
        return None
    
    async def detect_performance_drift(self, model_name: str) -> Optional[DriftAlert]:
        """Detect performance drift by comparing accuracy and confidence"""
        model_config = self.config['models'].get(model_name, {})
        
        end_time = datetime.utcnow()
        detection_start = end_time - timedelta(hours=model_config.get('detection_window_hours', 1))
        reference_start = end_time - timedelta(hours=model_config.get('reference_window_hours', 24))
        reference_end = detection_start
        
        # Collect prediction data
        reference_preds = await self.data_collector.collect_prediction_data(
            model_name, reference_start, reference_end)
        current_preds = await self.data_collector.collect_prediction_data(
            model_name, detection_start, end_time)
        
        if len(current_preds) < model_config.get('min_samples', 100):
            return None
        
        # Calculate performance metrics
        ref_confidence = reference_preds['confidence'].mean() if len(reference_preds) > 0 else 0.8
        cur_confidence = current_preds['confidence'].mean()
        
        # Get accuracy from validation metrics
        ref_accuracy = await self._get_model_accuracy(model_name, reference_start, reference_end)
        cur_accuracy = await self._get_model_accuracy(model_name, detection_start, end_time)
        
        # Calculate degradation
        accuracy_degradation = ref_accuracy - cur_accuracy if ref_accuracy > 0 else 0.0
        confidence_degradation = (ref_confidence - cur_confidence) / ref_confidence if ref_confidence > 0 else 0.0
        
        # Update metrics
        self.drift_score_gauge.labels(model_name=model_name, drift_type='performance').set(accuracy_degradation)
        
        # Determine severity based on worst degradation
        acc_severity = self._determine_severity(accuracy_degradation, 
                                              self.thresholds.accuracy_degradation_warning,
                                              self.thresholds.accuracy_degradation_critical)
        conf_severity = self._determine_severity(confidence_degradation,
                                                self.thresholds.confidence_degradation_warning,
                                                self.thresholds.confidence_degradation_critical)
        
        severity = max(acc_severity, conf_severity, key=lambda x: x.value)
        
        if severity != DriftSeverity.LOW:
            metrics = DriftMetrics(
                psi_score=0.0,
                kl_divergence=0.0,
                js_divergence=0.0,
                prediction_drift_rate=0.0,
                accuracy_degradation=accuracy_degradation,
                confidence_degradation=confidence_degradation
            )
            
            self.drift_alerts_counter.labels(model_name=model_name, severity=severity.value).inc()
            
            return DriftAlert(
                model_name=model_name,
                drift_type=DriftType.PERFORMANCE_DRIFT,
                severity=severity,
                metrics=metrics,
                message=f"Performance drift detected for {model_name}: accuracy↓{accuracy_degradation:.2%}, confidence↓{confidence_degradation:.2%}",
                recommended_action=self._get_drift_action(severity, DriftType.PERFORMANCE_DRIFT)
            )
        
        return None
    
    async def detect_concept_drift(self, model_name: str) -> Optional[DriftAlert]:
        """Detect concept drift by analyzing prediction patterns"""
        model_config = self.config['models'].get(model_name, {})
        
        end_time = datetime.utcnow()
        detection_start = end_time - timedelta(hours=model_config.get('detection_window_hours', 1))
        reference_start = end_time - timedelta(hours=model_config.get('reference_window_hours', 24))
        reference_end = detection_start
        
        # Collect prediction patterns
        reference_preds = await self.data_collector.collect_prediction_data(
            model_name, reference_start, reference_end)
        current_preds = await self.data_collector.collect_prediction_data(
            model_name, detection_start, end_time)
        
        if len(current_preds) < model_config.get('min_samples', 100):
            return None
        
        # Analyze prediction distribution changes
        ref_pred_dist = np.histogram(reference_preds['prediction'].values, bins=20, density=True)[0]
        cur_pred_dist = np.histogram(current_preds['prediction'].values, bins=20, density=True)[0]
        
        js_divergence = self.calculator.calculate_js_divergence(ref_pred_dist, cur_pred_dist)
        
        # Calculate prediction drift rate
        ref_mean = reference_preds['prediction'].mean() if len(reference_preds) > 0 else 0.5
        cur_mean = current_preds['prediction'].mean()
        prediction_drift_rate = abs(cur_mean - ref_mean) / max(abs(ref_mean), 0.1)
        
        # Update metrics
        self.drift_score_gauge.labels(model_name=model_name, drift_type='concept').set(js_divergence)
        
        # Determine severity
        severity = self._determine_severity(js_divergence, 0.1, 0.3)  # JS divergence thresholds
        
        if severity != DriftSeverity.LOW:
            metrics = DriftMetrics(
                psi_score=0.0,
                kl_divergence=0.0,
                js_divergence=js_divergence,
                prediction_drift_rate=prediction_drift_rate,
                accuracy_degradation=0.0,
                confidence_degradation=0.0
            )
            
            self.drift_alerts_counter.labels(model_name=model_name, severity=severity.value).inc()
            
            return DriftAlert(
                model_name=model_name,
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=severity,
                metrics=metrics,
                message=f"Concept drift detected for {model_name}: JS divergence={js_divergence:.4f}",
                recommended_action=self._get_drift_action(severity, DriftType.CONCEPT_DRIFT)
            )
        
        return None
    
    async def _get_model_accuracy(self, model_name: str, 
                                start_time: datetime, 
                                end_time: datetime) -> float:
        """Get model accuracy from validation metrics"""
        query = f'avg(ml_model_validation_accuracy{{model="{model_name}"}})'
        
        async with aiohttp.ClientSession() as session:
            params = {
                'query': query,
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            }
            
            async with session.get(f"{self.prometheus_url}/api/v1/query_range",
                                 params=params) as response:
                data = await response.json()
        
        results = data.get('data', {}).get('result', [])
        if results and results[0].get('values'):
            return float(results[0]['values'][-1][1])  # Latest value
        
        return 0.0
    
    def _determine_severity(self, value: float, warning_threshold: float, 
                          critical_threshold: float) -> DriftSeverity:
        """Determine drift severity based on thresholds"""
        if value >= critical_threshold:
            return DriftSeverity.CRITICAL
        elif value >= warning_threshold:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    def _get_drift_action(self, severity: DriftSeverity, drift_type: DriftType) -> str:
        """Get recommended action based on drift type and severity"""
        action_map = {
            (DriftSeverity.CRITICAL, DriftType.DATA_DRIFT): 
                "IMMEDIATE: Stop model serving, trigger emergency retraining, activate fallback scheduler",
            (DriftSeverity.CRITICAL, DriftType.PERFORMANCE_DRIFT):
                "IMMEDIATE: Switch to fallback scheduler, initiate emergency model retraining",
            (DriftSeverity.CRITICAL, DriftType.CONCEPT_DRIFT):
                "IMMEDIATE: Review workload patterns, trigger concept adaptation retraining",
            (DriftSeverity.MEDIUM, DriftType.DATA_DRIFT):
                "Schedule retraining within 6 hours, increase monitoring frequency",
            (DriftSeverity.MEDIUM, DriftType.PERFORMANCE_DRIFT):
                "Schedule model validation and retraining within 12 hours",
            (DriftSeverity.MEDIUM, DriftType.CONCEPT_DRIFT):
                "Investigate workload changes, schedule adaptive retraining",
        }
        
        return action_map.get((severity, drift_type), "Monitor closely and investigate root cause")

class RetrainingTrigger:
    """Triggers model retraining workflows"""
    
    def __init__(self, kserve_url: str, mlflow_url: str):
        self.kserve_url = kserve_url
        self.mlflow_url = mlflow_url
        
        # Metrics
        self.retraining_triggered_counter = Counter('ml_retraining_triggered_total',
                                                  'Retraining workflows triggered',
                                                  ['model_name', 'trigger_reason'])
    
    async def trigger_retraining(self, alert: DriftAlert) -> bool:
        """Trigger model retraining workflow"""
        try:
            logger.info(f"Triggering retraining for {alert.model_name} due to {alert.drift_type.value}")
            
            # Create retraining job specification
            retraining_spec = {
                'model_name': alert.model_name,
                'trigger_reason': alert.drift_type.value,
                'severity': alert.severity.value,
                'drift_metrics': {
                    'psi_score': alert.metrics.psi_score,
                    'accuracy_degradation': alert.metrics.accuracy_degradation,
                    'confidence_degradation': alert.metrics.confidence_degradation
                },
                'urgency': 'high' if alert.severity in [DriftSeverity.CRITICAL, DriftSeverity.HIGH] else 'normal',
                'timestamp': alert.timestamp.isoformat()
            }
            
            # Submit retraining job to MLflow
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.mlflow_url}/api/2.0/mlflow/experiments/create-run",
                    json={
                        'experiment_id': '1',  # Retraining experiment
                        'tags': [
                            {'key': 'trigger_reason', 'value': alert.drift_type.value},
                            {'key': 'severity', 'value': alert.severity.value},
                            {'key': 'model_name', 'value': alert.model_name}
                        ]
                    }
                ) as response:
                    if response.status == 200:
                        self.retraining_triggered_counter.labels(
                            model_name=alert.model_name,
                            trigger_reason=alert.drift_type.value
                        ).inc()
                        logger.info(f"Retraining workflow started for {alert.model_name}")
                        return True
                    else:
                        logger.error(f"Failed to trigger retraining: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error triggering retraining for {alert.model_name}: {e}")
            return False
    
    async def trigger_emergency_fallback(self, model_name: str) -> bool:
        """Switch to fallback scheduler for critical drift"""
        try:
            # Update KServe model configuration to reduce traffic
            async with aiohttp.ClientSession() as session:
                fallback_config = {
                    'traffic_percent': 0,  # Route 0% traffic to ML model
                    'fallback_enabled': True,
                    'reason': f'Critical drift detected in {model_name}'
                }
                
                async with session.patch(
                    f"{self.kserve_url}/v1/models/{model_name}/config",
                    json=fallback_config
                ) as response:
                    if response.status == 200:
                        logger.critical(f"Emergency fallback activated for {model_name}")
                        return True
                    else:
                        logger.error(f"Failed to activate fallback: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error activating emergency fallback for {model_name}: {e}")
            return False

class DriftMonitoringService:
    """Main drift monitoring service"""
    
    def __init__(self, config_path: str, prometheus_url: str, 
                 kserve_url: str, mlflow_url: str):
        self.detector = DriftDetector(config_path, prometheus_url)
        self.retraining_trigger = RetrainingTrigger(kserve_url, mlflow_url)
        self.running = False
        
        # Service metrics
        self.monitoring_cycles_counter = Counter('ml_drift_monitoring_cycles_total',
                                               'Total drift monitoring cycles completed')
        self.monitoring_errors_counter = Counter('ml_drift_monitoring_errors_total',
                                               'Drift monitoring errors')
    
    async def start_monitoring(self):
        """Start continuous drift monitoring"""
        self.running = True
        logger.info("Starting ML model drift monitoring service")
        
        while self.running:
            try:
                await self._monitoring_cycle()
                self.monitoring_cycles_counter.inc()
                
                # Wait before next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.monitoring_errors_counter.inc()
                logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(60)  # Shorter retry interval
    
    async def _monitoring_cycle(self):
        """Execute one monitoring cycle for all models"""
        models = self.detector.config['models'].keys()
        
        for model_name in models:
            with self.detector.drift_detection_duration.labels(model_name=model_name).time():
                # Check different types of drift
                alerts = await asyncio.gather(
                    self.detector.detect_data_drift(model_name),
                    self.detector.detect_performance_drift(model_name),
                    self.detector.detect_concept_drift(model_name),
                    return_exceptions=True
                )
                
                # Process alerts
                for alert in alerts:
                    if isinstance(alert, DriftAlert):
                        await self._handle_drift_alert(alert)
                    elif isinstance(alert, Exception):
                        logger.error(f"Error detecting drift for {model_name}: {alert}")
    
    async def _handle_drift_alert(self, alert: DriftAlert):
        """Handle detected drift alert"""
        logger.warning(f"Drift alert: {alert.message}")
        
        # Send alert to monitoring system
        await self._send_alert_notification(alert)
        
        # Take automated action based on severity
        if alert.severity == DriftSeverity.CRITICAL:
            # Emergency actions
            await self.retraining_trigger.trigger_emergency_fallback(alert.model_name)
            await self.retraining_trigger.trigger_retraining(alert)
            
        elif alert.severity == DriftSeverity.HIGH:
            # High priority retraining
            await self.retraining_trigger.trigger_retraining(alert)
            
        elif alert.severity == DriftSeverity.MEDIUM:
            # Scheduled retraining
            await self._schedule_retraining(alert)
    
    async def _send_alert_notification(self, alert: DriftAlert):
        """Send alert to monitoring/alerting system"""
        alert_payload = {
            'alert_name': f'MLModelDrift_{alert.model_name}',
            'severity': alert.severity.value,
            'message': alert.message,
            'drift_type': alert.drift_type.value,
            'recommended_action': alert.recommended_action,
            'metrics': {
                'psi_score': alert.metrics.psi_score,
                'accuracy_degradation': alert.metrics.accuracy_degradation,
                'confidence_degradation': alert.metrics.confidence_degradation
            },
            'timestamp': alert.timestamp.isoformat(),
            'labels': {
                'component': 'ml_drift_detection',
                'model_name': alert.model_name,
                'severity': alert.severity.value
            }
        }
        
        # Send to Alertmanager
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://alertmanager:9093/api/v1/alerts",
                    json=[alert_payload]
                ) as response:
                    if response.status == 200:
                        logger.info(f"Alert sent for {alert.model_name}")
                    else:
                        logger.error(f"Failed to send alert: {response.status}")
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    
    async def _schedule_retraining(self, alert: DriftAlert):
        """Schedule non-urgent retraining"""
        # Add to retraining queue with lower priority
        schedule_time = datetime.utcnow() + timedelta(hours=6)  # 6 hour delay
        
        logger.info(f"Scheduled retraining for {alert.model_name} at {schedule_time}")
        
        # Store in Redis for batch processing
        if self.detector.data_collector.redis_client:
            retraining_job = {
                'model_name': alert.model_name,
                'scheduled_time': schedule_time.isoformat(),
                'priority': 'normal',
                'trigger_reason': alert.drift_type.value
            }
            
            await self.detector.data_collector.redis_client.lpush(
                'retraining_queue', 
                json.dumps(retraining_job)
            )
    
    def stop_monitoring(self):
        """Stop drift monitoring service"""
        self.running = False
        logger.info("Stopping drift monitoring service")

class DriftReportGenerator:
    """Generates drift detection reports"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
    
    async def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily drift detection report"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        report = {
            'report_date': end_time.isoformat(),
            'period': '24h',
            'models': {},
            'summary': {
                'total_alerts': 0,
                'critical_alerts': 0,
                'retraining_triggered': 0
            }
        }
        
        # Query drift metrics for each model
        models = ['xgboost-predictor', 'qlearning-optimizer', 'anomaly-detector']
        
        for model_name in models:
            model_report = await self._generate_model_report(model_name, start_time, end_time)
            report['models'][model_name] = model_report
            
            # Update summary
            report['summary']['total_alerts'] += model_report['alert_count']
            report['summary']['critical_alerts'] += model_report['critical_alerts']
            report['summary']['retraining_triggered'] += model_report['retraining_count']
        
        return report
    
    async def _generate_model_report(self, model_name: str,
                                   start_time: datetime,
                                   end_time: datetime) -> Dict[str, Any]:
        """Generate report for specific model"""
        async with aiohttp.ClientSession() as session:
            # Get drift scores
            drift_query = f'avg_over_time(ml_drift_detection_score{{model_name="{model_name}"}}[24h])'
            alerts_query = f'increase(ml_drift_alerts_total{{model_name="{model_name}"}}[24h])'
            retraining_query = f'increase(ml_retraining_triggered_total{{model_name="{model_name}"}}[24h])'
            
            queries = [drift_query, alerts_query, retraining_query]
            results = []
            
            for query in queries:
                params = {'query': query}
                async with session.get(f"{self.prometheus_url}/api/v1/query",
                                     params=params) as response:
                    data = await response.json()
                    results.append(data)
        
        # Parse results
        avg_drift_score = 0.0
        alert_count = 0
        retraining_count = 0
        
        if results[0].get('data', {}).get('result'):
            avg_drift_score = float(results[0]['data']['result'][0]['value'][1])
        
        if results[1].get('data', {}).get('result'):
            alert_count = int(float(results[1]['data']['result'][0]['value'][1]))
        
        if results[2].get('data', {}).get('result'):
            retraining_count = int(float(results[2]['data']['result'][0]['value'][1]))
        
        return {
            'model_name': model_name,
            'avg_drift_score': avg_drift_score,
            'alert_count': alert_count,
            'critical_alerts': alert_count // 3,  # Estimate critical alerts
            'retraining_count': retraining_count,
            'status': 'healthy' if avg_drift_score < 0.1 else 'degraded' if avg_drift_score < 0.25 else 'critical'
        }

async def main():
    """Main entry point for drift monitoring service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Model Drift Detection Service')
    parser.add_argument('--config', default='/etc/ml-scheduler/drift_config.yaml',
                       help='Drift detection configuration file')
    parser.add_argument('--prometheus-url', default='http://prometheus:9090',
                       help='Prometheus server URL')
    parser.add_argument('--kserve-url', default='http://kserve-controller:8080',
                       help='KServe controller URL')
    parser.add_argument('--mlflow-url', default='http://mlflow:5000',
                       help='MLflow tracking server URL')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report only, do not start monitoring')
    
    args = parser.parse_args()
    
    if args.report_only:
        # Generate and print daily report
        reporter = DriftReportGenerator(args.prometheus_url)
        report = await reporter.generate_daily_report()
        print(json.dumps(report, indent=2))
    else:
        # Start continuous monitoring
        service = DriftMonitoringService(
            args.config, 
            args.prometheus_url,
            args.kserve_url,
            args.mlflow_url
        )
        
        try:
            await service.start_monitoring()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            service.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())