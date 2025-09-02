"""
Advanced Multi-Dimensional Anomaly Detection with Root Cause Analysis
Implements sophisticated anomaly detection beyond simple threshold-based alerting
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import networkx as nx
from prometheus_client import Counter, Histogram, Gauge
import yaml
import uuid

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    TEMPORAL = "temporal"
    MULTIVARIATE = "multivariate"

class AnomalySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"

class RootCauseCategory(Enum):
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SERVICE_DEGRADATION = "service_degradation"
    CONFIGURATION_DRIFT = "configuration_drift"
    WORKLOAD_SPIKE = "workload_spike"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    MODEL_DEGRADATION = "model_degradation"
    CASCADING_FAILURE = "cascading_failure"

@dataclass
class AnomalyDetection:
    """Detected anomaly with context"""
    anomaly_id: str
    type: AnomalyType
    severity: AnomalySeverity
    affected_components: List[str]
    confidence_score: float
    outlier_score: float
    description: str
    metrics_snapshot: Dict[str, float]
    temporal_context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RootCauseHypothesis:
    """Root cause analysis hypothesis"""
    hypothesis_id: str
    category: RootCauseCategory
    confidence: float
    contributing_factors: List[str]
    evidence: Dict[str, Any]
    recommended_investigation: List[str]
    estimated_impact: Dict[str, float]

@dataclass
class AnomalyInvestigation:
    """Complete anomaly investigation result"""
    anomaly: AnomalyDetection
    root_cause_hypotheses: List[RootCauseHypothesis]
    correlation_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    impact_assessment: Dict[str, Any]
    recommended_actions: List[str]

class MultiDimensionalAnomalyDetector:
    """Advanced multi-dimensional anomaly detection engine"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        
        # Anomaly detection models
        self.isolation_forest = IsolationForest(
            contamination=0.1, random_state=42, n_estimators=200)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.elliptic_envelope = EllipticEnvelope(contamination=0.1, random_state=42)
        
        # Scalers for different data types
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # PCA for dimensionality reduction
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        
        # Model training status
        self.models_trained = False
        self.last_training_time = None
        
        # Metrics
        self.anomalies_detected_counter = Counter('advanced_anomalies_detected_total',
                                                'Advanced anomalies detected',
                                                ['type', 'severity'])
        self.detection_duration = Histogram('anomaly_detection_duration_seconds',
                                          'Time spent on anomaly detection',
                                          ['detection_type'])
        self.anomaly_confidence_gauge = Gauge('anomaly_detection_confidence_score',
                                            'Confidence score of anomaly detections',
                                            ['anomaly_type'])
    
    async def collect_multi_dimensional_data(self, lookback_hours: int = 24) -> pd.DataFrame:
        """Collect multi-dimensional metrics for anomaly detection"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        # Define comprehensive metric queries
        metric_queries = {
            # Core cluster metrics
            'cpu_utilization': 'avg(100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))',
            'memory_utilization': 'avg((1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100)',
            'availability': 'avg(up{job="kubernetes-nodes"}) * 100',
            
            # Scheduler performance metrics
            'scheduling_latency_p99': 'histogram_quantile(0.99, rate(ml_scheduler_scheduling_duration_seconds_bucket[5m])) * 1000',
            'scheduling_latency_p95': 'histogram_quantile(0.95, rate(ml_scheduler_scheduling_duration_seconds_bucket[5m])) * 1000',
            'scheduling_latency_p50': 'histogram_quantile(0.50, rate(ml_scheduler_scheduling_duration_seconds_bucket[5m])) * 1000',
            'scheduling_success_rate': 'rate(ml_scheduler_scheduling_success_total[5m]) / rate(ml_scheduler_scheduling_requests_total[5m]) * 100',
            'scheduling_request_rate': 'rate(ml_scheduler_scheduling_requests_total[5m])',
            'fallback_rate': 'rate(ml_scheduler_fallback_triggered_total[5m]) / rate(ml_scheduler_scheduling_requests_total[5m]) * 100',
            
            # ML model performance
            'xgboost_latency': 'histogram_quantile(0.95, rate(xgboost_prediction_duration_seconds_bucket[5m])) * 1000',
            'xgboost_confidence': 'avg(xgboost_prediction_confidence)',
            'qlearning_latency': 'histogram_quantile(0.95, rate(qlearning_optimization_duration_seconds_bucket[5m])) * 1000',
            'qlearning_effectiveness': 'avg(qlearning_optimization_effectiveness)',
            'anomaly_detection_latency': 'histogram_quantile(0.95, rate(anomaly_detection_duration_seconds_bucket[5m])) * 1000',
            'anomaly_detection_precision': 'avg(anomaly_detection_precision)',
            
            # Cache performance
            'redis_hit_rate': 'rate(redis_keyspace_hits[5m]) / (rate(redis_keyspace_hits[5m]) + rate(redis_keyspace_misses[5m])) * 100',
            'redis_memory_usage': 'redis_memory_used_bytes / redis_memory_max_bytes * 100',
            'cache_latency': 'histogram_quantile(0.95, rate(redis_command_duration_seconds_bucket[5m])) * 1000',
            
            # Resource utilization patterns
            'pod_count': 'sum(kube_pod_info{scheduler="ml-scheduler"})',
            'node_balance_score': 'avg(ml_scheduler_node_balance_score)',
            'resource_efficiency': 'avg(cluster_resource_efficiency_percent)',
            'resource_waste': '100 - avg(cluster_resource_efficiency_percent)',
            
            # Network and I/O metrics
            'network_io_rate': 'rate(node_network_receive_bytes_total[5m]) + rate(node_network_transmit_bytes_total[5m])',
            'disk_io_rate': 'rate(node_disk_read_bytes_total[5m]) + rate(node_disk_written_bytes_total[5m])',
            'disk_utilization': 'avg((1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100)',
            
            # Business metrics
            'cost_efficiency_score': '((85 - avg(100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))) / 85 * 0.3 + (avg(up{job="kubernetes-nodes"}) - 0.952) / 0.045) * 1200',
            'sla_compliance': 'avg(up{job="kubernetes-nodes"}) >= bool 0.997'
        }
        
        # Collect all metrics in parallel
        all_data = []
        
        async with aiohttp.ClientSession() as session:
            for metric_name, query in metric_queries.items():
                try:
                    params = {
                        'query': query,
                        'start': start_time.isoformat(),
                        'end': end_time.isoformat(),
                        'step': '5m'  # 5-minute resolution
                    }
                    
                    async with session.get(f"{self.prometheus_url}/api/v1/query_range",
                                         params=params) as response:
                        data = await response.json()
                        
                    # Process time series data
                    results = data.get('data', {}).get('result', [])
                    if results:
                        values = results[0].get('values', [])
                        for timestamp, value in values:
                            all_data.append({
                                'timestamp': datetime.fromtimestamp(float(timestamp)),
                                'metric_name': metric_name,
                                'value': float(value) if value != 'NaN' else 0.0
                            })
                            
                except Exception as e:
                    logger.warning(f"Failed to collect metric {metric_name}: {e}")
        
        # Convert to wide format DataFrame
        df = pd.DataFrame(all_data)
        if len(df) > 0:
            df_wide = df.pivot(index='timestamp', columns='metric_name', values='value')
            df_wide = df_wide.fillna(method='forward').fillna(0)
            return df_wide.reset_index()
        
        return pd.DataFrame()
    
    async def train_anomaly_models(self, data: pd.DataFrame):
        """Train anomaly detection models on historical data"""
        if len(data) < 100:  # Minimum data requirement
            logger.warning("Insufficient data for anomaly model training")
            return
        
        try:
            # Prepare features (exclude timestamp)
            feature_cols = [col for col in data.columns if col != 'timestamp']
            X = data[feature_cols].values
            
            # Handle missing values and outliers
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Scale features
            X_standard = self.standard_scaler.fit_transform(X)
            X_robust = self.robust_scaler.fit_transform(X)
            
            # Apply PCA for dimensionality reduction
            X_pca = self.pca.fit_transform(X_standard)
            
            # Train different anomaly detection models
            self.isolation_forest.fit(X_standard)
            self.elliptic_envelope.fit(X_robust)
            
            # DBSCAN doesn't need explicit training
            
            self.models_trained = True
            self.last_training_time = datetime.utcnow()
            
            logger.info(f"Trained anomaly detection models on {len(data)} samples with {len(feature_cols)} features")
            
        except Exception as e:
            logger.error(f"Error training anomaly models: {e}")
    
    async def detect_statistical_anomalies(self, data: pd.DataFrame) -> List[AnomalyDetection]:
        """Detect statistical anomalies using multiple algorithms"""
        if not self.models_trained:
            await self.train_anomaly_models(data)
            
        anomalies = []
        
        try:
            # Prepare current data
            feature_cols = [col for col in data.columns if col != 'timestamp']
            X = data[feature_cols].values[-1:] if len(data) > 0 else np.array([])
            
            if len(X) == 0:
                return anomalies
            
            # Scale current data
            X_standard = self.standard_scaler.transform(X)
            X_robust = self.robust_scaler.transform(X)
            X_pca = self.pca.transform(X_standard)
            
            current_timestamp = data['timestamp'].iloc[-1] if len(data) > 0 else datetime.utcnow()
            current_metrics = dict(zip(feature_cols, X[0]))
            
            # Isolation Forest detection
            if_score = self.isolation_forest.decision_function(X_standard)[0]
            if_anomaly = self.isolation_forest.predict(X_standard)[0] == -1
            
            if if_anomaly:
                anomaly = AnomalyDetection(
                    anomaly_id=str(uuid.uuid4()),
                    type=AnomalyType.STATISTICAL,
                    severity=self._calculate_severity(abs(if_score)),
                    affected_components=['cluster'],
                    confidence_score=min(1.0, abs(if_score)),
                    outlier_score=abs(if_score),
                    description=f"Statistical anomaly detected (Isolation Forest score: {if_score:.4f})",
                    metrics_snapshot=current_metrics,
                    temporal_context={'algorithm': 'isolation_forest', 'score': if_score},
                    timestamp=current_timestamp
                )
                anomalies.append(anomaly)
            
            # Elliptic Envelope detection
            ee_anomaly = self.elliptic_envelope.predict(X_robust)[0] == -1
            
            if ee_anomaly:
                ee_score = self.elliptic_envelope.decision_function(X_robust)[0]
                
                anomaly = AnomalyDetection(
                    anomaly_id=str(uuid.uuid4()),
                    type=AnomalyType.STATISTICAL,
                    severity=self._calculate_severity(abs(ee_score)),
                    affected_components=['cluster'],
                    confidence_score=min(1.0, abs(ee_score) * 2),
                    outlier_score=abs(ee_score),
                    description=f"Multivariate anomaly detected (Elliptic Envelope score: {ee_score:.4f})",
                    metrics_snapshot=current_metrics,
                    temporal_context={'algorithm': 'elliptic_envelope', 'score': ee_score},
                    timestamp=current_timestamp
                )
                anomalies.append(anomaly)
            
            # Update metrics
            for anomaly in anomalies:
                self.anomalies_detected_counter.labels(
                    type=anomaly.type.value, severity=anomaly.severity.value).inc()
                self.anomaly_confidence_gauge.labels(
                    anomaly_type=anomaly.type.value).set(anomaly.confidence_score)
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
        
        return anomalies
    
    async def detect_behavioral_anomalies(self, data: pd.DataFrame) -> List[AnomalyDetection]:
        """Detect behavioral anomalies based on patterns and sequences"""
        anomalies = []
        
        if len(data) < 50:  # Need sufficient history
            return anomalies
        
        try:
            # Analyze scheduling pattern anomalies
            scheduling_anomalies = await self._detect_scheduling_pattern_anomalies(data)
            anomalies.extend(scheduling_anomalies)
            
            # Analyze resource usage pattern anomalies  
            resource_anomalies = await self._detect_resource_pattern_anomalies(data)
            anomalies.extend(resource_anomalies)
            
            # Analyze ML model behavior anomalies
            ml_anomalies = await self._detect_ml_behavior_anomalies(data)
            anomalies.extend(ml_anomalies)
            
        except Exception as e:
            logger.error(f"Error in behavioral anomaly detection: {e}")
        
        return anomalies
    
    async def _detect_scheduling_pattern_anomalies(self, data: pd.DataFrame) -> List[AnomalyDetection]:
        """Detect anomalies in scheduling patterns"""
        anomalies = []
        
        if 'scheduling_request_rate' not in data.columns or 'scheduling_latency_p99' not in data.columns:
            return anomalies
        
        # Analyze request rate patterns
        request_rates = data['scheduling_request_rate'].values
        latencies = data['scheduling_latency_p99'].values
        
        # Detect sudden spikes or drops in request rate
        if len(request_rates) > 10:
            rate_diff = np.diff(request_rates)
            rate_std = np.std(rate_diff)
            
            # Check for sudden changes
            recent_change = rate_diff[-1] if len(rate_diff) > 0 else 0
            
            if abs(recent_change) > 3 * rate_std and rate_std > 0:
                severity = AnomalySeverity.HIGH if abs(recent_change) > 5 * rate_std else AnomalySeverity.MEDIUM
                
                anomaly = AnomalyDetection(
                    anomaly_id=str(uuid.uuid4()),
                    type=AnomalyType.BEHAVIORAL,
                    severity=severity,
                    affected_components=['ml-scheduler'],
                    confidence_score=min(1.0, abs(recent_change) / (5 * rate_std)),
                    outlier_score=abs(recent_change) / rate_std,
                    description=f"Unusual scheduling request rate change: {recent_change:.2f} req/s",
                    metrics_snapshot={'scheduling_request_rate': request_rates[-1], 'rate_change': recent_change},
                    temporal_context={'pattern_type': 'sudden_rate_change', 'std_deviations': abs(recent_change) / rate_std}
                )
                anomalies.append(anomaly)
        
        # Detect latency-throughput anomalies
        if len(request_rates) > 5 and len(latencies) > 5:
            # Expected: higher request rate should correlate with higher latency
            correlation = np.corrcoef(request_rates[-20:], latencies[-20:])[0, 1] if len(request_rates) >= 20 else 0
            
            # Anomaly: high latency with low request rate (unusual)
            current_rate = request_rates[-1]
            current_latency = latencies[-1]
            
            if current_latency > 100 and current_rate < np.percentile(request_rates, 25):
                anomaly = AnomalyDetection(
                    anomaly_id=str(uuid.uuid4()),
                    type=AnomalyType.BEHAVIORAL,
                    severity=AnomalySeverity.HIGH,
                    affected_components=['ml-scheduler', 'ml-models'],
                    confidence_score=0.8,
                    outlier_score=(current_latency - 100) / 100,
                    description=f"High latency ({current_latency:.1f}ms) with low request rate ({current_rate:.2f} req/s)",
                    metrics_snapshot={'latency': current_latency, 'request_rate': current_rate, 'correlation': correlation},
                    temporal_context={'pattern_type': 'latency_throughput_anomaly', 'correlation': correlation}
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_resource_pattern_anomalies(self, data: pd.DataFrame) -> List[AnomalyDetection]:
        """Detect anomalies in resource usage patterns"""
        anomalies = []
        
        resource_metrics = ['cpu_utilization', 'memory_utilization', 'resource_efficiency', 'resource_waste']
        available_metrics = [col for col in resource_metrics if col in data.columns]
        
        if len(available_metrics) < 2:
            return anomalies
        
        # Detect resource imbalance anomalies
        if 'cpu_utilization' in data.columns and 'memory_utilization' in data.columns:
            cpu_util = data['cpu_utilization'].iloc[-1]
            memory_util = data['memory_utilization'].iloc[-1]
            
            # Anomaly: High CPU with very low memory or vice versa
            imbalance_ratio = abs(cpu_util - memory_util) / max(cpu_util, memory_util, 1)
            
            if imbalance_ratio > 0.5 and (cpu_util > 70 or memory_util > 70):
                anomaly = AnomalyDetection(
                    anomaly_id=str(uuid.uuid4()),
                    type=AnomalyType.BEHAVIORAL,
                    severity=AnomalySeverity.MEDIUM,
                    affected_components=['cluster', 'workload-placement'],
                    confidence_score=min(1.0, imbalance_ratio),
                    outlier_score=imbalance_ratio,
                    description=f"Resource imbalance detected: CPU {cpu_util:.1f}%, Memory {memory_util:.1f}%",
                    metrics_snapshot={'cpu_utilization': cpu_util, 'memory_utilization': memory_util},
                    temporal_context={'pattern_type': 'resource_imbalance', 'imbalance_ratio': imbalance_ratio}
                )
                anomalies.append(anomaly)
        
        # Detect efficiency degradation patterns
        if 'resource_efficiency' in data.columns:
            efficiency_values = data['resource_efficiency'].values
            
            if len(efficiency_values) > 10:
                # Check for steady degradation
                recent_trend = np.polyfit(range(len(efficiency_values[-10:])), efficiency_values[-10:], 1)[0]
                
                if recent_trend < -1.0:  # Efficiency dropping by >1% per time period
                    anomaly = AnomalyDetection(
                        anomaly_id=str(uuid.uuid4()),
                        type=AnomalyType.BEHAVIORAL,
                        severity=AnomalySeverity.MEDIUM,
                        affected_components=['ml-scheduler', 'resource-optimization'],
                        confidence_score=min(1.0, abs(recent_trend) / 5.0),
                        outlier_score=abs(recent_trend),
                        description=f"Resource efficiency degradation trend: {recent_trend:.2f}% per period",
                        metrics_snapshot={'resource_efficiency': efficiency_values[-1], 'trend': recent_trend},
                        temporal_context={'pattern_type': 'efficiency_degradation', 'trend_slope': recent_trend}
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_ml_behavior_anomalies(self, data: pd.DataFrame) -> List[AnomalyDetection]:
        """Detect anomalies in ML model behavior"""
        anomalies = []
        
        # Confidence score anomalies
        confidence_metrics = [col for col in data.columns if 'confidence' in col]
        
        for metric in confidence_metrics:
            if metric in data.columns:
                confidence_values = data[metric].values
                
                if len(confidence_values) > 10:
                    current_confidence = confidence_values[-1]
                    historical_mean = np.mean(confidence_values[:-5])  # Exclude recent values
                    historical_std = np.std(confidence_values[:-5])
                    
                    # Detect significant confidence drops
                    if historical_std > 0:
                        z_score = (historical_mean - current_confidence) / historical_std
                        
                        if z_score > 3.0:  # 3 standard deviations below normal
                            model_name = metric.replace('_confidence', '')
                            
                            anomaly = AnomalyDetection(
                                anomaly_id=str(uuid.uuid4()),
                                type=AnomalyType.BEHAVIORAL,
                                severity=AnomalySeverity.HIGH,
                                affected_components=[model_name],
                                confidence_score=min(1.0, z_score / 5.0),
                                outlier_score=z_score,
                                description=f"ML model confidence anomaly: {model_name} confidence dropped to {current_confidence:.3f}",
                                metrics_snapshot={metric: current_confidence, 'historical_mean': historical_mean},
                                temporal_context={'pattern_type': 'confidence_drop', 'z_score': z_score}
                            )
                            anomalies.append(anomaly)
        
        # Model latency anomalies
        latency_metrics = [col for col in data.columns if 'latency' in col and any(model in col for model in ['xgboost', 'qlearning', 'anomaly_detection'])]
        
        for metric in latency_metrics:
            if metric in data.columns:
                latency_values = data[metric].values
                
                if len(latency_values) > 10:
                    current_latency = latency_values[-1]
                    
                    # Detect sudden latency spikes
                    recent_values = latency_values[-5:]
                    historical_p95 = np.percentile(latency_values[:-5], 95)
                    
                    if current_latency > historical_p95 * 2:  # More than 2x historical P95
                        model_name = metric.split('_')[0]
                        
                        anomaly = AnomalyDetection(
                            anomaly_id=str(uuid.uuid4()),
                            type=AnomalyType.BEHAVIORAL,
                            severity=AnomalySeverity.HIGH,
                            affected_components=[model_name],
                            confidence_score=0.9,
                            outlier_score=current_latency / historical_p95,
                            description=f"ML model latency spike: {model_name} latency {current_latency:.1f}ms (historical P95: {historical_p95:.1f}ms)",
                            metrics_snapshot={metric: current_latency, 'historical_p95': historical_p95},
                            temporal_context={'pattern_type': 'latency_spike', 'spike_ratio': current_latency / historical_p95}
                        )
                        anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_contextual_anomalies(self, data: pd.DataFrame) -> List[AnomalyDetection]:
        """Detect contextual anomalies based on time and business context"""
        anomalies = []
        
        if len(data) == 0:
            return anomalies
        
        try:
            current_time = datetime.utcnow()
            current_metrics = {}
            
            # Get current metric values
            for col in data.columns:
                if col != 'timestamp':
                    current_metrics[col] = data[col].iloc[-1] if len(data) > 0 else 0
            
            # Business hours context (8 AM - 6 PM UTC)
            is_business_hours = 8 <= current_time.hour <= 18
            is_weekend = current_time.weekday() >= 5
            
            # CPU utilization context
            if 'cpu_utilization' in current_metrics:
                cpu_util = current_metrics['cpu_utilization']
                
                # Anomaly: Very low CPU during business hours (should be optimized to 65%)
                if is_business_hours and not is_weekend and cpu_util < 45:
                    anomaly = AnomalyDetection(
                        anomaly_id=str(uuid.uuid4()),
                        type=AnomalyType.CONTEXTUAL,
                        severity=AnomalySeverity.MEDIUM,
                        affected_components=['ml-scheduler', 'workload-optimization'],
                        confidence_score=0.8,
                        outlier_score=(65 - cpu_util) / 65,
                        description=f"Unexpectedly low CPU utilization during business hours: {cpu_util:.1f}%",
                        metrics_snapshot=current_metrics,
                        temporal_context={
                            'is_business_hours': is_business_hours,
                            'is_weekend': is_weekend,
                            'expected_range': '60-70%'
                        }
                    )
                    anomalies.append(anomaly)
                
                # Anomaly: Very high CPU during off-hours
                elif (not is_business_hours or is_weekend) and cpu_util > 80:
                    anomaly = AnomalyDetection(
                        anomaly_id=str(uuid.uuid4()),
                        type=AnomalyType.CONTEXTUAL,
                        severity=AnomalySeverity.HIGH,
                        affected_components=['cluster', 'workload-management'],
                        confidence_score=0.9,
                        outlier_score=(cpu_util - 80) / 20,
                        description=f"Unexpectedly high CPU utilization during off-hours: {cpu_util:.1f}%",
                        metrics_snapshot=current_metrics,
                        temporal_context={
                            'is_business_hours': is_business_hours,
                            'is_weekend': is_weekend,
                            'expected_range': '40-60%'
                        }
                    )
                    anomalies.append(anomaly)
            
            # Scheduling rate context
            if 'scheduling_request_rate' in current_metrics:
                request_rate = current_metrics['scheduling_request_rate']
                
                # Expected patterns based on time
                if is_business_hours and not is_weekend:
                    expected_min_rate = 10.0  # Minimum expected during business hours
                else:
                    expected_min_rate = 2.0   # Lower expected during off-hours
                
                if request_rate < expected_min_rate:
                    anomaly = AnomalyDetection(
                        anomaly_id=str(uuid.uuid4()),
                        type=AnomalyType.CONTEXTUAL,
                        severity=AnomalySeverity.MEDIUM,
                        affected_components=['workload-generation', 'applications'],
                        confidence_score=0.7,
                        outlier_score=(expected_min_rate - request_rate) / expected_min_rate,
                        description=f"Unexpectedly low scheduling request rate: {request_rate:.2f} req/s",
                        metrics_snapshot=current_metrics,
                        temporal_context={
                            'expected_min_rate': expected_min_rate,
                            'context': 'business_hours' if is_business_hours else 'off_hours'
                        }
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error in contextual anomaly detection: {e}")
        
        return anomalies
    
    def _calculate_severity(self, outlier_score: float) -> AnomalySeverity:
        """Calculate anomaly severity based on outlier score"""
        if outlier_score > 0.8:
            return AnomalySeverity.CRITICAL
        elif outlier_score > 0.6:
            return AnomalySeverity.HIGH
        elif outlier_score > 0.3:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

class RootCauseAnalyzer:
    """Advanced root cause analysis engine"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        
        # Component dependency graph
        self.dependency_graph = self._build_dependency_graph()
        
        # Metrics
        self.root_cause_analyses_counter = Counter('root_cause_analyses_total',
                                                 'Root cause analyses performed')
        self.rca_confidence_gauge = Gauge('root_cause_analysis_confidence',
                                        'Confidence in root cause analysis',
                                        ['category'])
    
    def _build_dependency_graph(self) -> nx.DiGraph:
        """Build component dependency graph for impact analysis"""
        G = nx.DiGraph()
        
        # Add nodes (components)
        components = [
            'ml-scheduler', 'xgboost-predictor', 'qlearning-optimizer', 
            'anomaly-detector', 'redis', 'kubernetes-api', 'prometheus',
            'nodes', 'network', 'storage'
        ]
        G.add_nodes_from(components)
        
        # Add edges (dependencies)
        dependencies = [
            ('ml-scheduler', 'xgboost-predictor'),
            ('ml-scheduler', 'qlearning-optimizer'),
            ('ml-scheduler', 'anomaly-detector'),
            ('ml-scheduler', 'redis'),
            ('ml-scheduler', 'kubernetes-api'),
            ('xgboost-predictor', 'redis'),
            ('qlearning-optimizer', 'redis'),
            ('anomaly-detector', 'redis'),
            ('xgboost-predictor', 'prometheus'),
            ('qlearning-optimizer', 'prometheus'),
            ('anomaly-detector', 'prometheus'),
            ('kubernetes-api', 'nodes'),
            ('nodes', 'network'),
            ('nodes', 'storage'),
            ('redis', 'nodes'),
            ('prometheus', 'nodes')
        ]
        G.add_edges_from(dependencies)
        
        return G
    
    async def analyze_root_cause(self, anomaly: AnomalyDetection) -> List[RootCauseHypothesis]:
        """Perform comprehensive root cause analysis"""
        self.root_cause_analyses_counter.inc()
        
        hypotheses = []
        
        try:
            # Collect additional context data
            context_data = await self._collect_root_cause_context(anomaly)
            
            # Generate hypotheses based on anomaly type and affected components
            if anomaly.type == AnomalyType.STATISTICAL:
                hypotheses.extend(await self._analyze_statistical_root_causes(anomaly, context_data))
            elif anomaly.type == AnomalyType.BEHAVIORAL:
                hypotheses.extend(await self._analyze_behavioral_root_causes(anomaly, context_data))
            elif anomaly.type == AnomalyType.CONTEXTUAL:
                hypotheses.extend(await self._analyze_contextual_root_causes(anomaly, context_data))
            
            # Add dependency-based analysis
            dependency_hypotheses = await self._analyze_dependency_root_causes(anomaly, context_data)
            hypotheses.extend(dependency_hypotheses)
            
            # Rank hypotheses by confidence
            hypotheses.sort(key=lambda h: h.confidence, reverse=True)
            
            # Update metrics
            for hypothesis in hypotheses:
                self.rca_confidence_gauge.labels(category=hypothesis.category.value).set(hypothesis.confidence)
            
        except Exception as e:
            logger.error(f"Error in root cause analysis: {e}")
        
        return hypotheses[:5]  # Return top 5 hypotheses
    
    async def _collect_root_cause_context(self, anomaly: AnomalyDetection) -> Dict[str, Any]:
        """Collect additional context for root cause analysis"""
        context = {
            'recent_changes': await self._detect_recent_changes(),
            'component_health': await self._assess_component_health(anomaly.affected_components),
            'resource_pressure': await self._assess_resource_pressure(),
            'external_factors': await self._assess_external_factors()
        }
        
        return context
    
    async def _detect_recent_changes(self) -> List[Dict[str, Any]]:
        """Detect recent configuration or deployment changes"""
        changes = []
        
        try:
            # Query for recent deployment changes (last 2 hours)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=2)
            
            # Check for deployment restarts
            restart_query = 'increase(kube_pod_container_status_restarts_total{namespace="ml-scheduler"}[2h])'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.prometheus_url}/api/v1/query",
                                     params={'query': restart_query}) as response:
                    data = await response.json()
                    
                results = data.get('data', {}).get('result', [])
                for result in results:
                    if float(result['value'][1]) > 0:
                        changes.append({
                            'type': 'pod_restart',
                            'component': result['metric'].get('pod', 'unknown'),
                            'count': float(result['value'][1]),
                            'impact': 'medium'
                        })
            
            # Check for configuration changes (would need custom metrics)
            # This would be implemented with ConfigMap change tracking
            
        except Exception as e:
            logger.error(f"Error detecting recent changes: {e}")
        
        return changes
    
    async def _assess_component_health(self, components: List[str]) -> Dict[str, Dict[str, Any]]:
        """Assess health of specific components"""
        health_assessment = {}
        
        for component in components:
            try:
                health_info = await self._get_component_health(component)
                health_assessment[component] = health_info
            except Exception as e:
                logger.error(f"Error assessing health for {component}: {e}")
                health_assessment[component] = {'status': 'unknown', 'error': str(e)}
        
        return health_assessment
    
    async def _get_component_health(self, component: str) -> Dict[str, Any]:
        """Get detailed health information for a component"""
        health_queries = {
            'ml-scheduler': {
                'up': 'up{job="ml-scheduler"}',
                'cpu_usage': 'rate(container_cpu_usage_seconds_total{pod=~"ml-scheduler-.*"}[5m]) * 100',
                'memory_usage': 'container_memory_working_set_bytes{pod=~"ml-scheduler-.*"} / container_spec_memory_limit_bytes{pod=~"ml-scheduler-.*"} * 100',
                'error_rate': 'rate(ml_scheduler_errors_total[5m])'
            },
            'redis': {
                'up': 'up{job="redis-exporter"}',
                'memory_usage': 'redis_memory_used_bytes / redis_memory_max_bytes * 100',
                'hit_rate': 'redis_keyspace_hits / (redis_keyspace_hits + redis_keyspace_misses) * 100',
                'connected_clients': 'redis_connected_clients'
            }
        }
        
        component_queries = health_queries.get(component, {'up': f'up{{job="{component}"}}'})
        health_data = {}
        
        async with aiohttp.ClientSession() as session:
            for metric_name, query in component_queries.items():
                try:
                    async with session.get(f"{self.prometheus_url}/api/v1/query",
                                         params={'query': query}) as response:
                        data = await response.json()
                        
                    results = data.get('data', {}).get('result', [])
                    if results:
                        health_data[metric_name] = float(results[0]['value'][1])
                    else:
                        health_data[metric_name] = 0.0
                        
                except Exception as e:
                    logger.warning(f"Failed to get {metric_name} for {component}: {e}")
                    health_data[metric_name] = 0.0
        
        # Determine overall health status
        if health_data.get('up', 0) == 0:
            status = 'down'
        elif any(v > 90 for k, v in health_data.items() if 'usage' in k):
            status = 'degraded'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'metrics': health_data,
            'assessment_time': datetime.utcnow().isoformat()
        }
    
    async def _assess_resource_pressure(self) -> Dict[str, float]:
        """Assess cluster-wide resource pressure"""
        pressure_queries = {
            'cpu_pressure': 'avg(rate(node_pressure_cpu_waiting_seconds_total[5m]))',
            'memory_pressure': 'avg(rate(node_pressure_memory_waiting_seconds_total[5m]))',
            'io_pressure': 'avg(rate(node_pressure_io_waiting_seconds_total[5m]))',
            'pod_pressure': 'sum(kube_pod_info) / sum(kube_node_status_allocatable{resource="pods"}) * 100'
        }
        
        pressure_data = {}
        
        async with aiohttp.ClientSession() as session:
            for pressure_type, query in pressure_queries.items():
                try:
                    async with session.get(f"{self.prometheus_url}/api/v1/query",
                                         params={'query': query}) as response:
                        data = await response.json()
                        
                    results = data.get('data', {}).get('result', [])
                    pressure_data[pressure_type] = float(results[0]['value'][1]) if results else 0.0
                    
                except Exception as e:
                    logger.warning(f"Failed to assess {pressure_type}: {e}")
                    pressure_data[pressure_type] = 0.0
        
        return pressure_data
    
    async def _assess_external_factors(self) -> Dict[str, Any]:
        """Assess external factors that might influence performance"""
        # This would integrate with external monitoring systems
        # For now, provide basic infrastructure assessment
        
        external_factors = {
            'network_latency': await self._check_network_latency(),
            'storage_latency': await self._check_storage_latency(),
            'dns_resolution': await self._check_dns_health(),
            'time_sync': await self._check_time_synchronization()
        }
        
        return external_factors
    
    async def _check_network_latency(self) -> Dict[str, float]:
        """Check network latency between components"""
        # Query network latency metrics
        latency_query = 'histogram_quantile(0.95, rate(probe_duration_seconds_bucket{job="blackbox"}[5m])) * 1000'
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.prometheus_url}/api/v1/query",
                                     params={'query': latency_query}) as response:
                    data = await response.json()
                    
            results = data.get('data', {}).get('result', [])
            return {'network_latency_p95_ms': float(results[0]['value'][1])} if results else {'network_latency_p95_ms': 0.0}
            
        except Exception as e:
            logger.warning(f"Failed to check network latency: {e}")
            return {'network_latency_p95_ms': 0.0}
    
    async def _check_storage_latency(self) -> Dict[str, float]:
        """Check storage I/O latency"""
        storage_query = 'histogram_quantile(0.95, rate(node_disk_io_time_seconds_total[5m])) * 1000'
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.prometheus_url}/api/v1/query",
                                     params={'query': storage_query}) as response:
                    data = await response.json()
                    
            results = data.get('data', {}).get('result', [])
            return {'storage_latency_p95_ms': float(results[0]['value'][1])} if results else {'storage_latency_p95_ms': 0.0}
            
        except Exception as e:
            logger.warning(f"Failed to check storage latency: {e}")
            return {'storage_latency_p95_ms': 0.0}
    
    async def _check_dns_health(self) -> Dict[str, Any]:
        """Check DNS resolution health"""
        # This would check DNS resolution times
        return {'dns_resolution_ms': 5.0, 'dns_errors': 0}
    
    async def _check_time_synchronization(self) -> Dict[str, Any]:
        """Check time synchronization across nodes"""
        # This would check NTP synchronization
        return {'time_drift_ms': 10.0, 'ntp_sync_status': 'healthy'}
    
    async def _analyze_statistical_root_causes(self, anomaly: AnomalyDetection,
                                             context: Dict[str, Any]) -> List[RootCauseHypothesis]:
        """Analyze root causes for statistical anomalies"""
        hypotheses = []
        
        # Resource exhaustion hypothesis
        resource_pressure = context['resource_pressure']
        
        if any(pressure > 0.1 for pressure in resource_pressure.values()):
            contributing_factors = [
                f"{pressure_type}: {pressure:.3f}" 
                for pressure_type, pressure in resource_pressure.items() 
                if pressure > 0.1
            ]
            
            hypothesis = RootCauseHypothesis(
                hypothesis_id=str(uuid.uuid4()),
                category=RootCauseCategory.RESOURCE_EXHAUSTION,
                confidence=0.8,
                contributing_factors=contributing_factors,
                evidence={'resource_pressure': resource_pressure},
                recommended_investigation=[
                    'Check node resource availability',
                    'Analyze pod resource requests vs limits',
                    'Review resource quota configurations'
                ],
                estimated_impact={'performance_degradation': 0.3, 'cost_increase': 0.1}
            )
            hypotheses.append(hypothesis)
        
        # Service degradation hypothesis
        component_health = context['component_health']
        degraded_components = [
            comp for comp, health in component_health.items()
            if health['status'] in ['degraded', 'down']
        ]
        
        if degraded_components:
            hypothesis = RootCauseHypothesis(
                hypothesis_id=str(uuid.uuid4()),
                category=RootCauseCategory.SERVICE_DEGRADATION,
                confidence=0.9,
                contributing_factors=[f"Degraded service: {comp}" for comp in degraded_components],
                evidence={'degraded_components': degraded_components, 'health_data': component_health},
                recommended_investigation=[
                    'Check service logs for errors',
                    'Verify service dependencies',
                    'Review recent deployments'
                ],
                estimated_impact={'availability_impact': 0.4, 'performance_degradation': 0.5}
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _analyze_behavioral_root_causes(self, anomaly: AnomalyDetection,
                                            context: Dict[str, Any]) -> List[RootCauseHypothesis]:
        """Analyze root causes for behavioral anomalies"""
        hypotheses = []
        
        # Workload spike hypothesis
        if 'scheduling_request_rate' in anomaly.metrics_snapshot:
            current_rate = anomaly.metrics_snapshot['scheduling_request_rate']
            
            # Check if this represents a workload spike
            if current_rate > 20:  # High request rate
                hypothesis = RootCauseHypothesis(
                    hypothesis_id=str(uuid.uuid4()),
                    category=RootCauseCategory.WORKLOAD_SPIKE,
                    confidence=0.7,
                    contributing_factors=[f"High scheduling request rate: {current_rate:.2f} req/s"],
                    evidence={'request_rate': current_rate, 'threshold': 20},
                    recommended_investigation=[
                        'Identify source of increased workload',
                        'Check for batch job submissions',
                        'Review application scaling policies'
                    ],
                    estimated_impact={'resource_pressure': 0.6, 'latency_increase': 0.4}
                )
                hypotheses.append(hypothesis)
        
        # Model degradation hypothesis
        confidence_metrics = [k for k in anomaly.metrics_snapshot.keys() if 'confidence' in k]
        
        for conf_metric in confidence_metrics:
            confidence_value = anomaly.metrics_snapshot[conf_metric]
            
            if confidence_value < 0.7:  # Low confidence threshold
                model_name = conf_metric.replace('_confidence', '')
                
                hypothesis = RootCauseHypothesis(
                    hypothesis_id=str(uuid.uuid4()),
                    category=RootCauseCategory.MODEL_DEGRADATION,
                    confidence=0.8,
                    contributing_factors=[f"Low {model_name} confidence: {confidence_value:.3f}"],
                    evidence={'confidence_score': confidence_value, 'model': model_name},
                    recommended_investigation=[
                        f'Check {model_name} model health',
                        'Review recent model updates',
                        'Analyze model input data quality'
                    ],
                    estimated_impact={'scheduling_accuracy': 0.5, 'fallback_rate_increase': 0.3}
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _analyze_contextual_root_causes(self, anomaly: AnomalyDetection,
                                            context: Dict[str, Any]) -> List[RootCauseHypothesis]:
        """Analyze root causes for contextual anomalies"""
        hypotheses = []
        
        # Configuration drift hypothesis
        if anomaly.temporal_context.get('is_business_hours', False):
            # During business hours, certain patterns are expected
            
            hypothesis = RootCauseHypothesis(
                hypothesis_id=str(uuid.uuid4()),
                category=RootCauseCategory.CONFIGURATION_DRIFT,
                confidence=0.6,
                contributing_factors=['Unexpected behavior during business hours'],
                evidence={'temporal_context': anomaly.temporal_context},
                recommended_investigation=[
                    'Compare current configuration with baseline',
                    'Check for recent configuration changes',
                    'Verify business hour scheduling policies'
                ],
                estimated_impact={'business_target_miss': 0.4}
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _analyze_dependency_root_causes(self, anomaly: AnomalyDetection,
                                            context: Dict[str, Any]) -> List[RootCauseHypothesis]:
        """Analyze root causes based on component dependencies"""
        hypotheses = []
        
        # Use dependency graph to find potential upstream causes
        affected_components = set(anomaly.affected_components)
        
        # Find upstream dependencies that could cause this anomaly
        upstream_components = set()
        for component in affected_components:
            if component in self.dependency_graph:
                predecessors = set(self.dependency_graph.predecessors(component))
                upstream_components.update(predecessors)
        
        # Check health of upstream components
        component_health = context['component_health']
        
        for upstream_comp in upstream_components:
            if upstream_comp in component_health:
                health = component_health[upstream_comp]
                
                if health['status'] != 'healthy':
                    hypothesis = RootCauseHypothesis(
                        hypothesis_id=str(uuid.uuid4()),
                        category=RootCauseCategory.CASCADING_FAILURE,
                        confidence=0.7,
                        contributing_factors=[f"Upstream component issues: {upstream_comp}"],
                        evidence={'upstream_component': upstream_comp, 'health_status': health},
                        recommended_investigation=[
                            f'Investigate {upstream_comp} health issues',
                            'Check dependency chain stability',
                            'Review circuit breaker configurations'
                        ],
                        estimated_impact={'cascading_failure_risk': 0.8}
                    )
                    hypotheses.append(hypothesis)
        
        return hypotheses

class AnomalyCorrelationEngine:
    """Correlates anomalies across time and components"""
    
    def __init__(self):
        self.anomaly_history: List[AnomalyDetection] = []
        self.correlation_window_hours = 2
        
        # Metrics
        self.correlations_found_counter = Counter('anomaly_correlations_found_total',
                                                'Anomaly correlations identified')
    
    def add_anomaly(self, anomaly: AnomalyDetection):
        """Add anomaly to correlation tracking"""
        self.anomaly_history.append(anomaly)
        
        # Clean up old anomalies
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.anomaly_history = [
            a for a in self.anomaly_history 
            if a.timestamp >= cutoff_time
        ]
    
    def find_correlated_anomalies(self, target_anomaly: AnomalyDetection) -> List[AnomalyDetection]:
        """Find anomalies correlated with the target anomaly"""
        correlations = []
        
        # Time-based correlation
        time_window = timedelta(hours=self.correlation_window_hours)
        time_correlated = [
            anomaly for anomaly in self.anomaly_history
            if abs((anomaly.timestamp - target_anomaly.timestamp).total_seconds()) <= time_window.total_seconds()
            and anomaly.anomaly_id != target_anomaly.anomaly_id
        ]
        
        # Component-based correlation
        target_components = set(target_anomaly.affected_components)
        
        for anomaly in time_correlated:
            anomaly_components = set(anomaly.affected_components)
            
            # Direct component overlap
            if target_components & anomaly_components:
                correlations.append(anomaly)
                continue
            
            # Related component correlation (e.g., scheduler <-> ML models)
            related_patterns = [
                ({'ml-scheduler'}, {'xgboost-predictor', 'qlearning-optimizer', 'anomaly-detector'}),
                ({'redis'}, {'ml-scheduler', 'xgboost-predictor'}),
                ({'nodes'}, {'cluster', 'kubernetes-api'})
            ]
            
            for primary_set, related_set in related_patterns:
                if (target_components & primary_set and anomaly_components & related_set) or \
                   (target_components & related_set and anomaly_components & primary_set):
                    correlations.append(anomaly)
                    break
        
        if correlations:
            self.correlations_found_counter.inc()
        
        return correlations

class AdvancedAnomalyDetectionService:
    """Main advanced anomaly detection service"""
    
    def __init__(self, config_path: str, prometheus_url: str):
        self.config = self._load_config(config_path)
        self.prometheus_url = prometheus_url
        
        # Initialize components
        self.anomaly_detector = MultiDimensionalAnomalyDetector(prometheus_url)
        self.root_cause_analyzer = RootCauseAnalyzer(prometheus_url)
        self.correlation_engine = AnomalyCorrelationEngine()
        
        # Service state
        self.running = False
        
        # Service metrics
        self.detection_cycles_counter = Counter('advanced_anomaly_detection_cycles_total',
                                              'Advanced anomaly detection cycles completed')
        self.investigations_counter = Counter('anomaly_investigations_total',
                                            'Anomaly investigations performed')
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load advanced anomaly detection configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'detection_settings': {
                    'check_interval_minutes': 5,
                    'lookback_hours': 24,
                    'min_confidence_threshold': 0.7,
                    'enable_all_detection_types': True
                },
                'root_cause_analysis': {
                    'enabled': True,
                    'max_hypotheses': 5,
                    'min_confidence_threshold': 0.5
                }
            }
    
    async def start_monitoring(self):
        """Start continuous advanced anomaly monitoring"""
        self.running = True
        logger.info("Starting advanced anomaly detection service")
        
        # Initial model training
        logger.info("Training initial anomaly detection models...")
        initial_data = await self.anomaly_detector.collect_multi_dimensional_data(
            lookback_hours=self.config['detection_settings']['lookback_hours'])
        
        if len(initial_data) > 0:
            await self.anomaly_detector.train_anomaly_models(initial_data)
        
        # Main monitoring loop
        while self.running:
            try:
                await self._detection_cycle()
                self.detection_cycles_counter.inc()
                
                # Wait before next cycle
                interval_minutes = self.config['detection_settings']['check_interval_minutes']
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in anomaly detection cycle: {e}")
                await asyncio.sleep(60)  # Shorter retry interval
    
    async def _detection_cycle(self):
        """Execute one complete anomaly detection cycle"""
        with self.anomaly_detector.detection_duration.labels(detection_type='full_cycle').time():
            # Collect current data
            data = await self.anomaly_detector.collect_multi_dimensional_data(lookback_hours=2)
            
            if len(data) == 0:
                logger.warning("No data collected for anomaly detection")
                return
            
            # Detect different types of anomalies
            all_anomalies = []
            
            if self.config['detection_settings']['enable_all_detection_types']:
                # Statistical anomalies
                statistical_anomalies = await self.anomaly_detector.detect_statistical_anomalies(data)
                all_anomalies.extend(statistical_anomalies)
                
                # Behavioral anomalies
                behavioral_anomalies = await self.anomaly_detector.detect_behavioral_anomalies(data)
                all_anomalies.extend(behavioral_anomalies)
                
                # Contextual anomalies
                contextual_anomalies = await self.anomaly_detector.detect_contextual_anomalies(data)
                all_anomalies.extend(contextual_anomalies)
            
            # Process detected anomalies
            for anomaly in all_anomalies:
                await self._process_anomaly(anomaly)
    
    async def _process_anomaly(self, anomaly: AnomalyDetection):
        """Process a detected anomaly with full investigation"""
        logger.warning(f"Processing anomaly: {anomaly.description}")
        
        # Add to correlation tracking
        self.correlation_engine.add_anomaly(anomaly)
        
        # Find correlated anomalies
        correlated_anomalies = self.correlation_engine.find_correlated_anomalies(anomaly)
        
        # Perform root cause analysis if confidence threshold met
        min_confidence = self.config['root_cause_analysis']['min_confidence_threshold']
        
        if (anomaly.confidence_score >= min_confidence and 
            self.config['root_cause_analysis']['enabled']):
            
            investigation = await self._perform_full_investigation(anomaly, correlated_anomalies)
            await self._report_investigation(investigation)
            
            self.investigations_counter.inc()
    
    async def _perform_full_investigation(self, primary_anomaly: AnomalyDetection,
                                        correlated_anomalies: List[AnomalyDetection]) -> AnomalyInvestigation:
        """Perform comprehensive anomaly investigation"""
        # Root cause analysis
        root_cause_hypotheses = await self.root_cause_analyzer.analyze_root_cause(primary_anomaly)
        
        # Correlation analysis
        correlation_analysis = {
            'correlated_count': len(correlated_anomalies),
            'correlation_timespan_minutes': self._calculate_correlation_timespan(correlated_anomalies),
            'affected_component_overlap': self._calculate_component_overlap(primary_anomaly, correlated_anomalies)
        }
        
        # Temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(primary_anomaly, correlated_anomalies)
        
        # Impact assessment
        impact_assessment = await self._assess_business_impact(primary_anomaly, root_cause_hypotheses)
        
        # Recommended actions
        recommended_actions = self._generate_investigation_actions(
            primary_anomaly, root_cause_hypotheses, impact_assessment)
        
        return AnomalyInvestigation(
            anomaly=primary_anomaly,
            root_cause_hypotheses=root_cause_hypotheses,
            correlation_analysis=correlation_analysis,
            temporal_analysis=temporal_analysis,
            impact_assessment=impact_assessment,
            recommended_actions=recommended_actions
        )
    
    def _calculate_correlation_timespan(self, anomalies: List[AnomalyDetection]) -> float:
        """Calculate timespan of correlated anomalies"""
        if len(anomalies) < 2:
            return 0.0
        
        timestamps = [a.timestamp for a in anomalies]
        timespan = (max(timestamps) - min(timestamps)).total_seconds() / 60  # Minutes
        
        return timespan
    
    def _calculate_component_overlap(self, primary: AnomalyDetection,
                                   correlated: List[AnomalyDetection]) -> float:
        """Calculate component overlap percentage"""
        if not correlated:
            return 0.0
        
        primary_components = set(primary.affected_components)
        
        overlaps = []
        for anomaly in correlated:
            anomaly_components = set(anomaly.affected_components)
            overlap = len(primary_components & anomaly_components)
            total = len(primary_components | anomaly_components)
            overlaps.append(overlap / total if total > 0 else 0.0)
        
        return sum(overlaps) / len(overlaps)
    
    def _analyze_temporal_patterns(self, primary: AnomalyDetection,
                                 correlated: List[AnomalyDetection]) -> Dict[str, Any]:
        """Analyze temporal patterns in anomaly occurrence"""
        all_anomalies = [primary] + correlated
        
        # Group by hour of day
        hour_distribution = {}
        for anomaly in all_anomalies:
            hour = anomaly.timestamp.hour
            hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
        
        # Identify peak anomaly hours
        peak_hours = sorted(hour_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'total_anomalies': len(all_anomalies),
            'timespan_minutes': self._calculate_correlation_timespan(correlated),
            'peak_anomaly_hours': [hour for hour, count in peak_hours],
            'hour_distribution': hour_distribution,
            'is_clustered_in_time': self._is_temporally_clustered(all_anomalies)
        }
    
    def _is_temporally_clustered(self, anomalies: List[AnomalyDetection]) -> bool:
        """Determine if anomalies are temporally clustered"""
        if len(anomalies) < 3:
            return False
        
        timestamps = [a.timestamp for a in anomalies]
        timestamps.sort()
        
        # Check if most anomalies occur within a short time window
        total_span = (timestamps[-1] - timestamps[0]).total_seconds() / 60  # Minutes
        
        # If 80% of anomalies occur within 25% of the total timespan, consider clustered
        cluster_window = total_span * 0.25
        
        clustered_count = 0
        for i in range(len(timestamps)):
            window_start = timestamps[i]
            window_end = window_start + timedelta(minutes=cluster_window)
            
            count_in_window = sum(1 for ts in timestamps if window_start <= ts <= window_end)
            if count_in_window >= len(timestamps) * 0.8:
                return True
        
        return False
    
    async def _assess_business_impact(self, anomaly: AnomalyDetection,
                                    hypotheses: List[RootCauseHypothesis]) -> Dict[str, Any]:
        """Assess business impact of anomaly"""
        impact = {
            'revenue_impact_hourly': 0.0,
            'cost_impact_monthly': 0.0,
            'sla_risk': 'low',
            'customer_impact': 'minimal',
            'operational_impact': 'low'
        }
        
        # Calculate revenue impact based on affected components
        if 'ml-scheduler' in anomaly.affected_components:
            # Scheduler issues directly impact scheduling efficiency
            if anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
                impact['revenue_impact_hourly'] = 2000  # $2k/hour for scheduler issues
                impact['sla_risk'] = 'high'
                impact['operational_impact'] = 'high'
        
        # Calculate cost impact from hypotheses
        for hypothesis in hypotheses:
            if hypothesis.category == RootCauseCategory.RESOURCE_EXHAUSTION:
                impact['cost_impact_monthly'] += 5000  # $5k/month for resource issues
            elif hypothesis.category == RootCauseCategory.SERVICE_DEGRADATION:
                impact['revenue_impact_hourly'] += 1000  # $1k/hour for service issues
        
        # SLA impact assessment
        if 'availability' in anomaly.metrics_snapshot:
            availability = anomaly.metrics_snapshot['availability']
            if availability < 99.5:
                impact['sla_risk'] = 'critical'
                impact['revenue_impact_hourly'] = max(impact['revenue_impact_hourly'], 5000)
        
        return impact
    
    def _generate_investigation_actions(self, anomaly: AnomalyDetection,
                                      hypotheses: List[RootCauseHypothesis],
                                      impact: Dict[str, Any]) -> List[str]:
        """Generate recommended investigation actions"""
        actions = []
        
        # High-priority actions based on severity
        if anomaly.severity == AnomalySeverity.CRITICAL:
            actions.append("IMMEDIATE: Activate incident response team")
            actions.append("IMMEDIATE: Consider activating fallback systems")
        
        # Actions based on top root cause hypothesis
        if hypotheses:
            top_hypothesis = hypotheses[0]
            actions.extend(top_hypothesis.recommended_investigation)
        
        # Business impact driven actions
        if impact['revenue_impact_hourly'] > 3000:
            actions.append("HIGH PRIORITY: Notify business stakeholders of potential revenue impact")
        
        if impact['sla_risk'] == 'critical':
            actions.append("URGENT: Implement SLA protection measures")
        
        # Component-specific actions
        for component in anomaly.affected_components:
            if component == 'ml-scheduler':
                actions.append(f"Check ML scheduler logs and configuration")
            elif component in ['xgboost-predictor', 'qlearning-optimizer', 'anomaly-detector']:
                actions.append(f"Validate {component} model health and performance")
            elif component == 'redis':
                actions.append("Check Redis cache health and memory usage")
        
        return list(set(actions))  # Remove duplicates
    
    async def _report_investigation(self, investigation: AnomalyInvestigation):
        """Report investigation results"""
        # Log investigation summary
        logger.info(f"Anomaly Investigation Complete: {investigation.anomaly.anomaly_id}")
        logger.info(f"Root cause hypotheses: {len(investigation.root_cause_hypotheses)}")
        logger.info(f"Business impact: ${investigation.impact_assessment['revenue_impact_hourly']}/hour")
        
        # Create investigation report
        report = {
            'investigation_id': investigation.anomaly.anomaly_id,
            'timestamp': investigation.anomaly.timestamp.isoformat(),
            'anomaly_summary': {
                'type': investigation.anomaly.type.value,
                'severity': investigation.anomaly.severity.value,
                'confidence': investigation.anomaly.confidence_score,
                'description': investigation.anomaly.description
            },
            'root_causes': [
                {
                    'category': h.category.value,
                    'confidence': h.confidence,
                    'factors': h.contributing_factors
                } for h in investigation.root_cause_hypotheses
            ],
            'business_impact': investigation.impact_assessment,
            'recommended_actions': investigation.recommended_actions,
            'correlation_summary': investigation.correlation_analysis
        }
        
        # Send to monitoring/alerting system
        await self._send_investigation_alert(report)
    
    async def _send_investigation_alert(self, report: Dict[str, Any]):
        """Send investigation report as alert"""
        alert_payload = {
            'alert_name': 'AdvancedAnomalyInvestigation',
            'severity': report['anomaly_summary']['severity'],
            'message': f"Advanced anomaly investigation completed: {report['anomaly_summary']['description']}",
            'investigation_id': report['investigation_id'],
            'business_impact': report['business_impact'],
            'recommended_actions': report['recommended_actions'],
            'timestamp': report['timestamp'],
            'labels': {
                'component': 'advanced_anomaly_detection',
                'anomaly_type': report['anomaly_summary']['type'],
                'severity': report['anomaly_summary']['severity']
            }
        }
        
        try:
            # Send to Alertmanager
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://alertmanager:9093/api/v1/alerts",
                    json=[alert_payload]
                ) as response:
                    if response.status == 200:
                        logger.info(f"Investigation alert sent: {report['investigation_id']}")
                    else:
                        logger.error(f"Failed to send investigation alert: {response.status}")
        except Exception as e:
            logger.error(f"Error sending investigation alert: {e}")
    
    def stop_monitoring(self):
        """Stop anomaly detection service"""
        self.running = False
        logger.info("Stopping advanced anomaly detection service")

async def main():
    """Main entry point for advanced anomaly detection service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Anomaly Detection Service')
    parser.add_argument('--config', default='/etc/ml-scheduler/anomaly_detection_config.yaml',
                       help='Anomaly detection configuration file')
    parser.add_argument('--prometheus-url', default='http://prometheus:9090',
                       help='Prometheus server URL')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (single detection cycle)')
    
    args = parser.parse_args()
    
    # Initialize service
    service = AdvancedAnomalyDetectionService(args.config, args.prometheus_url)
    
    if args.test_mode:
        # Run single detection cycle for testing
        logger.info("Running single anomaly detection cycle")
        await service._detection_cycle()
        logger.info("Test cycle completed")
    else:
        # Start continuous monitoring
        try:
            await service.start_monitoring()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            service.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())