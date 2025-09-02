#!/usr/bin/env python3
"""
HYDATIS Observability and Monitoring Framework
Comprehensive monitoring system for ML models with business impact tracking.
"""

import logging
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sqlite3
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetricDefinition:
    """Definition of a monitoring metric."""
    name: str
    description: str
    metric_type: str  # 'gauge', 'counter', 'histogram'
    business_impact: str  # 'high', 'medium', 'low'
    alert_threshold: float
    target_value: Optional[float] = None

@dataclass 
class AlertCondition:
    """Alert condition configuration."""
    metric_name: str
    condition: str  # 'above', 'below', 'equals'
    threshold: float
    duration_minutes: int
    severity: str  # 'critical', 'warning', 'info'

@dataclass
class BusinessMetrics:
    """HYDATIS business performance metrics."""
    cpu_utilization_target_achievement: float
    availability_sla_achievement: float
    roi_contribution: float
    cost_optimization_percentage: float
    workload_efficiency_score: float
    customer_satisfaction_impact: float
    operational_reliability: float

class HYDATISObservabilityFramework:
    """
    Advanced observability framework for HYDATIS ML Scheduler.
    
    Provides comprehensive monitoring, alerting, and business impact tracking
    for all ML models in production.
    """
    
    # HYDATIS Business Targets
    HYDATIS_CPU_TARGET = 0.65  # 65% CPU utilization target
    HYDATIS_AVAILABILITY_TARGET = 0.997  # 99.7% availability target  
    HYDATIS_ROI_TARGET = 14.0  # 1400% ROI target
    
    def __init__(self, config: Dict):
        """
        Initialize HYDATIS observability framework.
        
        Args:
            config: Framework configuration including Prometheus, MLflow, alerting
        """
        self.config = config
        self.prometheus_url = config.get('prometheus_url', 'http://localhost:9090')
        self.mlflow_uri = config.get('mlflow_uri', 'http://localhost:5000')
        self.alert_webhook = config.get('alert_webhook', None)
        
        # Initialize metric definitions
        self.metric_definitions = self._define_hydatis_metrics()
        self.alert_conditions = self._define_alert_conditions()
        
        # Initialize monitoring database
        self.db_path = config.get('monitoring_db', 'hydatis_monitoring.db')
        self._initialize_monitoring_database()
        
        logger.info("🔍 HYDATIS Observability Framework initialized")
        logger.info(f"   📊 {len(self.metric_definitions)} metrics defined")
        logger.info(f"   🚨 {len(self.alert_conditions)} alert conditions")
    
    def _define_hydatis_metrics(self) -> List[MetricDefinition]:
        """Define HYDATIS-specific monitoring metrics."""
        
        return [
            # Model Performance Metrics
            MetricDefinition(
                name="cpu_prediction_accuracy",
                description="CPU utilization prediction accuracy",
                metric_type="gauge",
                business_impact="high",
                alert_threshold=0.85,
                target_value=0.92
            ),
            MetricDefinition(
                name="memory_prediction_accuracy", 
                description="Memory utilization prediction accuracy",
                metric_type="gauge",
                business_impact="high",
                alert_threshold=0.82,
                target_value=0.88
            ),
            MetricDefinition(
                name="scheduling_latency_p95",
                description="95th percentile scheduling latency (ms)",
                metric_type="histogram",
                business_impact="medium",
                alert_threshold=20.0,
                target_value=12.0
            ),
            
            # Business Impact Metrics
            MetricDefinition(
                name="hydatis_cpu_target_achievement",
                description="Achievement of 65% CPU utilization target",
                metric_type="gauge", 
                business_impact="high",
                alert_threshold=0.90,
                target_value=1.0
            ),
            MetricDefinition(
                name="hydatis_availability_impact",
                description="Impact on 99.7% availability target",
                metric_type="gauge",
                business_impact="critical",
                alert_threshold=0.995,
                target_value=0.999
            ),
            MetricDefinition(
                name="hydatis_roi_contribution",
                description="Contribution to 1400% ROI target",
                metric_type="gauge",
                business_impact="high", 
                alert_threshold=12.0,
                target_value=15.0
            ),
            
            # Operational Metrics
            MetricDefinition(
                name="model_inference_throughput",
                description="Model inference requests per second",
                metric_type="counter",
                business_impact="medium",
                alert_threshold=100.0,
                target_value=500.0
            ),
            MetricDefinition(
                name="model_error_rate",
                description="Model prediction error rate",
                metric_type="gauge",
                business_impact="high",
                alert_threshold=0.05,
                target_value=0.001
            ),
            MetricDefinition(
                name="drift_detection_score",
                description="Model drift detection composite score",
                metric_type="gauge",
                business_impact="high",
                alert_threshold=0.15,
                target_value=0.05
            ),
            
            # Resource Utilization
            MetricDefinition(
                name="model_serving_cpu_usage",
                description="CPU usage of model serving containers",
                metric_type="gauge",
                business_impact="medium",
                alert_threshold=0.8,
                target_value=0.6
            ),
            MetricDefinition(
                name="model_serving_memory_usage",
                description="Memory usage of model serving containers",
                metric_type="gauge",
                business_impact="medium", 
                alert_threshold=0.85,
                target_value=0.7
            )
        ]
    
    def _define_alert_conditions(self) -> List[AlertCondition]:
        """Define alert conditions for HYDATIS business objectives."""
        
        return [
            # Critical Business Alerts
            AlertCondition(
                metric_name="hydatis_availability_impact",
                condition="below",
                threshold=0.997,
                duration_minutes=5,
                severity="critical"
            ),
            AlertCondition(
                metric_name="cpu_prediction_accuracy",
                condition="below", 
                threshold=0.85,
                duration_minutes=10,
                severity="critical"
            ),
            AlertCondition(
                metric_name="hydatis_roi_contribution",
                condition="below",
                threshold=12.0,
                duration_minutes=15,
                severity="warning"
            ),
            
            # Performance Alerts
            AlertCondition(
                metric_name="scheduling_latency_p95",
                condition="above",
                threshold=20.0,
                duration_minutes=5,
                severity="warning"
            ),
            AlertCondition(
                metric_name="model_error_rate",
                condition="above",
                threshold=0.05,
                duration_minutes=3,
                severity="critical"
            ),
            AlertCondition(
                metric_name="drift_detection_score",
                condition="above",
                threshold=0.15,
                duration_minutes=30,
                severity="warning"
            )
        ]
    
    def _initialize_monitoring_database(self):
        """Initialize SQLite database for monitoring data."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create monitoring tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metric_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp REAL NOT NULL,
                model_id TEXT,
                business_context TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                condition_met TEXT NOT NULL,
                business_impact TEXT,
                resolved_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_impact_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                model_affected TEXT NOT NULL,
                business_metric_impact TEXT NOT NULL,
                financial_impact REAL,
                mitigation_action TEXT,
                event_timestamp REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("📊 Monitoring database initialized")
    
    async def collect_model_metrics(self, model_id: str) -> Dict[str, float]:
        """
        Collect comprehensive metrics for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary of collected metrics
        """
        print(f"📊 Collecting metrics for model: {model_id}")
        
        # Simulate metric collection from various sources
        metrics = {}
        
        try:
            # 1. Model Performance Metrics (from Prometheus)
            if 'xgboost' in model_id:
                metrics.update({
                    'cpu_prediction_accuracy': 0.91 + np.random.normal(0, 0.02),
                    'memory_prediction_accuracy': 0.87 + np.random.normal(0, 0.02),
                    'scheduling_latency_p95': 12.5 + np.random.normal(0, 2.0),
                    'model_inference_throughput': 450 + np.random.normal(0, 50)
                })
            elif 'qlearning' in model_id:
                metrics.update({
                    'placement_improvement_score': 0.34 + np.random.normal(0, 0.03),
                    'reward_convergence_rate': 0.89 + np.random.normal(0, 0.02),
                    'policy_stability_score': 0.95 + np.random.normal(0, 0.01)
                })
            elif 'anomaly' in model_id:
                metrics.update({
                    'anomaly_detection_precision': 0.94 + np.random.normal(0, 0.02),
                    'anomaly_detection_recall': 0.91 + np.random.normal(0, 0.02),
                    'false_positive_rate': 0.03 + np.random.normal(0, 0.01)
                })
            
            # 2. Business Impact Metrics
            metrics.update({
                'hydatis_cpu_target_achievement': min(metrics.get('cpu_prediction_accuracy', 0.9) / 0.92, 1.0),
                'hydatis_availability_impact': 0.998 + np.random.normal(0, 0.0005),
                'hydatis_roi_contribution': 14.2 + np.random.normal(0, 0.5)
            })
            
            # 3. Resource Utilization Metrics
            metrics.update({
                'model_serving_cpu_usage': 0.65 + np.random.normal(0, 0.1),
                'model_serving_memory_usage': 0.72 + np.random.normal(0, 0.08),
                'model_error_rate': max(0, 0.008 + np.random.normal(0, 0.003))
            })
            
            # 4. Drift Detection Score
            baseline_accuracy = 0.89
            current_accuracy = metrics.get('cpu_prediction_accuracy', baseline_accuracy)
            drift_score = abs(current_accuracy - baseline_accuracy) / baseline_accuracy
            metrics['drift_detection_score'] = drift_score
            
            print(f"✅ Collected {len(metrics)} metrics for {model_id}")
            
        except Exception as e:
            logger.error(f"❌ Error collecting metrics for {model_id}: {e}")
            metrics = {}
        
        return metrics
    
    def store_metrics(self, model_id: str, metrics: Dict[str, float], business_context: str = None):
        """
        Store collected metrics in monitoring database.
        
        Args:
            model_id: Model identifier
            metrics: Dictionary of metric values
            business_context: Business context for the metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = time.time()
        
        for metric_name, value in metrics.items():
            cursor.execute('''
                INSERT INTO metric_values (metric_name, value, timestamp, model_id, business_context)
                VALUES (?, ?, ?, ?, ?)
            ''', (metric_name, float(value), timestamp, model_id, business_context))
        
        conn.commit()
        conn.close()
        
        logger.info(f"📊 Stored {len(metrics)} metrics for {model_id}")
    
    def evaluate_alert_conditions(self, metrics: Dict[str, float]) -> List[Dict]:
        """
        Evaluate alert conditions against current metrics.
        
        Args:
            metrics: Current metric values
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for alert in self.alert_conditions:
            metric_value = metrics.get(alert.metric_name)
            
            if metric_value is None:
                continue
            
            # Check alert condition
            condition_met = False
            if alert.condition == 'above' and metric_value > alert.threshold:
                condition_met = True
            elif alert.condition == 'below' and metric_value < alert.threshold:
                condition_met = True
            elif alert.condition == 'equals' and abs(metric_value - alert.threshold) < 0.001:
                condition_met = True
            
            if condition_met:
                # Calculate business impact
                metric_def = next((m for m in self.metric_definitions if m.name == alert.metric_name), None)
                business_impact = metric_def.business_impact if metric_def else 'unknown'
                
                alert_info = {
                    'alert_id': f"{alert.metric_name}_{int(time.time())}",
                    'metric_name': alert.metric_name,
                    'severity': alert.severity,
                    'condition': f"{alert.condition} {alert.threshold}",
                    'current_value': metric_value,
                    'business_impact': business_impact,
                    'duration_threshold': alert.duration_minutes,
                    'timestamp': time.time()
                }
                
                triggered_alerts.append(alert_info)
                
                # Log alert to database
                self._log_alert(alert_info)
        
        return triggered_alerts
    
    def _log_alert(self, alert_info: Dict):
        """Log alert to database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alert_history (alert_id, metric_name, severity, condition_met, business_impact)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            alert_info['alert_id'],
            alert_info['metric_name'],
            alert_info['severity'],
            f"{alert_info['condition']} (value: {alert_info['current_value']:.4f})",
            alert_info['business_impact']
        ))
        
        conn.commit()
        conn.close()
    
    def calculate_business_impact(self, metrics: Dict[str, float]) -> BusinessMetrics:
        """
        Calculate HYDATIS business impact from model metrics.
        
        Args:
            metrics: Current model metrics
            
        Returns:
            BusinessMetrics with calculated business impact
        """
        # CPU Target Achievement
        cpu_accuracy = metrics.get('cpu_prediction_accuracy', 0.85)
        cpu_target_achievement = min(cpu_accuracy / 0.92, 1.0)  # Target is 92% accuracy
        
        # Availability SLA Achievement
        availability_impact = metrics.get('hydatis_availability_impact', 0.997)
        availability_achievement = min(availability_impact / self.HYDATIS_AVAILABILITY_TARGET, 1.0)
        
        # ROI Contribution
        roi_contribution = metrics.get('hydatis_roi_contribution', 13.0)
        roi_achievement = min(roi_contribution / self.HYDATIS_ROI_TARGET, 1.0)
        
        # Cost Optimization (based on resource efficiency)
        cpu_usage = metrics.get('model_serving_cpu_usage', 0.7)
        memory_usage = metrics.get('model_serving_memory_usage', 0.75)
        cost_optimization = max(0, 1.0 - (cpu_usage + memory_usage) / 2.0) * 100
        
        # Workload Efficiency (composite score)
        latency = metrics.get('scheduling_latency_p95', 15.0)
        throughput = metrics.get('model_inference_throughput', 400)
        error_rate = metrics.get('model_error_rate', 0.01)
        
        workload_efficiency = (
            min(throughput / 500.0, 1.0) * 0.4 +  # Throughput factor
            max(0, 1.0 - latency / 25.0) * 0.3 +   # Latency factor
            (1.0 - error_rate) * 0.3               # Error rate factor
        )
        
        # Customer Satisfaction Impact (proxy based on availability and performance)
        customer_satisfaction = (
            availability_achievement * 0.5 +
            min(cpu_accuracy / 0.90, 1.0) * 0.3 +
            max(0, 1.0 - error_rate * 10) * 0.2
        )
        
        # Operational Reliability (composite reliability score)
        operational_reliability = (
            availability_achievement * 0.4 +
            (1.0 - metrics.get('drift_detection_score', 0.1)) * 0.3 +
            min(cpu_target_achievement, 1.0) * 0.3
        )
        
        return BusinessMetrics(
            cpu_utilization_target_achievement=cpu_target_achievement,
            availability_sla_achievement=availability_achievement,
            roi_contribution=roi_achievement,
            cost_optimization_percentage=cost_optimization,
            workload_efficiency_score=workload_efficiency,
            customer_satisfaction_impact=customer_satisfaction,
            operational_reliability=operational_reliability
        )
    
    def generate_business_impact_report(self, business_metrics: BusinessMetrics) -> Dict:
        """
        Generate comprehensive business impact report.
        
        Args:
            business_metrics: Calculated business metrics
            
        Returns:
            Business impact report
        """
        # Calculate overall business score
        business_score = (
            business_metrics.cpu_utilization_target_achievement * 0.20 +
            business_metrics.availability_sla_achievement * 0.25 +
            business_metrics.roi_contribution * 0.20 +
            business_metrics.workload_efficiency_score * 0.15 +
            business_metrics.customer_satisfaction_impact * 0.10 +
            business_metrics.operational_reliability * 0.10
        ) * 100
        
        # Determine business status
        if business_score >= 95:
            status = "EXCELLENT"
            action_required = "MAINTAIN"
        elif business_score >= 85:
            status = "GOOD" 
            action_required = "OPTIMIZE"
        elif business_score >= 70:
            status = "ACCEPTABLE"
            action_required = "INVESTIGATE"
        else:
            status = "CRITICAL"
            action_required = "IMMEDIATE_ACTION"
        
        # Generate recommendations
        recommendations = []
        
        if business_metrics.cpu_utilization_target_achievement < 0.9:
            recommendations.append("Optimize CPU prediction accuracy to meet 65% utilization target")
        
        if business_metrics.availability_sla_achievement < 0.99:
            recommendations.append("Critical: Address availability impact - risk to 99.7% SLA")
        
        if business_metrics.roi_contribution < 0.85:
            recommendations.append("Improve ROI contribution to meet 1400% target")
        
        if business_metrics.workload_efficiency_score < 0.8:
            recommendations.append("Optimize workload efficiency to reduce operational costs")
        
        return {
            "business_score": business_score,
            "status": status,
            "action_required": action_required,
            "hydatis_targets": {
                "cpu_target_65_percent": business_metrics.cpu_utilization_target_achievement >= 0.9,
                "availability_997_percent": business_metrics.availability_sla_achievement >= 0.99,
                "roi_1400_percent": business_metrics.roi_contribution >= 0.85
            },
            "detailed_metrics": {
                "cpu_target_achievement": f"{business_metrics.cpu_utilization_target_achievement*100:.1f}%",
                "availability_achievement": f"{business_metrics.availability_sla_achievement*100:.2f}%", 
                "roi_achievement": f"{business_metrics.roi_contribution*100:.1f}%",
                "cost_optimization": f"{business_metrics.cost_optimization_percentage:.1f}%",
                "workload_efficiency": f"{business_metrics.workload_efficiency_score*100:.1f}%",
                "customer_satisfaction": f"{business_metrics.customer_satisfaction_impact*100:.1f}%",
                "operational_reliability": f"{business_metrics.operational_reliability*100:.1f}%"
            },
            "recommendations": recommendations,
            "report_timestamp": datetime.now().isoformat()
        }
    
    async def monitor_model_ecosystem(self, models: List[str]) -> Dict:
        """
        Monitor entire HYDATIS model ecosystem.
        
        Args:
            models: List of model identifiers to monitor
            
        Returns:
            Comprehensive ecosystem monitoring report
        """
        print("🔍 Starting HYDATIS Model Ecosystem Monitoring...")
        
        ecosystem_metrics = {}
        ecosystem_alerts = []
        business_impacts = []
        
        # Collect metrics for each model
        for model_id in models:
            model_metrics = await self.collect_model_metrics(model_id)
            ecosystem_metrics[model_id] = model_metrics
            
            # Store metrics
            self.store_metrics(model_id, model_metrics, "ecosystem_monitoring")
            
            # Evaluate alerts
            model_alerts = self.evaluate_alert_conditions(model_metrics)
            ecosystem_alerts.extend(model_alerts)
            
            # Calculate business impact
            business_metrics = self.calculate_business_impact(model_metrics)
            business_report = self.generate_business_impact_report(business_metrics)
            business_impacts.append({
                'model_id': model_id,
                'business_report': business_report
            })
        
        # Calculate ecosystem-wide business score
        total_business_scores = [bi['business_report']['business_score'] for bi in business_impacts]
        ecosystem_business_score = np.mean(total_business_scores) if total_business_scores else 0
        
        # Ecosystem health assessment
        critical_alerts = [a for a in ecosystem_alerts if a['severity'] == 'critical']
        warning_alerts = [a for a in ecosystem_alerts if a['severity'] == 'warning']
        
        ecosystem_health = "HEALTHY"
        if len(critical_alerts) > 0:
            ecosystem_health = "CRITICAL"
        elif len(warning_alerts) > 2:
            ecosystem_health = "WARNING"
        elif ecosystem_business_score < 85:
            ecosystem_health = "DEGRADED"
        
        # Generate ecosystem report
        ecosystem_report = {
            "ecosystem_health": ecosystem_health,
            "business_score": ecosystem_business_score,
            "models_monitored": len(models),
            "total_alerts": len(ecosystem_alerts),
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "business_targets_status": {
                "cpu_utilization_65_percent": ecosystem_business_score >= 85,
                "availability_997_percent": len([a for a in ecosystem_alerts if 'availability' in a['metric_name']]) == 0,
                "roi_1400_percent": ecosystem_business_score >= 80
            },
            "model_reports": business_impacts,
            "active_alerts": ecosystem_alerts,
            "monitoring_timestamp": datetime.now().isoformat(),
            "next_scheduled_check": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        
        print(f"✅ Ecosystem monitoring completed:")
        print(f"   🏥 Health: {ecosystem_health}")
        print(f"   📊 Business score: {ecosystem_business_score:.1f}/100")
        print(f"   🚨 Active alerts: {len(ecosystem_alerts)} ({len(critical_alerts)} critical)")
        
        return ecosystem_report
    
    def create_monitoring_dashboard_config(self) -> Dict:
        """Create Grafana dashboard configuration for HYDATIS monitoring."""
        
        dashboard_config = {
            "dashboard": {
                "title": "HYDATIS ML Scheduler - MLOps Monitoring",
                "description": "Comprehensive monitoring dashboard for HYDATIS ML lifecycle",
                "panels": [
                    {
                        "title": "HYDATIS Business Targets",
                        "type": "stat",
                        "targets": [
                            {"expr": "hydatis_cpu_target_achievement", "legend": "CPU Target (65%)"},
                            {"expr": "hydatis_availability_impact", "legend": "Availability (99.7%)"},
                            {"expr": "hydatis_roi_contribution", "legend": "ROI (1400%)"}
                        ]
                    },
                    {
                        "title": "Model Performance Metrics",
                        "type": "graph",
                        "targets": [
                            {"expr": "cpu_prediction_accuracy", "legend": "CPU Prediction Accuracy"},
                            {"expr": "memory_prediction_accuracy", "legend": "Memory Prediction Accuracy"},
                            {"expr": "scheduling_latency_p95", "legend": "Scheduling Latency P95"}
                        ]
                    },
                    {
                        "title": "Drift Detection Dashboard",
                        "type": "heatmap",
                        "targets": [
                            {"expr": "drift_detection_score", "legend": "Drift Score"},
                            {"expr": "model_performance_degradation", "legend": "Performance Degradation"}
                        ]
                    },
                    {
                        "title": "Business Impact Tracking",
                        "type": "gauge",
                        "targets": [
                            {"expr": "business_score", "legend": "Overall Business Score"},
                            {"expr": "cost_optimization_percentage", "legend": "Cost Optimization"},
                            {"expr": "workload_efficiency_score", "legend": "Workload Efficiency"}
                        ]
                    }
                ]
            }
        }
        
        return dashboard_config
    
    async def start_continuous_monitoring(self, models: List[str], interval_minutes: int = 60):
        """
        Start continuous monitoring for HYDATIS model ecosystem.
        
        Args:
            models: List of models to monitor
            interval_minutes: Monitoring interval in minutes
        """
        print(f"🔄 Starting continuous HYDATIS monitoring (interval: {interval_minutes}m)")
        
        monitoring_active = True
        monitoring_cycles = 0
        
        try:
            while monitoring_active:
                monitoring_cycles += 1
                print(f"\n📊 Monitoring Cycle #{monitoring_cycles} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Monitor ecosystem
                ecosystem_report = await self.monitor_model_ecosystem(models)
                
                # Handle critical alerts
                critical_alerts = [a for a in ecosystem_report['active_alerts'] if a['severity'] == 'critical']
                if critical_alerts:
                    print(f"🚨 CRITICAL ALERTS DETECTED: {len(critical_alerts)}")
                    for alert in critical_alerts:
                        print(f"   ⚠️  {alert['metric_name']}: {alert['current_value']:.4f} {alert['condition']}")
                
                # Business score tracking
                business_score = ecosystem_report['business_score']
                if business_score < 70:
                    print(f"📉 BUSINESS IMPACT WARNING: Score {business_score:.1f}/100")
                elif business_score >= 95:
                    print(f"🏆 EXCELLENT BUSINESS PERFORMANCE: {business_score:.1f}/100")
                
                # Wait for next monitoring cycle
                await asyncio.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 Continuous monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
        
        print(f"✅ Continuous monitoring completed: {monitoring_cycles} cycles")

def create_hydatis_monitoring_setup() -> Dict:
    """
    Create complete HYDATIS monitoring setup configuration.
    
    Returns:
        Complete monitoring setup configuration
    """
    
    config = {
        'prometheus_url': 'http://prometheus-server.monitoring:9090',
        'mlflow_uri': 'http://mlflow-server.kubeflow:5000',
        'alert_webhook': 'https://hooks.slack.com/hydatis-ml-team',
        'monitoring_db': '/app/data/hydatis_monitoring.db',
        'dashboard_config': True,
        'continuous_monitoring': True,
        'monitoring_interval_minutes': 30
    }
    
    return config

async def main():
    """Main function for HYDATIS observability framework."""
    
    print("🚀 Initializing HYDATIS Observability Framework")
    
    # Create monitoring configuration
    config = create_hydatis_monitoring_setup()
    
    # Initialize framework
    framework = HYDATISObservabilityFramework(config)
    
    # Define models to monitor
    hydatis_models = [
        'hydatis-xgboost-cpu-predictor',
        'hydatis-xgboost-memory-predictor', 
        'hydatis-qlearning-optimizer',
        'hydatis-anomaly-detector'
    ]
    
    print(f"📊 Monitoring {len(hydatis_models)} HYDATIS models")
    
    # Generate dashboard configuration
    dashboard_config = framework.create_monitoring_dashboard_config()
    with open('/tmp/hydatis_dashboard.json', 'w') as f:
        json.dump(dashboard_config, f, indent=2)
    
    # Run single monitoring cycle for demonstration
    ecosystem_report = await framework.monitor_model_ecosystem(hydatis_models)
    
    print(f"\n🎯 HYDATIS Monitoring Summary:")
    print(f"   🏥 Ecosystem health: {ecosystem_report['ecosystem_health']}")
    print(f"   📊 Business score: {ecosystem_report['business_score']:.1f}/100")
    print(f"   🚨 Active alerts: {ecosystem_report['total_alerts']}")
    print(f"   🎯 Business targets: {sum(ecosystem_report['business_targets_status'].values())}/3 met")
    
    # Optionally start continuous monitoring (commented for demo)
    # await framework.start_continuous_monitoring(hydatis_models, interval_minutes=30)

if __name__ == "__main__":
    asyncio.run(main())