#!/usr/bin/env python3
"""
Real-time anomaly monitoring for HYDATIS ML Scheduler.
Implements continuous cluster monitoring with Isolation Forest detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path
import time
import threading
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .model import HYDATISIsolationForestDetector
from ...monitoring.prometheus_collector import HYDATISPrometheusCollector

logger = logging.getLogger(__name__)


class HYDATISAnomalyMonitor:
    """Real-time anomaly monitoring for HYDATIS cluster."""
    
    def __init__(self, 
                 model_dir: str = "/data/ml_scheduler_longhorn/models/isolation_forest",
                 monitoring_interval: int = 30):
        
        self.model_dir = Path(model_dir)
        self.monitoring_interval = monitoring_interval
        
        self.detector = HYDATISIsolationForestDetector()
        self.prometheus_collector = HYDATISPrometheusCollector()
        
        self.monitoring_config = {
            'monitoring_window_minutes': 60,
            'anomaly_buffer_size': 1000,
            'alert_cooldown_minutes': 15,
            'severity_thresholds': {
                'critical': -0.5,
                'high': -0.3,
                'medium': -0.1,
                'low': 0.0
            },
            'alert_frequency_limits': {
                'critical': 5,
                'high': 10,
                'medium': 20,
                'low': 50
            }
        }
        
        self.anomaly_buffer = deque(maxlen=self.monitoring_config['anomaly_buffer_size'])
        self.alert_history = deque(maxlen=500)
        self.monitoring_stats = {
            'monitoring_started': None,
            'total_samples_processed': 0,
            'anomalies_detected': 0,
            'alerts_generated': 0,
            'last_anomaly_time': None,
            'monitoring_uptime_minutes': 0
        }
        
        self.is_monitoring = False
        self.monitoring_thread = None
        self.model_loaded = False
        
        self._load_detector_models()
    
    def _load_detector_models(self) -> bool:
        """Load trained Isolation Forest models."""
        
        try:
            success = self.detector.load_models(str(self.model_dir))
            
            if success:
                self.model_loaded = True
                logger.info("Isolation Forest anomaly detection models loaded")
            else:
                logger.warning("Failed to load anomaly detection models")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def start_monitoring(self):
        """Start real-time anomaly monitoring."""
        
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        if not self.model_loaded:
            logger.error("Cannot start monitoring - models not loaded")
            return
        
        self.is_monitoring = True
        self.monitoring_stats['monitoring_started'] = datetime.now().isoformat()
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Anomaly monitoring started (interval: {self.monitoring_interval}s)")
    
    def stop_monitoring(self):
        """Stop real-time anomaly monitoring."""
        
        self.is_monitoring = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        uptime = self._calculate_monitoring_uptime()
        self.monitoring_stats['monitoring_uptime_minutes'] = uptime
        
        logger.info(f"Anomaly monitoring stopped (uptime: {uptime:.1f} minutes)")
    
    def _monitoring_loop(self):
        """Main monitoring loop for continuous anomaly detection."""
        
        logger.info("Starting continuous anomaly monitoring loop...")
        
        while self.is_monitoring:
            try:
                monitoring_start = time.time()
                
                cluster_metrics = self._collect_cluster_metrics()
                
                if cluster_metrics is not None and len(cluster_metrics) > 0:
                    anomaly_results = self._process_metrics_for_anomalies(cluster_metrics)
                    
                    self._update_monitoring_stats(anomaly_results)
                    
                    if anomaly_results['anomalies_detected'] > 0:
                        self._handle_detected_anomalies(anomaly_results)
                
                processing_time = (time.time() - monitoring_start) * 1000
                
                if processing_time > 5000:
                    logger.warning(f"Anomaly detection processing took {processing_time:.1f}ms")
                
                remaining_time = max(0, self.monitoring_interval - (processing_time / 1000))
                time.sleep(remaining_time)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_cluster_metrics(self) -> Optional[pd.DataFrame]:
        """Collect current cluster metrics for anomaly detection."""
        
        try:
            metrics = self.prometheus_collector.collect_cluster_metrics()
            
            if not metrics or 'node_metrics' not in metrics:
                return None
            
            node_data = []
            
            for node_name, node_metrics in metrics['node_metrics'].items():
                node_record = {
                    'timestamp': datetime.now(),
                    'instance': node_name,
                    'cpu_utilization': node_metrics.get('cpu_utilization', 0),
                    'memory_utilization': node_metrics.get('memory_utilization', 0),
                    'disk_utilization': node_metrics.get('disk_utilization', 0),
                    'load_1m': node_metrics.get('load_1m', 0),
                    'load_5m': node_metrics.get('load_5m', 0),
                    'load_15m': node_metrics.get('load_15m', 0),
                    'network_rx_bytes': node_metrics.get('network_rx_bytes_per_sec', 0),
                    'network_tx_bytes': node_metrics.get('network_tx_bytes_per_sec', 0)
                }
                
                node_data.append(node_record)
            
            cluster_df = pd.DataFrame(node_data)
            
            return cluster_df
            
        except Exception as e:
            logger.error(f"Error collecting cluster metrics: {e}")
            return None
    
    def _process_metrics_for_anomalies(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Process collected metrics for anomaly detection."""
        
        try:
            detection_results = self.detector.detect_anomalies(metrics_df)
            
            self.monitoring_stats['total_samples_processed'] += detection_results['total_samples']
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Error processing metrics for anomalies: {e}")
            return {
                'total_samples': len(metrics_df),
                'anomalies_detected': 0,
                'anomalies': [],
                'error': str(e)
            }
    
    def _handle_detected_anomalies(self, anomaly_results: Dict[str, Any]):
        """Handle detected anomalies and generate appropriate responses."""
        
        for anomaly in anomaly_results['anomalies']:
            
            self.anomaly_buffer.append({
                'detection_time': datetime.now().isoformat(),
                'anomaly_data': anomaly
            })
            
            if self._should_generate_alert(anomaly):
                alert = self._generate_anomaly_alert(anomaly)
                self.alert_history.append(alert)
                self.monitoring_stats['alerts_generated'] += 1
                
                logger.warning(f"Anomaly Alert: {alert['severity']} - {alert['description']}")
        
        self.monitoring_stats['anomalies_detected'] += anomaly_results['anomalies_detected']
        
        if anomaly_results['anomalies_detected'] > 0:
            self.monitoring_stats['last_anomaly_time'] = datetime.now().isoformat()
    
    def _should_generate_alert(self, anomaly: Dict[str, Any]) -> bool:
        """Determine if an anomaly should generate an alert."""
        
        severity = anomaly['severity']
        
        cooldown_minutes = self.monitoring_config['alert_cooldown_minutes']
        cooldown_threshold = datetime.now() - timedelta(minutes=cooldown_minutes)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if (datetime.fromisoformat(alert['alert_time']) > cooldown_threshold and
                alert['node'] == anomaly['node'] and
                alert['severity'] == severity)
        ]
        
        frequency_limit = self.monitoring_config['alert_frequency_limits'].get(severity, 10)
        
        return len(recent_alerts) < frequency_limit
    
    def _generate_anomaly_alert(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured alert for detected anomaly."""
        
        alert = {
            'alert_id': f"HYDATIS-ANOMALY-{int(time.time())}",
            'alert_time': datetime.now().isoformat(),
            'severity': anomaly['severity'],
            'node': anomaly['node'],
            'anomaly_type': anomaly['anomaly_type'],
            'anomaly_score': anomaly['anomaly_score'],
            'description': f"{anomaly['severity'].upper()} anomaly detected on {anomaly['node']}: {anomaly['anomaly_type']}",
            'affected_metrics': anomaly['affected_metrics'],
            'recommended_actions': self._get_recommended_actions(anomaly),
            'investigation_priority': self._calculate_investigation_priority(anomaly),
            'cluster_context': {
                'total_anomalies_last_hour': self._count_recent_anomalies(60),
                'node_anomaly_frequency': self._get_node_anomaly_frequency(anomaly['node'])
            }
        }
        
        return alert
    
    def _get_recommended_actions(self, anomaly: Dict[str, Any]) -> List[str]:
        """Get recommended actions for anomaly type."""
        
        anomaly_type = anomaly['anomaly_type']
        severity = anomaly['severity']
        
        actions = []
        
        if anomaly_type == 'cpu_spike':
            actions.extend([
                "Investigate high CPU processes on affected node",
                "Check for resource-intensive workloads",
                "Consider pod migration if sustained"
            ])
        elif anomaly_type == 'memory_spike':
            actions.extend([
                "Monitor memory usage patterns",
                "Check for memory leaks in applications",
                "Evaluate pod memory limits"
            ])
        elif anomaly_type in ['cpu_starvation', 'memory_starvation']:
            actions.extend([
                "Investigate resource allocation issues",
                "Check for failed or stuck processes",
                "Verify node health status"
            ])
        else:
            actions.extend([
                "Investigate cluster metrics for unusual patterns",
                "Check system logs for related events",
                "Monitor trend over next monitoring cycles"
            ])
        
        if severity in ['critical', 'high']:
            actions.insert(0, "Immediate investigation required")
        
        return actions
    
    def _calculate_investigation_priority(self, anomaly: Dict[str, Any]) -> int:
        """Calculate investigation priority (1-10 scale)."""
        
        severity_scores = {'critical': 10, 'high': 7, 'medium': 4, 'low': 2}
        base_priority = severity_scores.get(anomaly['severity'], 1)
        
        score_factor = min(abs(anomaly['anomaly_score']) * 2, 3)
        
        node_frequency = self._get_node_anomaly_frequency(anomaly['node'])
        frequency_factor = min(node_frequency, 2)
        
        priority = min(base_priority + score_factor + frequency_factor, 10)
        
        return int(priority)
    
    def _count_recent_anomalies(self, minutes: int) -> int:
        """Count anomalies detected in recent time window."""
        
        threshold_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_count = 0
        for anomaly_record in self.anomaly_buffer:
            detection_time = datetime.fromisoformat(anomaly_record['detection_time'])
            if detection_time > threshold_time:
                recent_count += 1
        
        return recent_count
    
    def _get_node_anomaly_frequency(self, node_name: str) -> int:
        """Get anomaly frequency for specific node."""
        
        node_anomalies = [
            record for record in self.anomaly_buffer
            if record['anomaly_data']['node'] == node_name
        ]
        
        return len(node_anomalies)
    
    def _calculate_monitoring_uptime(self) -> float:
        """Calculate monitoring uptime in minutes."""
        
        if not self.monitoring_stats['monitoring_started']:
            return 0.0
        
        start_time = datetime.fromisoformat(self.monitoring_stats['monitoring_started'])
        uptime = (datetime.now() - start_time).total_seconds() / 60
        
        return uptime
    
    def _update_monitoring_stats(self, anomaly_results: Dict[str, Any]):
        """Update monitoring statistics."""
        
        self.monitoring_stats['monitoring_uptime_minutes'] = self._calculate_monitoring_uptime()
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        
        uptime = self._calculate_monitoring_uptime()
        
        recent_anomalies_1h = self._count_recent_anomalies(60)
        recent_anomalies_24h = self._count_recent_anomalies(1440)
        
        anomaly_rate_1h = recent_anomalies_1h / (60 / (self.monitoring_interval / 60)) if uptime >= 60 else 0
        
        status = {
            'monitoring_active': self.is_monitoring,
            'model_loaded': self.model_loaded,
            'monitoring_interval_seconds': self.monitoring_interval,
            'uptime_minutes': uptime,
            'performance_stats': {
                'total_samples_processed': self.monitoring_stats['total_samples_processed'],
                'total_anomalies_detected': self.monitoring_stats['anomalies_detected'],
                'total_alerts_generated': self.monitoring_stats['alerts_generated'],
                'anomaly_detection_rate': (self.monitoring_stats['anomalies_detected'] / 
                                         max(self.monitoring_stats['total_samples_processed'], 1)),
                'recent_anomalies_1h': recent_anomalies_1h,
                'recent_anomalies_24h': recent_anomalies_24h,
                'anomaly_rate_per_hour': anomaly_rate_1h
            },
            'buffer_status': {
                'anomaly_buffer_size': len(self.anomaly_buffer),
                'alert_history_size': len(self.alert_history),
                'buffer_capacity': self.monitoring_config['anomaly_buffer_size']
            },
            'last_activity': {
                'last_anomaly_detected': self.monitoring_stats.get('last_anomaly_time'),
                'last_alert_generated': self.alert_history[-1]['alert_time'] if self.alert_history else None
            },
            'health_assessment': self._assess_monitoring_health(),
            'status_timestamp': datetime.now().isoformat()
        }
        
        return status
    
    def _assess_monitoring_health(self) -> Dict[str, Any]:
        """Assess monitoring system health."""
        
        uptime = self._calculate_monitoring_uptime()
        
        health_indicators = {
            'monitoring_active': self.is_monitoring,
            'model_available': self.model_loaded,
            'prometheus_connected': True,
            'processing_within_interval': True,
            'adequate_uptime': uptime > 5
        }
        
        health_score = sum(health_indicators.values()) / len(health_indicators)
        
        health_status = 'healthy'
        if health_score < 0.8:
            health_status = 'degraded'
        elif health_score < 0.6:
            health_status = 'unhealthy'
        
        assessment = {
            'health_status': health_status,
            'health_score': health_score,
            'health_indicators': health_indicators,
            'recommendations': self._generate_health_recommendations(health_indicators)
        }
        
        return assessment
    
    def _generate_health_recommendations(self, indicators: Dict[str, bool]) -> List[str]:
        """Generate health improvement recommendations."""
        
        recommendations = []
        
        if not indicators['monitoring_active']:
            recommendations.append("Start anomaly monitoring service")
        
        if not indicators['model_available']:
            recommendations.append("Load trained Isolation Forest models")
        
        if not indicators['prometheus_connected']:
            recommendations.append("Check Prometheus connection and metrics availability")
        
        if not indicators['processing_within_interval']:
            recommendations.append("Optimize monitoring performance or increase interval")
        
        if not recommendations:
            recommendations.append("Monitoring system healthy - no actions required")
        
        return recommendations
    
    def get_recent_anomalies(self, hours: int = 24) -> Dict[str, Any]:
        """Get recent anomalies within specified time window."""
        
        threshold_time = datetime.now() - timedelta(hours=hours)
        
        recent_anomalies = []
        for anomaly_record in self.anomaly_buffer:
            detection_time = datetime.fromisoformat(anomaly_record['detection_time'])
            if detection_time > threshold_time:
                recent_anomalies.append(anomaly_record)
        
        if not recent_anomalies:
            return {
                'time_window_hours': hours,
                'anomalies_found': 0,
                'anomalies': []
            }
        
        severity_counts = {}
        node_counts = {}
        anomaly_type_counts = {}
        
        for record in recent_anomalies:
            anomaly = record['anomaly_data']
            
            severity = anomaly['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            node = anomaly['node']
            node_counts[node] = node_counts.get(node, 0) + 1
            
            anomaly_type = anomaly['anomaly_type']
            anomaly_type_counts[anomaly_type] = anomaly_type_counts.get(anomaly_type, 0) + 1
        
        analysis = {
            'time_window_hours': hours,
            'anomalies_found': len(recent_anomalies),
            'anomalies': [record['anomaly_data'] for record in recent_anomalies],
            'summary_statistics': {
                'severity_distribution': severity_counts,
                'node_distribution': node_counts,
                'anomaly_type_distribution': anomaly_type_counts,
                'most_affected_node': max(node_counts.items(), key=lambda x: x[1])[0] if node_counts else None,
                'most_common_severity': max(severity_counts.items(), key=lambda x: x[1])[0] if severity_counts else None,
                'most_common_type': max(anomaly_type_counts.items(), key=lambda x: x[1])[0] if anomaly_type_counts else None
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def get_recent_alerts(self, hours: int = 24) -> Dict[str, Any]:
        """Get recent alerts within specified time window."""
        
        threshold_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['alert_time']) > threshold_time
        ]
        
        if not recent_alerts:
            return {
                'time_window_hours': hours,
                'alerts_found': 0,
                'alerts': []
            }
        
        severity_counts = {}
        for alert in recent_alerts:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        alert_summary = {
            'time_window_hours': hours,
            'alerts_found': len(recent_alerts),
            'alerts': recent_alerts,
            'severity_distribution': severity_counts,
            'critical_alerts': len([a for a in recent_alerts if a['severity'] == 'critical']),
            'high_priority_alerts': len([a for a in recent_alerts if a['severity'] in ['critical', 'high']]),
            'alert_frequency_per_hour': len(recent_alerts) / hours,
            'summary_timestamp': datetime.now().isoformat()
        }
        
        return alert_summary
    
    def run_anomaly_simulation(self, simulation_duration_minutes: int = 30) -> Dict[str, Any]:
        """Run anomaly detection simulation for testing."""
        
        logger.info(f"Running {simulation_duration_minutes} minute anomaly detection simulation...")
        
        simulation_start = datetime.now()
        simulation_results = {
            'simulation_start': simulation_start.isoformat(),
            'duration_minutes': simulation_duration_minutes,
            'monitoring_cycles': 0,
            'anomalies_detected': 0,
            'alerts_generated': 0,
            'performance_metrics': []
        }
        
        cycles = simulation_duration_minutes * 60 // self.monitoring_interval
        
        for cycle in range(cycles):
            cycle_start = time.time()
            
            simulated_metrics = self._generate_simulated_metrics()
            
            anomaly_results = self._process_metrics_for_anomalies(simulated_metrics)
            
            cycle_latency = (time.time() - cycle_start) * 1000
            
            simulation_results['monitoring_cycles'] += 1
            simulation_results['anomalies_detected'] += anomaly_results['anomalies_detected']
            
            simulation_results['performance_metrics'].append({
                'cycle': cycle + 1,
                'processing_latency_ms': cycle_latency,
                'anomalies_in_cycle': anomaly_results['anomalies_detected'],
                'samples_processed': anomaly_results['total_samples']
            })
            
            if anomaly_results['anomalies_detected'] > 0:
                self._handle_detected_anomalies(anomaly_results)
                simulation_results['alerts_generated'] += len([a for a in anomaly_results['anomalies'] if self._should_generate_alert(a)])
        
        simulation_results['simulation_completed'] = datetime.now().isoformat()
        simulation_results['average_cycle_latency_ms'] = np.mean([m['processing_latency_ms'] for m in simulation_results['performance_metrics']])
        
        logger.info(f"Simulation completed: {simulation_results['anomalies_detected']} anomalies, {simulation_results['alerts_generated']} alerts")
        
        return simulation_results
    
    def _generate_simulated_metrics(self) -> pd.DataFrame:
        """Generate simulated cluster metrics for testing."""
        
        node_data = []
        
        for node in ['worker-1', 'worker-2', 'worker-3']:
            
            base_cpu = np.random.uniform(0.08, 0.15)
            base_memory = np.random.uniform(0.35, 0.45)
            
            if np.random.random() < 0.05:
                base_cpu = np.random.uniform(0.85, 0.98)
            if np.random.random() < 0.03:
                base_memory = np.random.uniform(0.85, 0.95)
            
            node_record = {
                'timestamp': datetime.now(),
                'instance': node,
                'cpu_utilization': base_cpu,
                'memory_utilization': base_memory,
                'disk_utilization': np.random.uniform(0.2, 0.6),
                'load_1m': np.random.uniform(0.5, 2.0),
                'load_5m': np.random.uniform(0.4, 1.8),
                'load_15m': np.random.uniform(0.3, 1.5),
                'network_rx_bytes': np.random.uniform(1000, 50000),
                'network_tx_bytes': np.random.uniform(800, 30000)
            }
            
            node_data.append(node_record)
        
        return pd.DataFrame(node_data)


def main():
    """Main anomaly monitoring demonstration."""
    
    print("HYDATIS Real-time Anomaly Monitoring - Week 7")
    print("Target: 94% precision anomaly detection")
    
    monitor = HYDATISAnomalyMonitor()
    
    print(f"Model Loaded: {monitor.model_loaded}")
    print(f"Monitoring Interval: {monitor.monitoring_interval} seconds")
    print(f"Alert Cooldown: {monitor.monitoring_config['alert_cooldown_minutes']} minutes")
    print(f"Anomaly Buffer Size: {monitor.monitoring_config['anomaly_buffer_size']}")
    
    if monitor.model_loaded:
        status = monitor.get_monitoring_status()
        print(f"Monitoring Status: {status['health_assessment']['health_status']}")
    
    return monitor


if __name__ == "__main__":
    monitor = main()