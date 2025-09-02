#!/usr/bin/env python3
"""
Isolation Forest serving endpoint for HYDATIS ML Scheduler anomaly detection.
Provides real-time anomaly detection API for cluster monitoring integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
from pathlib import Path
from flask import Flask, request, jsonify
import time

from .model import HYDATISIsolationForestDetector
from .monitoring import HYDATISAnomalyMonitor
from ...monitoring.alert_manager import HYDATISAlertManager

logger = logging.getLogger(__name__)


class IsolationForestServingEngine:
    """Real-time serving engine for Isolation Forest anomaly detection."""
    
    def __init__(self, model_dir: str = "/data/ml_scheduler_longhorn/models/isolation_forest"):
        self.model_dir = Path(model_dir)
        self.detector = HYDATISIsolationForestDetector()
        self.anomaly_monitor = HYDATISAnomalyMonitor(str(self.model_dir))
        self.alert_manager = HYDATISAlertManager()
        
        self.serving_config = {
            'max_latency_ms': 100,
            'batch_processing_size': 50,
            'anomaly_confidence_threshold': 0.8,
            'real_time_monitoring': True,
            'alert_integration': True
        }
        
        self.performance_tracking = {
            'request_count': 0,
            'successful_detections': 0,
            'anomalies_detected': 0,
            'alerts_generated': 0,
            'total_latency': 0.0,
            'error_count': 0
        }
        
        self.model_loaded = False
        self._load_detection_models()
        
        if self.serving_config['real_time_monitoring']:
            self._start_monitoring()
    
    def _load_detection_models(self) -> bool:
        """Load trained Isolation Forest models."""
        
        try:
            success = self.detector.load_models(str(self.model_dir))
            
            if success:
                self.model_loaded = True
                logger.info("Isolation Forest models loaded successfully")
            else:
                logger.warning("Failed to load Isolation Forest models")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _start_monitoring(self):
        """Start real-time anomaly monitoring."""
        
        if self.model_loaded:
            self.anomaly_monitor.start_monitoring()
            logger.info("Real-time anomaly monitoring started")
    
    def detect_anomalies(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in provided metrics data."""
        
        start_time = time.time()
        
        try:
            self.performance_tracking['request_count'] += 1
            
            if not self.model_loaded:
                raise ValueError("Isolation Forest models not loaded")
            
            if isinstance(metrics_data, dict):
                if 'metrics' in metrics_data:
                    metrics_df = pd.DataFrame(metrics_data['metrics'])
                elif 'node_metrics' in metrics_data:
                    node_data = []
                    for node_name, node_metrics in metrics_data['node_metrics'].items():
                        node_record = node_metrics.copy()
                        node_record['instance'] = node_name
                        node_record['timestamp'] = datetime.now().isoformat()
                        node_data.append(node_record)
                    metrics_df = pd.DataFrame(node_data)
                else:
                    metrics_df = pd.DataFrame([metrics_data])
            else:
                metrics_df = pd.DataFrame(metrics_data)
            
            detection_results = self.detector.detect_anomalies(metrics_df)
            
            latency = (time.time() - start_time) * 1000
            
            if detection_results['anomalies_detected'] > 0:
                self.performance_tracking['anomalies_detected'] += detection_results['anomalies_detected']
                
                if self.serving_config['alert_integration']:
                    alert_results = self._process_anomalies_for_alerts(detection_results['anomalies'])
                    detection_results['alert_processing'] = alert_results
            
            self.performance_tracking['successful_detections'] += 1
            self.performance_tracking['total_latency'] += latency
            
            detection_results['serving_metrics'] = {
                'detection_latency_ms': round(latency, 2),
                'latency_target_met': latency < self.serving_config['max_latency_ms'],
                'batch_size': len(metrics_df),
                'model_confidence': 'high' if self.performance_tracking['successful_detections'] > 100 else 'medium'
            }
            
            return detection_results
            
        except Exception as e:
            self.performance_tracking['error_count'] += 1
            logger.error(f"Anomaly detection serving error: {e}")
            
            latency = (time.time() - start_time) * 1000
            self.performance_tracking['total_latency'] += latency
            
            return {
                'error': str(e),
                'serving_metrics': {
                    'detection_latency_ms': round(latency, 2),
                    'error_mode': True
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def _process_anomalies_for_alerts(self, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process detected anomalies for alert generation."""
        
        alerts_generated = []
        
        for anomaly in anomalies:
            
            if anomaly['severity'] in ['critical', 'high']:
                anomaly_alert = {
                    'alert_id': f"HYDATIS-ANOMALY-{int(time.time())}-{len(alerts_generated)}",
                    'alert_time': datetime.now().isoformat(),
                    'severity': anomaly['severity'],
                    'node': anomaly['node'],
                    'anomaly_type': anomaly['anomaly_type'],
                    'anomaly_score': anomaly['anomaly_score'],
                    'description': f"{anomaly['severity'].upper()} anomaly detected on {anomaly['node']}: {anomaly['anomaly_type']}",
                    'affected_metrics': anomaly.get('affected_metrics', []),
                    'recommended_actions': self._get_anomaly_actions(anomaly),
                    'investigation_priority': self._calculate_priority(anomaly)
                }
                
                processed_alert = self.alert_manager.process_anomaly_alert(anomaly_alert)
                alerts_generated.append(processed_alert)
                
                self.performance_tracking['alerts_generated'] += 1
        
        alert_processing_result = {
            'alerts_generated': len(alerts_generated),
            'alert_details': alerts_generated,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return alert_processing_result
    
    def _get_anomaly_actions(self, anomaly: Dict[str, Any]) -> List[str]:
        """Get recommended actions for anomaly."""
        
        anomaly_type = anomaly['anomaly_type']
        
        action_map = {
            'cpu_spike': [
                "Investigate high CPU processes",
                "Check for resource-intensive workloads",
                "Consider pod migration if sustained"
            ],
            'memory_spike': [
                "Monitor memory usage patterns",
                "Check for memory leaks",
                "Evaluate pod memory limits"
            ],
            'cpu_starvation': [
                "Investigate resource allocation",
                "Check for failed processes",
                "Verify node health"
            ],
            'memory_starvation': [
                "Check memory allocation issues",
                "Investigate system resource conflicts",
                "Review pod resource requests"
            ]
        }
        
        return action_map.get(anomaly_type, ["Investigate cluster metrics", "Check system logs"])
    
    def _calculate_priority(self, anomaly: Dict[str, Any]) -> int:
        """Calculate investigation priority for anomaly."""
        
        severity_scores = {'critical': 10, 'high': 7, 'medium': 4, 'low': 2}
        base_priority = severity_scores.get(anomaly['severity'], 1)
        
        score_adjustment = min(abs(anomaly['anomaly_score']) * 3, 3)
        
        return min(10, int(base_priority + score_adjustment))
    
    def predict_anomaly_probability(self, metrics_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Predict anomaly probability for single metrics sample."""
        
        start_time = time.time()
        
        try:
            if not self.model_loaded:
                raise ValueError("Isolation Forest models not loaded")
            
            if isinstance(metrics_sample, dict):
                metrics_df = pd.DataFrame([metrics_sample])
            else:
                metrics_df = pd.DataFrame(metrics_sample)
            
            prediction_results = self.detector.predict_anomaly_probability(metrics_df)
            
            latency = (time.time() - start_time) * 1000
            
            single_prediction = prediction_results['predictions'][0]
            
            result = {
                'node': metrics_sample.get('instance', 'unknown'),
                'anomaly_probability': single_prediction['anomaly_probability'],
                'anomaly_score': single_prediction['anomaly_score'],
                'is_anomaly': single_prediction['is_anomaly'],
                'confidence': single_prediction['confidence'],
                'severity': single_prediction['severity'],
                'prediction_timestamp': single_prediction['timestamp'],
                'serving_metrics': {
                    'prediction_latency_ms': round(latency, 2),
                    'latency_target_met': latency < self.serving_config['max_latency_ms']
                }
            }
            
            if single_prediction['is_anomaly']:
                result['anomaly_details'] = {
                    'anomaly_type': single_prediction.get('anomaly_type', 'general_anomaly'),
                    'contributing_features': single_prediction.get('contributing_features', []),
                    'recommended_actions': self._get_anomaly_actions(single_prediction)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Anomaly probability prediction error: {e}")
            
            latency = (time.time() - start_time) * 1000
            
            return {
                'error': str(e),
                'node': metrics_sample.get('instance', 'unknown'),
                'serving_metrics': {
                    'prediction_latency_ms': round(latency, 2),
                    'error_mode': True
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def get_cluster_anomaly_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster anomaly status."""
        
        try:
            monitoring_status = self.anomaly_monitor.get_monitoring_status()
            
            recent_anomalies = self.anomaly_monitor.get_recent_anomalies(hours=24)
            recent_alerts = self.alert_manager.get_recent_alerts(hours=24)
            
            cluster_status = {
                'anomaly_detection_status': {
                    'monitoring_active': monitoring_status['monitoring_active'],
                    'model_loaded': monitoring_status['model_loaded'],
                    'detection_uptime_minutes': monitoring_status['uptime_minutes'],
                    'total_samples_processed': monitoring_status['performance_stats']['total_samples_processed']
                },
                'recent_activity_24h': {
                    'anomalies_detected': recent_anomalies['anomalies_found'],
                    'alerts_generated': recent_alerts['alerts_found'],
                    'critical_alerts': recent_alerts.get('critical_alerts', 0),
                    'most_affected_node': recent_anomalies.get('summary_statistics', {}).get('most_affected_node'),
                    'most_common_anomaly_type': recent_anomalies.get('summary_statistics', {}).get('most_common_type')
                },
                'cluster_health_assessment': {
                    'overall_health': self._assess_overall_cluster_health(recent_anomalies, recent_alerts),
                    'anomaly_rate': monitoring_status['performance_stats']['anomaly_detection_rate'],
                    'alert_frequency': recent_alerts.get('alert_frequency_per_hour', 0),
                    'monitoring_system_health': monitoring_status['health_assessment']['health_status']
                },
                'active_issues': {
                    'active_alerts': len(self.alert_manager.active_alerts),
                    'critical_active': len([a for a in self.alert_manager.active_alerts.values() if a['severity'] == 'critical']),
                    'high_active': len([a for a in self.alert_manager.active_alerts.values() if a['severity'] == 'high'])
                },
                'status_timestamp': datetime.now().isoformat()
            }
            
            return cluster_status
            
        except Exception as e:
            logger.error(f"Cluster anomaly status error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _assess_overall_cluster_health(self, recent_anomalies: Dict, recent_alerts: Dict) -> str:
        """Assess overall cluster health based on anomaly patterns."""
        
        anomaly_count = recent_anomalies.get('anomalies_found', 0)
        critical_alerts = recent_alerts.get('critical_alerts', 0)
        alert_count = recent_alerts.get('alerts_found', 0)
        
        if critical_alerts > 0:
            return 'critical'
        elif alert_count > 20 or anomaly_count > 50:
            return 'degraded'
        elif alert_count > 5 or anomaly_count > 10:
            return 'warning'
        else:
            return 'healthy'
    
    def get_serving_health(self) -> Dict[str, Any]:
        """Get serving engine health and performance metrics."""
        
        avg_latency = (self.performance_tracking['total_latency'] / 
                      self.performance_tracking['request_count']) if self.performance_tracking['request_count'] > 0 else 0
        
        success_rate = (self.performance_tracking['successful_detections'] / 
                       self.performance_tracking['request_count']) if self.performance_tracking['request_count'] > 0 else 0
        
        error_rate = (self.performance_tracking['error_count'] / 
                     self.performance_tracking['request_count']) if self.performance_tracking['request_count'] > 0 else 0
        
        health_status = 'healthy'
        if not self.model_loaded:
            health_status = 'degraded'
        elif error_rate > 0.1 or avg_latency > self.serving_config['max_latency_ms']:
            health_status = 'degraded'
        
        health = {
            'status': health_status,
            'model_loaded': self.model_loaded,
            'serving_performance': {
                'total_requests': self.performance_tracking['request_count'],
                'success_rate': round(success_rate, 4),
                'error_rate': round(error_rate, 4),
                'average_latency_ms': round(avg_latency, 2),
                'latency_target_met': avg_latency < self.serving_config['max_latency_ms'],
                'anomalies_detected': self.performance_tracking['anomalies_detected'],
                'alerts_generated': self.performance_tracking['alerts_generated']
            },
            'monitoring_status': self.anomaly_monitor.get_monitoring_status(),
            'model_info': {
                'target_precision': self.detector.target_precision,
                'contamination_rate': self.detector.contamination,
                'feature_count': len(self.detector.feature_names),
                'anomaly_threshold': self.detector.anomaly_threshold
            },
            'alert_system': {
                'active_alerts': len(self.alert_manager.active_alerts),
                'total_alerts_generated': self.alert_manager.alert_statistics['total_alerts_generated']
            },
            'health_timestamp': datetime.now().isoformat()
        }
        
        return health
    
    def reload_models(self) -> Dict[str, Any]:
        """Reload Isolation Forest models."""
        
        reload_start = time.time()
        
        try:
            self.anomaly_monitor.stop_monitoring()
            
            success = self._load_detection_models()
            
            if success and self.serving_config['real_time_monitoring']:
                self._start_monitoring()
            
            reload_latency = (time.time() - reload_start) * 1000
            
            return {
                'reload_successful': success,
                'model_loaded': self.model_loaded,
                'monitoring_restarted': self.anomaly_monitor.is_monitoring,
                'reload_latency_ms': round(reload_latency, 2),
                'model_info': {
                    'target_precision': self.detector.target_precision,
                    'feature_count': len(self.detector.feature_names)
                } if success else {},
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            reload_latency = (time.time() - reload_start) * 1000
            logger.error(f"Model reload error: {e}")
            
            return {
                'reload_successful': False,
                'error': str(e),
                'reload_latency_ms': round(reload_latency, 2),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_anomaly_analysis(self, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive anomaly analysis on historical data."""
        
        try:
            historical_data_path = analysis_config.get('historical_data_path')
            analysis_window_hours = analysis_config.get('window_hours', 24)
            
            if historical_data_path and Path(historical_data_path).exists():
                historical_data = pd.read_csv(historical_data_path)
                
                if 'timestamp' in historical_data.columns:
                    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
                    
                    cutoff_time = datetime.now() - timedelta(hours=analysis_window_hours)
                    recent_data = historical_data[historical_data['timestamp'] > cutoff_time]
                else:
                    recent_data = historical_data.tail(1000)
                
                detection_results = self.detector.detect_anomalies(recent_data)
                
                analysis_results = {
                    'analysis_period_hours': analysis_window_hours,
                    'samples_analyzed': len(recent_data),
                    'anomalies_detected': detection_results['anomalies_detected'],
                    'anomaly_rate': detection_results['anomaly_rate'],
                    'cluster_health_score': detection_results['cluster_health_score'],
                    'anomaly_breakdown': detection_results.get('severity_distribution', {}),
                    'detailed_anomalies': detection_results['anomalies'],
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                if detection_results['anomalies_detected'] > 0:
                    trends = self._analyze_anomaly_trends(detection_results['anomalies'])
                    analysis_results['trend_analysis'] = trends
                
                return analysis_results
            else:
                return {
                    'error': 'Historical data path not provided or file not found',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Anomaly analysis error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_anomaly_trends(self, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in detected anomalies."""
        
        if not anomalies:
            return {'no_anomalies': True}
        
        timestamps = [datetime.fromisoformat(a['timestamp']) for a in anomalies]
        
        hourly_distribution = defaultdict(int)
        node_frequency = defaultdict(int)
        type_frequency = defaultdict(int)
        severity_timeline = []
        
        for i, anomaly in enumerate(anomalies):
            hour = timestamps[i].hour
            hourly_distribution[hour] += 1
            
            node_frequency[anomaly['node']] += 1
            type_frequency[anomaly['anomaly_type']] += 1
            
            severity_timeline.append({
                'timestamp': anomaly['timestamp'],
                'severity': anomaly['severity'],
                'score': anomaly['anomaly_score']
            })
        
        trend_analysis = {
            'temporal_patterns': {
                'peak_anomaly_hours': [hour for hour, count in hourly_distribution.items() if count == max(hourly_distribution.values())],
                'hourly_distribution': dict(hourly_distribution),
                'anomaly_frequency_trend': 'increasing' if len(anomalies[-10:]) > len(anomalies[-20:-10]) else 'stable'
            },
            'node_patterns': {
                'most_problematic_node': max(node_frequency.items(), key=lambda x: x[1])[0],
                'node_anomaly_distribution': dict(node_frequency),
                'nodes_affected': len(node_frequency)
            },
            'anomaly_type_patterns': {
                'most_common_type': max(type_frequency.items(), key=lambda x: x[1])[0],
                'type_distribution': dict(type_frequency),
                'diversity_score': len(type_frequency) / len(anomalies)
            },
            'severity_evolution': severity_timeline
        }
        
        return trend_analysis


app = Flask(__name__)
serving_engine = IsolationForestServingEngine()


@app.route('/detect/anomalies', methods=['POST'])
def detect_anomalies():
    """API endpoint for anomaly detection."""
    
    try:
        metrics_data = request.json
        result = serving_engine.detect_anomalies(metrics_data)
        
        status_code = 200 if 'error' not in result else 400
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/anomaly_probability', methods=['POST'])
def predict_anomaly_probability():
    """API endpoint for anomaly probability prediction."""
    
    try:
        metrics_sample = request.json
        result = serving_engine.predict_anomaly_probability(metrics_sample)
        
        status_code = 200 if 'error' not in result else 400
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/status/cluster', methods=['GET'])
def cluster_anomaly_status():
    """API endpoint for cluster anomaly status."""
    
    try:
        status = serving_engine.get_cluster_anomaly_status()
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analysis/historical', methods=['POST'])
def historical_anomaly_analysis():
    """API endpoint for historical anomaly analysis."""
    
    try:
        analysis_config = request.json
        result = serving_engine.run_anomaly_analysis(analysis_config)
        
        status_code = 200 if 'error' not in result else 400
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/alerts/recent', methods=['GET'])
def recent_alerts():
    """API endpoint for recent alerts."""
    
    try:
        hours = request.args.get('hours', 24, type=int)
        alerts = serving_engine.alert_manager.get_recent_alerts(hours)
        return jsonify(alerts)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/alerts/dashboard', methods=['GET'])
def alert_dashboard():
    """API endpoint for alert dashboard."""
    
    try:
        dashboard = serving_engine.alert_manager.get_alert_dashboard()
        return jsonify(dashboard)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    
    health = serving_engine.get_serving_health()
    status_code = 200 if health['status'] == 'healthy' else 503
    
    return jsonify(health), status_code


@app.route('/reload', methods=['POST'])
def reload_models():
    """Manually reload Isolation Forest models."""
    
    result = serving_engine.reload_models()
    status_code = 200 if result['reload_successful'] else 500
    
    return jsonify(result), status_code


def main():
    """Main serving application."""
    
    print("HYDATIS Isolation Forest Anomaly Detection Serving Engine")
    print(f"Model Directory: {serving_engine.model_dir}")
    print(f"Model Loaded: {serving_engine.model_loaded}")
    print(f"Target Precision: {serving_engine.detector.target_precision:.1%}")
    print(f"Target Latency: <{serving_engine.serving_config['max_latency_ms']}ms")
    print(f"Real-time Monitoring: {serving_engine.serving_config['real_time_monitoring']}")
    print("API Endpoints:")
    print("  POST /detect/anomalies - Anomaly detection")
    print("  POST /predict/anomaly_probability - Anomaly probability prediction")
    print("  GET /status/cluster - Cluster anomaly status")
    print("  POST /analysis/historical - Historical anomaly analysis")
    print("  GET /alerts/recent - Recent alerts")
    print("  GET /alerts/dashboard - Alert dashboard")
    print("  GET /health - Health check")
    print("  POST /reload - Reload models")
    
    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=8082, debug=False)


if __name__ == "__main__":
    main()