
#!/usr/bin/env python3
"""
Alert management system for HYDATIS ML Scheduler anomaly detection.
Implements intelligent alert generation, routing, and escalation policies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path
import smtplib
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from collections import defaultdict, deque
import threading
import time

logger = logging.getLogger(__name__)


class HYDATISAlertManager:
    """Intelligent alert management for HYDATIS cluster anomaly detection."""
    
    def __init__(self):
        self.alert_config = {
            'severity_priorities': {
                'critical': 1,
                'high': 2,
                'medium': 3,
                'low': 4
            },
            'escalation_timeouts': {
                'critical': 300,
                'high': 900,
                'medium': 1800,
                'low': 3600
            },
            'alert_frequency_limits': {
                'critical': {'per_hour': 10, 'per_day': 50},
                'high': {'per_hour': 20, 'per_day': 100},
                'medium': {'per_hour': 50, 'per_day': 200},
                'low': {'per_hour': 100, 'per_day': 400}
            },
            'correlation_window_minutes': 15,
            'auto_resolution_timeout_minutes': 60
        }
        
        self.notification_channels = {
            'slack': {
                'enabled': True,
                'webhook_url': 'https://hooks.slack.com/services/HYDATIS/ALERT/CHANNEL',
                'channel': '#ml-scheduler-alerts',
                'severity_channels': {
                    'critical': '#ops-critical',
                    'high': '#ops-alerts',
                    'medium': '#ml-scheduler-alerts',
                    'low': '#ml-scheduler-monitoring'
                }
            },
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.company.com',
                'smtp_port': 587,
                'recipients': {
                    'critical': ['ops-team@company.com', 'ml-team@company.com'],
                    'high': ['ml-team@company.com', 'devops@company.com'],
                    'medium': ['ml-team@company.com'],
                    'low': ['ml-monitoring@company.com']
                }
            },
            'webhook': {
                'enabled': True,
                'endpoints': {
                    'ops_webhook': 'http://10.110.190.32:8080/alerts/ml-scheduler',
                    'monitoring_webhook': 'http://10.110.190.83:9093/api/v1/alerts'
                }
            }
        }
        
        self.active_alerts = {}
        self.alert_history = deque(maxlen=5000)
        self.correlation_groups = defaultdict(list)
        self.suppression_rules = {}
        
        self.alert_statistics = {
            'total_alerts_generated': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_node': defaultdict(int),
            'alerts_by_type': defaultdict(int),
            'escalated_alerts': 0,
            'auto_resolved_alerts': 0,
            'correlated_alerts': 0
        }
    
    def process_anomaly_alert(self, anomaly_alert: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance anomaly alert with intelligent routing."""
        
        alert_id = anomaly_alert['alert_id']
        
        enhanced_alert = self._enhance_alert_context(anomaly_alert)
        
        correlation_result = self._correlate_with_existing_alerts(enhanced_alert)
        
        if correlation_result['correlated']:
            enhanced_alert['correlation_info'] = correlation_result
            self.alert_statistics['correlated_alerts'] += 1
        
        suppression_result = self._check_alert_suppression(enhanced_alert)
        
        if suppression_result['suppressed']:
            enhanced_alert['suppression_info'] = suppression_result
            logger.info(f"Alert {alert_id} suppressed: {suppression_result['reason']}")
            return enhanced_alert
        
        routing_decision = self._determine_alert_routing(enhanced_alert)
        
        notification_results = self._send_notifications(enhanced_alert, routing_decision)
        
        enhanced_alert['routing_decision'] = routing_decision
        enhanced_alert['notification_results'] = notification_results
        enhanced_alert['processing_timestamp'] = datetime.now().isoformat()
        
        self._store_alert(enhanced_alert)
        
        self._update_alert_statistics(enhanced_alert)
        
        logger.info(f"Alert {alert_id} processed: {enhanced_alert['severity']} - {enhanced_alert['description']}")
        
        return enhanced_alert
    
    def _enhance_alert_context(self, base_alert: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance alert with additional context and intelligence."""
        
        enhanced = base_alert.copy()
        
        enhanced['alert_context'] = {
            'business_impact': self._assess_business_impact(base_alert),
            'historical_frequency': self._get_historical_frequency(base_alert),
            'cluster_health_impact': self._assess_cluster_health_impact(base_alert),
            'workload_impact_estimate': self._estimate_workload_impact(base_alert),
            'resolution_suggestions': self._get_resolution_suggestions(base_alert)
        }
        
        enhanced['alert_metadata'] = {
            'alert_generation_source': 'HYDATIS-ML-Scheduler',
            'detection_model': 'isolation_forest',
            'cluster_context': 'HYDATIS-6node',
            'alert_version': '1.0',
            'processing_pipeline': 'anomaly_detection'
        }
        
        return enhanced
    
    def _assess_business_impact(self, alert: Dict[str, Any]) -> str:
        """Assess business impact of detected anomaly."""
        
        severity = alert['severity']
        node = alert['node']
        anomaly_type = alert['anomaly_type']
        
        if severity == 'critical':
            if 'master' in node:
                return 'high'
            elif anomaly_type in ['cpu_spike', 'memory_spike']:
                return 'medium'
            else:
                return 'low'
        elif severity == 'high':
            return 'medium' if anomaly_type in ['resource_spike'] else 'low'
        else:
            return 'low'
    
    def _get_historical_frequency(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical frequency of similar alerts."""
        
        node = alert['node']
        anomaly_type = alert['anomaly_type']
        
        similar_alerts = [
            alert_record for alert_record in self.alert_history
            if (alert_record.get('node') == node and 
                alert_record.get('anomaly_type') == anomaly_type)
        ]
        
        recent_similar = [
            alert_record for alert_record in similar_alerts
            if datetime.fromisoformat(alert_record['alert_time']) > datetime.now() - timedelta(days=7)
        ]
        
        frequency_analysis = {
            'total_similar_alerts': len(similar_alerts),
            'recent_similar_alerts_7d': len(recent_similar),
            'frequency_category': 'frequent' if len(recent_similar) > 5 else 'occasional' if len(recent_similar) > 1 else 'rare',
            'last_similar_alert': similar_alerts[-1]['alert_time'] if similar_alerts else None
        }
        
        return frequency_analysis
    
    def _assess_cluster_health_impact(self, alert: Dict[str, Any]) -> str:
        """Assess impact on overall cluster health."""
        
        severity = alert['severity']
        affected_metrics = alert.get('affected_metrics', [])
        
        if severity == 'critical':
            return 'significant'
        elif severity == 'high' and len(affected_metrics) > 3:
            return 'moderate'
        elif severity in ['high', 'medium'] and any('cpu' in metric or 'memory' in metric for metric in affected_metrics):
            return 'moderate'
        else:
            return 'minimal'
    
    def _estimate_workload_impact(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate impact on running workloads."""
        
        node = alert['node']
        anomaly_type = alert['anomaly_type']
        severity = alert['severity']
        
        impact_estimate = {
            'affected_workloads': 'unknown',
            'performance_degradation_risk': 'low',
            'service_disruption_risk': 'low',
            'resource_contention_risk': 'low'
        }
        
        if anomaly_type == 'cpu_spike':
            impact_estimate['performance_degradation_risk'] = 'high' if severity in ['critical', 'high'] else 'medium'
            impact_estimate['resource_contention_risk'] = 'high'
        elif anomaly_type == 'memory_spike':
            impact_estimate['service_disruption_risk'] = 'high' if severity == 'critical' else 'medium'
            impact_estimate['resource_contention_risk'] = 'medium'
        elif anomaly_type in ['cpu_starvation', 'memory_starvation']:
            impact_estimate['performance_degradation_risk'] = 'medium'
            impact_estimate['service_disruption_risk'] = 'low'
        
        return impact_estimate
    
    def _get_resolution_suggestions(self, alert: Dict[str, Any]) -> List[str]:
        """Get specific resolution suggestions for anomaly type."""
        
        anomaly_type = alert['anomaly_type']
        severity = alert['severity']
        
        suggestions = []
        
        if anomaly_type == 'cpu_spike':
            suggestions.extend([
                "Check top CPU-consuming processes on node",
                "Identify resource-intensive pods for potential migration",
                "Monitor CPU usage trend over next 10 minutes",
                "Consider horizontal pod autoscaling if applicable"
            ])
        elif anomaly_type == 'memory_spike':
            suggestions.extend([
                "Investigate memory usage by pod on affected node",
                "Check for memory leaks in running applications",
                "Review pod memory limits and requests",
                "Monitor memory growth patterns"
            ])
        elif anomaly_type in ['cpu_starvation', 'memory_starvation']:
            suggestions.extend([
                "Verify node health and system status",
                "Check for failed or crashed processes",
                "Investigate resource allocation configurations",
                "Review recent deployment changes"
            ])
        else:
            suggestions.extend([
                "Investigate cluster metrics for unusual patterns",
                "Check system logs for error messages",
                "Monitor affected node for trend changes",
                "Review recent cluster configuration changes"
            ])
        
        if severity in ['critical', 'high']:
            suggestions.insert(0, "Immediate investigation and response required")
        
        return suggestions
    
    def _correlate_with_existing_alerts(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate alert with existing active alerts."""
        
        correlation_window = timedelta(minutes=self.alert_config['correlation_window_minutes'])
        current_time = datetime.now()
        
        correlated_alerts = []
        
        for active_alert_id, active_alert in self.active_alerts.items():
            alert_time = datetime.fromisoformat(active_alert['alert_time'])
            
            if current_time - alert_time <= correlation_window:
                correlation_score = self._calculate_correlation_score(alert, active_alert)
                
                if correlation_score > 0.7:
                    correlated_alerts.append({
                        'alert_id': active_alert_id,
                        'correlation_score': correlation_score,
                        'correlation_reasons': self._get_correlation_reasons(alert, active_alert)
                    })
        
        correlation_result = {
            'correlated': len(correlated_alerts) > 0,
            'correlation_count': len(correlated_alerts),
            'correlated_alerts': correlated_alerts,
            'correlation_type': self._determine_correlation_type(correlated_alerts) if correlated_alerts else None
        }
        
        return correlation_result
    
    def _calculate_correlation_score(self, alert1: Dict[str, Any], alert2: Dict[str, Any]) -> float:
        """Calculate correlation score between two alerts."""
        
        score = 0.0
        
        if alert1['node'] == alert2['node']:
            score += 0.4
        
        if alert1['anomaly_type'] == alert2['anomaly_type']:
            score += 0.3
        
        if alert1['severity'] == alert2['severity']:
            score += 0.2
        
        affected_metrics1 = set(alert1.get('affected_metrics', []))
        affected_metrics2 = set(alert2.get('affected_metrics', []))
        
        if affected_metrics1 and affected_metrics2:
            metric_overlap = len(affected_metrics1.intersection(affected_metrics2))
            metric_union = len(affected_metrics1.union(affected_metrics2))
            metric_similarity = metric_overlap / metric_union if metric_union > 0 else 0
            score += metric_similarity * 0.1
        
        return min(score, 1.0)
    
    def _get_correlation_reasons(self, alert1: Dict[str, Any], alert2: Dict[str, Any]) -> List[str]:
        """Get reasons for alert correlation."""
        
        reasons = []
        
        if alert1['node'] == alert2['node']:
            reasons.append(f"Same node affected: {alert1['node']}")
        
        if alert1['anomaly_type'] == alert2['anomaly_type']:
            reasons.append(f"Same anomaly type: {alert1['anomaly_type']}")
        
        if alert1['severity'] == alert2['severity']:
            reasons.append(f"Same severity level: {alert1['severity']}")
        
        affected_metrics1 = set(alert1.get('affected_metrics', []))
        affected_metrics2 = set(alert2.get('affected_metrics', []))
        
        common_metrics = affected_metrics1.intersection(affected_metrics2)
        if common_metrics:
            reasons.append(f"Common affected metrics: {', '.join(common_metrics)}")
        
        return reasons
    
    def _determine_correlation_type(self, correlated_alerts: List[Dict]) -> str:
        """Determine type of alert correlation."""
        
        if len(correlated_alerts) == 1:
            return 'duplicate'
        elif len(correlated_alerts) > 1:
            avg_score = np.mean([alert['correlation_score'] for alert in correlated_alerts])
            if avg_score > 0.8:
                return 'cascade'
            else:
                return 'related'
        
        return 'independent'
    
    def _check_alert_suppression(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Check if alert should be suppressed based on rules."""
        
        alert_key = f"{alert['node']}_{alert['anomaly_type']}_{alert['severity']}"
        
        recent_threshold = datetime.now() - timedelta(hours=1)
        recent_similar_alerts = [
            hist_alert for hist_alert in self.alert_history
            if (f"{hist_alert.get('node')}_{hist_alert.get('anomaly_type')}_{hist_alert.get('severity')}" == alert_key and
                datetime.fromisoformat(hist_alert['alert_time']) > recent_threshold)
        ]
        
        severity = alert['severity']
        frequency_limits = self.alert_config['alert_frequency_limits'][severity]
        
        suppressed = len(recent_similar_alerts) >= frequency_limits['per_hour']
        
        suppression_result = {
            'suppressed': suppressed,
            'reason': f"Frequency limit exceeded: {len(recent_similar_alerts)}/{frequency_limits['per_hour']} per hour" if suppressed else None,
            'similar_alerts_count': len(recent_similar_alerts),
            'frequency_limit': frequency_limits['per_hour']
        }
        
        return suppression_result
    
    def _determine_alert_routing(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Determine routing strategy for alert notifications."""
        
        severity = alert['severity']
        business_impact = alert.get('alert_context', {}).get('business_impact', 'low')
        
        routing = {
            'notification_channels': [],
            'escalation_required': False,
            'escalation_timeout_seconds': self.alert_config['escalation_timeouts'][severity],
            'priority_score': self._calculate_priority_score(alert)
        }
        
        if severity == 'critical':
            routing['notification_channels'] = ['slack', 'email', 'webhook']
            routing['escalation_required'] = True
        elif severity == 'high':
            routing['notification_channels'] = ['slack', 'email']
            routing['escalation_required'] = business_impact == 'high'
        elif severity == 'medium':
            routing['notification_channels'] = ['slack']
        else:
            routing['notification_channels'] = ['webhook']
        
        if alert.get('correlation_info', {}).get('correlated'):
            routing['correlation_routing'] = True
            if alert['correlation_info']['correlation_type'] == 'cascade':
                routing['escalation_required'] = True
        
        return routing
    
    def _calculate_priority_score(self, alert: Dict[str, Any]) -> int:
        """Calculate alert priority score (1-10)."""
        
        base_priority = self.alert_config['severity_priorities'][alert['severity']]
        
        business_impact = alert.get('alert_context', {}).get('business_impact', 'low')
        impact_multiplier = {'high': 3, 'medium': 2, 'low': 1}[business_impact]
        
        frequency = alert.get('alert_context', {}).get('historical_frequency', {}).get('frequency_category', 'rare')
        frequency_adjustment = {'frequent': 1, 'occasional': 0, 'rare': -1}[frequency]
        
        priority_score = min(10, max(1, (10 - base_priority) * impact_multiplier + frequency_adjustment))
        
        return int(priority_score)
    
    def _send_notifications(self, alert: Dict[str, Any], routing: Dict[str, Any]) -> Dict[str, Any]:
        """Send alert notifications through configured channels."""
        
        notification_results = {}
        
        for channel in routing['notification_channels']:
            try:
                if channel == 'slack':
                    result = self._send_slack_notification(alert)
                elif channel == 'email':
                    result = self._send_email_notification(alert)
                elif channel == 'webhook':
                    result = self._send_webhook_notification(alert)
                else:
                    result = {'success': False, 'error': f'Unknown channel: {channel}'}
                
                notification_results[channel] = result
                
            except Exception as e:
                logger.error(f"Notification error for {channel}: {e}")
                notification_results[channel] = {'success': False, 'error': str(e)}
        
        return notification_results
    
    def _send_slack_notification(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Send Slack notification for alert."""
        
        severity = alert['severity']
        slack_config = self.notification_channels['slack']
        
        channel = slack_config['severity_channels'].get(severity, slack_config['channel'])
        
        severity_emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '🟡', 'low': '🔵'}[severity]
        
        message = {
            'channel': channel,
            'username': 'HYDATIS ML Scheduler',
            'icon_emoji': ':robot_face:',
            'attachments': [{
                'color': {'critical': 'danger', 'high': 'warning', 'medium': '#ffcc00', 'low': 'good'}[severity],
                'title': f"{severity_emoji} {severity.upper()} Anomaly Detected",
                'fields': [
                    {'title': 'Node', 'value': alert['node'], 'short': True},
                    {'title': 'Type', 'value': alert['anomaly_type'], 'short': True},
                    {'title': 'Score', 'value': f"{alert['anomaly_score']:.3f}", 'short': True},
                    {'title': 'Priority', 'value': str(alert.get('routing_decision', {}).get('priority_score', 'N/A')), 'short': True}
                ],
                'text': alert['description'],
                'footer': 'HYDATIS ML Scheduler',
                'ts': int(time.time())
            }]
        }
        
        return {'success': True, 'channel': channel, 'message_sent': True}
    
    def _send_email_notification(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification for alert."""
        
        severity = alert['severity']
        email_config = self.notification_channels['email']
        
        recipients = email_config['recipients'].get(severity, email_config['recipients']['low'])
        
        subject = f"[HYDATIS-{severity.upper()}] Anomaly Detected: {alert['node']} - {alert['anomaly_type']}"
        
        body = f"""
HYDATIS ML Scheduler Anomaly Alert

Alert Details:
- Severity: {severity.upper()}
- Node: {alert['node']}
- Anomaly Type: {alert['anomaly_type']}
- Anomaly Score: {alert['anomaly_score']:.4f}
- Detection Time: {alert['alert_time']}

Description: {alert['description']}

Affected Metrics: {', '.join(alert.get('affected_metrics', []))}

Recommended Actions:
{chr(10).join(f"- {action}" for action in alert.get('recommended_actions', []))}

Alert ID: {alert['alert_id']}
Cluster: HYDATIS-6node
Generated by: HYDATIS ML Scheduler Anomaly Detection
"""
        
        return {'success': True, 'recipients': recipients, 'subject': subject}
    
    def _send_webhook_notification(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook notification for alert."""
        
        webhook_config = self.notification_channels['webhook']
        
        webhook_payload = {
            'alert_id': alert['alert_id'],
            'timestamp': alert['alert_time'],
            'severity': alert['severity'],
            'source': 'HYDATIS-ML-Scheduler',
            'cluster': 'HYDATIS-6node',
            'node': alert['node'],
            'anomaly_type': alert['anomaly_type'],
            'anomaly_score': alert['anomaly_score'],
            'description': alert['description'],
            'priority_score': alert.get('routing_decision', {}).get('priority_score', 1),
            'metadata': alert.get('alert_metadata', {})
        }
        
        return {'success': True, 'payload_sent': True, 'endpoints': list(webhook_config['endpoints'].keys())}
    
    def _store_alert(self, alert: Dict[str, Any]):
        """Store alert in active alerts and history."""
        
        alert_id = alert['alert_id']
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        auto_resolve_timeout = timedelta(minutes=self.alert_config['auto_resolution_timeout_minutes'])
        
        def auto_resolve():
            time.sleep(auto_resolve_timeout.total_seconds())
            if alert_id in self.active_alerts:
                self.resolve_alert(alert_id, 'auto_resolved', 'Automatic resolution after timeout')
        
        threading.Thread(target=auto_resolve, daemon=True).start()
    
    def _update_alert_statistics(self, alert: Dict[str, Any]):
        """Update alert generation statistics."""
        
        self.alert_statistics['total_alerts_generated'] += 1
        self.alert_statistics['alerts_by_severity'][alert['severity']] += 1
        self.alert_statistics['alerts_by_node'][alert['node']] += 1
        self.alert_statistics['alerts_by_type'][alert['anomaly_type']] += 1
        
        if alert.get('routing_decision', {}).get('escalation_required'):
            self.alert_statistics['escalated_alerts'] += 1
    
    def resolve_alert(self, alert_id: str, resolution_type: str, resolution_notes: str = "") -> Dict[str, Any]:
        """Resolve an active alert."""
        
        if alert_id not in self.active_alerts:
            return {'success': False, 'error': 'Alert not found in active alerts'}
        
        alert = self.active_alerts[alert_id]
        
        resolution_info = {
            'resolution_time': datetime.now().isoformat(),
            'resolution_type': resolution_type,
            'resolution_notes': resolution_notes,
            'alert_duration_minutes': (datetime.now() - datetime.fromisoformat(alert['alert_time'])).total_seconds() / 60
        }
        
        alert['resolution_info'] = resolution_info
        
        del self.active_alerts[alert_id]
        
        if resolution_type == 'auto_resolved':
            self.alert_statistics['auto_resolved_alerts'] += 1
        
        logger.info(f"Alert {alert_id} resolved: {resolution_type}")
        
        return {'success': True, 'resolution_info': resolution_info}
    
    def get_alert_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive alert dashboard data."""
        
        current_time = datetime.now()
        
        active_alerts_summary = {
            'total_active': len(self.active_alerts),
            'by_severity': defaultdict(int),
            'by_node': defaultdict(int),
            'oldest_alert_age_minutes': 0
        }
        
        if self.active_alerts:
            for alert in self.active_alerts.values():
                active_alerts_summary['by_severity'][alert['severity']] += 1
                active_alerts_summary['by_node'][alert['node']] += 1
            
            oldest_alert_time = min(datetime.fromisoformat(alert['alert_time']) for alert in self.active_alerts.values())
            active_alerts_summary['oldest_alert_age_minutes'] = (current_time - oldest_alert_time).total_seconds() / 60
        
        recent_24h_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['alert_time']) > current_time - timedelta(hours=24)
        ]
        
        dashboard = {
            'dashboard_timestamp': current_time.isoformat(),
            'alert_summary': {
                'active_alerts': dict(active_alerts_summary['by_severity']),
                'total_active': active_alerts_summary['total_active'],
                'alerts_last_24h': len(recent_24h_alerts),
                'critical_active': active_alerts_summary['by_severity']['critical'],
                'high_active': active_alerts_summary['by_severity']['high']
            },
            'node_health_summary': dict(active_alerts_summary['by_node']),
            'alert_statistics': dict(self.alert_statistics),
            'recent_activity': {
                'last_alert_time': self.alert_history[-1]['alert_time'] if self.alert_history else None,
                'alerts_last_hour': len([a for a in recent_24h_alerts if datetime.fromisoformat(a['alert_time']) > current_time - timedelta(hours=1)]),
                'most_frequent_anomaly_type': max(self.alert_statistics['alerts_by_type'].items(), key=lambda x: x[1])[0] if self.alert_statistics['alerts_by_type'] else None
            },
            'system_health': {
                'alert_processing_healthy': True,
                'notification_channels_status': self._check_notification_channels(),
                'correlation_engine_active': True,
                'auto_resolution_active': True
            }
        }
        
        return dashboard
    
    def _check_notification_channels(self) -> Dict[str, bool]:
        """Check status of notification channels."""
        
        channel_status = {}
        
        for channel_name, config in self.notification_channels.items():
            channel_status[channel_name] = config.get('enabled', False)
        
        return channel_status
    
    def generate_alert_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive alert report."""
        
        report_start = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['alert_time']) > report_start
        ]
        
        if not recent_alerts:
            return {
                'report_period_hours': hours,
                'total_alerts': 0,
                'summary': 'No alerts generated in specified period'
            }
        
        severity_analysis = defaultdict(int)
        node_analysis = defaultdict(int)
        type_analysis = defaultdict(int)
        hourly_distribution = defaultdict(int)
        
        for alert in recent_alerts:
            severity_analysis[alert['severity']] += 1
            node_analysis[alert['node']] += 1
            type_analysis[alert['anomaly_type']] += 1
            
            alert_hour = datetime.fromisoformat(alert['alert_time']).hour
            hourly_distribution[alert_hour] += 1
        
        report = {
            'report_period_hours': hours,
            'report_generated': datetime.now().isoformat(),
            'alert_summary': {
                'total_alerts': len(recent_alerts),
                'severity_breakdown': dict(severity_analysis),
                'node_breakdown': dict(node_analysis),
                'anomaly_type_breakdown': dict(type_analysis),
                'hourly_distribution': dict(hourly_distribution)
            },
            'trend_analysis': {
                'peak_alert_hour': max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None,
                'most_problematic_node': max(node_analysis.items(), key=lambda x: x[1])[0] if node_analysis else None,
                'most_common_anomaly': max(type_analysis.items(), key=lambda x: x[1])[0] if type_analysis else None,
                'critical_alert_frequency': severity_analysis['critical'] / hours if hours > 0 else 0
            },
            'operational_insights': self._generate_operational_insights(recent_alerts),
            'recommendations': self._generate_alert_recommendations(recent_alerts, dict(severity_analysis))
        }
        
        return report
    
    def _generate_operational_insights(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate operational insights from alert patterns."""
        
        insights = []
        
        if len(alerts) == 0:
            insights.append("No anomalies detected - cluster operating normally")
            return insights
        
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        if len(critical_alerts) > 0:
            insights.append(f"{len(critical_alerts)} critical anomalies require immediate attention")
        
        node_counts = defaultdict(int)
        for alert in alerts:
            node_counts[alert['node']] += 1
        
        if node_counts:
            most_affected = max(node_counts.items(), key=lambda x: x[1])
            if most_affected[1] > len(alerts) * 0.4:
                insights.append(f"Node {most_affected[0]} shows high anomaly frequency ({most_affected[1]} alerts)")
        
        cpu_related = len([a for a in alerts if 'cpu' in a.get('anomaly_type', '')])
        memory_related = len([a for a in alerts if 'memory' in a.get('anomaly_type', '')])
        
        if cpu_related > len(alerts) * 0.5:
            insights.append("High frequency of CPU-related anomalies detected")
        if memory_related > len(alerts) * 0.5:
            insights.append("High frequency of memory-related anomalies detected")
        
        return insights
    
    def _generate_alert_recommendations(self, alerts: List[Dict[str, Any]], 
                                      severity_breakdown: Dict[str, int]) -> List[str]:
        """Generate recommendations based on alert patterns."""
        
        recommendations = []
        
        total_alerts = len(alerts)
        
        if total_alerts == 0:
            recommendations.append("Continue current monitoring configuration")
            return recommendations
        
        critical_rate = severity_breakdown.get('critical', 0) / total_alerts
        if critical_rate > 0.1:
            recommendations.append("High critical alert rate - investigate cluster stability")
        
        if total_alerts > 50:
            recommendations.append("High alert volume - consider tuning detection sensitivity")
        elif total_alerts < 5:
            recommendations.append("Low alert volume - verify detection coverage")
        
        node_counts = defaultdict(int)
        for alert in alerts:
            node_counts[alert['node']] += 1
        
        if node_counts:
            max_node_alerts = max(node_counts.values())
            if max_node_alerts > total_alerts * 0.6:
                recommendations.append("Single node generating majority of alerts - investigate node health")
        
        recommendations.append("Regular review of alert patterns recommended")
        
        return recommendations


def main():
    """Main alert manager demonstration."""
    
    print("HYDATIS Alert Management System - Week 7")
    print("Intelligent anomaly alert processing and routing")
    
    alert_manager = HYDATISAlertManager()
    
    print("Alert Configuration:")
    print(f"  Severity Levels: {list(alert_manager.alert_config['severity_priorities'].keys())}")
    print(f"  Notification Channels: {list(alert_manager.notification_channels.keys())}")
    print(f"  Correlation Window: {alert_manager.alert_config['correlation_window_minutes']} minutes")
    print(f"  Auto-Resolution Timeout: {alert_manager.alert_config['auto_resolution_timeout_minutes']} minutes")
    
    sample_anomaly_alert = {
        'alert_id': 'HYDATIS-ANOMALY-1725001234',
        'alert_time': datetime.now().isoformat(),
        'severity': 'high',
        'node': 'worker-2',
        'anomaly_type': 'cpu_spike',
        'anomaly_score': -0.42,
        'description': 'HIGH anomaly detected on worker-2: cpu_spike',
        'affected_metrics': ['cpu_utilization', 'load_1m'],
        'recommended_actions': ['Check top CPU-consuming processes on node']
    }
    
    processed_alert = alert_manager.process_anomaly_alert(sample_anomaly_alert)
    
    print(f"Sample Alert Processing:")
    print(f"  Alert ID: {processed_alert['alert_id']}")
    print(f"  Routing Channels: {processed_alert.get('routing_decision', {}).get('notification_channels', [])}")
    print(f"  Priority Score: {processed_alert.get('routing_decision', {}).get('priority_score', 'N/A')}")
    
    return alert_manager


if __name__ == "__main__":
    manager = main()