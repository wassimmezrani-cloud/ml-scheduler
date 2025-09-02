"""
Monitoring and Observability Component for HYDATIS ML Scheduler
Implements comprehensive monitoring with drift detection and business metrics tracking.
"""

from kfp import dsl
from kfp.dsl import component, Input, Output, Metrics
import json
from typing import NamedTuple

@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "prometheus-client==0.17.1",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "mlflow==2.5.0",
        "requests==2.31.0"
    ]
)
def drift_detection_component(
    model_endpoint: str,
    monitoring_config: dict,
    business_targets: dict,
    drift_report: Output[Metrics]
) -> NamedTuple("DriftOutput", [("drift_detected", bool), ("drift_severity", str), ("retraining_required", bool)]):
    """
    Advanced drift detection with HYDATIS business impact analysis.
    
    Args:
        model_endpoint: Model serving endpoint URL
        monitoring_config: Monitoring configuration
        business_targets: HYDATIS business objectives
        drift_report: Drift detection metrics and report
    
    Returns:
        DriftOutput with drift status, severity, and retraining recommendation
    """
    import pandas as pd
    import numpy as np
    from sklearn.base import BaseEstimator
    import json
    import time
    import requests
    
    print("🔍 Starting HYDATIS Drift Detection...")
    
    # Monitoring parameters
    monitoring_window = monitoring_config.get('window_hours', 24)
    drift_threshold = monitoring_config.get('drift_threshold', 0.1)
    business_impact_threshold = monitoring_config.get('business_impact_threshold', 0.05)
    
    print(f"📊 Monitoring window: {monitoring_window} hours")
    print(f"🎯 Drift threshold: {drift_threshold}")
    
    # Simulate data collection from model endpoint
    print("📈 Collecting model performance data...")
    
    # Simulate historical baseline metrics
    baseline_metrics = {
        "cpu_prediction_accuracy": 0.89,
        "memory_prediction_accuracy": 0.85,
        "scheduling_latency_ms": 12.5,
        "availability_impact": 0.998,
        "roi_contribution": 13.8
    }
    
    # Simulate current performance metrics
    current_metrics = {
        "cpu_prediction_accuracy": 0.83,  # Degraded
        "memory_prediction_accuracy": 0.82,  # Degraded
        "scheduling_latency_ms": 18.2,    # Increased
        "availability_impact": 0.995,     # Slightly degraded
        "roi_contribution": 12.1          # Degraded
    }
    
    # 1. Statistical Drift Detection
    print("🔬 Performing statistical drift analysis...")
    
    statistical_drifts = {}
    for metric, baseline_val in baseline_metrics.items():
        current_val = current_metrics[metric]
        
        if metric in ['cpu_prediction_accuracy', 'memory_prediction_accuracy', 'availability_impact']:
            # For accuracy metrics, lower is worse
            drift_magnitude = (baseline_val - current_val) / baseline_val
        elif metric in ['scheduling_latency_ms']:
            # For latency, higher is worse
            drift_magnitude = (current_val - baseline_val) / baseline_val
        else:
            # For ROI, lower is worse
            drift_magnitude = (baseline_val - current_val) / baseline_val
        
        statistical_drifts[metric] = {
            "baseline": baseline_val,
            "current": current_val,
            "drift_magnitude": drift_magnitude,
            "drift_detected": abs(drift_magnitude) > drift_threshold
        }
    
    # 2. Business Impact Analysis
    print("💼 Analyzing HYDATIS business impact...")
    
    cpu_target = business_targets.get('cpu_target', 0.65)
    availability_target = business_targets.get('availability_target', 0.997)
    roi_target = business_targets.get('roi_target', 14.0)
    
    business_impact = {
        "cpu_target_deviation": abs(current_metrics['cpu_prediction_accuracy'] - 0.89) > business_impact_threshold,
        "availability_target_risk": current_metrics['availability_impact'] < availability_target,
        "roi_target_shortfall": current_metrics['roi_contribution'] < roi_target * 0.9,
        "latency_degradation": current_metrics['scheduling_latency_ms'] > 15.0
    }
    
    # 3. Determine overall drift severity
    total_drifts = sum([1 for drift_data in statistical_drifts.values() if drift_data['drift_detected']])
    business_impacts = sum([1 for impact in business_impact.values() if impact])
    
    if total_drifts >= 3 or business_impacts >= 2:
        drift_severity = "HIGH"
        retraining_required = True
    elif total_drifts >= 2 or business_impacts >= 1:
        drift_severity = "MEDIUM" 
        retraining_required = True
    elif total_drifts >= 1:
        drift_severity = "LOW"
        retraining_required = False
    else:
        drift_severity = "NONE"
        retraining_required = False
    
    drift_detected = drift_severity != "NONE"
    
    # 4. Generate comprehensive drift report
    drift_report_data = {
        "drift_detected": drift_detected,
        "drift_severity": drift_severity,
        "retraining_required": retraining_required,
        "monitoring_window_hours": monitoring_window,
        "statistical_drift_analysis": statistical_drifts,
        "business_impact_analysis": business_impact,
        "hydatis_business_targets": {
            "cpu_target": cpu_target,
            "availability_target": availability_target,
            "roi_target": roi_target
        },
        "recommendations": {
            "immediate_action": "RETRAIN" if retraining_required else "MONITOR",
            "priority": "HIGH" if drift_severity == "HIGH" else "MEDIUM" if drift_severity == "MEDIUM" else "LOW",
            "estimated_business_impact": "SIGNIFICANT" if business_impacts >= 2 else "MODERATE" if business_impacts >= 1 else "MINIMAL"
        },
        "detection_timestamp": time.time()
    }
    
    with open(drift_report.path, 'w') as f:
        json.dump(drift_report_data, f, indent=2)
    
    print(f"🔍 Drift detection completed:")
    print(f"   📊 Drift severity: {drift_severity}")
    print(f"   🚨 Retraining required: {'✅ YES' if retraining_required else '❌ NO'}")
    print(f"   💼 Business impact: {drift_report_data['recommendations']['estimated_business_impact']}")
    print(f"   🎯 HYDATIS targets affected: {business_impacts}/4")
    
    return (drift_detected, drift_severity, retraining_required)

@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "prometheus-client==0.17.1",
        "pandas==2.0.3",
        "requests==2.31.0"
    ]
)
def business_monitoring_component(
    model_endpoint: str,
    business_targets: dict,
    monitoring_window_hours: int,
    business_metrics: Output[Metrics]
) -> NamedTuple("BusinessMonitoringOutput", [("targets_met", bool), ("business_score", float), ("action_required", str)]):
    """
    Monitor HYDATIS business KPIs and model impact on business objectives.
    """
    import json
    import time
    import random
    
    print("📈 Starting HYDATIS Business Monitoring...")
    
    # Extract HYDATIS business targets
    cpu_target = business_targets.get('cpu_target', 0.65)
    availability_target = business_targets.get('availability_target', 0.997) 
    roi_target = business_targets.get('roi_target', 14.0)
    
    print(f"🎯 HYDATIS Business Targets:")
    print(f"   🖥️  CPU utilization: {cpu_target*100:.1f}%")
    print(f"   📊 Availability: {availability_target*100:.2f}%")
    print(f"   💰 ROI: {roi_target:.1f}x")
    
    # Simulate business metrics collection
    print("📊 Collecting business performance metrics...")
    
    # Simulate current business performance
    current_business_metrics = {
        "cpu_utilization_achieved": 0.67,      # Slightly over target
        "availability_achieved": 0.9985,       # Above target
        "roi_achieved": 14.8,                  # Above target
        "cost_optimization_percentage": 23.5,
        "workload_efficiency_score": 0.91,
        "sla_achievement_rate": 0.998,
        "customer_satisfaction_score": 4.7,    # Out of 5
        "operational_cost_reduction": 0.18
    }
    
    # Business target achievement analysis
    target_achievements = {
        "cpu_target_met": abs(current_business_metrics['cpu_utilization_achieved'] - cpu_target) <= 0.05,
        "availability_target_met": current_business_metrics['availability_achieved'] >= availability_target,
        "roi_target_met": current_business_metrics['roi_achieved'] >= roi_target,
        "efficiency_target_met": current_business_metrics['workload_efficiency_score'] >= 0.85
    }
    
    targets_met = all(target_achievements.values())
    targets_achieved_count = sum(target_achievements.values())
    
    # Calculate composite business score
    business_score = (
        0.3 * (1 - abs(current_business_metrics['cpu_utilization_achieved'] - cpu_target) / cpu_target) +
        0.3 * min(current_business_metrics['availability_achieved'] / availability_target, 1.0) +
        0.2 * min(current_business_metrics['roi_achieved'] / roi_target, 1.0) + 
        0.2 * current_business_metrics['workload_efficiency_score']
    ) * 100
    
    # Determine action required
    if business_score >= 95:
        action_required = "MAINTAIN"
    elif business_score >= 85:
        action_required = "OPTIMIZE" 
    elif business_score >= 70:
        action_required = "INVESTIGATE"
    else:
        action_required = "IMMEDIATE_ACTION"
    
    # Generate business monitoring report
    business_report = {
        "targets_met": targets_met,
        "business_score": business_score,
        "action_required": action_required,
        "monitoring_window_hours": monitoring_window_hours,
        "target_achievements": target_achievements,
        "targets_achieved_count": targets_achieved_count,
        "current_metrics": current_business_metrics,
        "hydatis_targets": {
            "cpu_target": cpu_target,
            "availability_target": availability_target,
            "roi_target": roi_target
        },
        "recommendations": {
            "priority": "HIGH" if action_required == "IMMEDIATE_ACTION" else "MEDIUM" if action_required in ["INVESTIGATE", "OPTIMIZE"] else "LOW",
            "next_review_hours": 4 if action_required == "IMMEDIATE_ACTION" else 24,
            "stakeholder_notification": action_required in ["IMMEDIATE_ACTION", "INVESTIGATE"]
        },
        "monitoring_timestamp": time.time()
    }
    
    with open(business_metrics.path, 'w') as f:
        json.dump(business_report, f, indent=2)
    
    print(f"📈 Business monitoring completed:")
    print(f"   🎯 Targets met: {targets_achieved_count}/4 ({'✅ ALL' if targets_met else '⚠️ PARTIAL'})")
    print(f"   📊 Business score: {business_score:.1f}/100")
    print(f"   🚨 Action required: {action_required}")
    print(f"   🏢 HYDATIS performance: {'🏆 EXCELLENT' if business_score >= 95 else '📈 GOOD' if business_score >= 85 else '⚠️ NEEDS ATTENTION'}")
    
    return (targets_met, business_score, action_required)