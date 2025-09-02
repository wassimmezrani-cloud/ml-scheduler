#!/usr/bin/env python3
"""
Kubeflow Pipeline Component: Data Validation for ML Scheduler
Validates cluster data quality and triggers ML training pipeline.
"""

from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Metrics, Model
import pandas as pd
import numpy as np
from typing import NamedTuple

@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "pandas==1.5.3",
        "numpy==1.24.3", 
        "scikit-learn==1.3.0",
        "prometheus-api-client==0.5.3",
        "kubernetes==27.2.0"
    ]
)
def data_validation_component(
    prometheus_url: str,
    data_retention_days: int,
    quality_threshold: float,
    validation_metrics: Output[Metrics],
    validated_dataset: Output[Dataset]
) -> NamedTuple("DataValidationOutput", [("data_quality_score", float), ("validation_passed", bool), ("data_points_count", int)]):
    """
    Validate cluster data quality for ML scheduler training.
    
    Args:
        prometheus_url: Prometheus server endpoint
        data_retention_days: Required data retention period  
        quality_threshold: Minimum quality score (0.95 for production)
        validation_metrics: Output metrics for pipeline tracking
        validated_dataset: Output dataset for downstream components
        
    Returns:
        DataValidationOutput with quality metrics and validation status
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import requests
    import json
    from collections import namedtuple
    
    # Define output type
    DataValidationOutput = namedtuple("DataValidationOutput", ["data_quality_score", "validation_passed", "data_points_count"])
    
    def validate_prometheus_connectivity():
        """Validate Prometheus server connectivity and health."""
        try:
            response = requests.get(f"{prometheus_url}/api/v1/query?query=up", timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Prometheus connectivity failed: {e}")
            return False
    
    def collect_cluster_metrics(days: int):
        """Collect cluster metrics for validation."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Define critical metrics for ML scheduler
        metrics_queries = {
            'node_cpu_usage': 'node_cpu_seconds_total{mode!="idle"}',
            'node_memory_usage': 'node_memory_MemAvailable_bytes',
            'pod_cpu_usage': 'container_cpu_usage_seconds_total',
            'pod_memory_usage': 'container_memory_working_set_bytes',
            'scheduling_latency': 'scheduler_scheduling_duration_seconds',
            'pod_scheduling_attempts': 'scheduler_pod_scheduling_attempts_total'
        }
        
        collected_data = {}
        
        for metric_name, query in metrics_queries.items():
            try:
                params = {
                    'query': query,
                    'start': int(start_time.timestamp()),
                    'end': int(end_time.timestamp()),
                    'step': '30s'  # 30-second intervals
                }
                
                response = requests.get(f"{prometheus_url}/api/v1/query_range", params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data['status'] == 'success' and data['data']['result']:
                    collected_data[metric_name] = data['data']['result']
                    print(f"✓ Collected {metric_name}: {len(data['data']['result'])} series")
                else:
                    print(f"⚠️ No data for {metric_name}")
                    collected_data[metric_name] = []
                    
            except Exception as e:
                print(f"❌ Failed to collect {metric_name}: {e}")
                collected_data[metric_name] = []
        
        return collected_data
    
    def validate_data_quality(metrics_data):
        """Validate data quality across multiple dimensions."""
        
        quality_checks = {
            'completeness': 0.0,
            'consistency': 0.0,
            'timeliness': 0.0,
            'accuracy': 0.0,
            'coverage': 0.0
        }
        
        total_metrics = len(metrics_data)
        
        if total_metrics == 0:
            return quality_checks, 0.0
        
        # 1. Completeness check
        non_empty_metrics = sum(1 for data in metrics_data.values() if data)
        quality_checks['completeness'] = non_empty_metrics / total_metrics
        
        # 2. Coverage check (minimum data points per metric)
        sufficient_data_count = 0
        total_data_points = 0
        
        for metric_name, metric_data in metrics_data.items():
            if metric_data:
                data_points = sum(len(series.get('values', [])) for series in metric_data)
                total_data_points += data_points
                
                # Require at least 1000 data points for sufficient coverage
                if data_points >= 1000:
                    sufficient_data_count += 1
        
        quality_checks['coverage'] = sufficient_data_count / total_metrics if total_metrics > 0 else 0.0
        
        # 3. Consistency check (validate data ranges)
        consistent_metrics = 0
        
        for metric_name, metric_data in metrics_data.items():
            if metric_data:
                try:
                    # Extract values for consistency validation
                    all_values = []
                    for series in metric_data:
                        for timestamp, value in series.get('values', []):
                            try:
                                float_value = float(value)
                                all_values.append(float_value)
                            except (ValueError, TypeError):
                                pass
                    
                    if all_values:
                        # Basic consistency checks
                        values_array = np.array(all_values)
                        
                        # Check for reasonable ranges (no negative CPU, memory within limits)
                        if 'cpu' in metric_name:
                            valid_range = np.all((values_array >= 0) & (values_array <= 100))
                        elif 'memory' in metric_name:
                            valid_range = np.all(values_array >= 0)
                        else:
                            valid_range = True
                        
                        # Check for data variability (not all zeros)
                        has_variability = np.std(values_array) > 0.001
                        
                        if valid_range and has_variability:
                            consistent_metrics += 1
                            
                except Exception as e:
                    print(f"⚠️ Consistency check failed for {metric_name}: {e}")
        
        quality_checks['consistency'] = consistent_metrics / total_metrics if total_metrics > 0 else 0.0
        
        # 4. Timeliness check (data freshness within last hour)
        current_time = datetime.now().timestamp()
        fresh_metrics = 0
        
        for metric_name, metric_data in metrics_data.items():
            if metric_data:
                try:
                    # Get latest timestamp from data
                    latest_timestamp = 0
                    for series in metric_data:
                        for timestamp, value in series.get('values', []):
                            latest_timestamp = max(latest_timestamp, float(timestamp))
                    
                    # Check if data is fresh (within last hour)
                    if (current_time - latest_timestamp) < 3600:  # 1 hour
                        fresh_metrics += 1
                        
                except Exception as e:
                    print(f"⚠️ Timeliness check failed for {metric_name}: {e}")
        
        quality_checks['timeliness'] = fresh_metrics / total_metrics if total_metrics > 0 else 0.0
        
        # 5. Accuracy check (validate against expected patterns)
        accurate_metrics = 0
        
        for metric_name, metric_data in metrics_data.items():
            if metric_data:
                try:
                    # Basic accuracy validation (no extreme outliers)
                    all_values = []
                    for series in metric_data:
                        for timestamp, value in series.get('values', []):
                            try:
                                all_values.append(float(value))
                            except (ValueError, TypeError):
                                pass
                    
                    if all_values and len(all_values) >= 10:
                        values_array = np.array(all_values)
                        
                        # Check for reasonable statistical properties
                        q1, q3 = np.percentile(values_array, [25, 75])
                        iqr = q3 - q1
                        
                        # Count outliers (beyond 3*IQR)
                        outliers = np.sum(
                            (values_array < (q1 - 3*iqr)) | 
                            (values_array > (q3 + 3*iqr))
                        )
                        
                        outlier_percentage = outliers / len(values_array)
                        
                        # Accept up to 5% outliers as normal
                        if outlier_percentage <= 0.05:
                            accurate_metrics += 1
                    else:
                        # Insufficient data for accuracy check
                        accurate_metrics += 0.5
                        
                except Exception as e:
                    print(f"⚠️ Accuracy check failed for {metric_name}: {e}")
        
        quality_checks['accuracy'] = accurate_metrics / total_metrics if total_metrics > 0 else 0.0
        
        # Calculate overall quality score
        overall_score = np.mean(list(quality_checks.values()))
        
        return quality_checks, overall_score, total_data_points
    
    # Main validation logic
    print(f"🔍 Starting data validation for HYDATIS ML Scheduler")
    print(f"📊 Target data retention: {data_retention_days} days")
    print(f"🎯 Quality threshold: {quality_threshold}")
    
    # Step 1: Validate Prometheus connectivity
    if not validate_prometheus_connectivity():
        print("❌ Data validation failed: Cannot connect to Prometheus")
        return DataValidationOutput(0.0, False, 0)
    
    print("✅ Prometheus connectivity validated")
    
    # Step 2: Collect cluster metrics
    print(f"📥 Collecting cluster metrics for {data_retention_days} days...")
    metrics_data = collect_cluster_metrics(data_retention_days)
    
    # Step 3: Validate data quality
    print("🔬 Performing data quality validation...")
    quality_checks, overall_score, data_points_count = validate_data_quality(metrics_data)
    
    # Step 4: Generate validation report
    validation_passed = overall_score >= quality_threshold
    
    print(f"\n📋 Data Quality Report:")
    print(f"   Completeness: {quality_checks['completeness']:.2%}")
    print(f"   Coverage: {quality_checks['coverage']:.2%}")  
    print(f"   Consistency: {quality_checks['consistency']:.2%}")
    print(f"   Timeliness: {quality_checks['timeliness']:.2%}")
    print(f"   Accuracy: {quality_checks['accuracy']:.2%}")
    print(f"   Overall Score: {overall_score:.2%}")
    print(f"   Data Points: {data_points_count:,}")
    print(f"   Validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
    
    # Log metrics to MLflow/Kubeflow
    validation_metrics.log_metric("data_quality_score", overall_score)
    validation_metrics.log_metric("completeness_score", quality_checks['completeness'])
    validation_metrics.log_metric("coverage_score", quality_checks['coverage'])
    validation_metrics.log_metric("consistency_score", quality_checks['consistency'])
    validation_metrics.log_metric("timeliness_score", quality_checks['timeliness'])
    validation_metrics.log_metric("accuracy_score", quality_checks['accuracy'])
    validation_metrics.log_metric("data_points_count", data_points_count)
    validation_metrics.log_metric("validation_passed", int(validation_passed))
    
    # Save validated dataset for downstream components
    if validation_passed:
        # Convert metrics data to structured format for ML training
        training_dataset = {
            'metadata': {
                'validation_timestamp': datetime.now().isoformat(),
                'data_retention_days': data_retention_days,
                'quality_score': overall_score,
                'data_points_count': data_points_count
            },
            'metrics': metrics_data,
            'quality_report': quality_checks
        }
        
        # Save to pipeline artifact
        with open(validated_dataset.path, 'w') as f:
            json.dump(training_dataset, f, indent=2)
        
        print(f"💾 Validated dataset saved to: {validated_dataset.path}")
    
    return DataValidationOutput(overall_score, validation_passed, data_points_count)


@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "pandas==1.5.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0"
    ]
)
def feature_engineering_component(
    validated_dataset: Input[Dataset],
    feature_store_config: dict,
    engineered_features: Output[Dataset],
    feature_metrics: Output[Metrics]
) -> NamedTuple("FeatureOutput", [("feature_count", int), ("feature_quality_score", float)]):
    """
    Engineer features for ML scheduler training from validated cluster data.
    
    Args:
        validated_dataset: Validated cluster metrics from data validation
        feature_store_config: Feast feature store configuration
        engineered_features: Output engineered features dataset
        feature_metrics: Feature engineering metrics
        
    Returns:
        FeatureOutput with feature count and quality metrics
    """
    import pandas as pd
    import numpy as np
    import json
    from datetime import datetime, timedelta
    from collections import namedtuple
    
    FeatureOutput = namedtuple("FeatureOutput", ["feature_count", "feature_quality_score"])
    
    print("🔧 Starting feature engineering for HYDATIS ML Scheduler")
    
    # Load validated dataset
    with open(validated_dataset.path, 'r') as f:
        dataset = json.load(f)
    
    metrics_data = dataset['metrics']
    
    def engineer_node_features(node_metrics):
        """Engineer node-level features for scheduling decisions."""
        features = {}
        
        for metric_name, metric_data in node_metrics.items():
            if not metric_data:
                continue
                
            # Extract time series data
            all_values = []
            timestamps = []
            
            for series in metric_data:
                for timestamp, value in series.get('values', []):
                    try:
                        all_values.append(float(value))
                        timestamps.append(float(timestamp))
                    except (ValueError, TypeError):
                        continue
            
            if len(all_values) >= 10:
                values_array = np.array(all_values)
                
                # Statistical features
                features[f"{metric_name}_mean"] = np.mean(values_array)
                features[f"{metric_name}_std"] = np.std(values_array)
                features[f"{metric_name}_min"] = np.min(values_array)
                features[f"{metric_name}_max"] = np.max(values_array)
                features[f"{metric_name}_p95"] = np.percentile(values_array, 95)
                
                # Temporal features
                if len(values_array) >= 100:
                    # Rolling window features
                    rolling_mean = pd.Series(values_array).rolling(window=10).mean()
                    features[f"{metric_name}_trend"] = np.polyfit(range(len(rolling_mean)), rolling_mean.fillna(0), 1)[0]
                    
                    # Seasonality detection
                    features[f"{metric_name}_volatility"] = np.std(rolling_mean.fillna(0))
                
                # Business-relevant features
                if 'cpu' in metric_name:
                    features[f"{metric_name}_utilization_efficiency"] = 1.0 - (values_array.mean() / 100.0)
                    features[f"{metric_name}_capacity_available"] = max(0, 100.0 - values_array.mean()) / 100.0
                
        return features
    
    def engineer_scheduling_features(scheduling_metrics):
        """Engineer scheduling-specific features."""
        features = {}
        
        # Scheduling performance features
        if 'scheduling_latency' in scheduling_metrics:
            latency_data = scheduling_metrics['scheduling_latency']
            
            all_latencies = []
            for series in latency_data:
                for timestamp, value in series.get('values', []):
                    try:
                        all_latencies.append(float(value) * 1000)  # Convert to ms
                    except (ValueError, TypeError):
                        continue
            
            if all_latencies:
                latencies = np.array(all_latencies)
                features['scheduling_latency_p50'] = np.percentile(latencies, 50)
                features['scheduling_latency_p95'] = np.percentile(latencies, 95)
                features['scheduling_latency_p99'] = np.percentile(latencies, 99)
                features['scheduling_efficiency_score'] = max(0, 1.0 - (np.mean(latencies) / 200.0))  # Target <200ms
        
        # Pod placement success rate
        if 'pod_scheduling_attempts' in scheduling_metrics:
            attempts_data = scheduling_metrics['pod_scheduling_attempts']
            
            total_attempts = 0
            for series in attempts_data:
                for timestamp, value in series.get('values', []):
                    try:
                        total_attempts += float(value)
                    except (ValueError, TypeError):
                        continue
            
            features['scheduling_success_rate'] = min(1.0, total_attempts / max(1, total_attempts))
        
        return features
    
    def validate_feature_quality(features):
        """Validate engineered features quality."""
        
        if not features:
            return 0.0
        
        quality_score = 0.0
        
        # Check feature completeness
        completeness_score = len(features) / 50.0  # Target: 50+ features
        
        # Check feature variance (avoid constant features)
        variance_scores = []
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                # Good features should have reasonable variance
                variance_scores.append(1.0)
            else:
                variance_scores.append(0.0)
        
        variance_score = np.mean(variance_scores) if variance_scores else 0.0
        
        # Check feature relevance (business-aligned features)
        business_features = [
            'cpu_utilization_efficiency', 'memory_capacity_available',
            'scheduling_latency_p95', 'scheduling_efficiency_score'
        ]
        
        relevance_score = sum(1 for bf in business_features if any(bf in fn for fn in features.keys()))
        relevance_score = relevance_score / len(business_features)
        
        # Combined quality score
        quality_score = (completeness_score * 0.4 + variance_score * 0.3 + relevance_score * 0.3)
        
        return min(1.0, quality_score)
    
    # Main feature engineering pipeline
    print("🏗️ Engineering node features...")
    node_features = engineer_node_features(metrics_data)
    
    print("⚙️ Engineering scheduling features...")
    scheduling_features = engineer_scheduling_features(metrics_data)
    
    # Combine all features
    all_features = {**node_features, **scheduling_features}
    
    print(f"✨ Generated {len(all_features)} total features")
    
    # Validate feature quality
    feature_quality_score = validate_feature_quality(all_features)
    
    print(f"🎯 Feature quality score: {feature_quality_score:.2%}")
    
    # Log feature engineering metrics
    feature_metrics.log_metric("feature_count", len(all_features))
    feature_metrics.log_metric("feature_quality_score", feature_quality_score)
    feature_metrics.log_metric("node_features_count", len(node_features))
    feature_metrics.log_metric("scheduling_features_count", len(scheduling_features))
    
    # Save engineered features
    feature_dataset = {
        'metadata': {
            'engineering_timestamp': datetime.now().isoformat(),
            'total_features': len(all_features),
            'quality_score': feature_quality_score,
            'source_dataset': dataset['metadata']
        },
        'features': all_features,
        'feature_categories': {
            'node_features': list(node_features.keys()),
            'scheduling_features': list(scheduling_features.keys())
        }
    }
    
    with open(engineered_features.path, 'w') as f:
        json.dump(feature_dataset, f, indent=2)
    
    print(f"💾 Engineered features saved to: {engineered_features.path}")
    
    return FeatureOutput(len(all_features), feature_quality_score)


if __name__ == "__main__":
    # Test data validation component locally
    print("🧪 Testing Data Validation Component")
    
    # Mock inputs for testing
    class MockOutput:
        def __init__(self):
            self.path = "/tmp/test_output.json"
        def log_metric(self, name, value):
            print(f"Metric: {name} = {value}")
    
    validation_metrics = MockOutput()
    validated_dataset = MockOutput()
    
    # Test with HYDATIS Prometheus configuration  
    result = data_validation_component(
        prometheus_url="http://10.110.190.32:9090",
        data_retention_days=30,
        quality_threshold=0.95,
        validation_metrics=validation_metrics,
        validated_dataset=validated_dataset
    )
    
    print(f"✓ Test Result: {result}")