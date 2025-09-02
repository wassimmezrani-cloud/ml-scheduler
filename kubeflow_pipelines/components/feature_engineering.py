"""
Feature Engineering Component for HYDATIS ML Scheduler
Implements comprehensive feature engineering with business-aligned transformations.
"""

from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Metrics
import pandas as pd
from typing import NamedTuple

@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3", 
        "scikit-learn==1.3.0",
        "prometheus-client==0.17.1",
        "mlflow==2.5.0"
    ]
)
def feature_engineering_component(
    raw_data: Input[Dataset],
    business_targets: dict,
    feature_config: dict,
    engineered_features: Output[Dataset],
    feature_quality_metrics: Output[Metrics]
) -> NamedTuple("FeatureEngineeringOutput", [("feature_count", int), ("quality_score", float), ("business_alignment", str)]):
    """
    Advanced feature engineering with HYDATIS business alignment.
    
    Args:
        raw_data: Input dataset from data validation
        business_targets: HYDATIS business objectives (CPU, availability, ROI)
        feature_config: Feature engineering configuration
        engineered_features: Output engineered features dataset
        feature_quality_metrics: Feature quality metrics
    
    Returns:
        FeatureEngineeringOutput with feature count, quality score, and business alignment
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    import json
    import os
    
    print("🔧 Starting HYDATIS Feature Engineering...")
    
    # Load raw data
    df = pd.read_parquet(raw_data.path)
    print(f"📊 Loaded {len(df)} samples with {len(df.columns)} raw features")
    
    # HYDATIS-specific feature engineering
    hydatis_cpu_target = business_targets.get('cpu_target', 0.65)
    hydatis_availability_target = business_targets.get('availability_target', 0.997)
    
    # 1. CPU Utilization Features
    df['cpu_utilization_deviation'] = df['cpu_utilization'] - hydatis_cpu_target
    df['cpu_overload_risk'] = (df['cpu_utilization'] > 0.8).astype(int)
    df['cpu_underutilization'] = (df['cpu_utilization'] < 0.3).astype(int)
    df['cpu_target_proximity'] = 1 - abs(df['cpu_utilization'] - hydatis_cpu_target)
    
    # 2. Memory Efficiency Features
    df['memory_pressure_index'] = df['memory_utilization'] / df['memory_requests']
    df['memory_efficiency_score'] = (df['memory_utilization'] * df['cpu_utilization']) / 2
    df['memory_fragmentation'] = df['memory_available'] - df['memory_utilization']
    
    # 3. Availability and Reliability Features  
    df['availability_impact_score'] = df['pod_restarts'] * (1 - hydatis_availability_target)
    df['reliability_index'] = 1 - (df['failed_jobs'] / (df['successful_jobs'] + df['failed_jobs'] + 1))
    df['sla_achievement_score'] = (df['uptime'] / df['total_time']).clip(0, 1)
    
    # 4. Network and I/O Features
    df['network_efficiency'] = df['network_throughput'] / df['network_capacity']
    df['io_latency_impact'] = df['disk_io_wait'] * df['cpu_utilization']
    df['storage_efficiency'] = df['storage_used'] / df['storage_capacity']
    
    # 5. Temporal and Trend Features
    df['workload_volatility'] = df.groupby('node_id')['cpu_utilization'].rolling(window=5).std().reset_index(0, drop=True)
    df['trend_cpu_utilization'] = df.groupby('node_id')['cpu_utilization'].pct_change().fillna(0)
    df['peak_hour_indicator'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    # 6. Business ROI Features
    df['cost_efficiency'] = df['workload_completed'] / (df['cpu_cost'] + df['memory_cost'])
    df['roi_contribution'] = df['business_value_generated'] / df['resource_cost']
    df['performance_per_dollar'] = df['throughput'] / df['total_cost']
    
    # 7. Advanced Composite Features
    df['hydatis_optimization_score'] = (
        0.4 * df['cpu_target_proximity'] +
        0.3 * df['availability_impact_score'] + 
        0.2 * df['roi_contribution'] +
        0.1 * df['reliability_index']
    )
    
    df['cluster_health_index'] = (
        df['cpu_utilization'] * df['memory_efficiency_score'] * df['reliability_index']
    ).clip(0, 1)
    
    # Feature selection based on business importance
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'node_id']]
    
    # Apply scaling based on configuration
    scaler_type = feature_config.get('scaling_method', 'robust')
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    scaled_features = scaler.fit_transform(df[feature_columns])
    df_scaled = pd.DataFrame(scaled_features, columns=feature_columns, index=df.index)
    
    # Add back non-scaled columns
    for col in ['timestamp', 'node_id']:
        if col in df.columns:
            df_scaled[col] = df[col]
    
    # Feature selection for business objectives
    use_feature_selection = feature_config.get('feature_selection', True)
    if use_feature_selection and 'target' in df.columns:
        selector = SelectKBest(f_regression, k=min(25, len(feature_columns)))
        selected_features = selector.fit_transform(df_scaled[feature_columns], df['target'])
        selected_feature_names = [feature_columns[i] for i in selector.get_support(indices=True)]
        
        df_final = pd.DataFrame(selected_features, columns=selected_feature_names)
        for col in ['timestamp', 'node_id']:
            if col in df_scaled.columns:
                df_final[col] = df_scaled[col]
    else:
        df_final = df_scaled
    
    # Calculate feature quality metrics
    feature_count = len([col for col in df_final.columns if col not in ['timestamp', 'node_id']])
    
    # Business alignment score
    business_features = [
        'cpu_target_proximity', 'availability_impact_score', 'roi_contribution',
        'hydatis_optimization_score', 'cluster_health_index'
    ]
    business_alignment_score = sum([1 for bf in business_features if bf in df_final.columns]) / len(business_features)
    
    # Feature quality score
    non_null_ratio = df_final.isnull().sum().sum() / (len(df_final) * len(df_final.columns))
    feature_quality_score = (1 - non_null_ratio) * 100
    
    # Save engineered features
    df_final.to_parquet(engineered_features.path, index=False)
    
    # Save quality metrics
    quality_metrics = {
        "feature_count": feature_count,
        "quality_score": feature_quality_score,
        "business_alignment_score": business_alignment_score * 100,
        "hydatis_cpu_features": int('cpu_target_proximity' in df_final.columns),
        "hydatis_availability_features": int('availability_impact_score' in df_final.columns),
        "hydatis_roi_features": int('roi_contribution' in df_final.columns),
        "scaling_method": scaler_type,
        "feature_selection_applied": use_feature_selection
    }
    
    with open(feature_quality_metrics.path, 'w') as f:
        json.dump(quality_metrics, f)
    
    # Determine business alignment status
    if business_alignment_score >= 0.8:
        alignment_status = "EXCELLENT_HYDATIS_ALIGNMENT"
    elif business_alignment_score >= 0.6:
        alignment_status = "GOOD_HYDATIS_ALIGNMENT"
    else:
        alignment_status = "BASIC_HYDATIS_ALIGNMENT"
    
    print(f"✅ Feature engineering completed:")
    print(f"   📊 {feature_count} engineered features")
    print(f"   🎯 {business_alignment_score*100:.1f}% business alignment")
    print(f"   📈 {feature_quality_score:.1f}% quality score")
    print(f"   🏢 Status: {alignment_status}")
    
    return (feature_count, feature_quality_score, alignment_status)

@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0"
    ]
)
def feature_validation_component(
    engineered_features: Input[Dataset],
    validation_config: dict,
    feature_validation_report: Output[Metrics]
) -> NamedTuple("FeatureValidationOutput", [("validation_passed", bool), ("critical_issues", int)]):
    """
    Validate engineered features for HYDATIS business requirements.
    """
    import pandas as pd
    import numpy as np
    import json
    
    print("🔍 Validating HYDATIS engineered features...")
    
    # Load engineered features
    df = pd.read_parquet(engineered_features.path)
    
    # Validation checks
    issues = []
    critical_issues = 0
    
    # 1. Required HYDATIS features check
    required_features = [
        'cpu_target_proximity', 'availability_impact_score', 'roi_contribution',
        'hydatis_optimization_score'
    ]
    
    for feature in required_features:
        if feature not in df.columns:
            issues.append(f"Missing critical HYDATIS feature: {feature}")
            critical_issues += 1
    
    # 2. Feature distribution validation
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['timestamp', 'node_id']:
            if df[col].isnull().sum() > len(df) * 0.1:  # >10% null values
                issues.append(f"High null rate in {col}: {df[col].isnull().sum()/len(df)*100:.1f}%")
            
            if df[col].std() == 0:  # No variance
                issues.append(f"Zero variance in feature: {col}")
                critical_issues += 1
    
    # 3. HYDATIS business range validation
    if 'cpu_target_proximity' in df.columns:
        if df['cpu_target_proximity'].min() < 0 or df['cpu_target_proximity'].max() > 1:
            issues.append("CPU target proximity outside valid range [0,1]")
            critical_issues += 1
    
    # 4. Feature correlation analysis
    correlation_threshold = validation_config.get('max_correlation', 0.95)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        if high_corr_pairs:
            issues.append(f"High correlation detected in {len(high_corr_pairs)} feature pairs")
    
    # Generate validation report
    validation_passed = critical_issues == 0
    
    validation_report = {
        "validation_passed": validation_passed,
        "total_issues": len(issues),
        "critical_issues": critical_issues,
        "feature_count": len([col for col in df.columns if col not in ['timestamp', 'node_id']]),
        "hydatis_business_features": len([f for f in required_features if f in df.columns]),
        "issues": issues,
        "validation_timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(feature_validation_report.path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    status = "✅ PASSED" if validation_passed else "❌ FAILED"
    print(f"🔍 Feature validation: {status}")
    print(f"   📊 Issues found: {len(issues)} ({critical_issues} critical)")
    print(f"   🎯 HYDATIS features: {len([f for f in required_features if f in df.columns])}/{len(required_features)}")
    
    return (validation_passed, critical_issues)