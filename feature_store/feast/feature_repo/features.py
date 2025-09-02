#!/usr/bin/env python3
"""
Feast feature definitions for HYDATIS ML scheduler.
Defines features for real-time serving with <50ms latency target.
"""

from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource
from datetime import timedelta
import pandas as pd


# Define entities
node_entity = Entity(
    name="node",
    value_type=ValueType.STRING,
    description="HYDATIS cluster node identifier"
)

pod_entity = Entity(
    name="pod", 
    value_type=ValueType.STRING,
    description="Kubernetes pod identifier"
)

# Node metrics feature source
node_metrics_source = FileSource(
    path="/data/ml_scheduler_longhorn/processed/node_features.parquet",
    timestamp_field="timestamp"
)

# Node resource features
node_resource_features = FeatureView(
    name="node_resources",
    entities=["node"],
    ttl=timedelta(minutes=5),
    features=[
        Feature(name="cpu_usage_current", dtype=ValueType.FLOAT),
        Feature(name="memory_usage_current", dtype=ValueType.FLOAT),
        Feature(name="load_1m", dtype=ValueType.FLOAT),
        Feature(name="load_5m", dtype=ValueType.FLOAT),
        Feature(name="load_15m", dtype=ValueType.FLOAT),
        Feature(name="disk_usage", dtype=ValueType.FLOAT),
        Feature(name="network_rx_rate", dtype=ValueType.FLOAT),
        Feature(name="network_tx_rate", dtype=ValueType.FLOAT)
    ],
    online=True,
    source=node_metrics_source,
    tags={"team": "hydatis-mlops", "component": "ml-scheduler"}
)

# Temporal pattern features
temporal_features_source = FileSource(
    path="/data/ml_scheduler_longhorn/processed/temporal_features.parquet",
    timestamp_field="timestamp"
)

node_temporal_features = FeatureView(
    name="node_temporal_patterns",
    entities=["node"],
    ttl=timedelta(hours=1),
    features=[
        # Rolling averages
        Feature(name="cpu_rolling_mean_5m", dtype=ValueType.FLOAT),
        Feature(name="cpu_rolling_mean_15m", dtype=ValueType.FLOAT),
        Feature(name="cpu_rolling_mean_30m", dtype=ValueType.FLOAT),
        Feature(name="cpu_rolling_std_15m", dtype=ValueType.FLOAT),
        
        # Memory patterns
        Feature(name="memory_rolling_mean_5m", dtype=ValueType.FLOAT),
        Feature(name="memory_rolling_mean_15m", dtype=ValueType.FLOAT),
        Feature(name="memory_rolling_std_15m", dtype=ValueType.FLOAT),
        
        # Trend indicators
        Feature(name="cpu_trend_5m", dtype=ValueType.FLOAT),
        Feature(name="cpu_trend_15m", dtype=ValueType.FLOAT),
        Feature(name="memory_trend_5m", dtype=ValueType.FLOAT),
        
        # Seasonal patterns
        Feature(name="hour_sin", dtype=ValueType.FLOAT),
        Feature(name="hour_cos", dtype=ValueType.FLOAT),
        Feature(name="dow_sin", dtype=ValueType.FLOAT),
        Feature(name="dow_cos", dtype=ValueType.FLOAT),
        Feature(name="is_business_hours", dtype=ValueType.INT32),
        Feature(name="is_weekend", dtype=ValueType.INT32)
    ],
    online=True,
    source=temporal_features_source,
    tags={"team": "hydatis-mlops", "component": "temporal-analysis"}
)

# Node health and capacity features
node_health_source = FileSource(
    path="/data/ml_scheduler_longhorn/processed/node_health.parquet",
    timestamp_field="timestamp"
)

node_health_features = FeatureView(
    name="node_health_capacity",
    entities=["node"],
    ttl=timedelta(minutes=2),
    features=[
        # Capacity indicators
        Feature(name="cpu_capacity_remaining", dtype=ValueType.FLOAT),
        Feature(name="memory_capacity_remaining", dtype=ValueType.FLOAT),
        Feature(name="resource_pressure_score", dtype=ValueType.FLOAT),
        
        # Health indicators
        Feature(name="node_stability_score", dtype=ValueType.FLOAT),
        Feature(name="recent_failures", dtype=ValueType.INT32),
        Feature(name="average_pod_success_rate", dtype=ValueType.FLOAT),
        
        # Workload characteristics
        Feature(name="active_pods_count", dtype=ValueType.INT32),
        Feature(name="pending_pods_count", dtype=ValueType.INT32),
        Feature(name="cpu_memory_ratio", dtype=ValueType.FLOAT),
        Feature(name="load_cpu_efficiency", dtype=ValueType.FLOAT)
    ],
    online=True,
    source=node_health_source,
    tags={"team": "hydatis-mlops", "component": "node-health"}
)

# Scheduling performance features
scheduling_source = FileSource(
    path="/data/ml_scheduler_longhorn/processed/scheduling_features.parquet",
    timestamp_field="timestamp"
)

scheduling_performance_features = FeatureView(
    name="scheduling_performance",
    entities=["node"],
    ttl=timedelta(minutes=1),
    features=[
        # Current scheduler performance
        Feature(name="scheduling_latency_p95", dtype=ValueType.FLOAT),
        Feature(name="scheduling_success_rate", dtype=ValueType.FLOAT),
        Feature(name="pending_duration_avg", dtype=ValueType.FLOAT),
        
        # Historical performance
        Feature(name="scheduling_latency_trend", dtype=ValueType.FLOAT),
        Feature(name="success_rate_trend", dtype=ValueType.FLOAT),
        
        # Queue characteristics
        Feature(name="queue_depth", dtype=ValueType.INT32),
        Feature(name="queue_wait_time_avg", dtype=ValueType.FLOAT)
    ],
    online=True,
    source=scheduling_source,
    tags={"team": "hydatis-mlops", "component": "scheduling-performance"}
)


# Feature service for real-time ML serving
from feast import FeatureService

ml_scheduler_feature_service = FeatureService(
    name="ml_scheduler_features",
    features=[
        node_resource_features,
        node_temporal_features,
        node_health_features,
        scheduling_performance_features
    ],
    tags={"team": "hydatis-mlops", "use_case": "ml-scheduler"}
)


def get_feature_vector_for_node(feast_client, node_id: str) -> Dict:
    """Get complete feature vector for a node for ML inference."""
    
    entity_df = pd.DataFrame({
        "node": [node_id],
        "timestamp": [datetime.now()]
    })
    
    # Retrieve features from online store
    feature_vector = feast_client.get_online_features(
        features=[
            "node_resources:cpu_usage_current",
            "node_resources:memory_usage_current", 
            "node_resources:load_1m",
            "node_temporal_patterns:cpu_rolling_mean_15m",
            "node_temporal_patterns:memory_rolling_mean_15m",
            "node_temporal_patterns:cpu_trend_5m",
            "node_temporal_patterns:hour_sin",
            "node_temporal_patterns:hour_cos",
            "node_temporal_patterns:is_business_hours",
            "node_health_capacity:cpu_capacity_remaining",
            "node_health_capacity:memory_capacity_remaining",
            "node_health_capacity:resource_pressure_score",
            "scheduling_performance:scheduling_latency_p95",
            "scheduling_performance:scheduling_success_rate"
        ],
        entity_df=entity_df
    ).to_dict()
    
    return feature_vector


if __name__ == "__main__":
    print("HYDATIS ML Scheduler Feast Features Defined")
    print(f"Node Entity: {node_entity.name}")
    print(f"Feature Views: 4 views with {sum([len(fv.features) for fv in [node_resource_features, node_temporal_features, node_health_features, scheduling_performance_features]])} features")
    print("Target serving latency: <50ms")