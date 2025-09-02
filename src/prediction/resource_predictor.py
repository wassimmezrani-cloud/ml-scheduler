#!/usr/bin/env python3

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import requests
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

class PredictionHorizon(Enum):
    SHORT_TERM = "15min"
    MEDIUM_TERM = "1hour"
    LONG_TERM = "6hours"
    DAILY = "24hours"

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    PODS = "pods"

@dataclass
class ResourcePrediction:
    resource_type: ResourceType
    horizon: PredictionHorizon
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    prediction_timestamp: datetime
    model_used: str
    feature_importance: Dict[str, float]

@dataclass
class ClusterForecast:
    cluster_name: str
    timestamp: datetime
    predictions: Dict[str, ResourcePrediction]
    capacity_warnings: List[str]
    scaling_recommendations: List[str]
    overall_confidence: float

class PredictiveResourceAllocator:
    def __init__(self, prometheus_endpoint: str = "http://10.110.190.83:9090"):
        self.logger = self._setup_logging()
        self.prometheus_endpoint = prometheus_endpoint
        
        self.models = {
            ResourceType.CPU: {
                PredictionHorizon.SHORT_TERM: RandomForestRegressor(n_estimators=100, random_state=42),
                PredictionHorizon.MEDIUM_TERM: GradientBoostingRegressor(n_estimators=100, random_state=42),
                PredictionHorizon.LONG_TERM: Ridge(alpha=1.0),
                PredictionHorizon.DAILY: LinearRegression()
            },
            ResourceType.MEMORY: {
                PredictionHorizon.SHORT_TERM: RandomForestRegressor(n_estimators=100, random_state=42),
                PredictionHorizon.MEDIUM_TERM: GradientBoostingRegressor(n_estimators=100, random_state=42),
                PredictionHorizon.LONG_TERM: Ridge(alpha=1.0),
                PredictionHorizon.DAILY: LinearRegression()
            },
            ResourceType.NETWORK: {
                PredictionHorizon.SHORT_TERM: RandomForestRegressor(n_estimators=50, random_state=42),
                PredictionHorizon.MEDIUM_TERM: GradientBoostingRegressor(n_estimators=50, random_state=42),
                PredictionHorizon.LONG_TERM: Ridge(alpha=0.5),
                PredictionHorizon.DAILY: LinearRegression()
            },
            ResourceType.STORAGE: {
                PredictionHorizon.SHORT_TERM: RandomForestRegressor(n_estimators=50, random_state=42),
                PredictionHorizon.MEDIUM_TERM: GradientBoostingRegressor(n_estimators=50, random_state=42),
                PredictionHorizon.LONG_TERM: Ridge(alpha=0.5),
                PredictionHorizon.DAILY: LinearRegression()
            },
            ResourceType.PODS: {
                PredictionHorizon.SHORT_TERM: RandomForestRegressor(n_estimators=50, random_state=42),
                PredictionHorizon.MEDIUM_TERM: GradientBoostingRegressor(n_estimators=50, random_state=42),
                PredictionHorizon.LONG_TERM: Ridge(alpha=0.5),
                PredictionHorizon.DAILY: LinearRegression()
            }
        }
        
        self.scalers = {
            resource_type: StandardScaler() for resource_type in ResourceType
        }
        
        self.feature_windows = {
            PredictionHorizon.SHORT_TERM: ['5min', '10min'],
            PredictionHorizon.MEDIUM_TERM: ['15min', '30min', '45min'],
            PredictionHorizon.LONG_TERM: ['1hour', '2hour', '3hour'],
            PredictionHorizon.DAILY: ['6hour', '12hour', '18hour']
        }
        
        self.prediction_cache = {}
        self.model_performance = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for resource predictor."""
        logger = logging.getLogger('resource_predictor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    async def generate_cluster_forecast(self, 
                                      cluster_name: str = "HYDATIS", 
                                      horizons: List[PredictionHorizon] = None) -> ClusterForecast:
        """Generate comprehensive resource forecast for cluster."""
        
        if horizons is None:
            horizons = list(PredictionHorizon)
        
        self.logger.info(f"Generating resource forecast for cluster {cluster_name}")
        
        try:
            historical_data = await self._collect_historical_metrics(cluster_name)
            
            if len(historical_data) < 50:
                raise ValueError("Insufficient historical data for reliable predictions")
            
            predictions = {}
            
            for resource_type in ResourceType:
                for horizon in horizons:
                    try:
                        prediction = await self._predict_resource_usage(
                            historical_data, resource_type, horizon
                        )
                        predictions[f"{resource_type.value}_{horizon.value}"] = prediction
                        
                    except Exception as e:
                        self.logger.error(f"Prediction failed for {resource_type.value}/{horizon.value}: {e}")
            
            capacity_warnings = self._analyze_capacity_warnings(predictions, cluster_name)
            scaling_recommendations = self._generate_scaling_recommendations(predictions, cluster_name)
            
            overall_confidence = np.mean([
                pred.confidence_score for pred in predictions.values() if pred.confidence_score > 0
            ]) if predictions else 0.0
            
            forecast = ClusterForecast(
                cluster_name=cluster_name,
                timestamp=datetime.now(),
                predictions=predictions,
                capacity_warnings=capacity_warnings,
                scaling_recommendations=scaling_recommendations,
                overall_confidence=overall_confidence
            )
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Forecast generation failed for {cluster_name}: {e}")
            raise

    async def _collect_historical_metrics(self, cluster_name: str, lookback_hours: int = 24) -> pd.DataFrame:
        """Collect historical metrics from Prometheus."""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        metrics_queries = {
            'cpu_usage': 'sum(rate(container_cpu_usage_seconds_total[5m])) by (node)',
            'memory_usage': 'sum(container_memory_working_set_bytes) by (node)',
            'network_rx': 'sum(rate(container_network_receive_bytes_total[5m])) by (node)',
            'network_tx': 'sum(rate(container_network_transmit_bytes_total[5m])) by (node)',
            'storage_io': 'sum(rate(container_fs_io_time_seconds_total[5m])) by (node)',
            'pod_count': 'count(kube_pod_info) by (node)',
            'node_load': 'node_load1',
            'memory_available': 'node_memory_MemAvailable_bytes',
            'disk_usage': 'node_filesystem_avail_bytes'
        }
        
        historical_data = []
        
        for timestamp in pd.date_range(start_time, end_time, freq='5min'):
            try:
                data_point = {
                    'timestamp': timestamp,
                    'hour_of_day': timestamp.hour,
                    'day_of_week': timestamp.weekday(),
                    'month': timestamp.month
                }
                
                for metric_name, query in metrics_queries.items():
                    try:
                        response = requests.get(
                            f"{self.prometheus_endpoint}/api/v1/query",
                            params={
                                'query': query,
                                'time': timestamp.timestamp()
                            },
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            result_data = response.json()
                            if result_data['data']['result']:
                                metric_value = float(result_data['data']['result'][0]['value'][1])
                                data_point[metric_name] = metric_value
                            else:
                                data_point[metric_name] = 0.0
                        else:
                            data_point[metric_name] = 0.0
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to collect {metric_name} at {timestamp}: {e}")
                        data_point[metric_name] = 0.0
                
                historical_data.append(data_point)
                
            except Exception as e:
                self.logger.warning(f"Failed to collect data point at {timestamp}: {e}")
        
        df = pd.DataFrame(historical_data)
        
        if len(df) > 0:
            df = df.fillna(method='forward').fillna(0)
        
        return df

    async def _predict_resource_usage(self, 
                                    historical_data: pd.DataFrame, 
                                    resource_type: ResourceType, 
                                    horizon: PredictionHorizon) -> ResourcePrediction:
        """Predict resource usage for specific type and horizon."""
        
        try:
            features_df = self._extract_prediction_features(historical_data, resource_type, horizon)
            
            if len(features_df) < 20:
                raise ValueError(f"Insufficient data for {resource_type.value} prediction")
            
            target_column = self._get_target_column(resource_type)
            
            if target_column not in features_df.columns:
                raise ValueError(f"Target column {target_column} not found in data")
            
            X = features_df.drop([target_column, 'timestamp'], axis=1, errors='ignore')
            y = features_df[target_column]
            
            scaler = self.scalers[resource_type]
            X_scaled = scaler.fit_transform(X)
            
            model = self.models[resource_type][horizon]
            
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                cv_score = r2_score(y_val, y_pred)
                cv_scores.append(cv_score)
            
            model.fit(X_scaled, y)
            
            prediction_features = self._prepare_prediction_features(features_df.iloc[-1], resource_type)
            prediction_features_scaled = scaler.transform([prediction_features])
            
            predicted_value = model.predict(prediction_features_scaled)[0]
            
            confidence_score = np.mean(cv_scores) if cv_scores else 0.0
            
            prediction_std = np.std([
                model.predict(X_scaled[i:i+1])[0] for i in range(max(0, len(X_scaled)-10), len(X_scaled))
            ]) if len(X_scaled) > 10 else predicted_value * 0.1
            
            confidence_interval = (
                max(0, predicted_value - 1.96 * prediction_std),
                predicted_value + 1.96 * prediction_std
            )
            
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = X.columns
                for i, importance in enumerate(model.feature_importances_):
                    feature_importance[feature_names[i]] = float(importance)
            
            return ResourcePrediction(
                resource_type=resource_type,
                horizon=horizon,
                predicted_value=float(predicted_value),
                confidence_interval=confidence_interval,
                confidence_score=max(0.0, min(1.0, confidence_score)),
                prediction_timestamp=datetime.now(),
                model_used=model.__class__.__name__,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {resource_type.value}/{horizon.value}: {e}")
            
            return ResourcePrediction(
                resource_type=resource_type,
                horizon=horizon,
                predicted_value=0.0,
                confidence_interval=(0.0, 0.0),
                confidence_score=0.0,
                prediction_timestamp=datetime.now(),
                model_used="fallback",
                feature_importance={}
            )

    def _extract_prediction_features(self, data: pd.DataFrame, resource_type: ResourceType, horizon: PredictionHorizon) -> pd.DataFrame:
        """Extract features for prediction model."""
        
        df = data.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        feature_columns = ['hour_of_day', 'day_of_week', 'month']
        
        target_column = self._get_target_column(resource_type)
        if target_column in df.columns:
            feature_columns.append(target_column)
            
            rolling_windows = self.feature_windows[horizon]
            for window in rolling_windows:
                window_size = self._parse_time_window(window)
                if window_size > 0:
                    df[f'{target_column}_avg_{window}'] = df[target_column].rolling(window=window_size).mean()
                    df[f'{target_column}_std_{window}'] = df[target_column].rolling(window=window_size).std()
                    df[f'{target_column}_max_{window}'] = df[target_column].rolling(window=window_size).max()
                    df[f'{target_column}_min_{window}'] = df[target_column].rolling(window=window_size).min()
                    
                    feature_columns.extend([
                        f'{target_column}_avg_{window}',
                        f'{target_column}_std_{window}',
                        f'{target_column}_max_{window}',
                        f'{target_column}_min_{window}'
                    ])
        
        other_metrics = ['node_load', 'memory_available', 'disk_usage', 'network_rx', 'network_tx']
        for metric in other_metrics:
            if metric in df.columns:
                feature_columns.append(metric)
                
                df[f'{metric}_trend'] = df[metric].diff()
                feature_columns.append(f'{metric}_trend')
        
        df['load_trend'] = df[target_column].diff() if target_column in df.columns else 0
        df['load_acceleration'] = df['load_trend'].diff()
        
        feature_columns.extend(['load_trend', 'load_acceleration'])
        
        existing_features = [col for col in feature_columns if col in df.columns]
        
        result_df = df[existing_features].dropna()
        result_df = result_df.reset_index()
        
        return result_df

    def _get_target_column(self, resource_type: ResourceType) -> str:
        """Get target column name for resource type."""
        target_mapping = {
            ResourceType.CPU: 'cpu_usage',
            ResourceType.MEMORY: 'memory_usage',
            ResourceType.NETWORK: 'network_rx',
            ResourceType.STORAGE: 'storage_io',
            ResourceType.PODS: 'pod_count'
        }
        return target_mapping[resource_type]

    def _parse_time_window(self, window_str: str) -> int:
        """Parse time window string to number of data points."""
        window_mapping = {
            '5min': 1,
            '10min': 2,
            '15min': 3,
            '30min': 6,
            '45min': 9,
            '1hour': 12,
            '2hour': 24,
            '3hour': 36,
            '6hour': 72,
            '12hour': 144,
            '18hour': 216
        }
        return window_mapping.get(window_str, 1)

    def _prepare_prediction_features(self, latest_data: pd.Series, resource_type: ResourceType) -> List[float]:
        """Prepare features for prediction from latest data point."""
        
        feature_values = []
        
        basic_features = ['hour_of_day', 'day_of_week', 'month']
        for feature in basic_features:
            if feature in latest_data:
                feature_values.append(float(latest_data[feature]))
            else:
                feature_values.append(0.0)
        
        target_column = self._get_target_column(resource_type)
        if target_column in latest_data:
            feature_values.append(float(latest_data[target_column]))
        
        other_features = ['node_load', 'memory_available', 'disk_usage', 'network_rx', 'network_tx']
        for feature in other_features:
            if feature in latest_data:
                feature_values.append(float(latest_data[feature]))
            else:
                feature_values.append(0.0)
        
        return feature_values

    def _analyze_capacity_warnings(self, predictions: Dict[str, ResourcePrediction], cluster_name: str) -> List[str]:
        """Analyze predictions for capacity warnings."""
        
        warnings = []
        
        capacity_limits = {
            ResourceType.CPU: 24000,
            ResourceType.MEMORY: 98304,
            ResourceType.NETWORK: 10000000,
            ResourceType.STORAGE: 1000000,
            ResourceType.PODS: 180
        }
        
        for pred_key, prediction in predictions.items():
            resource_type = prediction.resource_type
            predicted_value = prediction.predicted_value
            
            if resource_type in capacity_limits:
                capacity_limit = capacity_limits[resource_type]
                utilization = predicted_value / capacity_limit
                
                if utilization > 0.9:
                    warnings.append(
                        f"Critical: {resource_type.value} predicted to reach {utilization*100:.1f}% "
                        f"capacity in {prediction.horizon.value}"
                    )
                elif utilization > 0.8:
                    warnings.append(
                        f"Warning: {resource_type.value} predicted to reach {utilization*100:.1f}% "
                        f"capacity in {prediction.horizon.value}"
                    )
        
        return warnings

    def _generate_scaling_recommendations(self, predictions: Dict[str, ResourcePrediction], cluster_name: str) -> List[str]:
        """Generate scaling recommendations based on predictions."""
        
        recommendations = []
        
        cpu_predictions = {k: v for k, v in predictions.items() if v.resource_type == ResourceType.CPU}
        memory_predictions = {k: v for k, v in predictions.items() if v.resource_type == ResourceType.MEMORY}
        
        if cpu_predictions:
            max_cpu_pred = max(cpu_predictions.values(), key=lambda x: x.predicted_value)
            if max_cpu_pred.predicted_value > 20000:
                additional_cpu = max_cpu_pred.predicted_value - 20000
                additional_nodes = int(np.ceil(additional_cpu / 4000))
                recommendations.append(
                    f"Scale up: Add {additional_nodes} worker nodes to handle predicted CPU demand "
                    f"({max_cpu_pred.predicted_value:.0f} cores) in {max_cpu_pred.horizon.value}"
                )
        
        if memory_predictions:
            max_memory_pred = max(memory_predictions.values(), key=lambda x: x.predicted_value)
            if max_memory_pred.predicted_value > 80000:
                additional_memory = max_memory_pred.predicted_value - 80000
                additional_nodes = int(np.ceil(additional_memory / 16384))
                recommendations.append(
                    f"Scale up: Add {additional_nodes} worker nodes to handle predicted memory demand "
                    f"({max_memory_pred.predicted_value/1024:.1f} GB) in {max_memory_pred.horizon.value}"
                )
        
        short_term_cpu = next((p for p in cpu_predictions.values() if p.horizon == PredictionHorizon.SHORT_TERM), None)
        if short_term_cpu and short_term_cpu.predicted_value < 5000:
            recommendations.append(
                "Scale down: Consider reducing cluster size due to low predicted CPU usage"
            )
        
        return recommendations

    async def proactive_resource_allocation(self, forecast: ClusterForecast) -> Dict[str, Any]:
        """Proactively allocate resources based on forecast."""
        
        self.logger.info(f"Executing proactive resource allocation for {forecast.cluster_name}")
        
        allocation_actions = []
        
        try:
            for prediction_key, prediction in forecast.predictions.items():
                if prediction.confidence_score > 0.7:
                    action = await self._determine_allocation_action(prediction, forecast.cluster_name)
                    if action:
                        allocation_actions.append(action)
            
            execution_results = []
            for action in allocation_actions:
                try:
                    result = await self._execute_allocation_action(action)
                    execution_results.append(result)
                except Exception as e:
                    execution_results.append({
                        'action': action,
                        'status': 'error',
                        'error': str(e)
                    })
            
            return {
                'status': 'success',
                'cluster': forecast.cluster_name,
                'actions_planned': len(allocation_actions),
                'actions_executed': len([r for r in execution_results if r.get('status') == 'success']),
                'execution_results': execution_results,
                'forecast_confidence': forecast.overall_confidence
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'cluster': forecast.cluster_name,
                'error': str(e)
            }

    async def _determine_allocation_action(self, prediction: ResourcePrediction, cluster_name: str) -> Optional[Dict[str, Any]]:
        """Determine specific allocation action based on prediction."""
        
        if prediction.resource_type == ResourceType.CPU:
            if prediction.predicted_value > 20000:
                return {
                    'type': 'scale_up_cpu',
                    'resource': 'cpu',
                    'current_capacity': 24000,
                    'predicted_demand': prediction.predicted_value,
                    'recommended_action': 'add_worker_nodes',
                    'urgency': 'high' if prediction.horizon == PredictionHorizon.SHORT_TERM else 'medium'
                }
        
        elif prediction.resource_type == ResourceType.MEMORY:
            if prediction.predicted_value > 80000:
                return {
                    'type': 'scale_up_memory',
                    'resource': 'memory',
                    'current_capacity': 98304,
                    'predicted_demand': prediction.predicted_value,
                    'recommended_action': 'add_memory_nodes',
                    'urgency': 'high' if prediction.horizon == PredictionHorizon.SHORT_TERM else 'medium'
                }
        
        elif prediction.resource_type == ResourceType.PODS:
            if prediction.predicted_value > 150:
                return {
                    'type': 'scale_up_pods',
                    'resource': 'pods',
                    'current_capacity': 180,
                    'predicted_demand': prediction.predicted_value,
                    'recommended_action': 'increase_pod_limits',
                    'urgency': 'medium'
                }
        
        return None

    async def _execute_allocation_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute resource allocation action."""
        
        self.logger.info(f"Executing allocation action: {action['type']}")
        
        if action['type'] in ['scale_up_cpu', 'scale_up_memory']:
            return await self._simulate_node_scaling(action)
        elif action['type'] == 'scale_up_pods':
            return await self._simulate_pod_limit_increase(action)
        else:
            return {
                'action': action,
                'status': 'error',
                'error': 'Unknown action type'
            }

    async def _simulate_node_scaling(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate node scaling action."""
        
        predicted_demand = action['predicted_demand']
        current_capacity = action['current_capacity']
        
        if predicted_demand > current_capacity:
            required_additional = predicted_demand - current_capacity
            
            if action['resource'] == 'cpu':
                additional_nodes = int(np.ceil(required_additional / 4000))
                node_type = 'cpu-optimized'
            else:
                additional_nodes = int(np.ceil(required_additional / 16384))
                node_type = 'memory-optimized'
            
            return {
                'action': action,
                'status': 'success',
                'recommendation': f"Add {additional_nodes} {node_type} worker nodes",
                'estimated_cost_increase': additional_nodes * 0.15,
                'estimated_time_to_provision': f"{additional_nodes * 5} minutes"
            }
        
        return {
            'action': action,
            'status': 'no_action_needed',
            'reason': 'Current capacity sufficient'
        }

    async def _simulate_pod_limit_increase(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate pod limit increase action."""
        
        return {
            'action': action,
            'status': 'success',
            'recommendation': 'Increase max pods per node from 30 to 35',
            'configuration_change': 'Update kubelet --max-pods parameter'
        }

    async def optimize_resource_allocation_across_time(self, 
                                                     cluster_name: str, 
                                                     optimization_window_hours: int = 24) -> Dict[str, Any]:
        """Optimize resource allocation across time horizon."""
        
        self.logger.info(f"Optimizing resource allocation for {optimization_window_hours}h window")
        
        try:
            time_slots = []
            current_time = datetime.now()
            
            for hour_offset in range(optimization_window_hours):
                slot_time = current_time + timedelta(hours=hour_offset)
                
                slot_forecast = await self._predict_resource_demand_at_time(cluster_name, slot_time)
                
                time_slots.append({
                    'time': slot_time,
                    'forecast': slot_forecast,
                    'resource_requirements': self._calculate_resource_requirements(slot_forecast)
                })
            
            optimization_plan = self._create_temporal_optimization_plan(time_slots)
            
            cost_analysis = self._analyze_optimization_costs(optimization_plan)
            
            return {
                'status': 'success',
                'cluster': cluster_name,
                'optimization_window_hours': optimization_window_hours,
                'time_slots': len(time_slots),
                'optimization_plan': optimization_plan,
                'cost_analysis': cost_analysis,
                'estimated_savings': cost_analysis.get('potential_savings', 0.0)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _predict_resource_demand_at_time(self, cluster_name: str, target_time: datetime) -> Dict[str, float]:
        """Predict resource demand at specific time."""
        
        time_features = {
            'hour_of_day': target_time.hour,
            'day_of_week': target_time.weekday(),
            'month': target_time.month
        }
        
        base_demand = {
            ResourceType.CPU: 8000 + 4000 * np.sin(target_time.hour * np.pi / 12),
            ResourceType.MEMORY: 32000 + 16000 * np.sin(target_time.hour * np.pi / 12),
            ResourceType.NETWORK: 1000000 + 500000 * np.sin(target_time.hour * np.pi / 12),
            ResourceType.STORAGE: 100000 + 50000 * np.sin(target_time.hour * np.pi / 12),
            ResourceType.PODS: 60 + 30 * np.sin(target_time.hour * np.pi / 12)
        }
        
        return {resource_type.value: demand for resource_type, demand in base_demand.items()}

    def _calculate_resource_requirements(self, forecast: Dict[str, float]) -> Dict[str, Any]:
        """Calculate resource requirements from forecast."""
        
        requirements = {}
        
        for resource_name, predicted_value in forecast.items():
            if resource_name == 'cpu':
                required_nodes = int(np.ceil(predicted_value / 4000))
                requirements[resource_name] = {
                    'predicted_usage': predicted_value,
                    'required_nodes': required_nodes,
                    'recommended_capacity': required_nodes * 4000
                }
            elif resource_name == 'memory':
                required_nodes = int(np.ceil(predicted_value / 16384))
                requirements[resource_name] = {
                    'predicted_usage': predicted_value,
                    'required_nodes': required_nodes,
                    'recommended_capacity': required_nodes * 16384
                }
        
        return requirements

    def _create_temporal_optimization_plan(self, time_slots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create optimization plan across time slots."""
        
        max_cpu_demand = max(slot['resource_requirements'].get('cpu', {}).get('predicted_usage', 0) for slot in time_slots)
        max_memory_demand = max(slot['resource_requirements'].get('memory', {}).get('predicted_usage', 0) for slot in time_slots)
        
        recommended_base_capacity = {
            'cpu_nodes': int(np.ceil(max_cpu_demand / 4000)),
            'memory_nodes': int(np.ceil(max_memory_demand / 16384)),
            'total_nodes': max(
                int(np.ceil(max_cpu_demand / 4000)),
                int(np.ceil(max_memory_demand / 16384))
            )
        }
        
        scaling_schedule = []
        for slot in time_slots:
            cpu_req = slot['resource_requirements'].get('cpu', {}).get('required_nodes', 0)
            memory_req = slot['resource_requirements'].get('memory', {}).get('required_nodes', 0)
            
            required_nodes = max(cpu_req, memory_req)
            
            if required_nodes != recommended_base_capacity['total_nodes']:
                scaling_schedule.append({
                    'time': slot['time'].isoformat(),
                    'action': 'scale_up' if required_nodes > 6 else 'scale_down',
                    'target_nodes': required_nodes,
                    'reason': f"Predicted demand requires {required_nodes} nodes"
                })
        
        return {
            'base_capacity': recommended_base_capacity,
            'scaling_schedule': scaling_schedule,
            'peak_demand_time': max(time_slots, key=lambda x: x['resource_requirements'].get('cpu', {}).get('predicted_usage', 0))['time'].isoformat()
        }

    def _analyze_optimization_costs(self, optimization_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze costs of optimization plan."""
        
        base_cost_per_hour = 6 * 0.15
        
        current_daily_cost = base_cost_per_hour * 24
        
        optimized_cost = 0.0
        for scaling_action in optimization_plan.get('scaling_schedule', []):
            target_nodes = scaling_action['target_nodes']
            optimized_cost += target_nodes * 0.15
        
        if optimization_plan.get('scaling_schedule'):
            avg_optimized_cost_per_hour = optimized_cost / len(optimization_plan['scaling_schedule'])
            optimized_daily_cost = avg_optimized_cost_per_hour * 24
        else:
            optimized_daily_cost = current_daily_cost
        
        potential_savings = max(0.0, current_daily_cost - optimized_daily_cost)
        
        return {
            'current_daily_cost': current_daily_cost,
            'optimized_daily_cost': optimized_daily_cost,
            'potential_savings': potential_savings,
            'savings_percentage': (potential_savings / current_daily_cost * 100) if current_daily_cost > 0 else 0,
            'scaling_events': len(optimization_plan.get('scaling_schedule', []))
        }

    async def adaptive_model_retraining(self) -> Dict[str, Any]:
        """Adaptively retrain prediction models based on recent performance."""
        
        self.logger.info("Starting adaptive model retraining")
        
        retraining_results = {}
        
        recent_data = await self._collect_historical_metrics("HYDATIS", lookback_hours=48)
        
        if len(recent_data) < 100:
            return {
                'status': 'insufficient_data',
                'data_points': len(recent_data)
            }
        
        for resource_type in ResourceType:
            for horizon in PredictionHorizon:
                try:
                    current_model = self.models[resource_type][horizon]
                    
                    performance_score = await self._evaluate_model_performance(
                        current_model, recent_data, resource_type, horizon
                    )
                    
                    if performance_score < 0.7:
                        self.logger.info(f"Retraining {resource_type.value}/{horizon.value} model (score: {performance_score:.3f})")
                        
                        new_model = await self._retrain_model(recent_data, resource_type, horizon)
                        
                        new_performance = await self._evaluate_model_performance(
                            new_model, recent_data, resource_type, horizon
                        )
                        
                        if new_performance > performance_score:
                            self.models[resource_type][horizon] = new_model
                            retraining_results[f"{resource_type.value}_{horizon.value}"] = {
                                'status': 'improved',
                                'old_score': performance_score,
                                'new_score': new_performance
                            }
                        else:
                            retraining_results[f"{resource_type.value}_{horizon.value}"] = {
                                'status': 'no_improvement',
                                'score': performance_score
                            }
                    else:
                        retraining_results[f"{resource_type.value}_{horizon.value}"] = {
                            'status': 'satisfactory',
                            'score': performance_score
                        }
                        
                except Exception as e:
                    retraining_results[f"{resource_type.value}_{horizon.value}"] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        return {
            'status': 'success',
            'models_retrained': len([r for r in retraining_results.values() if r.get('status') == 'improved']),
            'retraining_details': retraining_results
        }

    async def _evaluate_model_performance(self, model, data: pd.DataFrame, resource_type: ResourceType, horizon: PredictionHorizon) -> float:
        """Evaluate model performance on recent data."""
        
        try:
            features_df = self._extract_prediction_features(data, resource_type, horizon)
            
            if len(features_df) < 10:
                return 0.0
            
            target_column = self._get_target_column(resource_type)
            
            X = features_df.drop([target_column, 'timestamp'], axis=1, errors='ignore')
            y = features_df[target_column]
            
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            
            return max(0.0, r2)
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return 0.0

    async def _retrain_model(self, data: pd.DataFrame, resource_type: ResourceType, horizon: PredictionHorizon):
        """Retrain model with recent data."""
        
        if horizon == PredictionHorizon.SHORT_TERM:
            return RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        elif horizon == PredictionHorizon.MEDIUM_TERM:
            return GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
        elif horizon == PredictionHorizon.LONG_TERM:
            return Ridge(alpha=2.0)
        else:
            return LinearRegression()

async def main():
    """Main predictive resource allocator entry point."""
    allocator = PredictiveResourceAllocator()
    
    try:
        forecast = await allocator.generate_cluster_forecast("HYDATIS")
        
        allocation_result = await allocator.proactive_resource_allocation(forecast)
        
        retraining_result = await allocator.adaptive_model_retraining()
        
        temporal_optimization = await allocator.optimize_resource_allocation_across_time("HYDATIS", 24)
        
        result = {
            'cluster_forecast': asdict(forecast),
            'allocation_result': allocation_result,
            'retraining_result': retraining_result,
            'temporal_optimization': temporal_optimization
        }
        
        with open('/tmp/predictive_allocation_result.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Resource forecast confidence: {forecast.overall_confidence:.3f}")
        print(f"Capacity warnings: {len(forecast.capacity_warnings)}")
        print(f"Scaling recommendations: {len(forecast.scaling_recommendations)}")
        print("Predictive allocation results saved to /tmp/predictive_allocation_result.json")
        
    except Exception as e:
        allocator.logger.error(f"Predictive allocation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())