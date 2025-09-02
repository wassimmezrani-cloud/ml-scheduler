"""
Predictive Analytics Engine for HYDATIS Cluster
Forecasts capacity needs, cost optimization, and performance trends
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from prometheus_client import Counter, Histogram, Gauge
import yaml

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    CAPACITY_FORECAST = "capacity_forecast"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_TREND = "performance_trend"
    RESOURCE_DEMAND = "resource_demand"
    AVAILABILITY_PREDICTION = "availability_prediction"

class ForecastHorizon(Enum):
    SHORT_TERM = "24h"      # 24 hours
    MEDIUM_TERM = "7d"      # 7 days
    LONG_TERM = "30d"       # 30 days

@dataclass
class PredictionResult:
    """Result of a predictive analytics query"""
    prediction_type: PredictionType
    horizon: ForecastHorizon
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CapacityForecast:
    """Capacity forecasting result"""
    forecast_date: datetime
    predicted_cpu_utilization: float
    predicted_memory_utilization: float
    predicted_pod_count: int
    recommended_node_count: int
    scaling_confidence: float
    cost_impact: float
    recommendations: List[str]

@dataclass
class CostOptimizationForecast:
    """Cost optimization forecasting result"""
    current_monthly_cost: float
    predicted_monthly_cost: float
    potential_savings: float
    roi_projection: float
    optimization_opportunities: List[Dict[str, Any]]
    confidence_score: float

class TimeSeriesPredictor:
    """Time series prediction engine"""
    
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scalers = {name: StandardScaler() for name in self.models.keys()}
        self.trained_models = {}
        
    def prepare_features(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series features for prediction"""
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Create time-based features
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['day_of_month'] = data['timestamp'].dt.day
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:  # 1h, 2h, 3h, 6h, 12h, 24h lags
            data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            data[f'{target_col}_rolling_mean_{window}'] = data[target_col].rolling(window).mean()
            data[f'{target_col}_rolling_std_{window}'] = data[target_col].rolling(window).std()
        
        # Trend features
        data[f'{target_col}_trend'] = data[target_col].diff()
        data[f'{target_col}_trend_ma'] = data[f'{target_col}_trend'].rolling(6).mean()
        
        # Drop rows with NaN values (due to lag/rolling operations)
        data = data.dropna()
        
        # Feature columns (exclude timestamp and target)
        feature_cols = [col for col in data.columns 
                       if col not in ['timestamp', target_col]]
        
        X = data[feature_cols].values
        y = data[target_col].values
        
        return X, y
    
    def train_model(self, data: pd.DataFrame, target_col: str, 
                   model_type: str = 'forest') -> Dict[str, Any]:
        """Train prediction model on historical data"""
        try:
            X, y = self.prepare_features(data, target_col)
            
            if len(X) < 50:  # Minimum samples for training
                raise ValueError(f"Insufficient training data: {len(X)} samples")
            
            # Split data for validation
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = self.scalers[model_type]
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = self.models[model_type]
            model.fit(X_train_scaled, y_train)
            
            # Validate
            val_predictions = model.predict(X_val_scaled)
            mae = mean_absolute_error(y_val, val_predictions)
            mse = mean_squared_error(y_val, val_predictions)
            
            # Store trained model
            model_key = f"{target_col}_{model_type}"
            self.trained_models[model_key] = {
                'model': model,
                'scaler': scaler,
                'target_col': target_col,
                'trained_at': datetime.utcnow(),
                'validation_mae': mae,
                'validation_mse': mse,
                'feature_count': X.shape[1]
            }
            
            logger.info(f"Trained {model_type} model for {target_col}: MAE={mae:.4f}, MSE={mse:.4f}")
            
            return {
                'success': True,
                'model_key': model_key,
                'validation_mae': mae,
                'validation_mse': mse,
                'training_samples': len(X_train)
            }
            
        except Exception as e:
            logger.error(f"Error training model for {target_col}: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, data: pd.DataFrame, target_col: str, 
               steps_ahead: int, model_type: str = 'forest') -> np.ndarray:
        """Make predictions for future time steps"""
        model_key = f"{target_col}_{model_type}"
        
        if model_key not in self.trained_models:
            raise ValueError(f"Model {model_key} not trained")
        
        model_info = self.trained_models[model_key]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Prepare initial features
        X, _ = self.prepare_features(data, target_col)
        
        if len(X) == 0:
            raise ValueError("No valid feature data for prediction")
        
        predictions = []
        current_data = data.copy()
        
        for step in range(steps_ahead):
            # Get latest features
            X_current, _ = self.prepare_features(current_data, target_col)
            if len(X_current) == 0:
                break
                
            # Scale and predict
            X_scaled = scaler.transform(X_current[-1:])
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)
            
            # Update data with prediction for next step
            next_timestamp = current_data['timestamp'].max() + timedelta(hours=1)
            new_row = pd.DataFrame({
                'timestamp': [next_timestamp],
                target_col: [pred]
            })
            current_data = pd.concat([current_data, new_row], ignore_index=True)
            
        return np.array(predictions)

class CapacityPredictor:
    """Predicts cluster capacity requirements"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        self.predictor = TimeSeriesPredictor()
        
        # Metrics
        self.capacity_predictions_counter = Counter('capacity_predictions_total',
                                                  'Capacity predictions generated',
                                                  ['resource_type', 'horizon'])
        self.capacity_forecast_accuracy = Gauge('capacity_forecast_accuracy',
                                              'Accuracy of capacity forecasts',
                                              ['resource_type'])
    
    async def predict_cluster_capacity(self, horizon: ForecastHorizon) -> CapacityForecast:
        """Predict cluster capacity requirements"""
        # Collect historical data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=14)  # 2 weeks of historical data
        
        cpu_data = await self._collect_cpu_utilization_data(start_time, end_time)
        memory_data = await self._collect_memory_utilization_data(start_time, end_time)
        pod_data = await self._collect_pod_count_data(start_time, end_time)
        
        # Train models if not already trained
        await self._ensure_models_trained(cpu_data, memory_data, pod_data)
        
        # Determine prediction steps
        horizon_hours = {'24h': 24, '7d': 168, '30d': 720}[horizon.value]
        
        # Make predictions
        cpu_predictions = self.predictor.predict(cpu_data, 'cpu_utilization', 
                                                horizon_hours, 'forest')
        memory_predictions = self.predictor.predict(memory_data, 'memory_utilization',
                                                   horizon_hours, 'forest')
        pod_predictions = self.predictor.predict(pod_data, 'pod_count',
                                               horizon_hours, 'forest')
        
        # Calculate forecast metrics
        predicted_cpu = float(np.mean(cpu_predictions))
        predicted_memory = float(np.mean(memory_predictions))
        predicted_pods = int(np.mean(pod_predictions))
        
        # Capacity planning calculations
        current_node_count = await self._get_current_node_count()
        recommended_nodes = self._calculate_recommended_nodes(
            predicted_cpu, predicted_memory, predicted_pods, current_node_count)
        
        # Calculate scaling confidence
        cpu_variance = float(np.var(cpu_predictions))
        memory_variance = float(np.var(memory_predictions))
        scaling_confidence = 1.0 / (1.0 + cpu_variance + memory_variance)
        
        # Cost impact calculation
        cost_impact = self._calculate_capacity_cost_impact(
            current_node_count, recommended_nodes)
        
        # Generate recommendations
        recommendations = self._generate_capacity_recommendations(
            predicted_cpu, predicted_memory, predicted_pods, 
            recommended_nodes, horizon)
        
        self.capacity_predictions_counter.labels(
            resource_type='cluster', horizon=horizon.value).inc()
        
        return CapacityForecast(
            forecast_date=end_time + timedelta(hours=horizon_hours),
            predicted_cpu_utilization=predicted_cpu,
            predicted_memory_utilization=predicted_memory,
            predicted_pod_count=predicted_pods,
            recommended_node_count=recommended_nodes,
            scaling_confidence=scaling_confidence,
            cost_impact=cost_impact,
            recommendations=recommendations
        )
    
    async def _collect_cpu_utilization_data(self, start_time: datetime, 
                                          end_time: datetime) -> pd.DataFrame:
        """Collect CPU utilization time series data"""
        query = 'avg(100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))'
        return await self._collect_prometheus_timeseries(query, start_time, end_time, 'cpu_utilization')
    
    async def _collect_memory_utilization_data(self, start_time: datetime,
                                             end_time: datetime) -> pd.DataFrame:
        """Collect memory utilization time series data"""
        query = 'avg((1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100)'
        return await self._collect_prometheus_timeseries(query, start_time, end_time, 'memory_utilization')
    
    async def _collect_pod_count_data(self, start_time: datetime,
                                    end_time: datetime) -> pd.DataFrame:
        """Collect pod count time series data"""
        query = 'sum(kube_pod_info{scheduler="ml-scheduler"})'
        return await self._collect_prometheus_timeseries(query, start_time, end_time, 'pod_count')
    
    async def _collect_prometheus_timeseries(self, query: str, start_time: datetime,
                                           end_time: datetime, metric_name: str) -> pd.DataFrame:
        """Collect time series data from Prometheus"""
        async with aiohttp.ClientSession() as session:
            params = {
                'query': query,
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'step': '1h'  # 1 hour resolution
            }
            
            async with session.get(f"{self.prometheus_url}/api/v1/query_range",
                                 params=params) as response:
                data = await response.json()
        
        # Convert to DataFrame
        time_series_data = []
        results = data.get('data', {}).get('result', [])
        
        if results:
            values = results[0].get('values', [])
            for timestamp, value in values:
                time_series_data.append({
                    'timestamp': datetime.fromtimestamp(float(timestamp)),
                    metric_name: float(value)
                })
        
        return pd.DataFrame(time_series_data)
    
    async def _ensure_models_trained(self, cpu_data: pd.DataFrame,
                                   memory_data: pd.DataFrame,
                                   pod_data: pd.DataFrame):
        """Ensure prediction models are trained"""
        datasets = [
            (cpu_data, 'cpu_utilization'),
            (memory_data, 'memory_utilization'),
            (pod_data, 'pod_count')
        ]
        
        for data, target_col in datasets:
            if len(data) > 50:  # Minimum data requirement
                await asyncio.get_event_loop().run_in_executor(
                    None, self.predictor.train_model, data, target_col)
    
    async def _get_current_node_count(self) -> int:
        """Get current number of nodes in cluster"""
        query = 'count(up{job="kubernetes-nodes"})'
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.prometheus_url}/api/v1/query",
                                 params={'query': query}) as response:
                data = await response.json()
        
        results = data.get('data', {}).get('result', [])
        if results:
            return int(float(results[0]['value'][1]))
        return 6  # Default HYDATIS cluster size
    
    def _calculate_recommended_nodes(self, predicted_cpu: float, predicted_memory: float,
                                   predicted_pods: int, current_nodes: int) -> int:
        """Calculate recommended node count based on predictions"""
        # HYDATIS cluster node specifications
        node_cpu_capacity = 16    # 16 vCPUs per node
        node_memory_capacity = 64 # 64GB memory per node
        pods_per_node = 110       # Max pods per node
        
        # Calculate required nodes for each resource
        cpu_nodes_needed = max(1, int(np.ceil(predicted_cpu * current_nodes / 65.0)))  # 65% target
        memory_nodes_needed = max(1, int(np.ceil(predicted_memory * current_nodes / 80.0)))  # 80% memory target
        pod_nodes_needed = max(1, int(np.ceil(predicted_pods / pods_per_node)))
        
        # Take maximum with buffer
        recommended = max(cpu_nodes_needed, memory_nodes_needed, pod_nodes_needed)
        
        # Add 20% buffer for safety
        recommended = int(recommended * 1.2)
        
        # Ensure minimum cluster size
        return max(recommended, 3)  # Minimum 3 nodes for HA
    
    def _calculate_capacity_cost_impact(self, current_nodes: int, recommended_nodes: int) -> float:
        """Calculate cost impact of capacity changes"""
        # HYDATIS node cost: $500/month per node
        node_cost_monthly = 500.0
        
        node_difference = recommended_nodes - current_nodes
        monthly_cost_impact = node_difference * node_cost_monthly
        
        return monthly_cost_impact
    
    def _generate_capacity_recommendations(self, predicted_cpu: float, predicted_memory: float,
                                         predicted_pods: int, recommended_nodes: int,
                                         horizon: ForecastHorizon) -> List[str]:
        """Generate capacity planning recommendations"""
        recommendations = []
        current_nodes = 6  # HYDATIS baseline
        
        if recommended_nodes > current_nodes:
            recommendations.append(
                f"Scale cluster to {recommended_nodes} nodes within {horizon.value} "
                f"to handle predicted workload increase"
            )
            
        if predicted_cpu > 70:
            recommendations.append(
                f"CPU utilization predicted to exceed target (65%): {predicted_cpu:.1f}% - "
                "consider workload optimization or additional nodes"
            )
            
        if predicted_memory > 85:
            recommendations.append(
                f"Memory utilization high ({predicted_memory:.1f}%) - "
                "review memory-intensive workloads"
            )
            
        if predicted_pods > 600:  # 6 nodes * 110 pods per node * 0.9 safety margin
            recommendations.append(
                f"Pod count approaching cluster limits ({predicted_pods}) - "
                "plan node scaling or workload consolidation"
            )
        
        if not recommendations:
            recommendations.append("Current capacity appears sufficient for predicted workload")
            
        return recommendations

class CostOptimizationPredictor:
    """Predicts cost optimization opportunities"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        self.predictor = TimeSeriesPredictor()
        
        # HYDATIS business targets
        self.target_cpu_utilization = 65.0
        self.target_availability = 99.7
        self.monthly_baseline_cost = 30000  # $30k baseline
        self.target_annual_roi = 1400  # 1400% ROI target
        
        # Metrics
        self.cost_predictions_counter = Counter('cost_optimization_predictions_total',
                                              'Cost optimization predictions generated')
        self.roi_projection_gauge = Gauge('cost_optimization_roi_projection',
                                        'Projected ROI percentage')
    
    async def predict_cost_optimization(self, horizon: ForecastHorizon) -> CostOptimizationForecast:
        """Predict cost optimization opportunities"""
        # Collect current performance data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)  # 1 week history
        
        cpu_data = await self._collect_cost_metrics(start_time, end_time)
        
        # Current state analysis
        current_cpu = await self._get_current_cpu_utilization()
        current_availability = await self._get_current_availability()
        current_monthly_cost = await self._calculate_current_monthly_cost()
        
        # Predict future performance
        horizon_days = {'24h': 1, '7d': 7, '30d': 30}[horizon.value]
        
        # ROI calculation components
        cpu_efficiency_gain = max(0, 85.0 - current_cpu)  # Efficiency from 85% to current
        availability_gain = max(0, current_availability - 95.2)  # Availability improvement
        
        # Calculate cost savings
        cpu_cost_savings = (cpu_efficiency_gain / 85.0) * 0.3 * self.monthly_baseline_cost
        availability_cost_savings = (availability_gain / 4.5) * 0.2 * self.monthly_baseline_cost
        
        predicted_monthly_cost = self.monthly_baseline_cost - cpu_cost_savings - availability_cost_savings
        potential_savings = current_monthly_cost - predicted_monthly_cost
        
        # ROI projection (annual)
        annual_savings = potential_savings * 12
        investment_cost = 150000  # $150k HYDATIS investment
        roi_projection = (annual_savings / investment_cost) * 100 if investment_cost > 0 else 0
        
        # Identify optimization opportunities
        optimization_opportunities = await self._identify_optimization_opportunities(
            current_cpu, current_availability)
        
        # Calculate confidence
        confidence_score = self._calculate_forecast_confidence(
            cpu_data, horizon_days, current_cpu, current_availability)
        
        # Update metrics
        self.cost_predictions_counter.inc()
        self.roi_projection_gauge.set(roi_projection)
        
        return CostOptimizationForecast(
            current_monthly_cost=current_monthly_cost,
            predicted_monthly_cost=predicted_monthly_cost,
            potential_savings=potential_savings,
            roi_projection=roi_projection,
            optimization_opportunities=optimization_opportunities,
            confidence_score=confidence_score
        )
    
    async def _collect_cost_metrics(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Collect metrics relevant to cost optimization"""
        queries = {
            'cpu_utilization': 'avg(100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))',
            'availability': 'avg(up{job="kubernetes-nodes"}) * 100',
            'efficiency_score': 'avg(ml_scheduler_efficiency_percent)',
            'resource_waste': '100 - avg(cluster_resource_efficiency_percent)'
        }
        
        combined_data = []
        
        for metric_name, query in queries.items():
            data = await self._collect_prometheus_timeseries_simple(
                query, start_time, end_time, metric_name)
            if not combined_data:
                combined_data = data
            else:
                combined_data = pd.merge(combined_data, data, on='timestamp', how='outer')
        
        return combined_data.fillna(method='forward')
    
    async def _collect_prometheus_timeseries_simple(self, query: str, start_time: datetime,
                                                  end_time: datetime, metric_name: str) -> pd.DataFrame:
        """Simplified Prometheus time series collection"""
        async with aiohttp.ClientSession() as session:
            params = {
                'query': query,
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'step': '1h'
            }
            
            async with session.get(f"{self.prometheus_url}/api/v1/query_range",
                                 params=params) as response:
                data = await response.json()
        
        time_series_data = []
        results = data.get('data', {}).get('result', [])
        
        if results:
            values = results[0].get('values', [])
            for timestamp, value in values:
                time_series_data.append({
                    'timestamp': datetime.fromtimestamp(float(timestamp)),
                    metric_name: float(value)
                })
        
        return pd.DataFrame(time_series_data)
    
    async def _get_current_cpu_utilization(self) -> float:
        """Get current CPU utilization"""
        query = 'avg(100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))'
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.prometheus_url}/api/v1/query",
                                 params={'query': query}) as response:
                data = await response.json()
        
        results = data.get('data', {}).get('result', [])
        return float(results[0]['value'][1]) if results else 85.0
    
    async def _get_current_availability(self) -> float:
        """Get current cluster availability"""
        query = 'avg(up{job="kubernetes-nodes"}) * 100'
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.prometheus_url}/api/v1/query",
                                 params={'query': query}) as response:
                data = await response.json()
        
        results = data.get('data', {}).get('result', [])
        return float(results[0]['value'][1]) if results else 95.2
    
    async def _calculate_current_monthly_cost(self) -> float:
        """Calculate current monthly operational cost"""
        # HYDATIS cost structure
        node_count = await self._get_current_node_count()
        base_cost_per_node = 500  # $500/month per node
        
        # Additional costs: monitoring, storage, networking
        additional_costs = 2000  # $2k/month additional
        
        return (node_count * base_cost_per_node) + additional_costs
    
    async def _get_current_node_count(self) -> int:
        """Get current node count"""
        query = 'count(up{job="kubernetes-nodes"})'
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.prometheus_url}/api/v1/query",
                                 params={'query': query}) as response:
                data = await response.json()
        
        results = data.get('data', {}).get('result', [])
        return int(float(results[0]['value'][1])) if results else 6
    
    async def _identify_optimization_opportunities(self, current_cpu: float,
                                                 current_availability: float) -> List[Dict[str, Any]]:
        """Identify specific cost optimization opportunities"""
        opportunities = []
        
        # CPU optimization opportunities
        if current_cpu < 60:
            cpu_gap = 65 - current_cpu  # Target is 65%
            potential_savings = (cpu_gap / 65) * 0.3 * self.monthly_baseline_cost
            opportunities.append({
                'type': 'cpu_optimization',
                'description': f'Increase CPU utilization from {current_cpu:.1f}% to 65% target',
                'potential_monthly_savings': potential_savings,
                'effort': 'medium',
                'timeline': '2-4 weeks',
                'actions': [
                    'Tune ML scheduler aggressiveness parameters',
                    'Optimize pod resource requests',
                    'Implement workload consolidation policies'
                ]
            })
        
        elif current_cpu > 70:
            over_utilization = current_cpu - 65
            risk_cost = (over_utilization / 65) * 0.1 * self.monthly_baseline_cost
            opportunities.append({
                'type': 'cpu_risk_mitigation',
                'description': f'Reduce CPU utilization from {current_cpu:.1f}% to 65% target',
                'potential_monthly_cost': risk_cost,
                'effort': 'low',
                'timeline': '1-2 weeks',
                'actions': [
                    'Scale cluster horizontally',
                    'Reduce ML scheduler aggressiveness',
                    'Implement CPU throttling policies'
                ]
            })
        
        # Availability optimization
        if current_availability < 99.7:
            availability_gap = 99.7 - current_availability
            revenue_impact = (availability_gap / 4.5) * 5000 * 24 * 30  # Monthly revenue impact
            opportunities.append({
                'type': 'availability_improvement',
                'description': f'Improve availability from {current_availability:.2f}% to 99.7% SLA',
                'potential_monthly_revenue': revenue_impact,
                'effort': 'high',
                'timeline': '4-6 weeks',
                'actions': [
                    'Implement advanced health checking',
                    'Add node redundancy and auto-healing',
                    'Optimize ML scheduler resilience'
                ]
            })
        
        # Resource efficiency opportunities
        resource_efficiency = await self._get_resource_efficiency()
        if resource_efficiency < 85:
            efficiency_gap = 85 - resource_efficiency
            efficiency_savings = (efficiency_gap / 85) * 0.2 * self.monthly_baseline_cost
            opportunities.append({
                'type': 'resource_efficiency',
                'description': f'Improve resource efficiency from {resource_efficiency:.1f}% to 85%',
                'potential_monthly_savings': efficiency_savings,
                'effort': 'medium',
                'timeline': '3-5 weeks',
                'actions': [
                    'Implement bin-packing optimization',
                    'Tune ML model scoring algorithms',
                    'Add intelligent workload placement'
                ]
            })
        
        return opportunities
    
    async def _get_resource_efficiency(self) -> float:
        """Get current resource efficiency score"""
        query = 'avg(cluster_resource_efficiency_percent)'
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.prometheus_url}/api/v1/query",
                                 params={'query': query}) as response:
                data = await response.json()
        
        results = data.get('data', {}).get('result', [])
        return float(results[0]['value'][1]) if results else 75.0
    
    def _calculate_forecast_confidence(self, historical_data: pd.DataFrame,
                                     horizon_days: int, current_cpu: float,
                                     current_availability: float) -> float:
        """Calculate confidence in cost optimization forecast"""
        # Base confidence on data quality and horizon
        data_quality_score = min(1.0, len(historical_data) / 168)  # 168 hours = 1 week
        horizon_penalty = 1.0 / (1.0 + horizon_days / 30.0)  # Longer horizon = lower confidence
        
        # Performance stability factor
        if len(historical_data) > 24:
            cpu_variance = historical_data['cpu_utilization'].var() if 'cpu_utilization' in historical_data.columns else 100
            stability_score = 1.0 / (1.0 + cpu_variance / 100)
        else:
            stability_score = 0.5
        
        # Business target alignment factor
        cpu_alignment = 1.0 - abs(current_cpu - 65) / 65
        availability_alignment = min(1.0, current_availability / 99.7)
        target_alignment = (cpu_alignment + availability_alignment) / 2
        
        # Overall confidence
        confidence = (data_quality_score * 0.3 + 
                     horizon_penalty * 0.2 + 
                     stability_score * 0.3 + 
                     target_alignment * 0.2)
        
        return max(0.1, min(1.0, confidence))

class PerformanceTrendAnalyzer:
    """Analyzes performance trends and predicts degradation"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        
        # Metrics
        self.trend_analysis_counter = Counter('performance_trend_analysis_total',
                                            'Performance trend analyses completed',
                                            ['metric_type'])
        self.trend_confidence_gauge = Gauge('performance_trend_confidence',
                                          'Confidence in performance trend predictions',
                                          ['metric_type'])
    
    async def analyze_scheduling_performance_trend(self, horizon: ForecastHorizon) -> PredictionResult:
        """Analyze scheduling performance trends"""
        # Collect scheduling performance data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=14)
        
        latency_data = await self._collect_scheduling_latency_data(start_time, end_time)
        success_rate_data = await self._collect_success_rate_data(start_time, end_time)
        
        # Trend analysis
        latency_trend = self._calculate_trend(latency_data, 'latency')
        success_trend = self._calculate_trend(success_rate_data, 'success_rate')
        
        # Predict future performance
        horizon_hours = {'24h': 24, '7d': 168, '30d': 720}[horizon.value]
        future_latency = self._extrapolate_trend(latency_trend, horizon_hours)
        future_success_rate = self._extrapolate_trend(success_trend, horizon_hours)
        
        # Business impact assessment
        recommendations = self._generate_performance_recommendations(
            future_latency, future_success_rate, horizon)
        
        # Calculate confidence
        confidence = self._calculate_trend_confidence(latency_data, success_rate_data)
        
        self.trend_analysis_counter.labels(metric_type='scheduling_performance').inc()
        self.trend_confidence_gauge.labels(metric_type='scheduling_performance').set(confidence)
        
        return PredictionResult(
            prediction_type=PredictionType.PERFORMANCE_TREND,
            horizon=horizon,
            predicted_value=future_latency,
            confidence_interval=(future_latency * 0.8, future_latency * 1.2),
            confidence_score=confidence,
            trend_direction=self._determine_trend_direction(latency_trend),
            recommendations=recommendations,
            metadata={
                'predicted_latency_ms': future_latency,
                'predicted_success_rate': future_success_rate,
                'current_latency_ms': latency_data['latency'].iloc[-1] if len(latency_data) > 0 else 0,
                'current_success_rate': success_rate_data['success_rate'].iloc[-1] if len(success_rate_data) > 0 else 0
            }
        )
    
    async def _collect_scheduling_latency_data(self, start_time: datetime, 
                                             end_time: datetime) -> pd.DataFrame:
        """Collect scheduling latency time series"""
        query = 'histogram_quantile(0.99, rate(ml_scheduler_scheduling_duration_seconds_bucket[5m])) * 1000'
        
        async with aiohttp.ClientSession() as session:
            params = {
                'query': query,
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'step': '1h'
            }
            
            async with session.get(f"{self.prometheus_url}/api/v1/query_range",
                                 params=params) as response:
                data = await response.json()
        
        time_series_data = []
        results = data.get('data', {}).get('result', [])
        
        if results:
            values = results[0].get('values', [])
            for timestamp, value in values:
                time_series_data.append({
                    'timestamp': datetime.fromtimestamp(float(timestamp)),
                    'latency': float(value)
                })
        
        return pd.DataFrame(time_series_data)
    
    async def _collect_success_rate_data(self, start_time: datetime,
                                       end_time: datetime) -> pd.DataFrame:
        """Collect scheduling success rate time series"""
        query = 'rate(ml_scheduler_scheduling_success_total[5m]) / rate(ml_scheduler_scheduling_requests_total[5m]) * 100'
        
        async with aiohttp.ClientSession() as session:
            params = {
                'query': query,
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'step': '1h'
            }
            
            async with session.get(f"{self.prometheus_url}/api/v1/query_range",
                                 params=params) as response:
                data = await response.json()
        
        time_series_data = []
        results = data.get('data', {}).get('result', [])
        
        if results:
            values = results[0].get('values', [])
            for timestamp, value in values:
                time_series_data.append({
                    'timestamp': datetime.fromtimestamp(float(timestamp)),
                    'success_rate': float(value)
                })
        
        return pd.DataFrame(time_series_data)
    
    def _calculate_trend(self, data: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """Calculate trend statistics"""
        if len(data) < 10:
            return {'slope': 0.0, 'r_squared': 0.0, 'trend_strength': 0.0}
        
        # Convert timestamps to numeric for regression
        data = data.sort_values('timestamp')
        x = np.arange(len(data)).reshape(-1, 1)
        y = data[target_col].values
        
        # Linear regression for trend
        model = LinearRegression()
        model.fit(x, y)
        
        slope = model.coef_[0]
        r_squared = model.score(x, y)
        
        # Trend strength (normalized slope)
        value_range = y.max() - y.min()
        trend_strength = abs(slope) / (value_range / len(data)) if value_range > 0 else 0
        
        return {
            'slope': slope,
            'r_squared': r_squared,
            'trend_strength': trend_strength
        }
    
    def _extrapolate_trend(self, trend: Dict[str, float], hours_ahead: int) -> float:
        """Extrapolate trend to future time point"""
        slope = trend['slope']
        baseline_value = 50.0  # Baseline assumption
        
        # Project trend forward
        future_value = baseline_value + (slope * hours_ahead)
        
        # Apply confidence dampening for longer horizons
        confidence_factor = 1.0 / (1.0 + hours_ahead / 168.0)  # Dampen for longer horizons
        
        return future_value * confidence_factor + baseline_value * (1 - confidence_factor)
    
    def _determine_trend_direction(self, trend: Dict[str, float]) -> str:
        """Determine trend direction"""
        slope = trend['slope']
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _calculate_trend_confidence(self, latency_data: pd.DataFrame,
                                  success_data: pd.DataFrame) -> float:
        """Calculate confidence in trend predictions"""
        # Data quality factors
        latency_quality = min(1.0, len(latency_data) / 168)  # 1 week of hourly data
        success_quality = min(1.0, len(success_data) / 168)
        
        # Data variance (lower variance = higher confidence)
        latency_variance = latency_data['latency'].var() if len(latency_data) > 0 else 1000
        success_variance = success_data['success_rate'].var() if len(success_data) > 0 else 100
        
        latency_stability = 1.0 / (1.0 + latency_variance / 100)
        success_stability = 1.0 / (1.0 + success_variance / 10)
        
        # Overall confidence
        confidence = (latency_quality * 0.3 + success_quality * 0.3 +
                     latency_stability * 0.2 + success_stability * 0.2)
        
        return max(0.1, min(1.0, confidence))
    
    def _generate_performance_recommendations(self, future_latency: float,
                                            future_success_rate: float,
                                            horizon: ForecastHorizon) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if future_latency > 100:  # Target: <100ms P99
            recommendations.append(
                f"Predicted latency ({future_latency:.1f}ms) exceeds target - "
                "optimize ML model inference or increase cache hit rate"
            )
        
        if future_success_rate < 99:  # Target: >99% success rate
            recommendations.append(
                f"Predicted success rate ({future_success_rate:.2f}%) below target - "
                "investigate ML service reliability and fallback mechanisms"
            )
        
        if future_latency > 150:
            recommendations.append(
                "Consider implementing latency-based auto-scaling for ML services"
            )
        
        if future_success_rate < 95:
            recommendations.append(
                "Critical: Review ML service health and implement redundancy"
            )
        
        if not recommendations:
            recommendations.append(
                f"Performance metrics projected to remain within targets for {horizon.value}"
            )
        
        return recommendations

class PredictiveAnalyticsEngine:
    """Main predictive analytics engine"""
    
    def __init__(self, config_path: str, prometheus_url: str):
        self.config = self._load_config(config_path)
        self.prometheus_url = prometheus_url
        
        # Initialize predictors
        self.capacity_predictor = CapacityPredictor(prometheus_url)
        self.cost_predictor = CostOptimizationPredictor(prometheus_url)
        self.performance_analyzer = PerformanceTrendAnalyzer(prometheus_url)
        
        # Service metrics
        self.predictions_generated_counter = Counter('predictive_analytics_predictions_total',
                                                   'Total predictions generated',
                                                   ['prediction_type', 'horizon'])
        self.prediction_accuracy_gauge = Gauge('predictive_analytics_accuracy',
                                             'Historical prediction accuracy',
                                             ['prediction_type'])
        self.analytics_cycle_duration = Histogram('predictive_analytics_cycle_duration_seconds',
                                                'Time spent on analytics cycles')
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load predictive analytics configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'prediction_schedule': {
                    'capacity_forecast': {'interval_hours': 6, 'horizons': ['24h', '7d', '30d']},
                    'cost_optimization': {'interval_hours': 12, 'horizons': ['7d', '30d']},
                    'performance_trend': {'interval_hours': 4, 'horizons': ['24h', '7d']}
                },
                'business_targets': {
                    'cpu_utilization': 65.0,
                    'availability': 99.7,
                    'annual_roi': 1400
                }
            }
    
    async def generate_capacity_forecast(self, horizon: ForecastHorizon) -> CapacityForecast:
        """Generate capacity forecast"""
        forecast = await self.capacity_predictor.predict_cluster_capacity(horizon)
        
        self.predictions_generated_counter.labels(
            prediction_type='capacity', horizon=horizon.value).inc()
        
        return forecast
    
    async def generate_cost_optimization_forecast(self, horizon: ForecastHorizon) -> CostOptimizationForecast:
        """Generate cost optimization forecast"""
        forecast = await self.cost_predictor.predict_cost_optimization(horizon)
        
        self.predictions_generated_counter.labels(
            prediction_type='cost', horizon=horizon.value).inc()
        
        return forecast
    
    async def generate_performance_forecast(self, horizon: ForecastHorizon) -> PredictionResult:
        """Generate performance trend forecast"""
        forecast = await self.performance_analyzer.analyze_scheduling_performance_trend(horizon)
        
        self.predictions_generated_counter.labels(
            prediction_type='performance', horizon=horizon.value).inc()
        
        return forecast
    
    async def generate_comprehensive_forecast(self) -> Dict[str, Any]:
        """Generate comprehensive forecast across all dimensions"""
        with self.analytics_cycle_duration.time():
            # Generate forecasts for different horizons
            forecasts = {}
            
            # Short-term forecasts (24h)
            forecasts['24h'] = {
                'capacity': await self.generate_capacity_forecast(ForecastHorizon.SHORT_TERM),
                'performance': await self.generate_performance_forecast(ForecastHorizon.SHORT_TERM)
            }
            
            # Medium-term forecasts (7d)
            forecasts['7d'] = {
                'capacity': await self.generate_capacity_forecast(ForecastHorizon.MEDIUM_TERM),
                'cost_optimization': await self.generate_cost_optimization_forecast(ForecastHorizon.MEDIUM_TERM),
                'performance': await self.generate_performance_forecast(ForecastHorizon.MEDIUM_TERM)
            }
            
            # Long-term forecasts (30d)
            forecasts['30d'] = {
                'capacity': await self.generate_capacity_forecast(ForecastHorizon.LONG_TERM),
                'cost_optimization': await self.generate_cost_optimization_forecast(ForecastHorizon.LONG_TERM)
            }
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(forecasts)
            
            return {
                'generated_at': datetime.utcnow().isoformat(),
                'forecasts': forecasts,
                'executive_summary': executive_summary,
                'business_alignment': await self._assess_business_alignment(forecasts)
            }
    
    async def _generate_executive_summary(self, forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of predictions"""
        # Extract key metrics
        roi_7d = forecasts['7d']['cost_optimization'].roi_projection
        roi_30d = forecasts['30d']['cost_optimization'].roi_projection
        
        capacity_scaling_24h = forecasts['24h']['capacity'].recommended_node_count
        capacity_scaling_30d = forecasts['30d']['capacity'].recommended_node_count
        
        potential_savings_7d = forecasts['7d']['cost_optimization'].potential_savings
        potential_savings_30d = forecasts['30d']['cost_optimization'].potential_savings
        
        return {
            'roi_trajectory': {
                '7_day': roi_7d,
                '30_day': roi_30d,
                'on_track_for_target': roi_30d > 1200  # Within 200% of 1400% target
            },
            'capacity_planning': {
                'immediate_scaling_needed': capacity_scaling_24h > 6,
                'long_term_scaling_needed': capacity_scaling_30d > 8,
                'recommended_action': 'scale' if capacity_scaling_30d > 6 else 'optimize'
            },
            'cost_optimization': {
                'monthly_savings_potential': potential_savings_30d,
                'annual_savings_projection': potential_savings_30d * 12,
                'payback_period_months': 150000 / max(potential_savings_30d, 1000)  # Investment / monthly savings
            },
            'key_risks': await self._identify_key_risks(forecasts),
            'recommended_actions': await self._generate_executive_actions(forecasts)
        }
    
    async def _assess_business_alignment(self, forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """Assess alignment with HYDATIS business targets"""
        targets = self.config['business_targets']
        
        # Extract predicted metrics
        predicted_cpu_30d = forecasts['30d']['capacity'].predicted_cpu_utilization
        roi_30d = forecasts['30d']['cost_optimization'].roi_projection
        
        return {
            'cpu_target_alignment': {
                'target': targets['cpu_utilization'],
                'predicted': predicted_cpu_30d,
                'deviation': abs(predicted_cpu_30d - targets['cpu_utilization']),
                'on_target': abs(predicted_cpu_30d - targets['cpu_utilization']) <= 5.0
            },
            'roi_target_alignment': {
                'target': targets['annual_roi'],
                'predicted': roi_30d,
                'achievement_percentage': (roi_30d / targets['annual_roi']) * 100,
                'on_target': roi_30d >= targets['annual_roi'] * 0.9  # 90% of target
            },
            'overall_score': self._calculate_business_alignment_score(
                predicted_cpu_30d, roi_30d, targets)
        }
    
    def _calculate_business_alignment_score(self, predicted_cpu: float,
                                          predicted_roi: float,
                                          targets: Dict[str, float]) -> float:
        """Calculate overall business alignment score"""
        # CPU alignment (0-50 points)
        cpu_alignment = max(0, 50 - abs(predicted_cpu - targets['cpu_utilization']) * 2)
        
        # ROI alignment (0-50 points)
        roi_ratio = predicted_roi / targets['annual_roi']
        roi_alignment = min(50, roi_ratio * 50)
        
        return (cpu_alignment + roi_alignment) / 100.0
    
    async def _identify_key_risks(self, forecasts: Dict[str, Any]) -> List[str]:
        """Identify key business risks from forecasts"""
        risks = []
        
        # Capacity risks
        capacity_30d = forecasts['30d']['capacity']
        if capacity_30d.recommended_node_count > 10:
            risks.append(f"Significant cluster scaling required: {capacity_30d.recommended_node_count} nodes")
        
        # Cost risks
        cost_30d = forecasts['30d']['cost_optimization']
        if cost_30d.roi_projection < 1000:
            risks.append(f"ROI projection below business case: {cost_30d.roi_projection:.0f}%")
        
        # Performance risks
        perf_24h = forecasts['24h']['performance']
        if perf_24h.predicted_value > 150:  # Latency > 150ms
            risks.append("Performance degradation predicted: latency exceeding targets")
        
        return risks
    
    async def _generate_executive_actions(self, forecasts: Dict[str, Any]) -> List[str]:
        """Generate executive-level recommended actions"""
        actions = []
        
        # ROI-focused actions
        roi_30d = forecasts['30d']['cost_optimization'].roi_projection
        if roi_30d < 1200:
            actions.append("Accelerate cost optimization initiatives to meet ROI targets")
        
        # Capacity actions
        capacity_30d = forecasts['30d']['capacity']
        if capacity_30d.cost_impact > 5000:
            actions.append(f"Budget approval needed: ${capacity_30d.cost_impact:.0f}/month capacity expansion")
        
        # Performance actions
        if any('Critical' in rec for rec in forecasts['24h']['performance'].recommendations):
            actions.append("Immediate performance optimization required")
        
        return actions

async def main():
    """Main entry point for predictive analytics service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predictive Analytics Engine')
    parser.add_argument('--config', default='/etc/ml-scheduler/predictive_config.yaml',
                       help='Predictive analytics configuration file')
    parser.add_argument('--prometheus-url', default='http://prometheus:9090',
                       help='Prometheus server URL')
    parser.add_argument('--output-format', choices=['json', 'yaml'], default='json',
                       help='Output format for forecasts')
    parser.add_argument('--horizon', choices=['24h', '7d', '30d'], default='7d',
                       help='Forecast horizon')
    
    args = parser.parse_args()
    
    # Initialize analytics engine
    engine = PredictiveAnalyticsEngine(args.config, args.prometheus_url)
    
    try:
        # Generate comprehensive forecast
        forecast = await engine.generate_comprehensive_forecast()
        
        # Output results
        if args.output_format == 'json':
            print(json.dumps(forecast, indent=2, default=str))
        else:
            print(yaml.dump(forecast, default_flow_style=False))
            
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))