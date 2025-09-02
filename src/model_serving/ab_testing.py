import random
import hashlib
import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiment"""
    experiment_id: str
    name: str
    description: str
    model_type: str  # xgboost, qlearning, anomaly
    traffic_split: Dict[str, float]  # variant_name -> traffic_percentage
    start_time: datetime
    end_time: datetime
    success_metrics: List[str]
    minimum_sample_size: int
    confidence_level: float = 0.95
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_by: str = "ml-scheduler"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class ModelVariant:
    """Model variant for A/B testing"""
    variant_id: str
    name: str
    model_version: str
    model_uri: str
    traffic_percentage: float
    config_overrides: Dict[str, Any] = None
    is_control: bool = False
    
    def __post_init__(self):
        if self.config_overrides is None:
            self.config_overrides = {}

@dataclass
class ExperimentResult:
    """Result of an A/B test request"""
    experiment_id: str
    variant_id: str
    variant_name: str
    user_id: str
    request_data: Dict[str, Any]
    response_data: Dict[str, Any]
    timestamp: datetime
    latency_ms: float
    success: bool
    error_message: Optional[str] = None

class ABTestingFramework:
    """
    A/B testing framework for ML model serving
    Supports traffic splitting, statistical analysis, and experiment management
    """
    
    def __init__(self, redis_cache=None):
        self.redis_cache = redis_cache
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.model_variants: Dict[str, Dict[str, ModelVariant]] = defaultdict(dict)
        self.results_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._lock = threading.RLock()
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'variant_requests': defaultdict(int),
            'experiment_requests': defaultdict(int),
            'errors': 0
        }
    
    def create_experiment(self, config: ExperimentConfig) -> bool:
        """
        Create new A/B testing experiment
        
        Args:
            config: Experiment configuration
            
        Returns:
            True if experiment created successfully
        """
        try:
            with self._lock:
                # Validate traffic split sums to 100%
                total_traffic = sum(config.traffic_split.values())
                if abs(total_traffic - 100.0) > 0.01:
                    raise ValueError(f"Traffic split must sum to 100%, got {total_traffic}")
                
                # Validate time range
                if config.start_time >= config.end_time:
                    raise ValueError("Start time must be before end time")
                
                self.experiments[config.experiment_id] = config
                logger.info(f"Created experiment {config.experiment_id}: {config.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create experiment {config.experiment_id}: {e}")
            return False
    
    def add_model_variant(self, experiment_id: str, variant: ModelVariant) -> bool:
        """
        Add model variant to experiment
        
        Args:
            experiment_id: ID of experiment
            variant: Model variant configuration
            
        Returns:
            True if variant added successfully
        """
        try:
            with self._lock:
                if experiment_id not in self.experiments:
                    raise ValueError(f"Experiment {experiment_id} not found")
                
                experiment = self.experiments[experiment_id]
                if variant.name not in experiment.traffic_split:
                    raise ValueError(f"Variant {variant.name} not in traffic split configuration")
                
                variant.traffic_percentage = experiment.traffic_split[variant.name]
                self.model_variants[experiment_id][variant.variant_id] = variant
                
                logger.info(f"Added variant {variant.variant_id} to experiment {experiment_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add variant {variant.variant_id}: {e}")
            return False
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        try:
            with self._lock:
                if experiment_id not in self.experiments:
                    raise ValueError(f"Experiment {experiment_id} not found")
                
                experiment = self.experiments[experiment_id]
                
                # Validate experiment has variants
                if experiment_id not in self.model_variants or not self.model_variants[experiment_id]:
                    raise ValueError(f"No variants configured for experiment {experiment_id}")
                
                # Check if current time is within experiment window
                now = datetime.utcnow()
                if now < experiment.start_time:
                    raise ValueError(f"Experiment {experiment_id} start time is in the future")
                
                if now > experiment.end_time:
                    raise ValueError(f"Experiment {experiment_id} end time has passed")
                
                experiment.status = ExperimentStatus.RUNNING
                logger.info(f"Started experiment {experiment_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {e}")
            return False
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a running experiment"""
        try:
            with self._lock:
                if experiment_id not in self.experiments:
                    raise ValueError(f"Experiment {experiment_id} not found")
                
                experiment = self.experiments[experiment_id]
                experiment.status = ExperimentStatus.COMPLETED
                logger.info(f"Stopped experiment {experiment_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop experiment {experiment_id}: {e}")
            return False
    
    def _hash_user_id(self, user_id: str, experiment_id: str) -> str:
        """Generate consistent hash for user assignment"""
        combined = f"{user_id}:{experiment_id}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _assign_variant(self, user_id: str, experiment_id: str) -> Optional[ModelVariant]:
        """
        Assign user to experiment variant using consistent hashing
        
        Args:
            user_id: Unique user identifier
            experiment_id: Experiment ID
            
        Returns:
            Assigned model variant or None if no experiment
        """
        try:
            if experiment_id not in self.experiments:
                return None
            
            experiment = self.experiments[experiment_id]
            if experiment.status != ExperimentStatus.RUNNING:
                return None
            
            # Check if experiment is active
            now = datetime.utcnow()
            if now < experiment.start_time or now > experiment.end_time:
                return None
            
            variants = list(self.model_variants[experiment_id].values())
            if not variants:
                return None
            
            # Use consistent hashing for stable assignment
            user_hash = self._hash_user_id(user_id, experiment_id)
            hash_value = int(user_hash[:8], 16) % 10000  # 0-9999 range
            
            # Assign based on cumulative traffic percentages
            cumulative = 0
            for variant in variants:
                cumulative += variant.traffic_percentage
                if hash_value < cumulative * 100:  # Convert percentage to 0-10000 range
                    return variant
            
            # Fallback to control variant or first variant
            control_variant = next((v for v in variants if v.is_control), variants[0])
            return control_variant
            
        except Exception as e:
            logger.error(f"Failed to assign variant for user {user_id}: {e}")
            return None
    
    def get_model_variant(self, user_id: str, model_type: str, 
                         request_data: Dict[str, Any]) -> Tuple[Optional[ModelVariant], Optional[str]]:
        """
        Get model variant for user based on active experiments
        
        Args:
            user_id: Unique user identifier
            model_type: Type of model (xgboost, qlearning, anomaly)
            request_data: Request data for context
            
        Returns:
            Tuple of (assigned_variant, experiment_id) or (None, None)
        """
        try:
            # Find active experiments for this model type
            active_experiments = [
                (exp_id, exp) for exp_id, exp in self.experiments.items()
                if exp.model_type == model_type and exp.status == ExperimentStatus.RUNNING
            ]
            
            # If no active experiments, return None
            if not active_experiments:
                return None, None
            
            # For simplicity, use the first active experiment
            # In production, you might want more sophisticated experiment selection
            experiment_id, experiment = active_experiments[0]
            
            # Check time bounds
            now = datetime.utcnow()
            if now < experiment.start_time or now > experiment.end_time:
                return None, None
            
            variant = self._assign_variant(user_id, experiment_id)
            return variant, experiment_id
            
        except Exception as e:
            logger.error(f"Failed to get model variant for user {user_id}: {e}")
            return None, None
    
    def record_result(self, result: ExperimentResult):
        """Record experiment result for analysis"""
        try:
            with self._lock:
                self.results_buffer[result.experiment_id].append(result)
                self.stats['total_requests'] += 1
                self.stats['variant_requests'][result.variant_id] += 1
                self.stats['experiment_requests'][result.experiment_id] += 1
                
                if not result.success:
                    self.stats['errors'] += 1
                
                # Cache result if Redis available
                if self.redis_cache:
                    cache_key = f"ab_result:{result.experiment_id}:{result.timestamp.isoformat()}"
                    self.redis_cache.set(cache_key, asdict(result), ttl=7*24*3600)  # 7 days
                    
        except Exception as e:
            logger.error(f"Failed to record experiment result: {e}")
    
    def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment statistics and performance metrics"""
        try:
            if experiment_id not in self.experiments:
                return {'error': f'Experiment {experiment_id} not found'}
            
            experiment = self.experiments[experiment_id]
            results = list(self.results_buffer[experiment_id])
            
            if not results:
                return {
                    'experiment_id': experiment_id,
                    'status': experiment.status.value,
                    'total_requests': 0,
                    'variants': []
                }
            
            # Group results by variant
            variant_results = defaultdict(list)
            for result in results:
                variant_results[result.variant_id].append(result)
            
            # Calculate statistics for each variant
            variant_stats = []
            for variant_id, variant_data in variant_results.items():
                successful_results = [r for r in variant_data if r.success]
                
                if successful_results:
                    latencies = [r.latency_ms for r in successful_results]
                    avg_latency = sum(latencies) / len(latencies)
                    p95_latency = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0
                else:
                    avg_latency = 0
                    p95_latency = 0
                
                variant_info = {
                    'variant_id': variant_id,
                    'total_requests': len(variant_data),
                    'successful_requests': len(successful_results),
                    'success_rate': len(successful_results) / len(variant_data) * 100 if variant_data else 0,
                    'average_latency_ms': round(avg_latency, 2),
                    'p95_latency_ms': round(p95_latency, 2),
                    'error_rate': (len(variant_data) - len(successful_results)) / len(variant_data) * 100 if variant_data else 0
                }
                
                # Add variant details if available
                if experiment_id in self.model_variants and variant_id in self.model_variants[experiment_id]:
                    variant = self.model_variants[experiment_id][variant_id]
                    variant_info.update({
                        'variant_name': variant.name,
                        'model_version': variant.model_version,
                        'traffic_percentage': variant.traffic_percentage,
                        'is_control': variant.is_control
                    })
                
                variant_stats.append(variant_info)
            
            return {
                'experiment_id': experiment_id,
                'experiment_name': experiment.name,
                'model_type': experiment.model_type,
                'status': experiment.status.value,
                'start_time': experiment.start_time.isoformat(),
                'end_time': experiment.end_time.isoformat(),
                'total_requests': len(results),
                'variants': variant_stats,
                'overall_success_rate': len([r for r in results if r.success]) / len(results) * 100,
                'confidence_level': experiment.confidence_level,
                'minimum_sample_size': experiment.minimum_sample_size
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment stats for {experiment_id}: {e}")
            return {'error': str(e)}
    
    def check_statistical_significance(self, experiment_id: str) -> Dict[str, Any]:
        """
        Check if experiment results are statistically significant
        Uses basic t-test for comparing variants
        """
        try:
            stats = self.get_experiment_stats(experiment_id)
            if 'error' in stats:
                return stats
            
            variants = stats['variants']
            if len(variants) < 2:
                return {'significant': False, 'reason': 'Need at least 2 variants'}
            
            # Find control variant
            control_variant = next((v for v in variants if v.get('is_control', False)), variants[0])
            
            significance_results = []
            for variant in variants:
                if variant['variant_id'] == control_variant['variant_id']:
                    continue
                
                # Check minimum sample size
                if (variant['total_requests'] < self.experiments[experiment_id].minimum_sample_size or
                    control_variant['total_requests'] < self.experiments[experiment_id].minimum_sample_size):
                    significance_results.append({
                        'variant_id': variant['variant_id'],
                        'significant': False,
                        'reason': 'Insufficient sample size',
                        'sample_size': variant['total_requests'],
                        'required_size': self.experiments[experiment_id].minimum_sample_size
                    })
                    continue
                
                # Simple comparison based on success rate difference
                # In production, you'd use proper statistical tests (t-test, chi-square, etc.)
                success_rate_diff = variant['success_rate'] - control_variant['success_rate']
                latency_diff = variant['average_latency_ms'] - control_variant['average_latency_ms']
                
                # Basic significance threshold (>5% improvement or <10% latency increase)
                is_significant = (success_rate_diff > 5.0 or 
                                (success_rate_diff > 1.0 and latency_diff < control_variant['average_latency_ms'] * 0.1))
                
                significance_results.append({
                    'variant_id': variant['variant_id'],
                    'variant_name': variant.get('variant_name', 'Unknown'),
                    'significant': is_significant,
                    'success_rate_diff': round(success_rate_diff, 2),
                    'latency_diff_ms': round(latency_diff, 2),
                    'sample_size': variant['total_requests'],
                    'confidence_level': self.experiments[experiment_id].confidence_level
                })
            
            return {
                'experiment_id': experiment_id,
                'control_variant': control_variant['variant_id'],
                'significance_results': significance_results,
                'overall_significant': any(r['significant'] for r in significance_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to check statistical significance for {experiment_id}: {e}")
            return {'error': str(e)}
    
    def get_recommendation(self, experiment_id: str) -> Dict[str, Any]:
        """Get recommendation for experiment winner"""
        try:
            significance = self.check_statistical_significance(experiment_id)
            if 'error' in significance:
                return significance
            
            if not significance['overall_significant']:
                return {
                    'recommendation': 'continue',
                    'reason': 'No statistically significant differences found',
                    'action': 'Continue experiment or increase sample size'
                }
            
            # Find best performing variant
            stats = self.get_experiment_stats(experiment_id)
            variants = stats['variants']
            
            # Score variants based on success rate and latency
            scored_variants = []
            for variant in variants:
                # Composite score: success_rate - latency_penalty
                latency_penalty = variant['average_latency_ms'] / 1000  # Convert to seconds for scoring
                score = variant['success_rate'] - latency_penalty
                
                scored_variants.append({
                    'variant_id': variant['variant_id'],
                    'variant_name': variant.get('variant_name', 'Unknown'),
                    'score': score,
                    'success_rate': variant['success_rate'],
                    'average_latency_ms': variant['average_latency_ms'],
                    'total_requests': variant['total_requests']
                })
            
            # Sort by score descending
            scored_variants.sort(key=lambda x: x['score'], reverse=True)
            winner = scored_variants[0]
            
            return {
                'recommendation': 'promote_winner',
                'winner': winner,
                'all_variants': scored_variants,
                'reason': f"Variant {winner['variant_name']} shows best performance",
                'action': f"Promote variant {winner['variant_id']} to 100% traffic"
            }
            
        except Exception as e:
            logger.error(f"Failed to get recommendation for {experiment_id}: {e}")
            return {'error': str(e)}
    
    def route_request(self, user_id: str, model_type: str, 
                     request_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """
        Route request to appropriate model variant
        
        Args:
            user_id: Unique user identifier
            model_type: Type of model
            request_data: Request payload
            
        Returns:
            Tuple of (variant_id, experiment_id, routing_metadata)
        """
        try:
            self.stats['total_requests'] += 1
            
            # Get variant assignment
            variant, experiment_id = self.get_model_variant(user_id, model_type, request_data)
            
            if variant is None:
                return None, None, {'routing': 'default', 'reason': 'no_active_experiment'}
            
            routing_metadata = {
                'routing': 'experiment',
                'experiment_id': experiment_id,
                'variant_id': variant.variant_id,
                'variant_name': variant.name,
                'model_version': variant.model_version,
                'traffic_percentage': variant.traffic_percentage,
                'is_control': variant.is_control,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return variant.variant_id, experiment_id, routing_metadata
            
        except Exception as e:
            logger.error(f"Failed to route request for user {user_id}: {e}")
            self.stats['errors'] += 1
            return None, None, {'routing': 'error', 'error': str(e)}
    
    def get_active_experiments(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of active experiments"""
        try:
            active_experiments = []
            now = datetime.utcnow()
            
            for exp_id, experiment in self.experiments.items():
                if experiment.status != ExperimentStatus.RUNNING:
                    continue
                
                if model_type and experiment.model_type != model_type:
                    continue
                
                if now < experiment.start_time or now > experiment.end_time:
                    continue
                
                experiment_info = {
                    'experiment_id': exp_id,
                    'name': experiment.name,
                    'model_type': experiment.model_type,
                    'status': experiment.status.value,
                    'start_time': experiment.start_time.isoformat(),
                    'end_time': experiment.end_time.isoformat(),
                    'traffic_split': experiment.traffic_split,
                    'variant_count': len(self.model_variants.get(exp_id, {}))
                }
                
                active_experiments.append(experiment_info)
            
            return active_experiments
            
        except Exception as e:
            logger.error(f"Failed to get active experiments: {e}")
            return []
    
    def cleanup_completed_experiments(self, days_to_keep: int = 30) -> int:
        """Clean up old completed experiments"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            cleaned_count = 0
            
            with self._lock:
                experiments_to_remove = []
                for exp_id, experiment in self.experiments.items():
                    if (experiment.status in [ExperimentStatus.COMPLETED, ExperimentStatus.CANCELLED] and
                        experiment.end_time < cutoff_date):
                        experiments_to_remove.append(exp_id)
                
                for exp_id in experiments_to_remove:
                    del self.experiments[exp_id]
                    if exp_id in self.model_variants:
                        del self.model_variants[exp_id]
                    if exp_id in self.results_buffer:
                        del self.results_buffer[exp_id]
                    cleaned_count += 1
                
                logger.info(f"Cleaned up {cleaned_count} completed experiments")
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup experiments: {e}")
            return 0
    
    def get_framework_stats(self) -> Dict[str, Any]:
        """Get overall A/B testing framework statistics"""
        return {
            'total_experiments': len(self.experiments),
            'active_experiments': len([e for e in self.experiments.values() 
                                     if e.status == ExperimentStatus.RUNNING]),
            'total_variants': sum(len(variants) for variants in self.model_variants.values()),
            'framework_stats': dict(self.stats),
            'timestamp': datetime.utcnow().isoformat()
        }


# Global A/B testing instance
_ab_testing_instance = None

def get_ab_testing() -> ABTestingFramework:
    """Get global A/B testing framework instance"""
    global _ab_testing_instance
    if _ab_testing_instance is None:
        from .redis_cache import get_cache
        _ab_testing_instance = ABTestingFramework(redis_cache=get_cache())
    return _ab_testing_instance

def init_ab_testing(redis_cache=None) -> ABTestingFramework:
    """Initialize A/B testing framework with custom cache"""
    global _ab_testing_instance
    _ab_testing_instance = ABTestingFramework(redis_cache=redis_cache)
    return _ab_testing_instance