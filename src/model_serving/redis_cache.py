import redis
import json
import pickle
import hashlib
import logging
import time
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class RedisCache:
    """
    High-performance Redis caching layer for ML model predictions
    Supports TTL, compression, and intelligent cache invalidation
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 default_ttl: int = 3600,
                 max_retries: int = 3,
                 retry_delay: float = 0.1):
        """
        Initialize Redis cache connection
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password if authentication required
            default_ttl: Default time-to-live in seconds
            max_retries: Maximum retry attempts for failed operations
            retry_delay: Delay between retries in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_pool = None
        self.redis_client = None
        self._lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'sets': 0,
            'deletes': 0
        }
        
        self._connect()
    
    def _connect(self):
        """Establish Redis connection with connection pooling"""
        try:
            self.connection_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20
            )
            
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            raise
    
    def _generate_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """Generate consistent cache key from data"""
        # Sort keys for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True, separators=(',', ':'))
        hash_object = hashlib.md5(sorted_data.encode())
        return f"{prefix}:{hash_object.hexdigest()}"
    
    @contextmanager
    def _retry_operation(self):
        """Context manager for retrying Redis operations"""
        for attempt in range(self.max_retries):
            try:
                yield
                break
            except (redis.ConnectionError, redis.TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Redis operation failed after {self.max_retries} attempts: {e}")
                    self.stats['errors'] += 1
                    raise
                else:
                    logger.warning(f"Redis operation failed (attempt {attempt + 1}), retrying: {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected Redis error: {e}")
                self.stats['errors'] += 1
                raise
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for Redis storage with compression"""
        try:
            # Use pickle for better performance with complex Python objects
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            raise
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from Redis storage"""
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.redis_client:
            return None
        
        try:
            with self._retry_operation():
                with self._lock:
                    data = self.redis_client.get(key)
                    if data is not None:
                        self.stats['hits'] += 1
                        return self._deserialize_data(data)
                    else:
                        self.stats['misses'] += 1
                        return None
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default: use default_ttl)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            with self._retry_operation():
                with self._lock:
                    serialized_data = self._serialize_data(value)
                    ttl = ttl or self.default_ttl
                    
                    result = self.redis_client.setex(key, ttl, serialized_data)
                    if result:
                        self.stats['sets'] += 1
                    return bool(result)
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False
        
        try:
            with self._retry_operation():
                with self._lock:
                    result = self.redis_client.delete(key)
                    if result:
                        self.stats['deletes'] += 1
                    return bool(result)
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    def get_prediction(self, model_type: str, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction for specific model and features
        
        Args:
            model_type: Type of model (xgboost, qlearning, anomaly)
            features: Feature dictionary for prediction
            
        Returns:
            Cached prediction result or None
        """
        cache_key = self._generate_key(f"prediction:{model_type}", features)
        return self.get(cache_key)
    
    def set_prediction(self, model_type: str, features: Dict[str, Any], 
                      prediction: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Cache prediction result for specific model and features
        
        Args:
            model_type: Type of model (xgboost, qlearning, anomaly)
            features: Feature dictionary used for prediction
            prediction: Prediction result to cache
            ttl: Time-to-live for this prediction
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._generate_key(f"prediction:{model_type}", features)
        
        # Add timestamp to prediction
        prediction_with_meta = {
            'prediction': prediction,
            'timestamp': datetime.utcnow().isoformat(),
            'model_type': model_type
        }
        
        return self.set(cache_key, prediction_with_meta, ttl)
    
    def invalidate_model_cache(self, model_type: str) -> int:
        """
        Invalidate all cached predictions for a specific model
        
        Args:
            model_type: Type of model to invalidate
            
        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0
        
        try:
            with self._retry_operation():
                pattern = f"prediction:{model_type}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    logger.info(f"Invalidated {deleted} cache entries for {model_type}")
                    return deleted
                return 0
        except Exception as e:
            logger.error(f"Cache invalidation failed for {model_type}: {e}")
            return 0
    
    def get_batch_predictions(self, model_type: str, 
                            feature_list: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """
        Get multiple cached predictions in batch
        
        Args:
            model_type: Type of model
            feature_list: List of feature dictionaries
            
        Returns:
            List of cached predictions (None for cache misses)
        """
        if not self.redis_client or not feature_list:
            return [None] * len(feature_list)
        
        try:
            with self._retry_operation():
                cache_keys = [self._generate_key(f"prediction:{model_type}", features) 
                             for features in feature_list]
                
                with self._lock:
                    # Use pipeline for efficient batch operations
                    pipe = self.redis_client.pipeline()
                    for key in cache_keys:
                        pipe.get(key)
                    
                    results = pipe.execute()
                    
                    predictions = []
                    for result in results:
                        if result is not None:
                            self.stats['hits'] += 1
                            predictions.append(self._deserialize_data(result))
                        else:
                            self.stats['misses'] += 1
                            predictions.append(None)
                    
                    return predictions
        except Exception as e:
            logger.error(f"Batch cache get failed: {e}")
            return [None] * len(feature_list)
    
    def set_batch_predictions(self, model_type: str, 
                            feature_prediction_pairs: List[tuple],
                            ttl: Optional[int] = None) -> int:
        """
        Set multiple predictions in batch
        
        Args:
            model_type: Type of model
            feature_prediction_pairs: List of (features, prediction) tuples
            ttl: Time-to-live for predictions
            
        Returns:
            Number of successful cache sets
        """
        if not self.redis_client or not feature_prediction_pairs:
            return 0
        
        try:
            with self._retry_operation():
                ttl = ttl or self.default_ttl
                
                with self._lock:
                    # Use pipeline for efficient batch operations
                    pipe = self.redis_client.pipeline()
                    
                    for features, prediction in feature_prediction_pairs:
                        cache_key = self._generate_key(f"prediction:{model_type}", features)
                        prediction_with_meta = {
                            'prediction': prediction,
                            'timestamp': datetime.utcnow().isoformat(),
                            'model_type': model_type
                        }
                        
                        serialized_data = self._serialize_data(prediction_with_meta)
                        pipe.setex(cache_key, ttl, serialized_data)
                    
                    results = pipe.execute()
                    successful_sets = sum(1 for result in results if result)
                    self.stats['sets'] += successful_sets
                    
                    return successful_sets
        except Exception as e:
            logger.error(f"Batch cache set failed: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            **self.stats,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests
        }
        
        # Add Redis server info if available
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'N/A'),
                    'redis_connected_clients': info.get('connected_clients', 0),
                    'redis_keyspace_hits': info.get('keyspace_hits', 0),
                    'redis_keyspace_misses': info.get('keyspace_misses', 0)
                })
            except Exception as e:
                logger.warning(f"Could not get Redis info: {e}")
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Check Redis connection health"""
        if not self.redis_client:
            return {
                'status': 'error',
                'message': 'Redis client not initialized',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        try:
            start_time = time.time()
            self.redis_client.ping()
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'status': 'healthy',
                'latency_ms': round(latency, 2),
                'timestamp': datetime.utcnow().isoformat(),
                'stats': self.get_cache_stats()
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def clear_expired_keys(self, pattern: str = "prediction:*") -> int:
        """
        Clear expired keys matching pattern
        
        Args:
            pattern: Key pattern to match
            
        Returns:
            Number of keys cleared
        """
        if not self.redis_client:
            return 0
        
        try:
            with self._retry_operation():
                keys = self.redis_client.keys(pattern)
                if not keys:
                    return 0
                
                # Check which keys are expired and remove them
                pipe = self.redis_client.pipeline()
                for key in keys:
                    pipe.ttl(key)
                
                ttls = pipe.execute()
                expired_keys = [key for key, ttl in zip(keys, ttls) if ttl == -2]
                
                if expired_keys:
                    deleted = self.redis_client.delete(*expired_keys)
                    logger.info(f"Cleared {deleted} expired cache keys")
                    return deleted
                
                return 0
        except Exception as e:
            logger.error(f"Failed to clear expired keys: {e}")
            return 0
    
    def close(self):
        """Close Redis connection"""
        if self.connection_pool:
            self.connection_pool.disconnect()
            logger.info("Redis connection closed")


class ModelCacheManager:
    """
    Specialized cache manager for ML model predictions
    Provides model-specific caching strategies and invalidation
    """
    
    def __init__(self, redis_cache: RedisCache):
        self.cache = redis_cache
        
        # Model-specific TTL configurations
        self.model_ttls = {
            'xgboost': 300,      # 5 minutes - load predictions change frequently
            'qlearning': 180,    # 3 minutes - placement decisions need frequent updates
            'anomaly': 60        # 1 minute - anomaly detection needs real-time responsiveness
        }
    
    def get_load_prediction(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached XGBoost load prediction"""
        return self.cache.get_prediction('xgboost', features)
    
    def set_load_prediction(self, features: Dict[str, Any], prediction: Dict[str, Any]) -> bool:
        """Cache XGBoost load prediction"""
        return self.cache.set_prediction(
            'xgboost', features, prediction, 
            ttl=self.model_ttls['xgboost']
        )
    
    def get_placement_optimization(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached Q-Learning placement optimization"""
        return self.cache.get_prediction('qlearning', request)
    
    def set_placement_optimization(self, request: Dict[str, Any], 
                                 optimization: Dict[str, Any]) -> bool:
        """Cache Q-Learning placement optimization"""
        return self.cache.set_prediction(
            'qlearning', request, optimization,
            ttl=self.model_ttls['qlearning']
        )
    
    def get_anomaly_detection(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached anomaly detection result"""
        return self.cache.get_prediction('anomaly', metrics)
    
    def set_anomaly_detection(self, metrics: Dict[str, Any], 
                            detection: Dict[str, Any]) -> bool:
        """Cache anomaly detection result"""
        return self.cache.set_prediction(
            'anomaly', metrics, detection,
            ttl=self.model_ttls['anomaly']
        )
    
    def invalidate_all_models(self) -> Dict[str, int]:
        """Invalidate cache for all models"""
        results = {}
        for model_type in self.model_ttls.keys():
            results[model_type] = self.cache.invalidate_model_cache(model_type)
        return results
    
    def get_model_cache_stats(self, model_type: str) -> Dict[str, Any]:
        """Get cache statistics for specific model"""
        if not self.cache.redis_client:
            return {'error': 'Redis not available'}
        
        try:
            pattern = f"prediction:{model_type}:*"
            keys = self.cache.redis_client.keys(pattern)
            
            # Sample a few keys to get average TTL
            sample_size = min(10, len(keys))
            if sample_size > 0:
                sample_keys = keys[:sample_size]
                pipe = self.cache.redis_client.pipeline()
                for key in sample_keys:
                    pipe.ttl(key)
                
                ttls = pipe.execute()
                avg_ttl = sum(ttl for ttl in ttls if ttl > 0) / len([ttl for ttl in ttls if ttl > 0]) if any(ttl > 0 for ttl in ttls) else 0
            else:
                avg_ttl = 0
            
            return {
                'model_type': model_type,
                'cached_keys': len(keys),
                'average_ttl_seconds': round(avg_ttl, 2),
                'configured_ttl': self.model_ttls.get(model_type, 'unknown')
            }
        except Exception as e:
            return {'error': str(e)}


# Global cache instance
_cache_instance = None
_cache_manager_instance = None

def get_cache() -> RedisCache:
    """Get global Redis cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance

def get_cache_manager() -> ModelCacheManager:
    """Get global model cache manager instance"""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = ModelCacheManager(get_cache())
    return _cache_manager_instance

def init_cache(host: str = 'localhost', port: int = 6379, **kwargs) -> RedisCache:
    """Initialize global cache with custom configuration"""
    global _cache_instance, _cache_manager_instance
    _cache_instance = RedisCache(host=host, port=port, **kwargs)
    _cache_manager_instance = ModelCacheManager(_cache_instance)
    return _cache_instance