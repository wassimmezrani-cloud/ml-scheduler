package caching

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// CacheManager handles Redis caching for the scheduler plugin
type CacheManager struct {
	client          *redis.Client
	defaultTTL      time.Duration
	
	// Metrics
	cacheHits       prometheus.Counter
	cacheMisses     prometheus.Counter
	cacheErrors     prometheus.Counter
	cacheOperations prometheus.Counter
	
	// Internal state
	mu              sync.RWMutex
	connectionHealthy bool
	lastHealthCheck  time.Time
}

// CachedScore represents a cached scoring result
type CachedScore struct {
	Score      int64     `json:"score"`
	Confidence float64   `json:"confidence"`
	NodeName   string    `json:"node_name"`
	PodName    string    `json:"pod_name"`
	Timestamp  time.Time `json:"timestamp"`
	TTL        int64     `json:"ttl"`
}

// NewCacheManager creates a new cache manager
func NewCacheManager(redisURL string) (*CacheManager, error) {
	// Parse Redis URL
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Redis URL: %w", err)
	}
	
	// Configure connection options
	opts.PoolSize = 10
	opts.MinIdleConns = 2
	opts.MaxRetries = 3
	opts.RetryDelay = 100 * time.Millisecond
	opts.DialTimeout = 5 * time.Second
	opts.ReadTimeout = 3 * time.Second
	opts.WriteTimeout = 3 * time.Second
	opts.PoolTimeout = 4 * time.Second
	opts.IdleTimeout = 5 * time.Minute
	
	client := redis.NewClient(opts)
	
	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	_, err = client.Ping(ctx).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}
	
	cm := &CacheManager{
		client:            client,
		defaultTTL:        5 * time.Minute,
		connectionHealthy: true,
		lastHealthCheck:   time.Now(),
	}
	
	// Initialize metrics
	cm.initMetrics()
	
	// Start health check routine
	go cm.healthCheckRoutine()
	
	klog.InfoS("Cache manager initialized", "redisURL", redisURL)
	return cm, nil
}

func (cm *CacheManager) initMetrics() {
	cm.cacheHits = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_cache_hits_total",
		Help: "Total number of cache hits",
	})
	
	cm.cacheMisses = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_cache_misses_total",
		Help: "Total number of cache misses",
	})
	
	cm.cacheErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_cache_errors_total",
		Help: "Total number of cache errors",
	})
	
	cm.cacheOperations = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_cache_operations_total",
		Help: "Total number of cache operations",
	})
}

// generateCacheKey generates a consistent cache key for pod/node combination
func (cm *CacheManager) generateCacheKey(pod *v1.Pod, nodeName string) string {
	// Create a hash from pod spec and node name for consistent caching
	podSpec := map[string]interface{}{
		"name":      pod.Name,
		"namespace": pod.Namespace,
		"requests":  pod.Spec.Containers[0].Resources.Requests,
		"limits":    pod.Spec.Containers[0].Resources.Limits,
		"nodeSelector": pod.Spec.NodeSelector,
		"tolerations":  pod.Spec.Tolerations,
		"affinity":     pod.Spec.Affinity,
		"priorityClassName": pod.Spec.PriorityClassName,
	}
	
	cacheData := map[string]interface{}{
		"pod_spec": podSpec,
		"node":     nodeName,
	}
	
	jsonData, _ := json.Marshal(cacheData)
	hash := md5.Sum(jsonData)
	return fmt.Sprintf("scheduler:score:%s", hex.EncodeToString(hash[:]))
}

// GetScore retrieves a cached score for pod/node combination
func (cm *CacheManager) GetScore(pod *v1.Pod, nodeName string) (int64, bool) {
	cm.cacheOperations.Inc()
	
	// Check if connection is healthy
	cm.mu.RLock()
	isHealthy := cm.connectionHealthy
	cm.mu.RUnlock()
	
	if !isHealthy {
		cm.cacheMisses.Inc()
		return 0, false
	}
	
	cacheKey := cm.generateCacheKey(pod, nodeName)
	
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	
	result, err := cm.client.Get(ctx, cacheKey).Result()
	if err != nil {
		if err == redis.Nil {
			cm.cacheMisses.Inc()
			klog.V(6).InfoS("Cache miss", "key", cacheKey)
		} else {
			cm.cacheErrors.Inc()
			klog.ErrorS(err, "Cache get error", "key", cacheKey)
		}
		return 0, false
	}
	
	var cachedScore CachedScore
	if err := json.Unmarshal([]byte(result), &cachedScore); err != nil {
		cm.cacheErrors.Inc()
		klog.ErrorS(err, "Failed to unmarshal cached score", "key", cacheKey)
		return 0, false
	}
	
	cm.cacheHits.Inc()
	klog.V(6).InfoS("Cache hit", 
		"key", cacheKey,
		"score", cachedScore.Score,
		"age", time.Since(cachedScore.Timestamp))
	
	return cachedScore.Score, true
}

// SetScore caches a score for pod/node combination
func (cm *CacheManager) SetScore(pod *v1.Pod, nodeName string, score int64, ttl time.Duration) {
	cm.cacheOperations.Inc()
	
	// Check if connection is healthy
	cm.mu.RLock()
	isHealthy := cm.connectionHealthy
	cm.mu.RUnlock()
	
	if !isHealthy {
		return
	}
	
	cacheKey := cm.generateCacheKey(pod, nodeName)
	
	cachedScore := CachedScore{
		Score:      score,
		Confidence: 1.0, // Default confidence
		NodeName:   nodeName,
		PodName:    fmt.Sprintf("%s/%s", pod.Namespace, pod.Name),
		Timestamp:  time.Now(),
		TTL:        int64(ttl.Seconds()),
	}
	
	jsonData, err := json.Marshal(cachedScore)
	if err != nil {
		cm.cacheErrors.Inc()
		klog.ErrorS(err, "Failed to marshal cache data", "key", cacheKey)
		return
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	
	if ttl == 0 {
		ttl = cm.defaultTTL
	}
	
	err = cm.client.Set(ctx, cacheKey, jsonData, ttl).Err()
	if err != nil {
		cm.cacheErrors.Inc()
		klog.ErrorS(err, "Cache set error", "key", cacheKey)
		return
	}
	
	klog.V(6).InfoS("Score cached", 
		"key", cacheKey,
		"score", score,
		"ttl", ttl)
}

// InvalidateNode invalidates all cached scores for a specific node
func (cm *CacheManager) InvalidateNode(nodeName string) error {
	cm.mu.RLock()
	isHealthy := cm.connectionHealthy
	cm.mu.RUnlock()
	
	if !isHealthy {
		return fmt.Errorf("cache not healthy")
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	pattern := fmt.Sprintf("scheduler:score:*")
	
	// Scan for keys and delete those matching the node
	iter := cm.client.Scan(ctx, 0, pattern, 100).Iterator()
	var keysToDelete []string
	
	for iter.Next(ctx) {
		key := iter.Val()
		
		// Get the cached score to check if it's for this node
		result, err := cm.client.Get(ctx, key).Result()
		if err != nil {
			continue
		}
		
		var cachedScore CachedScore
		if err := json.Unmarshal([]byte(result), &cachedScore); err != nil {
			continue
		}
		
		if cachedScore.NodeName == nodeName {
			keysToDelete = append(keysToDelete, key)
		}
	}
	
	if err := iter.Err(); err != nil {
		return fmt.Errorf("scan iteration error: %w", err)
	}
	
	// Delete matching keys
	if len(keysToDelete) > 0 {
		deleted, err := cm.client.Del(ctx, keysToDelete...).Result()
		if err != nil {
			return fmt.Errorf("failed to delete keys: %w", err)
		}
		
		klog.InfoS("Invalidated node cache", 
			"node", nodeName,
			"keysDeleted", deleted)
	}
	
	return nil
}

// InvalidatePod invalidates all cached scores for a specific pod
func (cm *CacheManager) InvalidatePod(pod *v1.Pod) error {
	cm.mu.RLock()
	isHealthy := cm.connectionHealthy
	cm.mu.RUnlock()
	
	if !isHealthy {
		return fmt.Errorf("cache not healthy")
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	pattern := fmt.Sprintf("scheduler:score:*")
	podFullName := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)
	
	// Scan for keys and delete those matching the pod
	iter := cm.client.Scan(ctx, 0, pattern, 100).Iterator()
	var keysToDelete []string
	
	for iter.Next(ctx) {
		key := iter.Val()
		
		// Get the cached score to check if it's for this pod
		result, err := cm.client.Get(ctx, key).Result()
		if err != nil {
			continue
		}
		
		var cachedScore CachedScore
		if err := json.Unmarshal([]byte(result), &cachedScore); err != nil {
			continue
		}
		
		if cachedScore.PodName == podFullName {
			keysToDelete = append(keysToDelete, key)
		}
	}
	
	if err := iter.Err(); err != nil {
		return fmt.Errorf("scan iteration error: %w", err)
	}
	
	// Delete matching keys
	if len(keysToDelete) > 0 {
		deleted, err := cm.client.Del(ctx, keysToDelete...).Result()
		if err != nil {
			return fmt.Errorf("failed to delete keys: %w", err)
		}
		
		klog.InfoS("Invalidated pod cache", 
			"pod", klog.KObj(pod),
			"keysDeleted", deleted)
	}
	
	return nil
}

// GetCacheStats returns cache performance statistics
func (cm *CacheManager) GetCacheStats() map[string]interface{} {
	cm.mu.RLock()
	isHealthy := cm.connectionHealthy
	lastCheck := cm.lastHealthCheck
	cm.mu.RUnlock()
	
	stats := map[string]interface{}{
		"healthy":           isHealthy,
		"last_health_check": lastCheck,
	}
	
	if !isHealthy {
		return stats
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	
	// Get Redis info
	info, err := cm.client.Info(ctx, "memory", "stats", "keyspace").Result()
	if err != nil {
		stats["error"] = err.Error()
		return stats
	}
	
	// Parse Redis info (simplified)
	stats["redis_info"] = "available"
	
	// Get cache key count
	keys, err := cm.client.Keys(ctx, "scheduler:score:*").Result()
	if err == nil {
		stats["cached_scores"] = len(keys)
	}
	
	return stats
}

func (cm *CacheManager) healthCheckRoutine() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_, err := cm.client.Ping(ctx).Result()
			cancel()
			
			cm.mu.Lock()
			cm.connectionHealthy = (err == nil)
			cm.lastHealthCheck = time.Now()
			cm.mu.Unlock()
			
			if err != nil {
				klog.ErrorS(err, "Redis health check failed")
			} else {
				klog.V(6).InfoS("Redis health check passed")
			}
		}
	}
}

// Close closes the Redis connection
func (cm *CacheManager) Close() error {
	return cm.client.Close()
}