package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"k8s.io/klog/v2"
)

// MetricsCollector collects and exposes metrics for the ML scheduler plugin
type MetricsCollector struct {
	// Scheduling metrics
	SchedulingRequests    prometheus.Counter
	SchedulingDuration    prometheus.Histogram
	SchedulingErrors      prometheus.Counter
	SchedulingSuccess     prometheus.Counter
	
	// ML scoring metrics
	MLScoringRequests     prometheus.Counter
	MLScoringDuration     prometheus.Histogram
	MLScoringErrors       prometheus.Counter
	MLConfidenceScores    prometheus.Histogram
	
	// Cache metrics
	CacheHits             prometheus.Counter
	CacheMisses           prometheus.Counter
	CacheErrors           prometheus.Counter
	CacheLatency          prometheus.Histogram
	
	// Fallback metrics
	FallbackTriggered     prometheus.CounterVec
	FallbackLatency       prometheus.Histogram
	
	// Node scoring metrics
	NodeScoreDistribution prometheus.Histogram
	NodesEvaluated        prometheus.Counter
	NodesFiltered         prometheus.CounterVec
	
	// Business metrics
	ClusterUtilization    prometheus.GaugeVec
	SchedulingEfficiency  prometheus.Gauge
	PodPlacementLatency   prometheus.Histogram
	
	// Performance tracking
	mu                    sync.RWMutex
	recentSchedulingTimes []time.Duration
	recentMLScores        []float64
	windowSize            int
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	mc := &MetricsCollector{
		windowSize: 1000, // Keep track of last 1000 operations
	}
	
	mc.initMetrics()
	
	klog.InfoS("Metrics collector initialized")
	return mc
}

func (mc *MetricsCollector) initMetrics() {
	// Scheduling metrics
	mc.SchedulingRequests = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_scheduling_requests_total",
		Help: "Total number of scheduling requests processed",
	})
	
	mc.SchedulingDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "ml_scheduler_scheduling_duration_seconds",
		Help: "Duration of scheduling operations",
		Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5},
	})
	
	mc.SchedulingErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_scheduling_errors_total",
		Help: "Total number of scheduling errors",
	})
	
	mc.SchedulingSuccess = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_scheduling_success_total",
		Help: "Total number of successful scheduling operations",
	})
	
	// ML scoring metrics
	mc.MLScoringRequests = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_ml_requests_total",
		Help: "Total number of ML scoring requests",
	})
	
	mc.MLScoringDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "ml_scheduler_ml_duration_seconds",
		Help: "Duration of ML scoring operations",
		Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1},
	})
	
	mc.MLScoringErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_ml_errors_total",
		Help: "Total number of ML scoring errors",
	})
	
	mc.MLConfidenceScores = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "ml_scheduler_ml_confidence_scores",
		Help: "Distribution of ML confidence scores",
		Buckets: []float64{0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0},
	})
	
	// Cache metrics
	mc.CacheHits = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_cache_hits_total",
		Help: "Total number of cache hits",
	})
	
	mc.CacheMisses = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_cache_misses_total",
		Help: "Total number of cache misses",
	})
	
	mc.CacheErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_cache_errors_total",
		Help: "Total number of cache errors",
	})
	
	mc.CacheLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "ml_scheduler_cache_duration_seconds",
		Help: "Duration of cache operations",
		Buckets: []float64{.0001, .0005, .001, .005, .01, .025, .05},
	})
	
	// Fallback metrics
	mc.FallbackTriggered = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "ml_scheduler_fallback_triggered_total",
		Help: "Total number of times fallback was triggered",
	}, []string{"reason"})
	
	mc.FallbackLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "ml_scheduler_fallback_duration_seconds",
		Help: "Duration of fallback scheduling operations",
		Buckets: []float64{.001, .005, .01, .025, .05, .1},
	})
	
	// Node scoring metrics
	mc.NodeScoreDistribution = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "ml_scheduler_node_scores",
		Help: "Distribution of node scores",
		Buckets: []float64{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
	})
	
	mc.NodesEvaluated = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_nodes_evaluated_total",
		Help: "Total number of nodes evaluated for scheduling",
	})
	
	mc.NodesFiltered = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "ml_scheduler_nodes_filtered_total",
		Help: "Total number of nodes filtered out",
	}, []string{"reason"})
	
	// Business metrics
	mc.ClusterUtilization = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "ml_scheduler_cluster_utilization_percent",
		Help: "Current cluster resource utilization",
	}, []string{"resource"})
	
	mc.SchedulingEfficiency = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "ml_scheduler_efficiency_percent",
		Help: "Overall scheduling efficiency percentage",
	})
	
	mc.PodPlacementLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "ml_scheduler_pod_placement_duration_seconds",
		Help: "End-to-end pod placement latency",
		Buckets: []float64{.1, .25, .5, 1, 2.5, 5, 10, 25, 60},
	})
}

// RecordSchedulingRequest records a scheduling request
func (mc *MetricsCollector) RecordSchedulingRequest(duration time.Duration, success bool) {
	mc.SchedulingRequests.Inc()
	mc.SchedulingDuration.Observe(duration.Seconds())
	
	if success {
		mc.SchedulingSuccess.Inc()
	} else {
		mc.SchedulingErrors.Inc()
	}
	
	// Track recent scheduling times for efficiency calculation
	mc.mu.Lock()
	if len(mc.recentSchedulingTimes) >= mc.windowSize {
		mc.recentSchedulingTimes = mc.recentSchedulingTimes[1:]
	}
	mc.recentSchedulingTimes = append(mc.recentSchedulingTimes, duration)
	mc.mu.Unlock()
	
	// Update efficiency metric
	mc.updateSchedulingEfficiency()
}

// RecordMLScoring records ML scoring operation
func (mc *MetricsCollector) RecordMLScoring(duration time.Duration, confidence float64, success bool) {
	mc.MLScoringRequests.Inc()
	mc.MLScoringDuration.Observe(duration.Seconds())
	mc.MLConfidenceScores.Observe(confidence)
	
	if !success {
		mc.MLScoringErrors.Inc()
	}
	
	// Track recent ML scores
	mc.mu.Lock()
	if len(mc.recentMLScores) >= mc.windowSize {
		mc.recentMLScores = mc.recentMLScores[1:]
	}
	mc.recentMLScores = append(mc.recentMLScores, confidence)
	mc.mu.Unlock()
}

// RecordCacheOperation records cache hit/miss/error
func (mc *MetricsCollector) RecordCacheOperation(duration time.Duration, hit bool, err error) {
	mc.CacheLatency.Observe(duration.Seconds())
	
	if err != nil {
		mc.CacheErrors.Inc()
	} else if hit {
		mc.CacheHits.Inc()
	} else {
		mc.CacheMisses.Inc()
	}
}

// RecordFallback records fallback scheduler usage
func (mc *MetricsCollector) RecordFallback(reason FallbackReason, duration time.Duration) {
	mc.FallbackTriggered.WithLabelValues(string(reason)).Inc()
	mc.FallbackLatency.Observe(duration.Seconds())
}

// RecordNodeEvaluation records node evaluation metrics
func (mc *MetricsCollector) RecordNodeEvaluation(nodeName string, score int64, filtered bool, filterReason string) {
	mc.NodesEvaluated.Inc()
	
	if !filtered {
		mc.NodeScoreDistribution.Observe(float64(score))
	} else {
		mc.NodesFiltered.WithLabelValues(filterReason).Inc()
	}
}

// UpdateClusterUtilization updates cluster utilization metrics
func (mc *MetricsCollector) UpdateClusterUtilization(cpuPercent, memoryPercent, diskPercent float64) {
	mc.ClusterUtilization.WithLabelValues("cpu").Set(cpuPercent)
	mc.ClusterUtilization.WithLabelValues("memory").Set(memoryPercent)
	mc.ClusterUtilization.WithLabelValues("disk").Set(diskPercent)
}

// updateSchedulingEfficiency calculates and updates scheduling efficiency
func (mc *MetricsCollector) updateSchedulingEfficiency() {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	
	if len(mc.recentSchedulingTimes) == 0 {
		return
	}
	
	// Calculate efficiency based on recent performance
	var totalDuration time.Duration
	fastOperations := 0
	
	for _, duration := range mc.recentSchedulingTimes {
		totalDuration += duration
		if duration < 100*time.Millisecond { // Target: <100ms scheduling
			fastOperations++
		}
	}
	
	avgDuration := totalDuration / time.Duration(len(mc.recentSchedulingTimes))
	fastOperationPercent := float64(fastOperations) / float64(len(mc.recentSchedulingTimes)) * 100
	
	// Efficiency based on speed and ML confidence
	avgConfidence := mc.calculateAverageConfidence()
	
	// Combined efficiency metric
	efficiency := (fastOperationPercent + avgConfidence*100) / 2
	mc.SchedulingEfficiency.Set(efficiency)
	
	klog.V(6).InfoS("Scheduling efficiency updated", 
		"efficiency", efficiency,
		"avgDuration", avgDuration,
		"fastOperationPercent", fastOperationPercent,
		"avgConfidence", avgConfidence)
}

func (mc *MetricsCollector) calculateAverageConfidence() float64 {
	if len(mc.recentMLScores) == 0 {
		return 0.0
	}
	
	total := 0.0
	for _, score := range mc.recentMLScores {
		total += score
	}
	
	return total / float64(len(mc.recentMLScores))
}

// GetMetricsSnapshot returns a snapshot of current metrics
func (mc *MetricsCollector) GetMetricsSnapshot() map[string]interface{} {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	
	snapshot := map[string]interface{}{
		"timestamp": time.Now().Unix(),
		"window_size": mc.windowSize,
		"recent_operations": len(mc.recentSchedulingTimes),
	}
	
	if len(mc.recentSchedulingTimes) > 0 {
		var totalDuration time.Duration
		for _, d := range mc.recentSchedulingTimes {
			totalDuration += d
		}
		avgDuration := totalDuration / time.Duration(len(mc.recentSchedulingTimes))
		snapshot["average_scheduling_duration_ms"] = avgDuration.Milliseconds()
	}
	
	if len(mc.recentMLScores) > 0 {
		snapshot["average_ml_confidence"] = mc.calculateAverageConfidence()
	}
	
	return snapshot
}

// FallbackReason represents why fallback scheduling was triggered
type FallbackReason string

const (
	MLServiceUnavailable FallbackReason = "ml_service_unavailable"
	MLServiceTimeout     FallbackReason = "ml_service_timeout"
	MLServiceError       FallbackReason = "ml_service_error"
	LowConfidence        FallbackReason = "low_confidence"
	CacheError           FallbackReason = "cache_error"
	ConfigurationError   FallbackReason = "configuration_error"
	NetworkError         FallbackReason = "network_error"
)

// Global metrics instance
var (
	globalMetrics *MetricsCollector
	metricsOnce   sync.Once
)

// GetMetrics returns the global metrics collector instance
func GetMetrics() *MetricsCollector {
	metricsOnce.Do(func() {
		globalMetrics = NewMetricsCollector()
	})
	return globalMetrics
}