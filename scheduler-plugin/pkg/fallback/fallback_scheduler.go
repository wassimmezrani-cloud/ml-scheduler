package fallback

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// FallbackScheduler provides fallback scheduling when ML models fail or are unavailable
type FallbackScheduler struct {
	handle         framework.Handle
	enabled        bool
	nodeResources  framework.ScorePlugin
	
	// Metrics
	fallbackUsed   prometheus.Counter
	fallbackErrors prometheus.Counter
	fallbackLatency prometheus.Histogram
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

// NewFallbackScheduler creates a new fallback scheduler
func NewFallbackScheduler(handle framework.Handle, enabled bool) *FallbackScheduler {
	fs := &FallbackScheduler{
		handle:  handle,
		enabled: enabled,
	}
	
	// Initialize the default node resources plugin for fallback scoring
	nodeResourcesPlugin, err := noderesources.NewFit(nil, handle)
	if err != nil {
		klog.ErrorS(err, "Failed to initialize node resources plugin")
	} else {
		if scorePlugin, ok := nodeResourcesPlugin.(framework.ScorePlugin); ok {
			fs.nodeResources = scorePlugin
		}
	}
	
	// Initialize metrics
	fs.initMetrics()
	
	klog.InfoS("Fallback scheduler initialized", "enabled", enabled)
	return fs
}

func (fs *FallbackScheduler) initMetrics() {
	fs.fallbackUsed = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "ml_scheduler_fallback_used_total",
		Help: "Total number of times fallback scheduler was used",
	}, []string{"reason"}).WithLabelValues("total")
	
	fs.fallbackErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_fallback_errors_total",
		Help: "Total number of fallback scheduler errors",
	})
	
	fs.fallbackLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "ml_scheduler_fallback_duration_seconds",
		Help: "Duration of fallback scheduling operations",
		Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1},
	})
}

// ScoreNode scores a node using fallback algorithms
func (fs *FallbackScheduler) ScoreNode(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	
	startTime := time.Now()
	defer func() {
		fs.fallbackLatency.Observe(time.Since(startTime).Seconds())
		fs.fallbackUsed.Inc()
	}()
	
	if !fs.enabled {
		klog.V(3).InfoS("Fallback scheduler disabled", 
			"pod", klog.KObj(pod), "node", nodeName)
		return 0, framework.NewStatus(framework.Error, "fallback scheduler disabled")
	}
	
	klog.V(4).InfoS("Using fallback scheduler", 
		"pod", klog.KObj(pod), "node", nodeName)
	
	// Try node resources plugin first
	if fs.nodeResources != nil {
		score, status := fs.nodeResources.Score(ctx, state, pod, nodeName)
		if status.IsSuccess() {
			klog.V(5).InfoS("Fallback score from node resources", 
				"pod", klog.KObj(pod),
				"node", nodeName,
				"score", score)
			return score, status
		}
	}
	
	// If node resources plugin fails, use simple resource-based scoring
	return fs.simpleResourceScore(pod, nodeName)
}

// simpleResourceScore provides basic resource-based scoring
func (fs *FallbackScheduler) simpleResourceScore(pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo := fs.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if nodeInfo == nil {
		fs.fallbackErrors.Inc()
		return 0, framework.NewStatus(framework.Error, 
			fmt.Sprintf("node %s not found", nodeName))
	}
	
	node := nodeInfo.Node()
	
	// Basic scoring algorithm based on available resources
	allocatable := node.Status.Allocatable
	allocated := nodeInfo.Requested
	
	// Get resource capacity
	cpuCapacity := allocatable[v1.ResourceCPU].MilliValue()
	memoryCapacity := allocatable[v1.ResourceMemory].Value()
	
	// Get current usage
	cpuUsed := allocated.MilliCPU
	memoryUsed := allocated.Memory
	
	// Calculate available resources as percentage
	cpuAvailable := float64(cpuCapacity-cpuUsed) / float64(cpuCapacity) * 100
	memoryAvailable := float64(memoryCapacity-memoryUsed) / float64(memoryCapacity) * 100
	
	// Ensure values are within bounds
	if cpuAvailable < 0 {
		cpuAvailable = 0
	}
	if memoryAvailable < 0 {
		memoryAvailable = 0
	}
	
	// Score based on least utilized resources (prefer balanced utilization)
	// Optimal utilization target: 60-70%
	cpuScore := calculateUtilizationScore(100 - cpuAvailable)
	memoryScore := calculateUtilizationScore(100 - memoryAvailable)
	
	// Weighted average (CPU 60%, Memory 40%)
	finalScore := int64(cpuScore*0.6 + memoryScore*0.4)
	
	// Apply node condition penalties
	if !isNodeReady(node) {
		finalScore = finalScore / 2
	}
	
	if hasResourcePressure(node) {
		finalScore = finalScore * 3 / 4
	}
	
	klog.V(5).InfoS("Simple resource score calculated", 
		"node", nodeName,
		"cpuAvailable", cpuAvailable,
		"memoryAvailable", memoryAvailable,
		"score", finalScore)
	
	return finalScore, nil
}

// calculateUtilizationScore calculates score based on resource utilization
// Prefers nodes with 60-70% utilization for optimal cluster efficiency
func calculateUtilizationScore(utilizationPercent float64) float64 {
	if utilizationPercent < 0 {
		return 0
	}
	if utilizationPercent > 95 {
		return 10 // Very low score for overutilized nodes
	}
	
	// Optimal range: 60-70%
	if utilizationPercent >= 60 && utilizationPercent <= 70 {
		return 100
	}
	
	// Gradual decrease outside optimal range
	if utilizationPercent < 60 {
		// Prefer some utilization over idle nodes
		return 50 + (utilizationPercent/60)*50
	}
	
	// Above 70% - decreasing score
	if utilizationPercent <= 85 {
		return 100 - (utilizationPercent-70)*3 // 3 points per percent over 70%
	}
	
	// 85-95% - rapid score decrease
	return 55 - (utilizationPercent-85)*4.5 // 4.5 points per percent over 85%
}

// isNodeReady checks if node is in ready state
func isNodeReady(node *v1.Node) bool {
	for _, condition := range node.Status.Conditions {
		if condition.Type == v1.NodeReady {
			return condition.Status == v1.ConditionTrue
		}
	}
	return false
}

// hasResourcePressure checks if node has resource pressure conditions
func hasResourcePressure(node *v1.Node) bool {
	for _, condition := range node.Status.Conditions {
		if condition.Type == v1.NodeMemoryPressure ||
		   condition.Type == v1.NodeDiskPressure ||
		   condition.Type == v1.NodePIDPressure {
			if condition.Status == v1.ConditionTrue {
				return true
			}
		}
	}
	return false
}

// ScoreWithReason scores a node and provides the fallback reason
func (fs *FallbackScheduler) ScoreWithReason(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, nodeName string, reason FallbackReason) (int64, *framework.Status) {
	
	// Record the specific fallback reason
	fallbackReasonCounter := promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "ml_scheduler_fallback_reason_total",
		Help: "Total number of fallback uses by reason",
	}, []string{"reason"})
	
	fallbackReasonCounter.WithLabelValues(string(reason)).Inc()
	
	klog.V(3).InfoS("Fallback scheduler triggered", 
		"pod", klog.KObj(pod),
		"node", nodeName,
		"reason", reason)
	
	return fs.ScoreNode(ctx, state, pod, nodeName)
}

// IsEnabled returns whether fallback scheduling is enabled
func (fs *FallbackScheduler) IsEnabled() bool {
	return fs.enabled
}

// GetFallbackStats returns fallback scheduler statistics
func (fs *FallbackScheduler) GetFallbackStats() map[string]interface{} {
	return map[string]interface{}{
		"enabled":              fs.enabled,
		"node_resources_available": fs.nodeResources != nil,
		"timestamp":            time.Now(),
	}
}