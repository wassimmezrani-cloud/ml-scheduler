package logging

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

// SchedulingEvent represents a scheduling event for structured logging
type SchedulingEvent struct {
	EventType    string                 `json:"event_type"`
	PodName      string                 `json:"pod_name"`
	PodNamespace string                 `json:"pod_namespace"`
	NodeName     string                 `json:"node_name"`
	RequestID    string                 `json:"request_id"`
	Timestamp    time.Time              `json:"timestamp"`
	Duration     time.Duration          `json:"duration"`
	Success      bool                   `json:"success"`
	Score        int64                  `json:"score,omitempty"`
	Confidence   float64                `json:"confidence,omitempty"`
	Reason       string                 `json:"reason,omitempty"`
	MLResponse   map[string]interface{} `json:"ml_response,omitempty"`
	CacheHit     bool                   `json:"cache_hit,omitempty"`
	FallbackUsed bool                   `json:"fallback_used,omitempty"`
}

// MLSchedulerLogger provides structured logging for the ML scheduler
type MLSchedulerLogger struct {
	component string
}

// NewMLSchedulerLogger creates a new logger instance
func NewMLSchedulerLogger(component string) *MLSchedulerLogger {
	return &MLSchedulerLogger{
		component: component,
	}
}

// LogSchedulingAttempt logs a scheduling attempt
func (l *MLSchedulerLogger) LogSchedulingAttempt(ctx context.Context, pod *v1.Pod, 
	nodeName, requestID string) {
	
	event := SchedulingEvent{
		EventType:    "scheduling_attempt",
		PodName:      pod.Name,
		PodNamespace: pod.Namespace,
		NodeName:     nodeName,
		RequestID:    requestID,
		Timestamp:    time.Now(),
	}
	
	l.logEvent(event)
}

// LogSchedulingSuccess logs successful scheduling
func (l *MLSchedulerLogger) LogSchedulingSuccess(ctx context.Context, pod *v1.Pod, 
	nodeName, requestID string, score int64, confidence float64, duration time.Duration,
	cacheHit bool, mlResponse map[string]interface{}) {
	
	event := SchedulingEvent{
		EventType:    "scheduling_success",
		PodName:      pod.Name,
		PodNamespace: pod.Namespace,
		NodeName:     nodeName,
		RequestID:    requestID,
		Timestamp:    time.Now(),
		Duration:     duration,
		Success:      true,
		Score:        score,
		Confidence:   confidence,
		MLResponse:   mlResponse,
		CacheHit:     cacheHit,
	}
	
	l.logEvent(event)
}

// LogSchedulingFailure logs scheduling failure
func (l *MLSchedulerLogger) LogSchedulingFailure(ctx context.Context, pod *v1.Pod, 
	nodeName, requestID, reason string, duration time.Duration, fallbackUsed bool) {
	
	event := SchedulingEvent{
		EventType:    "scheduling_failure",
		PodName:      pod.Name,
		PodNamespace: pod.Namespace,
		NodeName:     nodeName,
		RequestID:    requestID,
		Timestamp:    time.Now(),
		Duration:     duration,
		Success:      false,
		Reason:       reason,
		FallbackUsed: fallbackUsed,
	}
	
	l.logEvent(event)
}

// LogMLScoringRequest logs ML scoring request
func (l *MLSchedulerLogger) LogMLScoringRequest(ctx context.Context, requestID string, 
	nodeCount int, podSpec map[string]interface{}) {
	
	klog.V(4).InfoS("ML scoring request", 
		"component", l.component,
		"requestID", requestID,
		"nodeCount", nodeCount,
		"podNamespace", podSpec["namespace"],
		"podName", extractPodName(podSpec))
}

// LogMLScoringResponse logs ML scoring response
func (l *MLSchedulerLogger) LogMLScoringResponse(ctx context.Context, requestID string, 
	duration time.Duration, success bool, confidence float64, 
	recommendedNode string, fallbackRequired bool) {
	
	if success {
		klog.V(4).InfoS("ML scoring successful", 
			"component", l.component,
			"requestID", requestID,
			"duration", duration,
			"confidence", confidence,
			"recommendedNode", recommendedNode,
			"fallbackRequired", fallbackRequired)
	} else {
		klog.WarningS(nil, "ML scoring failed", 
			"component", l.component,
			"requestID", requestID,
			"duration", duration)
	}
}

// LogCacheOperation logs cache operations
func (l *MLSchedulerLogger) LogCacheOperation(operation string, key string, 
	hit bool, duration time.Duration, err error) {
	
	if err != nil {
		klog.ErrorS(err, "Cache operation failed", 
			"component", l.component,
			"operation", operation,
			"key", key,
			"duration", duration)
	} else {
		klog.V(6).InfoS("Cache operation", 
			"component", l.component,
			"operation", operation,
			"key", key,
			"hit", hit,
			"duration", duration)
	}
}

// LogFallbackUsage logs fallback scheduler usage
func (l *MLSchedulerLogger) LogFallbackUsage(ctx context.Context, pod *v1.Pod, 
	nodeName, reason string, duration time.Duration, score int64) {
	
	klog.V(3).InfoS("Fallback scheduler used", 
		"component", l.component,
		"pod", klog.KObj(pod),
		"node", nodeName,
		"reason", reason,
		"duration", duration,
		"score", score)
}

// LogAnomalyDetection logs anomaly detection results
func (l *MLSchedulerLogger) LogAnomalyDetection(ctx context.Context, nodeName string, 
	anomalies []interface{}, severity string) {
	
	if len(anomalies) > 0 {
		klog.WarningS(nil, "Node anomalies detected", 
			"component", l.component,
			"node", nodeName,
			"anomalyCount", len(anomalies),
			"severity", severity)
		
		for i, anomaly := range anomalies {
			if i < 5 { // Log first 5 anomalies
				klog.V(4).InfoS("Anomaly detail", 
					"component", l.component,
					"node", nodeName,
					"anomaly", anomaly)
			}
		}
	}
}

// LogNodeFiltering logs node filtering decisions
func (l *MLSchedulerLogger) LogNodeFiltering(ctx context.Context, pod *v1.Pod, 
	nodeName, reason string, basicResourceCheck bool) {
	
	klog.V(5).InfoS("Node filtered", 
		"component", l.component,
		"pod", klog.KObj(pod),
		"node", nodeName,
		"reason", reason,
		"basicResourcesOK", basicResourceCheck)
}

// LogClusterScaling logs cluster scaling suggestions
func (l *MLSchedulerLogger) LogClusterScaling(ctx context.Context, pod *v1.Pod, 
	suggestion string, filteredNodeCount int) {
	
	klog.InfoS("Cluster scaling suggestion", 
		"component", l.component,
		"pod", klog.KObj(pod),
		"suggestion", suggestion,
		"filteredNodes", filteredNodeCount)
}

// LogPluginHealth logs plugin health status
func (l *MLSchedulerLogger) LogPluginHealth(healthy bool, mlServiceHealthy bool, 
	cacheHealthy bool, lastError error) {
	
	if healthy {
		klog.V(5).InfoS("Plugin health check", 
			"component", l.component,
			"status", "healthy",
			"mlService", mlServiceHealthy,
			"cache", cacheHealthy)
	} else {
		klog.WarningS(lastError, "Plugin health check failed", 
			"component", l.component,
			"status", "unhealthy",
			"mlService", mlServiceHealthy,
			"cache", cacheHealthy)
	}
}

// logEvent logs a structured scheduling event
func (l *MLSchedulerLogger) logEvent(event SchedulingEvent) {
	// Log at different levels based on event type and success
	switch event.EventType {
	case "scheduling_success":
		klog.V(3).InfoS("Scheduling event", 
			"component", l.component,
			"event", event.EventType,
			"pod", fmt.Sprintf("%s/%s", event.PodNamespace, event.PodName),
			"node", event.NodeName,
			"requestID", event.RequestID,
			"duration", event.Duration,
			"score", event.Score,
			"confidence", event.Confidence,
			"cacheHit", event.CacheHit)
		
	case "scheduling_failure":
		klog.WarningS(nil, "Scheduling event", 
			"component", l.component,
			"event", event.EventType,
			"pod", fmt.Sprintf("%s/%s", event.PodNamespace, event.PodName),
			"node", event.NodeName,
			"requestID", event.RequestID,
			"duration", event.Duration,
			"reason", event.Reason,
			"fallbackUsed", event.FallbackUsed)
		
	default:
		klog.V(4).InfoS("Scheduling event", 
			"component", l.component,
			"event", event.EventType,
			"pod", fmt.Sprintf("%s/%s", event.PodNamespace, event.PodName),
			"node", event.NodeName,
			"requestID", event.RequestID)
	}
	
	// For debugging, also log as JSON at higher verbosity
	if klog.V(8).Enabled() {
		if jsonData, err := json.Marshal(event); err == nil {
			klog.V(8).InfoS("Detailed scheduling event", 
				"component", l.component,
				"eventJSON", string(jsonData))
		}
	}
}

// Helper function to extract pod name from pod spec
func extractPodName(podSpec map[string]interface{}) string {
	if metadata, ok := podSpec["metadata"].(map[string]interface{}); ok {
		if name, ok := metadata["name"].(string); ok {
			return name
		}
	}
	return "unknown"
}

// Global logger instances
var (
	pluginLogger    *MLSchedulerLogger
	scoringLogger   *MLSchedulerLogger
	cachingLogger   *MLSchedulerLogger
	fallbackLogger  *MLSchedulerLogger
)

// GetPluginLogger returns the plugin logger
func GetPluginLogger() *MLSchedulerLogger {
	if pluginLogger == nil {
		pluginLogger = NewMLSchedulerLogger("plugin")
	}
	return pluginLogger
}

// GetScoringLogger returns the scoring logger
func GetScoringLogger() *MLSchedulerLogger {
	if scoringLogger == nil {
		scoringLogger = NewMLSchedulerLogger("scoring")
	}
	return scoringLogger
}

// GetCachingLogger returns the caching logger
func GetCachingLogger() *MLSchedulerLogger {
	if cachingLogger == nil {
		cachingLogger = NewMLSchedulerLogger("caching")
	}
	return cachingLogger
}

// GetFallbackLogger returns the fallback logger
func GetFallbackLogger() *MLSchedulerLogger {
	if fallbackLogger == nil {
		fallbackLogger = NewMLSchedulerLogger("fallback")
	}
	return fallbackLogger
}