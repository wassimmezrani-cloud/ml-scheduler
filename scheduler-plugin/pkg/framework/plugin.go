package framework

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"

	"github.com/hydatis/ml-scheduler-plugin/pkg/scoring"
	"github.com/hydatis/ml-scheduler-plugin/pkg/caching"
	"github.com/hydatis/ml-scheduler-plugin/pkg/fallback"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "MLSchedulerPlugin"
	
	// Default scoring timeout
	DefaultScoringTimeoutMs = 5000
	
	// Default confidence threshold
	DefaultConfidenceThreshold = 0.7
)

// PluginConfiguration holds the configuration for the ML scheduler plugin
type PluginConfiguration struct {
	MLServicesURL       string  `json:"mlServicesURL"`
	RedisURL           string  `json:"redisURL"`
	ScoringTimeoutMs   int     `json:"scoringTimeoutMs"`
	EnableFallback     bool    `json:"enableFallback"`
	LogLevel          string  `json:"logLevel"`
	ConfidenceThreshold float64 `json:"confidenceThreshold"`
}

// Global plugin configuration
var PluginConfig = PluginConfiguration{
	MLServicesURL:       "http://combined-ml-scorer:8080",
	RedisURL:           "redis://redis-cache-service:6379",
	ScoringTimeoutMs:   DefaultScoringTimeoutMs,
	EnableFallback:     true,
	LogLevel:          "info",
	ConfidenceThreshold: DefaultConfidenceThreshold,
}

// MLSchedulerPlugin is the main plugin struct
type MLSchedulerPlugin struct {
	handle       framework.Handle
	mlScorer     *scoring.MLScorer
	cacheManager *caching.CacheManager
	fallback     *fallback.FallbackScheduler
	config       PluginConfiguration
}

// Name returns the name of the plugin
func (pl *MLSchedulerPlugin) Name() string {
	return Name
}

// New initializes a new plugin and returns it.
func New(obj runtime.Object, h framework.Handle) (framework.Plugin, error) {
	config := PluginConfig
	
	// Override with configuration from object if provided
	if obj != nil {
		if configData, ok := obj.(*PluginConfiguration); ok {
			config = *configData
		}
	}
	
	klog.InfoS("Initializing ML Scheduler Plugin", 
		"mlServicesURL", config.MLServicesURL,
		"redisURL", config.RedisURL,
		"scoringTimeout", config.ScoringTimeoutMs,
		"enableFallback", config.EnableFallback,
		"confidenceThreshold", config.ConfidenceThreshold)
	
	// Initialize ML scorer
	mlScorer, err := scoring.NewMLScorer(config.MLServicesURL, 
		time.Duration(config.ScoringTimeoutMs)*time.Millisecond)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ML scorer: %w", err)
	}
	
	// Initialize cache manager
	cacheManager, err := caching.NewCacheManager(config.RedisURL)
	if err != nil {
		klog.ErrorS(err, "Failed to initialize Redis cache, continuing without cache")
		cacheManager = nil
	}
	
	// Initialize fallback scheduler
	fallbackScheduler := fallback.NewFallbackScheduler(h, config.EnableFallback)
	
	plugin := &MLSchedulerPlugin{
		handle:       h,
		mlScorer:     mlScorer,
		cacheManager: cacheManager,
		fallback:     fallbackScheduler,
		config:       config,
	}
	
	klog.InfoS("ML Scheduler Plugin initialized successfully")
	return plugin, nil
}

// Score is called on each filterable node. It must return success and an integer
// indicating the preference of the given node.
func (pl *MLSchedulerPlugin) Score(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	
	startTime := time.Now()
	defer func() {
		duration := time.Since(startTime)
		klog.V(4).InfoS("ML scoring completed", 
			"pod", klog.KObj(pod),
			"node", nodeName,
			"duration", duration)
	}()
	
	// Get node info
	nodeInfo := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if nodeInfo == nil {
		klog.ErrorS(nil, "Node not found", "node", nodeName)
		return pl.fallback.ScoreNode(ctx, state, pod, nodeName)
	}
	
	// Check cache first
	if pl.cacheManager != nil {
		if cachedScore, found := pl.cacheManager.GetScore(pod, nodeName); found {
			klog.V(5).InfoS("Using cached score", 
				"pod", klog.KObj(pod),
				"node", nodeName,
				"score", cachedScore)
			return cachedScore, nil
		}
	}
	
	// Prepare scoring request
	scoringRequest, err := pl.prepareScoringRequest(pod, nodeInfo, state)
	if err != nil {
		klog.ErrorS(err, "Failed to prepare scoring request", 
			"pod", klog.KObj(pod), "node", nodeName)
		return pl.fallback.ScoreNode(ctx, state, pod, nodeName)
	}
	
	// Get ML score
	mlScore, confidence, err := pl.mlScorer.ScoreNode(ctx, scoringRequest)
	if err != nil {
		klog.ErrorS(err, "ML scoring failed, using fallback", 
			"pod", klog.KObj(pod), "node", nodeName)
		return pl.fallback.ScoreNode(ctx, state, pod, nodeName)
	}
	
	// Check confidence threshold
	if confidence < pl.config.ConfidenceThreshold {
		klog.V(3).InfoS("Low confidence ML score, using fallback", 
			"pod", klog.KObj(pod),
			"node", nodeName,
			"confidence", confidence,
			"threshold", pl.config.ConfidenceThreshold)
		return pl.fallback.ScoreNode(ctx, state, pod, nodeName)
	}
	
	// Convert ML score (0.0-1.0) to scheduler score (0-100)
	schedulerScore := int64(mlScore * 100)
	
	// Cache the result
	if pl.cacheManager != nil {
		pl.cacheManager.SetScore(pod, nodeName, schedulerScore, 5*time.Minute)
	}
	
	klog.V(4).InfoS("ML scoring successful", 
		"pod", klog.KObj(pod),
		"node", nodeName,
		"mlScore", mlScore,
		"confidence", confidence,
		"schedulerScore", schedulerScore)
	
	return schedulerScore, nil
}

// ScoreExtensions returns a ScoreExtensions interface if it implements one, or nil if it does not.
func (pl *MLSchedulerPlugin) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// NormalizeScore normalizes the scores for all nodes.
func (pl *MLSchedulerPlugin) NormalizeScore(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	
	if len(scores) == 0 {
		return nil
	}
	
	// Find min and max scores
	var minScore, maxScore int64 = scores[0].Score, scores[0].Score
	for _, score := range scores {
		if score.Score < minScore {
			minScore = score.Score
		}
		if score.Score > maxScore {
			maxScore = score.Score
		}
	}
	
	// Normalize scores to 0-100 range
	if maxScore > minScore {
		for i := range scores {
			scores[i].Score = (scores[i].Score - minScore) * 100 / (maxScore - minScore)
		}
	}
	
	klog.V(5).InfoS("Score normalization completed", 
		"pod", klog.KObj(pod),
		"nodeCount", len(scores),
		"originalRange", fmt.Sprintf("%d-%d", minScore, maxScore))
	
	return nil
}

// Filter is called to filter out nodes that cannot fit the pod.
func (pl *MLSchedulerPlugin) Filter(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	
	// First check basic resource requirements
	if !pl.checkBasicResources(pod, nodeInfo) {
		return framework.NewStatus(framework.Unschedulable, 
			fmt.Sprintf("Node %s has insufficient resources", nodeInfo.Node().Name))
	}
	
	// Check for critical anomalies
	if pl.mlScorer != nil {
		if hasAnomalies, reason := pl.checkNodeAnomalies(ctx, nodeInfo.Node().Name); hasAnomalies {
			klog.V(3).InfoS("Node filtered due to anomalies", 
				"node", nodeInfo.Node().Name,
				"reason", reason)
			return framework.NewStatus(framework.Unschedulable, reason)
		}
	}
	
	return nil
}

// PreFilter is called at the beginning of the scheduling cycle.
func (pl *MLSchedulerPlugin) PreFilter(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod) (*framework.PreFilterResult, *framework.Status) {
	
	klog.V(4).InfoS("Pre-filter phase started", "pod", klog.KObj(pod))
	
	// Store pod information in cycle state for later use
	podInfo := &PodSchedulingInfo{
		Pod:       pod,
		StartTime: time.Now(),
		RequestID: fmt.Sprintf("%s-%s-%d", pod.Namespace, pod.Name, time.Now().Unix()),
	}
	
	state.Write(podInfoKey, podInfo)
	
	return nil, nil
}

// PreFilterExtensions returns a PreFilterExtensions interface if the plugin implements one.
func (pl *MLSchedulerPlugin) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

// PostFilter is called after Filter phase, only when no feasible nodes are left.
func (pl *MLSchedulerPlugin) PostFilter(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, filteredNodeStatusMap framework.NodeToStatusMap) (*framework.PostFilterResult, *framework.Status) {
	
	klog.InfoS("Post-filter phase: no feasible nodes found", "pod", klog.KObj(pod))
	
	// Log the filtering reasons for debugging
	for nodeName, status := range filteredNodeStatusMap {
		klog.V(3).InfoS("Node filtered", 
			"pod", klog.KObj(pod),
			"node", nodeName,
			"reason", status.Message())
	}
	
	// Try to suggest cluster scaling if needed
	if len(filteredNodeStatusMap) > 0 {
		pl.suggestClusterScaling(pod, filteredNodeStatusMap)
	}
	
	return nil, framework.NewStatus(framework.Unschedulable, 
		"No nodes available after ML filtering")
}

// Reserve reserves resources on the given node for the given pod.
func (pl *MLSchedulerPlugin) Reserve(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, nodeName string) *framework.Status {
	
	klog.V(4).InfoS("Reserve phase", "pod", klog.KObj(pod), "node", nodeName)
	
	// Record scheduling decision for ML model feedback
	if podInfo := pl.getPodInfo(state); podInfo != nil {
		pl.recordSchedulingDecision(podInfo, nodeName, true)
	}
	
	return nil
}

// Unreserve unreserves resources on the given node for the given pod.
func (pl *MLSchedulerPlugin) Unreserve(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, nodeName string) {
	
	klog.V(4).InfoS("Unreserve phase", "pod", klog.KObj(pod), "node", nodeName)
	
	// Record failed scheduling for ML model feedback
	if podInfo := pl.getPodInfo(state); podInfo != nil {
		pl.recordSchedulingDecision(podInfo, nodeName, false)
	}
}

// Permit is called before a pod is bound to a node.
func (pl *MLSchedulerPlugin) Permit(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, nodeName string) (*framework.Status, time.Duration) {
	
	klog.V(4).InfoS("Permit phase", "pod", klog.KObj(pod), "node", nodeName)
	
	// Final validation - check for any last-minute anomalies
	if pl.mlScorer != nil {
		if hasAnomalies, reason := pl.checkNodeAnomalies(ctx, nodeName); hasAnomalies {
			klog.WarningS(nil, "Node anomaly detected during permit phase", 
				"pod", klog.KObj(pod),
				"node", nodeName,
				"reason", reason)
			return framework.NewStatus(framework.Unschedulable, reason), 0
		}
	}
	
	return nil, 0
}

// Helper types and functions

type PodSchedulingInfo struct {
	Pod       *v1.Pod
	StartTime time.Time
	RequestID string
}

var podInfoKey = "ml-scheduler/pod-info"

func (pl *MLSchedulerPlugin) getPodInfo(state *framework.CycleState) *PodSchedulingInfo {
	if info, err := state.Read(podInfoKey); err == nil {
		if podInfo, ok := info.(*PodSchedulingInfo); ok {
			return podInfo
		}
	}
	return nil
}

func (pl *MLSchedulerPlugin) prepareScoringRequest(pod *v1.Pod, nodeInfo *framework.NodeInfo, 
	state *framework.CycleState) (*scoring.ScoringRequest, error) {
	
	podInfo := pl.getPodInfo(state)
	if podInfo == nil {
		return nil, fmt.Errorf("pod info not found in cycle state")
	}
	
	// Convert pod spec to map
	podSpecBytes, err := json.Marshal(pod.Spec)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal pod spec: %w", err)
	}
	
	var podSpecMap map[string]interface{}
	if err := json.Unmarshal(podSpecBytes, &podSpecMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal pod spec: %w", err)
	}
	
	// Convert node info to map
	nodeBytes, err := json.Marshal(nodeInfo.Node())
	if err != nil {
		return nil, fmt.Errorf("failed to marshal node: %w", err)
	}
	
	var nodeMap map[string]interface{}
	if err := json.Unmarshal(nodeBytes, &nodeMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal node: %w", err)
	}
	
	// Get current cluster metrics
	currentMetrics, err := pl.mlScorer.GetClusterMetrics(context.Background())
	if err != nil {
		klog.WarningS(err, "Failed to get cluster metrics, using empty metrics")
		currentMetrics = map[string]interface{}{}
	}
	
	request := &scoring.ScoringRequest{
		RequestID:      podInfo.RequestID,
		PodSpec:       podSpecMap,
		NodeCandidates: []map[string]interface{}{nodeMap},
		CurrentMetrics: currentMetrics,
		UserID:        "scheduler",
		PriorityClass:  pod.Spec.PriorityClassName,
	}
	
	// Set deadline if pod has scheduling deadline
	if pod.Spec.ActiveDeadlineSeconds != nil {
		deadlineSeconds := int(*pod.Spec.ActiveDeadlineSeconds)
		request.DeadlineSeconds = &deadlineSeconds
	}
	
	return request, nil
}

func (pl *MLSchedulerPlugin) checkBasicResources(pod *v1.Pod, nodeInfo *framework.NodeInfo) bool {
	// Check CPU and memory requests
	podRequests := pod.Spec.Containers[0].Resources.Requests
	if podRequests != nil {
		cpuRequest := podRequests[v1.ResourceCPU]
		memoryRequest := podRequests[v1.ResourceMemory]
		
		nodeAllocatable := nodeInfo.Node().Status.Allocatable
		nodeCPU := nodeAllocatable[v1.ResourceCPU]
		nodeMemory := nodeAllocatable[v1.ResourceMemory]
		
		// Check if node has enough allocatable resources
		if cpuRequest.Cmp(nodeCPU) > 0 || memoryRequest.Cmp(nodeMemory) > 0 {
			return false
		}
		
		// Check current usage
		allocatedCPU := nodeInfo.Requested.MilliCPU
		allocatedMemory := nodeInfo.Requested.Memory
		
		// Allow up to 90% utilization
		maxUsableCPU := nodeCPU.MilliValue() * 90 / 100
		maxUsableMemory := nodeMemory.Value() * 90 / 100
		
		if allocatedCPU+cpuRequest.MilliValue() > maxUsableCPU ||
		   allocatedMemory+memoryRequest.Value() > maxUsableMemory {
			return false
		}
	}
	
	return true
}

func (pl *MLSchedulerPlugin) checkNodeAnomalies(ctx context.Context, nodeName string) (bool, string) {
	if pl.mlScorer == nil {
		return false, ""
	}
	
	// Quick anomaly check for the specific node
	anomalies, err := pl.mlScorer.CheckNodeAnomalies(ctx, nodeName)
	if err != nil {
		klog.V(3).InfoS("Failed to check node anomalies", "node", nodeName, "error", err)
		return false, ""
	}
	
	// Check for critical anomalies
	for _, anomaly := range anomalies {
		if anomaly.Severity == "critical" || anomaly.Severity == "high" {
			return true, fmt.Sprintf("Node has %s anomaly: %s", anomaly.Severity, anomaly.Description)
		}
	}
	
	return false, ""
}

func (pl *MLSchedulerPlugin) recordSchedulingDecision(podInfo *PodSchedulingInfo, 
	nodeName string, success bool) {
	
	decision := scoring.SchedulingDecision{
		RequestID:   podInfo.RequestID,
		PodName:     podInfo.Pod.Name,
		PodNamespace: podInfo.Pod.Namespace,
		NodeName:    nodeName,
		Success:     success,
		Timestamp:   time.Now(),
		Duration:    time.Since(podInfo.StartTime),
	}
	
	// Send feedback to ML scorer for model improvement
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		
		if err := pl.mlScorer.RecordSchedulingDecision(ctx, decision); err != nil {
			klog.ErrorS(err, "Failed to record scheduling decision")
		}
	}()
}

func (pl *MLSchedulerPlugin) suggestClusterScaling(pod *v1.Pod, 
	filteredNodes framework.NodeToStatusMap) {
	
	// Analyze why nodes were filtered
	resourceShortage := false
	anomalyBlocked := false
	
	for _, status := range filteredNodes {
		reason := status.Message()
		if contains(reason, "insufficient resources") {
			resourceShortage = true
		}
		if contains(reason, "anomaly") {
			anomalyBlocked = true
		}
	}
	
	// Log scaling suggestions
	if resourceShortage {
		klog.InfoS("Cluster scaling suggested: insufficient resources", 
			"pod", klog.KObj(pod),
			"suggestion", "Add more worker nodes or increase node capacity")
	}
	
	if anomalyBlocked {
		klog.InfoS("Cluster health issue detected", 
			"pod", klog.KObj(pod),
			"suggestion", "Investigate and resolve node anomalies")
	}
}

func contains(str, substr string) bool {
	return len(str) >= len(substr) && 
		   (str == substr || 
		    (len(str) > len(substr) && 
		     (str[:len(substr)] == substr || 
		      str[len(str)-len(substr):] == substr ||
		      findSubstring(str, substr))))
}

func findSubstring(str, substr string) bool {
	for i := 0; i <= len(str)-len(substr); i++ {
		if str[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}