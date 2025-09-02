package scoring

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"k8s.io/klog/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// ScoringRequest represents a request to score a pod placement
type ScoringRequest struct {
	RequestID      string                 `json:"request_id"`
	PodSpec       map[string]interface{} `json:"pod_spec"`
	NodeCandidates []map[string]interface{} `json:"node_candidates"`
	CurrentMetrics map[string]interface{} `json:"current_metrics"`
	UserID        string                 `json:"user_id"`
	PriorityClass string                 `json:"priority_class"`
	DeadlineSeconds *int                 `json:"deadline_seconds,omitempty"`
}

// ScoringResponse represents the response from ML scoring service
type ScoringResponse struct {
	RequestID           string             `json:"request_id"`
	NodeScores         []NodeScore        `json:"node_scores"`
	RecommendedNode    string             `json:"recommended_node"`
	FallbackRequired   bool               `json:"fallback_required"`
	ProcessingTimeMs   float64            `json:"processing_time_ms"`
	AnomalyAlerts      []AnomalyAlert     `json:"anomaly_alerts"`
	ABTestMetadata     map[string]interface{} `json:"ab_test_metadata"`
}

// NodeScore represents the ML score for a specific node
type NodeScore struct {
	NodeName              string             `json:"node_name"`
	TotalScore           float64            `json:"total_score"`
	ComponentScores      map[string]float64 `json:"component_scores"`
	Confidence           float64            `json:"confidence"`
	PlacementRecommendation bool           `json:"placement_recommendation"`
	Reasoning            []string           `json:"reasoning"`
}

// AnomalyAlert represents an anomaly detection alert
type AnomalyAlert struct {
	Severity    string    `json:"severity"`
	NodeName    string    `json:"node_name"`
	Description string    `json:"description"`
	Timestamp   time.Time `json:"timestamp"`
	AlertID     string    `json:"alert_id"`
}

// SchedulingDecision represents feedback for ML model improvement
type SchedulingDecision struct {
	RequestID    string        `json:"request_id"`
	PodName      string        `json:"pod_name"`
	PodNamespace string        `json:"pod_namespace"`
	NodeName     string        `json:"node_name"`
	Success      bool          `json:"success"`
	Timestamp    time.Time     `json:"timestamp"`
	Duration     time.Duration `json:"duration"`
}

// MLScorer handles communication with ML scoring services
type MLScorer struct {
	baseURL    string
	timeout    time.Duration
	httpClient *http.Client
	
	// Metrics
	requestsTotal    prometheus.Counter
	requestDuration  prometheus.Histogram
	errorTotal       prometheus.Counter
	cacheHits        prometheus.Counter
	fallbackTotal    prometheus.Counter
	
	// Internal state
	mu             sync.RWMutex
	lastHealthCheck time.Time
	healthy        bool
}

// NewMLScorer creates a new ML scorer instance
func NewMLScorer(baseURL string, timeout time.Duration) (*MLScorer, error) {
	if baseURL == "" {
		return nil, fmt.Errorf("base URL cannot be empty")
	}
	
	scorer := &MLScorer{
		baseURL: baseURL,
		timeout: timeout,
		httpClient: &http.Client{
			Timeout: timeout,
		},
		healthy: true,
	}
	
	// Initialize Prometheus metrics
	scorer.initMetrics()
	
	// Start health check routine
	go scorer.healthCheckRoutine()
	
	klog.InfoS("ML Scorer initialized", "baseURL", baseURL, "timeout", timeout)
	return scorer, nil
}

func (ms *MLScorer) initMetrics() {
	ms.requestsTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_requests_total",
		Help: "Total number of ML scoring requests",
	})
	
	ms.requestDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "ml_scheduler_request_duration_seconds",
		Help: "Duration of ML scoring requests",
		Buckets: prometheus.DefBuckets,
	})
	
	ms.errorTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_errors_total",
		Help: "Total number of ML scoring errors",
	})
	
	ms.cacheHits = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_cache_hits_total",
		Help: "Total number of cache hits",
	})
	
	ms.fallbackTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "ml_scheduler_fallback_total",
		Help: "Total number of fallback scheduler uses",
	})
}

// ScoreNode scores a single node for pod placement
func (ms *MLScorer) ScoreNode(ctx context.Context, request *ScoringRequest) (float64, float64, error) {
	startTime := time.Now()
	defer func() {
		ms.requestDuration.Observe(time.Since(startTime).Seconds())
		ms.requestsTotal.Inc()
	}()
	
	// Check if service is healthy
	ms.mu.RLock()
	isHealthy := ms.healthy
	ms.mu.RUnlock()
	
	if !isHealthy {
		ms.errorTotal.Inc()
		return 0.0, 0.0, fmt.Errorf("ML scoring service is unhealthy")
	}
	
	// Make request to combined scorer
	response, err := ms.callScoringService(ctx, request)
	if err != nil {
		ms.errorTotal.Inc()
		return 0.0, 0.0, fmt.Errorf("scoring service call failed: %w", err)
	}
	
	// Extract score for the target node
	if len(request.NodeCandidates) == 0 {
		return 0.0, 0.0, fmt.Errorf("no node candidates provided")
	}
	
	targetNodeName := extractNodeName(request.NodeCandidates[0])
	
	for _, nodeScore := range response.NodeScores {
		if nodeScore.NodeName == targetNodeName {
			return nodeScore.TotalScore, nodeScore.Confidence, nil
		}
	}
	
	return 0.0, 0.0, fmt.Errorf("score not found for node %s", targetNodeName)
}

func (ms *MLScorer) callScoringService(ctx context.Context, request *ScoringRequest) (*ScoringResponse, error) {
	// Prepare JSON payload
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	// Create HTTP request
	url := fmt.Sprintf("%s/v1/score", ms.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}
	
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("User-Agent", "ml-scheduler-plugin/1.0")
	
	// Make HTTP request
	resp, err := ms.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()
	
	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}
	
	// Parse response
	var scoringResponse ScoringResponse
	if err := json.Unmarshal(body, &scoringResponse); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	return &scoringResponse, nil
}

// GetClusterMetrics gets current cluster metrics for scoring
func (ms *MLScorer) GetClusterMetrics(ctx context.Context) (map[string]interface{}, error) {
	url := fmt.Sprintf("%s/v1/cluster/metrics", ms.baseURL)
	
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create metrics request: %w", err)
	}
	
	resp, err := ms.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("metrics request failed: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("metrics request returned HTTP %d", resp.StatusCode)
	}
	
	var metrics map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&metrics); err != nil {
		return nil, fmt.Errorf("failed to decode metrics response: %w", err)
	}
	
	return metrics, nil
}

// CheckNodeAnomalies checks for anomalies on a specific node
func (ms *MLScorer) CheckNodeAnomalies(ctx context.Context, nodeName string) ([]AnomalyAlert, error) {
	url := fmt.Sprintf("%s/v1/anomalies/node/%s", ms.baseURL, nodeName)
	
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create anomaly request: %w", err)
	}
	
	resp, err := ms.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anomaly request failed: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("anomaly request returned HTTP %d", resp.StatusCode)
	}
	
	var anomalies []AnomalyAlert
	if err := json.NewDecoder(resp.Body).Decode(&anomalies); err != nil {
		return nil, fmt.Errorf("failed to decode anomaly response: %w", err)
	}
	
	return anomalies, nil
}

// RecordSchedulingDecision sends scheduling feedback to ML services
func (ms *MLScorer) RecordSchedulingDecision(ctx context.Context, decision SchedulingDecision) error {
	url := fmt.Sprintf("%s/v1/feedback", ms.baseURL)
	
	jsonData, err := json.Marshal(decision)
	if err != nil {
		return fmt.Errorf("failed to marshal decision: %w", err)
	}
	
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create feedback request: %w", err)
	}
	
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := ms.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("feedback request failed: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("feedback request returned HTTP %d: %s", resp.StatusCode, string(body))
	}
	
	return nil
}

// HealthCheck performs health check on ML scoring service
func (ms *MLScorer) HealthCheck(ctx context.Context) error {
	url := fmt.Sprintf("%s/health", ms.baseURL)
	
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}
	
	resp, err := ms.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned HTTP %d", resp.StatusCode)
	}
	
	return nil
}

func (ms *MLScorer) healthCheckRoutine() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			err := ms.HealthCheck(ctx)
			cancel()
			
			ms.mu.Lock()
			ms.healthy = (err == nil)
			ms.lastHealthCheck = time.Now()
			ms.mu.Unlock()
			
			if err != nil {
				klog.ErrorS(err, "ML scoring service health check failed")
			} else {
				klog.V(5).InfoS("ML scoring service health check passed")
			}
		}
	}
}

// GetHealthStatus returns current health status
func (ms *MLScorer) GetHealthStatus() (bool, time.Time) {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	return ms.healthy, ms.lastHealthCheck
}

// Helper function to extract node name from node candidate
func extractNodeName(nodeCandidate map[string]interface{}) string {
	if metadata, ok := nodeCandidate["metadata"].(map[string]interface{}); ok {
		if name, ok := metadata["name"].(string); ok {
			return name
		}
	}
	
	// Fallback: try direct name field
	if name, ok := nodeCandidate["name"].(string); ok {
		return name
	}
	
	return "unknown"
}