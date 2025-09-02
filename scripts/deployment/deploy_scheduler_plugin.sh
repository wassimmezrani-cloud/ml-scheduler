#!/bin/bash

# ML Scheduler Plugin Deployment Script
# Deploys the complete ML scheduler plugin to HYDATIS cluster

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAMESPACE="${NAMESPACE:-kube-system}"
ML_NAMESPACE="${ML_NAMESPACE:-ml-scheduler}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-localhost:5000}"
SCHEDULER_IMAGE="${SCHEDULER_IMAGE:-ml-scheduler-plugin:latest}"
TIMEOUT="${TIMEOUT:-600}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required but not installed"
        exit 1
    fi
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        log_error "docker is required but not installed"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if ML services namespace exists
    if ! kubectl get namespace "$ML_NAMESPACE" &> /dev/null; then
        log_warning "ML services namespace '$ML_NAMESPACE' not found"
        read -p "Create ML services namespace? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kubectl create namespace "$ML_NAMESPACE"
            log_success "Created namespace '$ML_NAMESPACE'"
        else
            log_error "ML services namespace is required"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Build and push scheduler image
build_scheduler_image() {
    log_info "Building ML scheduler plugin image..."
    
    cd "$PROJECT_ROOT/scheduler-plugin"
    
    # Build the Go binary
    make build
    
    # Build Docker image
    docker build -t "$IMAGE_REGISTRY/ml-scheduler-plugin:latest" .
    
    # Push to registry if not localhost
    if [[ "$IMAGE_REGISTRY" != "localhost:5000" ]]; then
        docker push "$IMAGE_REGISTRY/ml-scheduler-plugin:latest"
        log_success "Pushed image to registry"
    else
        log_info "Using local registry, skipping push"
    fi
    
    log_success "Scheduler image built and ready"
}

# Deploy ML services dependencies
deploy_ml_services() {
    log_info "Deploying ML services dependencies..."
    
    # Deploy Redis cache
    log_info "Deploying Redis cache..."
    kubectl apply -f "$PROJECT_ROOT/k8s_configs/ml_services/redis-deployment.yaml" -n "$ML_NAMESPACE"
    
    # Wait for Redis to be ready
    kubectl rollout status deployment/redis-cache -n "$ML_NAMESPACE" --timeout="${TIMEOUT}s"
    
    # Deploy KServe services
    log_info "Deploying KServe inference services..."
    kubectl apply -f "$PROJECT_ROOT/kserve_configs/" -n "$ML_NAMESPACE"
    
    # Wait for inference services to be ready
    log_info "Waiting for inference services to be ready..."
    
    # Check XGBoost service
    kubectl wait --for=condition=Ready inferenceservice/xgboost-load-predictor \
        -n "$ML_NAMESPACE" --timeout="${TIMEOUT}s" || {
        log_warning "XGBoost service not ready, continuing..."
    }
    
    # Check Q-Learning service
    kubectl wait --for=condition=Ready inferenceservice/qlearning-placement-optimizer \
        -n "$ML_NAMESPACE" --timeout="${TIMEOUT}s" || {
        log_warning "Q-Learning service not ready, continuing..."
    }
    
    # Check Anomaly Detection service
    kubectl wait --for=condition=Ready inferenceservice/anomaly-detector \
        -n "$ML_NAMESPACE" --timeout="${TIMEOUT}s" || {
        log_warning "Anomaly detection service not ready, continuing..."
    }
    
    log_success "ML services deployed"
}

# Deploy scheduler plugin
deploy_scheduler() {
    log_info "Deploying ML scheduler plugin..."
    
    # Update image in deployment manifest
    sed -i.bak "s|ml-scheduler-plugin:latest|$IMAGE_REGISTRY/ml-scheduler-plugin:latest|g" \
        "$PROJECT_ROOT/scheduler-plugin/manifests/scheduler-deployment.yaml"
    
    # Apply scheduler deployment
    kubectl apply -f "$PROJECT_ROOT/scheduler-plugin/manifests/scheduler-deployment.yaml" -n "$NAMESPACE"
    
    # Wait for scheduler to be ready
    log_info "Waiting for scheduler deployment to be ready..."
    kubectl rollout status deployment/ml-scheduler -n "$NAMESPACE" --timeout="${TIMEOUT}s"
    
    # Restore original manifest
    if [[ -f "$PROJECT_ROOT/scheduler-plugin/manifests/scheduler-deployment.yaml.bak" ]]; then
        mv "$PROJECT_ROOT/scheduler-plugin/manifests/scheduler-deployment.yaml.bak" \
           "$PROJECT_ROOT/scheduler-plugin/manifests/scheduler-deployment.yaml"
    fi
    
    log_success "ML scheduler plugin deployed"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check scheduler pods
    log_info "Checking scheduler pods..."
    kubectl get pods -l app=ml-scheduler -n "$NAMESPACE" -o wide
    
    # Check ML services
    log_info "Checking ML services..."
    kubectl get inferenceservices -n "$ML_NAMESPACE"
    
    # Check Redis
    log_info "Checking Redis cache..."
    kubectl get pods -l app=redis-cache -n "$ML_NAMESPACE"
    
    # Test scheduler health
    log_info "Testing scheduler health..."
    local scheduler_pod
    scheduler_pod=$(kubectl get pods -l app=ml-scheduler -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$scheduler_pod" ]]; then
        if kubectl exec "$scheduler_pod" -n "$NAMESPACE" -- /usr/local/bin/kube-scheduler --version &> /dev/null; then
            log_success "Scheduler health check passed"
        else
            log_warning "Scheduler health check failed"
        fi
    else
        log_warning "No scheduler pods found for health check"
    fi
    
    # Show recent events
    log_info "Recent events:"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -5
    
    log_success "Deployment verification completed"
}

# Test scheduler functionality
test_scheduler() {
    log_info "Testing scheduler functionality..."
    
    # Create a test pod with ML scheduler
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ml-scheduler-test-pod
  namespace: default
spec:
  schedulerName: ml-scheduler
  containers:
  - name: test-container
    image: nginx:alpine
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
      limits:
        cpu: 200m
        memory: 256Mi
  restartPolicy: Never
EOF

    # Wait for pod to be scheduled
    log_info "Waiting for test pod to be scheduled..."
    
    local max_wait=60
    local wait_time=0
    
    while [[ $wait_time -lt $max_wait ]]; do
        local phase
        phase=$(kubectl get pod ml-scheduler-test-pod -n default -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
        
        if [[ "$phase" == "Running" ]] || [[ "$phase" == "Succeeded" ]]; then
            log_success "Test pod scheduled successfully"
            kubectl get pod ml-scheduler-test-pod -n default -o wide
            break
        elif [[ "$phase" == "Failed" ]]; then
            log_error "Test pod failed to schedule"
            kubectl describe pod ml-scheduler-test-pod -n default
            break
        fi
        
        sleep 2
        ((wait_time+=2))
    done
    
    if [[ $wait_time -ge $max_wait ]]; then
        log_warning "Test pod scheduling timed out"
        kubectl describe pod ml-scheduler-test-pod -n default
    fi
    
    # Clean up test pod
    kubectl delete pod ml-scheduler-test-pod -n default --ignore-not-found=true
    
    log_info "Scheduler functionality test completed"
}

# Show scheduler metrics
show_metrics() {
    log_info "Fetching scheduler metrics..."
    
    local scheduler_pod
    scheduler_pod=$(kubectl get pods -l app=ml-scheduler -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$scheduler_pod" ]]; then
        log_info "Port forwarding metrics from pod: $scheduler_pod"
        kubectl port-forward pod/"$scheduler_pod" -n "$NAMESPACE" 10251:10251 &
        local port_forward_pid=$!
        
        sleep 3
        
        echo "=== ML Scheduler Metrics ==="
        curl -s http://localhost:10251/metrics | grep ml_scheduler | head -20 || {
            log_warning "Could not fetch metrics"
        }
        
        # Clean up port forward
        kill $port_forward_pid 2>/dev/null || true
    else
        log_warning "No scheduler pods found for metrics"
    fi
}

# Main deployment function
main() {
    local action="${1:-deploy}"
    
    echo "=================================="
    echo "ML Scheduler Plugin Deployment"
    echo "=================================="
    echo "Action: $action"
    echo "Namespace: $NAMESPACE"
    echo "ML Namespace: $ML_NAMESPACE"
    echo "Image: $IMAGE_REGISTRY/ml-scheduler-plugin:latest"
    echo "=================================="
    echo
    
    case "$action" in
        "deploy")
            check_prerequisites
            build_scheduler_image
            deploy_ml_services
            deploy_scheduler
            verify_deployment
            test_scheduler
            show_metrics
            log_success "ML scheduler plugin deployment completed successfully!"
            ;;
        "undeploy")
            log_info "Undeploying ML scheduler plugin..."
            kubectl delete -f "$PROJECT_ROOT/scheduler-plugin/manifests/scheduler-deployment.yaml" -n "$NAMESPACE" --ignore-not-found=true
            log_success "ML scheduler plugin undeployed"
            ;;
        "status")
            verify_deployment
            ;;
        "test")
            test_scheduler
            ;;
        "metrics")
            show_metrics
            ;;
        "logs")
            kubectl logs -l app=ml-scheduler -n "$NAMESPACE" --tail=100 -f
            ;;
        *)
            echo "Usage: $0 {deploy|undeploy|status|test|metrics|logs}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy the complete ML scheduler plugin"
            echo "  undeploy - Remove the ML scheduler plugin"
            echo "  status   - Check deployment status"
            echo "  test     - Test scheduler functionality"
            echo "  metrics  - Show scheduler metrics"
            echo "  logs     - Show scheduler logs"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"