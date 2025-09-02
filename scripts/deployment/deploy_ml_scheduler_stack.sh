#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NAMESPACE="ml-scheduler"
CLUSTER_NAME="HYDATIS"

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1"
}

check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    CURRENT_CONTEXT=$(kubectl config current-context)
    log_info "Using Kubernetes context: ${CURRENT_CONTEXT}"
    
    if ! kubectl get storageclass longhorn &> /dev/null; then
        log_error "Longhorn storage class not found in cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

build_container_images() {
    log_info "Building container images for ML scheduler components..."
    
    IMAGES=(
        "xgboost-predictor:${PROJECT_ROOT}/docker/xgboost_predictor"
        "qlearning-optimizer:${PROJECT_ROOT}/docker/qlearning_optimizer"
        "anomaly-detector:${PROJECT_ROOT}/docker/anomaly_detector"
        "ml-gateway:${PROJECT_ROOT}/docker/ml_gateway"
        "scheduler-plugin:${PROJECT_ROOT}/docker/scheduler_plugin"
        "monitoring-dashboard:${PROJECT_ROOT}/docker/monitoring_dashboard"
        "alert-manager:${PROJECT_ROOT}/docker/alert_manager"
    )
    
    for image_spec in "${IMAGES[@]}"; do
        IFS=':' read -r image_name dockerfile_path <<< "${image_spec}"
        
        if [ -d "${dockerfile_path}" ]; then
            log_info "Building hydatis/${image_name}..."
            
            docker build \
                -t "hydatis/${image_name}" \
                -f "${dockerfile_path}/Dockerfile" \
                "${PROJECT_ROOT}"
            
            if [ $? -eq 0 ]; then
                log_info "Successfully built hydatis/${image_name}"
            else
                log_error "Failed to build hydatis/${image_name}"
                exit 1
            fi
        else
            log_warn "Dockerfile path not found: ${dockerfile_path}"
        fi
    done
    
    log_info "All container images built successfully"
}

create_namespace() {
    log_info "Creating namespace: ${NAMESPACE}"
    
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl label namespace "${NAMESPACE}" \
        cluster="${CLUSTER_NAME}" \
        component="ml-scheduler" \
        --overwrite
    
    log_info "Namespace ${NAMESPACE} ready"
}

deploy_infrastructure_components() {
    log_info "Deploying infrastructure components..."
    
    log_info "Deploying persistent volumes for Longhorn storage..."
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: xgboost-model-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: longhorn
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qlearning-model-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: longhorn
  resources:
    requests:
      storage: 15Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qlearning-replay-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: longhorn
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: anomaly-model-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: longhorn
  resources:
    requests:
      storage: 8Gi
EOF

    log_info "Creating ConfigMaps for component configurations..."
    
    kubectl create configmap scheduler-plugin-config \
        --namespace="${NAMESPACE}" \
        --from-literal=scheduler-name="hydatis-ml-scheduler" \
        --from-literal=webhook-timeout="30" \
        --from-literal=cluster-name="${CLUSTER_NAME}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create configmap alert-manager-config \
        --namespace="${NAMESPACE}" \
        --from-literal=correlation-window="300" \
        --from-literal=escalation-delay="600" \
        --from-literal=max-alerts-per-hour="10" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_info "Infrastructure components deployed"
}

run_orchestrated_deployment() {
    log_info "Running orchestrated deployment using Python orchestrator..."
    
    cd "${PROJECT_ROOT}"
    
    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
    
    python3 "${PROJECT_ROOT}/src/orchestration/deployment_orchestrator.py"
    
    if [ $? -eq 0 ]; then
        log_info "Orchestrated deployment completed successfully"
    else
        log_error "Orchestrated deployment failed"
        exit 1
    fi
}

verify_deployment() {
    log_info "Verifying deployment status..."
    
    log_info "Checking pod status..."
    kubectl get pods -n "${NAMESPACE}" -o wide
    
    log_info "Checking service status..."
    kubectl get services -n "${NAMESPACE}"
    
    log_info "Checking persistent volume claims..."
    kubectl get pvc -n "${NAMESPACE}"
    
    log_info "Waiting for all pods to be ready..."
    kubectl wait --for=condition=ready pod \
        --selector=cluster=hydatis \
        --namespace="${NAMESPACE}" \
        --timeout=300s
    
    log_info "Testing service endpoints..."
    
    SERVICES=(
        "ml-gateway-service:8000:/health"
        "xgboost-service:8001:/health"
        "qlearning-service:8002:/health"
        "anomaly-service:8003:/health"
        "monitoring-dashboard-service:8080:/health"
    )
    
    for service_spec in "${SERVICES[@]}"; do
        IFS=':' read -r service_name port path <<< "${service_spec}"
        
        log_info "Testing ${service_name} health endpoint..."
        
        kubectl run test-pod-${RANDOM} \
            --rm -i --restart=Never \
            --namespace="${NAMESPACE}" \
            --image=curlimages/curl \
            --command -- curl -f "http://${service_name}.${NAMESPACE}.svc.cluster.local:${port}${path}" \
            --max-time 10 || log_warn "Health check failed for ${service_name}"
    done
    
    log_info "Deployment verification completed"
}

generate_access_information() {
    log_info "Generating access information..."
    
    cat <<EOF

================================================================================
HYDATIS ML SCHEDULER DEPLOYMENT COMPLETE
================================================================================

Cluster: ${CLUSTER_NAME}
Namespace: ${NAMESPACE}

Service Endpoints (Internal):
- ML Gateway API: http://ml-gateway-service.${NAMESPACE}.svc.cluster.local:8000
- XGBoost Predictor: http://xgboost-service.${NAMESPACE}.svc.cluster.local:8001
- Q-Learning Optimizer: http://qlearning-service.${NAMESPACE}.svc.cluster.local:8002
- Anomaly Detector: http://anomaly-service.${NAMESPACE}.svc.cluster.local:8003
- Monitoring Dashboard: http://monitoring-dashboard-service.${NAMESPACE}.svc.cluster.local:8080
- Scheduler Plugin: https://hydatis-scheduler-plugin.${NAMESPACE}.svc.cluster.local:9443

External Services:
- Prometheus: http://10.110.190.83:9090
- MLflow: http://10.110.190.32:31380

Useful Commands:
- Check deployment status: kubectl get all -n ${NAMESPACE}
- View logs: kubectl logs -f deployment/<component-name> -n ${NAMESPACE}
- Scale component: kubectl scale deployment <component-name> --replicas=<count> -n ${NAMESPACE}
- Port forward for local access: kubectl port-forward service/<service-name> <local-port>:<service-port> -n ${NAMESPACE}

Deployment Report: /tmp/hydatis_deployment_report.json

================================================================================

EOF
}

cleanup_deployment() {
    log_warn "Cleaning up existing deployment..."
    
    kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true
    
    log_info "Waiting for namespace cleanup..."
    while kubectl get namespace "${NAMESPACE}" &> /dev/null; do
        sleep 2
    done
    
    log_info "Cleanup completed"
}

main() {
    log_info "Starting HYDATIS ML Scheduler deployment"
    log_info "Deployment target: ${CLUSTER_NAME} cluster"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cleanup)
                cleanup_deployment
                exit 0
                ;;
            --build-only)
                check_prerequisites
                build_container_images
                exit 0
                ;;
            --verify-only)
                verify_deployment
                exit 0
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: $0 [--cleanup|--build-only|--verify-only|--skip-build]"
                exit 1
                ;;
        esac
    done
    
    check_prerequisites
    
    if [ "${SKIP_BUILD:-false}" != "true" ]; then
        build_container_images
    fi
    
    create_namespace
    deploy_infrastructure_components
    run_orchestrated_deployment
    verify_deployment
    generate_access_information
    
    log_info "HYDATIS ML Scheduler deployment completed successfully!"
}

trap 'log_error "Deployment interrupted"; exit 1' INT TERM

main "$@"