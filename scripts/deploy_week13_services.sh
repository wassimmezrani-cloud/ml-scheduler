#!/bin/bash

# Week 13 - Advanced Monitoring & AIOps Deployment Script
# Deploys drift detection, predictive analytics, AIOps, and advanced anomaly detection

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="ml-scheduler"
CONFIG_DIR="/etc/ml-scheduler"
WEEK13_COMPONENTS=(
    "drift-detection"
    "predictive-analytics" 
    "aiops-remediation"
    "advanced-anomaly-detection"
)

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check if kubectl is available and cluster is accessible
    if ! kubectl cluster-info &>/dev/null; then
        error "kubectl not available or cluster not accessible"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace $NAMESPACE &>/dev/null; then
        log "Creating namespace $NAMESPACE"
        kubectl create namespace $NAMESPACE
    fi
    
    # Check if required directories exist
    sudo mkdir -p $CONFIG_DIR
    
    success "Prerequisites check completed"
}

# Deploy configuration files
deploy_configurations() {
    log "Deploying Week 13 configuration files..."
    
    # Copy configuration files to system location
    sudo cp config/drift_detection_config.yaml $CONFIG_DIR/
    sudo cp config/predictive_analytics_config.yaml $CONFIG_DIR/
    sudo cp config/aiops_config.yaml $CONFIG_DIR/
    sudo cp config/anomaly_detection_config.yaml $CONFIG_DIR/
    
    # Create Kubernetes ConfigMaps
    kubectl create configmap drift-detection-config \
        --from-file=$CONFIG_DIR/drift_detection_config.yaml \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create configmap predictive-analytics-config \
        --from-file=$CONFIG_DIR/predictive_analytics_config.yaml \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create configmap aiops-config \
        --from-file=$CONFIG_DIR/aiops_config.yaml \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create configmap anomaly-detection-config \
        --from-file=$CONFIG_DIR/anomaly_detection_config.yaml \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    success "Configuration files deployed"
}

# Create Kubernetes deployments
create_deployments() {
    log "Creating Week 13 service deployments..."
    
    # Drift Detection Service
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: drift-detection-service
  namespace: $NAMESPACE
  labels:
    app: drift-detection
    component: monitoring
    week: "13"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: drift-detection
  template:
    metadata:
      labels:
        app: drift-detection
        component: monitoring
    spec:
      containers:
      - name: drift-detection
        image: python:3.9-slim
        command: ["python", "/app/drift_detection.py"]
        env:
        - name: PROMETHEUS_URL
          value: "http://prometheus:9090"
        - name: KSERVE_URL
          value: "http://kserve-controller:8080"
        - name: MLFLOW_URL
          value: "http://mlflow:5000"
        ports:
        - containerPort: 8080
          name: metrics
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: config
          mountPath: /etc/ml-scheduler
        resources:
          requests:
            memory: "512Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: app-code
        configMap:
          name: drift-detection-code
      - name: config
        configMap:
          name: drift-detection-config
---
apiVersion: v1
kind: Service
metadata:
  name: drift-detection-service
  namespace: $NAMESPACE
spec:
  selector:
    app: drift-detection
  ports:
  - port: 8080
    targetPort: 8080
    name: metrics
EOF

    # Predictive Analytics Service
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: predictive-analytics-service
  namespace: $NAMESPACE
  labels:
    app: predictive-analytics
    component: analytics
    week: "13"
spec:
  replicas: 1
  selector:
    matchLabels:
      app: predictive-analytics
  template:
    metadata:
      labels:
        app: predictive-analytics
        component: analytics
    spec:
      containers:
      - name: predictive-analytics
        image: python:3.9-slim
        command: ["python", "/app/predictive_engine.py"]
        env:
        - name: PROMETHEUS_URL
          value: "http://prometheus:9090"
        ports:
        - containerPort: 8080
          name: metrics
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: config
          mountPath: /etc/ml-scheduler
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
      volumes:
      - name: app-code
        configMap:
          name: predictive-analytics-code
      - name: config
        configMap:
          name: predictive-analytics-config
---
apiVersion: v1
kind: Service
metadata:
  name: predictive-analytics-service
  namespace: $NAMESPACE
spec:
  selector:
    app: predictive-analytics
  ports:
  - port: 8080
    targetPort: 8080
    name: metrics
EOF

    # AIOps Remediation Service
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aiops-remediation-service
  namespace: $NAMESPACE
  labels:
    app: aiops-remediation
    component: automation
    week: "13"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: aiops-remediation
  template:
    metadata:
      labels:
        app: aiops-remediation
        component: automation
    spec:
      serviceAccountName: aiops-service-account
      containers:
      - name: aiops-remediation
        image: python:3.9-slim
        command: ["python", "/app/automated_remediation.py"]
        env:
        - name: PROMETHEUS_URL
          value: "http://prometheus:9090"
        - name: KSERVE_URL
          value: "http://kserve-controller:8080"
        - name: REDIS_URL
          value: "redis://redis:6379"
        ports:
        - containerPort: 8080
          name: webhook
        - containerPort: 9090
          name: metrics
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: config
          mountPath: /etc/ml-scheduler
        resources:
          requests:
            memory: "512Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: app-code
        configMap:
          name: aiops-remediation-code
      - name: config
        configMap:
          name: aiops-config
---
apiVersion: v1
kind: Service
metadata:
  name: aiops-remediation-service
  namespace: $NAMESPACE
spec:
  selector:
    app: aiops-remediation
  ports:
  - port: 8080
    targetPort: 8080
    name: webhook
  - port: 9090
    targetPort: 9090
    name: metrics
EOF

    # Advanced Anomaly Detection Service
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: advanced-anomaly-detection
  namespace: $NAMESPACE
  labels:
    app: advanced-anomaly-detection
    component: monitoring
    week: "13"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: advanced-anomaly-detection
  template:
    metadata:
      labels:
        app: advanced-anomaly-detection
        component: monitoring
    spec:
      containers:
      - name: anomaly-detection
        image: python:3.9-slim
        command: ["python", "/app/advanced_anomaly_detection.py"]
        env:
        - name: PROMETHEUS_URL
          value: "http://prometheus:9090"
        ports:
        - containerPort: 8080
          name: metrics
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: config
          mountPath: /etc/ml-scheduler
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
      volumes:
      - name: app-code
        configMap:
          name: advanced-anomaly-detection-code
      - name: config
        configMap:
          name: anomaly-detection-config
---
apiVersion: v1
kind: Service
metadata:
  name: advanced-anomaly-detection-service
  namespace: $NAMESPACE
spec:
  selector:
    app: advanced-anomaly-detection
  ports:
  - port: 8080
    targetPort: 8080
    name: metrics
EOF

    success "Kubernetes deployments created"
}

# Create service account and RBAC for AIOps
create_rbac() {
    log "Creating RBAC for AIOps automation..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: aiops-service-account
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: aiops-automation-role
rules:
- apiGroups: [""]
  resources: ["pods", "nodes", "configmaps", "services"]
  verbs: ["get", "list", "watch", "patch", "update"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "patch", "update", "scale"]
- apiGroups: ["autoscaling"]
  resources: ["horizontalpodautoscalers"]
  verbs: ["get", "list", "watch", "patch", "update"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["*"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: aiops-automation-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: aiops-automation-role
subjects:
- kind: ServiceAccount
  name: aiops-service-account
  namespace: $NAMESPACE
EOF

    success "RBAC configuration created"
}

# Create code ConfigMaps
create_code_configmaps() {
    log "Creating code ConfigMaps..."
    
    # Drift Detection code
    kubectl create configmap drift-detection-code \
        --from-file=src/monitoring/drift_detection.py \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Predictive Analytics code
    kubectl create configmap predictive-analytics-code \
        --from-file=src/analytics/predictive_engine.py \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # AIOps Remediation code
    kubectl create configmap aiops-remediation-code \
        --from-file=src/aiops/automated_remediation.py \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Advanced Anomaly Detection code
    kubectl create configmap advanced-anomaly-detection-code \
        --from-file=src/monitoring/advanced_anomaly_detection.py \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    success "Code ConfigMaps created"
}

# Configure Alertmanager webhook for AIOps
configure_alertmanager_webhook() {
    log "Configuring Alertmanager webhook for AIOps..."
    
    # Add AIOps webhook to Alertmanager configuration
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-aiops-config
  namespace: monitoring
data:
  alertmanager.yml: |
    global:
      smtp_smarthost: 'localhost:587'
      smtp_from: 'alertmanager@hydatis.com'
    
    route:
      group_by: ['alertname']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'web.hook'
      routes:
      - match:
          severity: critical
        receiver: 'aiops-webhook'
      - match:
          component: ml_scheduler
        receiver: 'aiops-webhook'
    
    receivers:
    - name: 'web.hook'
      webhook_configs:
      - url: 'http://aiops-remediation-service.ml-scheduler:8080/webhook'
        send_resolved: true
        
    - name: 'aiops-webhook'
      webhook_configs:
      - url: 'http://aiops-remediation-service.ml-scheduler:8080/webhook'
        send_resolved: true
        http_config:
          timeout: 30s
EOF

    success "Alertmanager webhook configured"
}

# Create monitoring ServiceMonitors
create_service_monitors() {
    log "Creating Prometheus ServiceMonitors for Week 13 services..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: week13-services-monitoring
  namespace: $NAMESPACE
  labels:
    app: ml-scheduler
    week: "13"
spec:
  selector:
    matchLabels:
      component: monitoring
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: aiops-monitoring
  namespace: $NAMESPACE
  labels:
    app: aiops-remediation
    week: "13"
spec:
  selector:
    matchLabels:
      app: aiops-remediation
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
EOF

    success "ServiceMonitors created"
}

# Validate deployment
validate_deployment() {
    log "Validating Week 13 deployment..."
    
    # Check if all deployments are ready
    for component in "${WEEK13_COMPONENTS[@]}"; do
        deployment_name="${component}-service"
        if [[ "$component" == "advanced-anomaly-detection" ]]; then
            deployment_name="advanced-anomaly-detection"
        fi
        
        log "Checking deployment: $deployment_name"
        
        # Wait for deployment to be ready
        if kubectl wait --for=condition=available --timeout=300s deployment/$deployment_name -n $NAMESPACE; then
            success "$deployment_name is ready"
        else
            error "$deployment_name failed to become ready"
            kubectl describe deployment $deployment_name -n $NAMESPACE
            return 1
        fi
    done
    
    # Verify services are accessible
    log "Verifying service accessibility..."
    
    # Test drift detection service
    if kubectl run test-drift --rm -i --restart=Never --image=curlimages/curl -- \
        curl -s http://drift-detection-service.ml-scheduler:8080/health; then
        success "Drift detection service accessible"
    else
        warning "Drift detection service health check failed"
    fi
    
    # Test predictive analytics service  
    if kubectl run test-analytics --rm -i --restart=Never --image=curlimages/curl -- \
        curl -s http://predictive-analytics-service.ml-scheduler:8080/metrics; then
        success "Predictive analytics service accessible"
    else
        warning "Predictive analytics service health check failed"
    fi
    
    # Test AIOps webhook endpoint
    if kubectl run test-aiops --rm -i --restart=Never --image=curlimages/curl -- \
        curl -s http://aiops-remediation-service.ml-scheduler:8080/health; then
        success "AIOps remediation service accessible"
    else
        warning "AIOps remediation service health check failed"
    fi
    
    success "Deployment validation completed"
}

# Setup monitoring dashboards for Week 13
setup_week13_dashboards() {
    log "Setting up Week 13 monitoring dashboards..."
    
    # Create AIOps effectiveness dashboard
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: aiops-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  aiops-effectiveness.json: |
    {
      "dashboard": {
        "title": "Week 13 - AIOps Effectiveness Dashboard",
        "tags": ["week13", "aiops", "automation"],
        "refresh": "30s",
        "panels": [
          {
            "title": "Automated Remediation Success Rate",
            "type": "stat",
            "targets": [{
              "expr": "aiops_remediation_success_rate",
              "legendFormat": "Success Rate"
            }],
            "fieldConfig": {
              "defaults": {
                "unit": "percent",
                "thresholds": {
                  "steps": [
                    {"color": "red", "value": null},
                    {"color": "yellow", "value": 70},
                    {"color": "green", "value": 90}
                  ]
                }
              }
            }
          },
          {
            "title": "Mean Time to Resolution",
            "type": "stat", 
            "targets": [{
              "expr": "aiops_mean_time_to_resolution_seconds",
              "legendFormat": "MTTR"
            }],
            "fieldConfig": {
              "defaults": {
                "unit": "s",
                "thresholds": {
                  "steps": [
                    {"color": "green", "value": null},
                    {"color": "yellow", "value": 300},
                    {"color": "red", "value": 600}
                  ]
                }
              }
            }
          },
          {
            "title": "Anomaly Detection Confidence",
            "type": "timeseries",
            "targets": [{
              "expr": "anomaly_detection_confidence_score",
              "legendFormat": "{{anomaly_type}}"
            }],
            "fieldConfig": {
              "defaults": {
                "unit": "percent"
              }
            }
          }
        ]
      }
    }
EOF

    success "Week 13 dashboards configured"
}

# Main deployment function
main() {
    log "Starting Week 13 - Advanced Monitoring & AIOps deployment"
    
    check_prerequisites
    deploy_configurations
    create_rbac
    create_code_configmaps
    create_deployments
    configure_alertmanager_webhook
    create_service_monitors
    setup_week13_dashboards
    validate_deployment
    
    success "Week 13 deployment completed successfully!"
    
    log "Week 13 Services Deployed:"
    log "  ✅ Model Drift Detection - Monitors ML model performance degradation"
    log "  ✅ Predictive Analytics Engine - Forecasts capacity, cost, and performance"
    log "  ✅ AIOps Automated Remediation - Intelligent incident response"
    log "  ✅ Advanced Anomaly Detection - Multi-dimensional anomaly detection with RCA"
    
    log ""
    log "Next Steps:"
    log "  1. Monitor service health: kubectl get pods -n $NAMESPACE"
    log "  2. View AIOps webhook logs: kubectl logs -f deployment/aiops-remediation-service -n $NAMESPACE"
    log "  3. Check anomaly detection: kubectl logs -f deployment/advanced-anomaly-detection -n $NAMESPACE"
    log "  4. Access dashboards at: http://grafana:3000"
    log ""
    log "Week 14 (Final): Documentation & Handover"
}

# Execute main function
main "$@"