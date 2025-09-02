# HYDATIS ML Scheduler - Deployment Guide

## Prerequisites

### Infrastructure Requirements
- **Kubernetes Cluster:** v1.24+ with 6 nodes minimum
  - 3 master nodes (16 vCPU, 64GB RAM each)
  - 3 worker nodes (16 vCPU, 64GB RAM each)
- **Storage:** Longhorn distributed storage (1TB+ available)
- **Networking:** Pod network with 10.244.0.0/16 CIDR
- **Load Balancer:** Ingress controller for external access

### Software Dependencies
```bash
# Required components
kubectl version --client  # v1.24+
helm version              # v3.8+
docker --version         # v20.10+

# Optional but recommended
istioctl version         # Service mesh (optional)
```

### Environment Variables
```bash
export HYDATIS_NAMESPACE="ml-scheduler"
export PROMETHEUS_URL="http://prometheus:9090"
export GRAFANA_URL="http://grafana:3000"
export MLFLOW_URL="http://mlflow:5000"
export REDIS_URL="redis://redis:6379"
```

## Deployment Phases

### Phase 1: Infrastructure Setup

#### 1.1 Create Namespace and RBAC
```bash
# Create dedicated namespace
kubectl create namespace $HYDATIS_NAMESPACE

# Apply RBAC configuration
kubectl apply -f k8s_configs/rbac/
```

#### 1.2 Deploy Storage Infrastructure
```bash
# Deploy Longhorn storage
kubectl apply -f k8s_configs/storage/longhorn-ml-data-storage.yaml

# Verify storage classes
kubectl get storageclass
```

#### 1.3 Deploy Monitoring Stack
```bash
# Deploy Prometheus with extended retention
kubectl apply -f k8s_configs/monitoring/prometheus-extended-retention.yaml

# Deploy Grafana with ML scheduler dashboards
kubectl apply -f monitoring/dashboards/

# Verify monitoring stack
kubectl get pods -n monitoring
```

### Phase 2: Data Infrastructure

#### 2.1 Deploy Redis Cache
```bash
# Deploy Redis with optimized configuration
kubectl apply -f k8s_configs/ml_services/redis-deployment.yaml

# Verify Redis connectivity
kubectl run redis-test --rm -i --restart=Never --image=redis:7 -- redis-cli -h redis.ml-scheduler ping
```

#### 2.2 Deploy Feature Store (Feast)
```bash
# Apply Feast configuration
kubectl apply -f feature_store/feast/

# Initialize feature store
kubectl run feast-init --rm -i --restart=Never --image=feast-core:latest -- feast apply
```

#### 2.3 Deploy MLflow
```bash
# Deploy MLflow tracking server
kubectl apply -f mlflow_configs/

# Verify MLflow API
curl http://mlflow:5000/api/2.0/mlflow/experiments/list
```

### Phase 3: ML Model Serving

#### 3.1 Deploy KServe Runtime
```bash
# Deploy KServe serving runtime
kubectl apply -f kserve_configs/serving-runtime.yaml

# Verify KServe controller
kubectl get pods -n kserve
```

#### 3.2 Deploy ML Models
```bash
# Deploy XGBoost predictor
kubectl apply -f kserve_configs/xgboost-isvc.yaml

# Deploy Q-Learning optimizer  
kubectl apply -f kserve_configs/qlearning-isvc.yaml

# Deploy Anomaly detector
kubectl apply -f kserve_configs/anomaly-isvc.yaml

# Verify model serving
kubectl get inferenceservices -n $HYDATIS_NAMESPACE
```

#### 3.3 Validate Model Endpoints
```bash
# Test XGBoost endpoint
curl -X POST http://xgboost-predictor.ml-scheduler/v1/models/xgboost:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.0, 2.0, 3.0]]}'

# Test Q-Learning endpoint
curl -X POST http://qlearning-optimizer.ml-scheduler/v1/models/qlearning:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[0.5, 0.8, 0.2]]}'
```

### Phase 4: ML Scheduler Plugin

#### 4.1 Build Scheduler Plugin
```bash
cd scheduler-plugin/

# Build Docker image
make docker-build

# Push to registry (if using external registry)
make docker-push
```

#### 4.2 Deploy Scheduler Plugin
```bash
# Apply scheduler deployment
kubectl apply -f scheduler-plugin/manifests/scheduler-deployment.yaml

# Verify scheduler pods
kubectl get pods -n $HYDATIS_NAMESPACE -l app=ml-scheduler
```

#### 4.3 Configure Default Scheduler
```bash
# Create scheduler configuration
kubectl apply -f config/deployment_config.yaml

# Restart kube-scheduler with ML scheduler profile
# (This step varies by cluster setup)
```

### Phase 5: Advanced Monitoring (Week 13)

#### 5.1 Deploy Week 13 Services
```bash
# Execute comprehensive Week 13 deployment
./scripts/deploy_week13_services.sh

# Verify advanced monitoring services
kubectl get pods -n $HYDATIS_NAMESPACE -l week=13
```

#### 5.2 Configure AIOps Integration
```bash
# Setup Alertmanager webhook
kubectl apply -f config/alertmanager-aiops-config.yaml

# Test webhook endpoint
curl -X POST http://aiops-remediation-service.ml-scheduler:8080/webhook \
  -H "Content-Type: application/json" \
  -d '{"alerts": [{"alertname": "TestAlert", "status": "firing"}]}'
```

## Configuration Management

### Configuration Files Structure
```
config/
├── deployment_config.yaml          # Main scheduler configuration
├── drift_detection_config.yaml     # Model drift thresholds
├── predictive_analytics_config.yaml # Forecasting parameters
├── aiops_config.yaml               # Automated remediation rules
└── anomaly_detection_config.yaml   # Advanced anomaly detection
```

### Environment-Specific Configuration
```yaml
# Production
production:
  replicas:
    ml_scheduler: 3
    models: {xgboost: 4, qlearning: 2, anomaly: 3}
  resources:
    requests: {cpu: high, memory: high}
  monitoring:
    retention: 30d
    alerting: enabled

# Staging  
staging:
  replicas:
    ml_scheduler: 2
    models: {xgboost: 2, qlearning: 1, anomaly: 2}
  resources:
    requests: {cpu: medium, memory: medium}
  monitoring:
    retention: 7d
    alerting: reduced

# Development
development:
  replicas:
    ml_scheduler: 1
    models: {xgboost: 1, qlearning: 1, anomaly: 1}
  resources:
    requests: {cpu: low, memory: low}
  monitoring:
    retention: 1d
    alerting: disabled
```

## Validation and Testing

### Post-Deployment Validation
```bash
# 1. Verify all pods are running
kubectl get pods -n $HYDATIS_NAMESPACE

# 2. Check scheduler is registered
kubectl get schedulers

# 3. Test scheduling with ML scheduler
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-ml-scheduling
  namespace: $HYDATIS_NAMESPACE
spec:
  schedulerName: ml-scheduler
  containers:
  - name: test
    image: nginx:latest
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
EOF

# 4. Verify pod was scheduled by ML scheduler
kubectl describe pod test-ml-scheduling -n $HYDATIS_NAMESPACE | grep "Scheduled"

# 5. Check scheduling metrics
curl http://ml-scheduler-service:8080/metrics | grep ml_scheduler_scheduling
```

### Performance Testing
```bash
# Execute load testing
python tests/performance/load_tests/scheduler_load_test.py \
  --scenario high_load \
  --duration 300 \
  --target-pods 200

# Execute latency testing
python tests/performance/latency_tests/scheduling_latency_test.py \
  --samples 1000 \
  --report-format json

# Execute stress testing
python tests/performance/stress_tests/cluster_stress_test.py \
  --scenario gradual_ramp \
  --max-pods 2000
```

### Business Metrics Validation
```bash
# Validate business targets achievement
python scripts/validation/validate_business_metrics.py \
  --target-cpu 65.0 \
  --target-availability 99.7 \
  --target-roi 1400 \
  --report-format detailed

# Generate ROI report
python scripts/validation/validate_business_metrics.py \
  --roi-report \
  --export-format pdf
```

## Monitoring Setup

### Dashboard Configuration
```bash
# Import business metrics dashboard
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/business_metrics.json

# Import technical performance dashboard
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/ml_performance.json

# Import scheduler operations dashboard
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/scheduler_metrics.json
```

### Alert Configuration
```bash
# Apply ML model alerts
kubectl apply -f monitoring/alerts/ml_model_alerts.yaml

# Apply business alerts
kubectl apply -f monitoring/alerts/business_alerts.yaml

# Verify alert rules
curl http://prometheus:9090/api/v1/rules
```

## Troubleshooting

### Common Issues

#### Scheduler Plugin Not Starting
```bash
# Check scheduler logs
kubectl logs -f deployment/ml-scheduler -n $HYDATIS_NAMESPACE

# Common causes:
# 1. Missing RBAC permissions
# 2. ML model services not ready
# 3. Redis connection issues
# 4. Configuration errors
```

#### ML Models Not Responding
```bash
# Check KServe inference services
kubectl get inferenceservices -n $HYDATIS_NAMESPACE

# Check model service logs
kubectl logs -f deployment/xgboost-predictor -n $HYDATIS_NAMESPACE

# Common fixes:
# 1. Restart inference service
# 2. Check model artifacts in MLflow
# 3. Verify resource allocation
```

#### Performance Issues
```bash
# Check cache hit rates
kubectl exec -it deployment/redis -n $HYDATIS_NAMESPACE -- redis-cli info stats

# Check scheduling latency
curl http://ml-scheduler-service:8080/metrics | grep scheduling_duration

# Common optimizations:
# 1. Increase cache TTL
# 2. Scale ML services
# 3. Optimize feature extraction
```

## Rollback Procedures

### Emergency Rollback to Default Scheduler
```bash
# 1. Stop ML scheduler
kubectl scale deployment ml-scheduler --replicas=0 -n $HYDATIS_NAMESPACE

# 2. Update existing pods to use default scheduler
kubectl patch deployments --all -p '{"spec":{"template":{"spec":{"schedulerName":"default-scheduler"}}}}'

# 3. Verify default scheduling
kubectl get events --field-selector reason=Scheduled
```

### Progressive Rollback
```bash
# Use progressive rollout system for gradual rollback
python src/deployment/progressive_rollout.py \
  --rollout-id emergency-rollback \
  --source-scheduler ml-scheduler \
  --target-scheduler default-scheduler \
  --phases "50,25,0"
```

## Maintenance

### Regular Maintenance Tasks
```bash
# Daily: Check system health
./scripts/health_check.sh

# Weekly: Model retraining validation
python scripts/validation/validate_model_performance.py

# Monthly: Business metrics review
python scripts/validation/validate_business_metrics.py --monthly-report

# Quarterly: Full system audit
./scripts/audit/comprehensive_audit.sh
```

### Update Procedures
```bash
# Update ML models
kubectl patch inferenceservice xgboost-predictor -n $HYDATIS_NAMESPACE \
  --type merge -p '{"spec":{"predictor":{"model":{"modelUri":"models:/xgboost/v2.1"}}}}'

# Update scheduler configuration
kubectl patch configmap ml-scheduler-config -n $HYDATIS_NAMESPACE \
  --patch-file config/updates/scheduler-config-patch.yaml

# Rolling update scheduler plugin
kubectl set image deployment/ml-scheduler ml-scheduler=ml-scheduler:v1.2 -n $HYDATIS_NAMESPACE
```

## Security Considerations

### Network Security
- All inter-service communication over TLS
- Network policies restrict unnecessary traffic
- Ingress controls for external access

### Data Security
- No sensitive data in logs or metrics
- Encrypted storage for model artifacts
- Regular security scanning of container images

### Access Control
- Role-based access control (RBAC)
- Service account per component
- Audit logging for all administrative actions

---

**Document Version:** 1.0  
**Deployment Target:** HYDATIS Production Cluster  
**Review Date:** Weekly during initial deployment, monthly thereafter