# Technical Training Materials - HYDATIS ML Scheduler

## Core Technology Stack Training

### Kubernetes Scheduler Framework

#### Concepts Review
```yaml
# Scheduler Configuration Example
apiVersion: kubescheduler.config.k8s.io/v1beta3
kind: KubeSchedulerConfiguration
profiles:
- schedulerName: ml-scheduler
  plugins:
    score:
      enabled:
      - name: MLResourceOptimizer
    filter:
      enabled:
      - name: MLNodeSelector
```

#### Extension Points
- **PreFilter:** Early pod filtering based on ML predictions
- **Filter:** Node eligibility based on resource predictions
- **Score:** ML-powered node scoring and ranking
- **Bind:** Custom binding with optimization logging

### Machine Learning Pipeline

#### XGBoost Load Predictor
```python
# Feature Engineering Pipeline
features = [
    'node_cpu_utilization', 'node_memory_utilization',
    'pod_cpu_request', 'pod_memory_request',
    'historical_placement_success', 'node_network_latency'
]

# Model Training
model = XGBoostRegressor(
    n_estimators=500,
    max_depth=12,
    learning_rate=0.1,
    subsample=0.8
)
model.fit(X_train, y_train)

# Prediction Integration
predicted_load = model.predict(node_features)
scheduling_score = 1.0 / (1.0 + predicted_load)
```

#### Q-Learning Placement Optimizer
```python
# Reward Function Design
def calculate_reward(placement_outcome):
    resource_efficiency = 1.0 - (actual_utilization / predicted_utilization)
    latency_penalty = -max(0, (actual_latency - target_latency) / target_latency)
    availability_bonus = 1.0 if availability > 0.997 else 0.0
    
    return resource_efficiency + latency_penalty + availability_bonus

# Q-Learning Update
Q[state][action] = Q[state][action] + alpha * (
    reward + gamma * max(Q[next_state]) - Q[state][action]
)
```

#### Anomaly Detection System
```python
# Multi-dimensional Anomaly Detection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Statistical Anomaly Detection
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
detector = IsolationForest(contamination=0.1, random_state=42)
anomaly_scores = detector.decision_function(features_normalized)

# Contextual Anomaly Analysis
def analyze_anomaly_context(anomaly_data):
    temporal_context = extract_temporal_patterns(anomaly_data)
    resource_context = analyze_resource_correlations(anomaly_data)
    business_context = assess_business_impact(anomaly_data)
    return AnomalyContext(temporal_context, resource_context, business_context)
```

## Practical Exercises

### Exercise 1: End-to-End Deployment
**Objective:** Deploy complete ML scheduler system from scratch

**Prerequisites:**
- Kubernetes cluster access (minikube or development cluster)
- Python 3.9+ with required dependencies
- Access to HYDATIS configuration files

**Steps:**
```bash
# 1. Setup development environment
git clone <repository-url>
cd ml-scheduler
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Deploy infrastructure components
kubectl apply -f configs/namespace.yaml
kubectl apply -f configs/rbac/
kubectl apply -f configs/storage/

# 3. Deploy ML models
kubectl apply -f configs/model-serving/
python scripts/model_management/deploy_models.py --environment dev

# 4. Deploy scheduler plugin
kubectl apply -f configs/scheduler/
kubectl apply -f configs/monitoring/

# 5. Validate deployment
python scripts/validation/validate_deployment.py --comprehensive
./scripts/validation/test_scheduling.py
```

### Exercise 2: Model Training and Optimization
**Objective:** Train and optimize ML models for scheduling decisions

```python
# 1. Data collection and preprocessing
from src.data.collectors.cluster_metrics import ClusterMetricsCollector
collector = ClusterMetricsCollector()
training_data = collector.collect_historical_data(days=30)

# 2. Feature engineering
from src.ml.features.feature_engineer import SchedulingFeatureEngineer
engineer = SchedulingFeatureEngineer()
features, labels = engineer.prepare_training_data(training_data)

# 3. Model training with hyperparameter optimization
from src.ml.training.hyperparameter_optimizer import HyperparameterOptimizer
optimizer = HyperparameterOptimizer()
best_params = optimizer.optimize_xgboost(features, labels, cv_folds=5)

# 4. Model validation and deployment
model = XGBoostPredictor(**best_params)
model.fit(features, labels)
validation_metrics = model.validate(test_features, test_labels)
model.deploy_to_production()
```

### Exercise 3: Performance Tuning
**Objective:** Optimize system performance for HYDATIS targets

```bash
# 1. Baseline performance measurement
python scripts/performance/benchmark_scheduler.py --baseline

# 2. Load testing with different scenarios
python scripts/performance/load_tests/scheduler_load_test.py --scenario high_load
python scripts/performance/load_tests/latency_test.py --target-latency 120ms

# 3. Resource optimization
kubectl top nodes
python scripts/optimization/resource_analyzer.py --target-cpu 65

# 4. Business metrics validation
python scripts/validation/validate_business_metrics.py --target-availability 99.7
```

### Exercise 4: Troubleshooting Scenarios
**Objective:** Diagnose and resolve common system issues

**Scenario 1: High CPU Utilization**
```bash
# Problem: CPU utilization above 75% for >30 minutes
# Investigation:
kubectl top nodes
kubectl describe nodes | grep -E "(Allocatable|Allocated resources)"
python scripts/analytics/resource_analysis.py --high-cpu-investigation

# Resolution:
kubectl patch configmap ml-scheduler-config -n $HYDATIS_NAMESPACE \
  --patch '{"data":{"aggressiveness":"0.3"}}'
kubectl rollout restart deployment/ml-scheduler -n $HYDATIS_NAMESPACE
```

**Scenario 2: Model Performance Degradation**
```python
# Problem: Scheduling accuracy dropping below 90%
# Investigation:
from src.monitoring.drift_detection import DriftDetector
detector = DriftDetector()
drift_report = detector.analyze_recent_performance(days=7)

# Resolution:
if drift_report.requires_retraining:
    from src.ml.training.auto_trainer import AutoTrainer
    trainer = AutoTrainer()
    trainer.trigger_retraining('xgboost-predictor')
```

## Development Workflows

### Code Contribution Process

#### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/scheduling-optimization

# Development workflow
python -m pytest tests/unit/
python scripts/validation/code_quality.py
python scripts/security/security_scan.py

# Integration testing
./scripts/testing/integration_test.sh
python scripts/validation/validate_business_metrics.py
```

#### 2. Code Review Guidelines
- **Performance Impact:** Validate scheduling latency <120ms
- **Resource Efficiency:** Ensure CPU utilization targets maintained
- **Business Metrics:** Verify availability and cost optimization
- **Security Compliance:** RBAC, network policies, secret management
- **Documentation:** Update relevant guides and runbooks

#### 3. Deployment Process
```bash
# Staging deployment
./scripts/deployment/production_deployment.sh --environment staging
python scripts/validation/staging_validation.py

# Production deployment (progressive)
./scripts/deployment/production_deployment.sh --environment production
# Automatic progression: 10% → 50% → 100% with validation gates
```

### Testing Strategies

#### Unit Testing
```python
# Test ML model components
pytest tests/unit/ml/test_xgboost_predictor.py
pytest tests/unit/ml/test_qlearning_optimizer.py
pytest tests/unit/ml/test_anomaly_detector.py

# Test scheduler logic
pytest tests/unit/scheduler/test_scoring_algorithm.py
pytest tests/unit/scheduler/test_node_selection.py
```

#### Integration Testing
```bash
# End-to-end scheduler testing
python tests/integration/test_complete_scheduling_flow.py

# Business metrics integration
python tests/integration/test_business_metrics_collection.py

# Model serving integration
python tests/integration/test_model_serving_integration.py
```

#### Performance Testing
```bash
# Load testing scenarios
python tests/performance/load_tests/scheduler_load_test.py --scenario burst_load
python tests/performance/load_tests/latency_test.py --duration 3600

# Business validation testing
python tests/performance/business_validation/test_cost_optimization.py
python tests/performance/business_validation/test_availability_targets.py
```

## Troubleshooting Workshop

### Common Issues and Solutions

#### Issue 1: Scheduler Plugin Not Loading
**Symptoms:**
```bash
kubectl logs deployment/ml-scheduler -n $HYDATIS_NAMESPACE
# Error: failed to create scheduler: plugin "MLResourceOptimizer" not found
```

**Diagnosis:**
```bash
# Check plugin registration
kubectl describe configmap ml-scheduler-config -n $HYDATIS_NAMESPACE

# Verify plugin binary
kubectl exec -it deployment/ml-scheduler -n $HYDATIS_NAMESPACE -- \
  ls -la /usr/local/bin/scheduler-plugins
```

**Resolution:**
```bash
# Rebuild and redeploy plugin
./scripts/build/build_scheduler_plugin.sh
kubectl apply -f configs/scheduler/ml-scheduler-deployment.yaml
kubectl rollout status deployment/ml-scheduler -n $HYDATIS_NAMESPACE
```

#### Issue 2: Model Serving Timeouts
**Symptoms:**
```bash
curl http://xgboost-predictor.ml-scheduler/v1/models/xgboost:predict -m 5
# Error: timeout after 5 seconds
```

**Diagnosis:**
```python
# Check model serving health
from src.monitoring.model_health import ModelHealthChecker
checker = ModelHealthChecker()
health_status = checker.check_model_health('xgboost-predictor')
print(f"Model health: {health_status}")

# Analyze model performance
python scripts/analytics/model_performance_analysis.py --model xgboost-predictor
```

**Resolution:**
```bash
# Scale model serving replicas
kubectl scale deployment xgboost-predictor --replicas=5 -n $HYDATIS_NAMESPACE

# Optimize model configuration
kubectl patch inferenceservice xgboost-predictor -n $HYDATIS_NAMESPACE \
  --patch '{"spec":{"predictor":{"minReplicas":3,"maxReplicas":10}}}'
```

### Debugging Tools

#### Log Analysis Tools
```bash
# Centralized logging with filtering
kubectl logs -f deployment/ml-scheduler -n $HYDATIS_NAMESPACE | grep -E "(ERROR|WARN)"

# Business metrics logging
python scripts/monitoring/log_analyzer.py --business-metrics --last-24h

# Performance profiling
python scripts/debugging/performance_profiler.py --component scheduler
```

#### Monitoring and Alerting
```bash
# Real-time metrics monitoring
watch 'kubectl top nodes && kubectl top pods -n $HYDATIS_NAMESPACE'

# Business dashboard access
curl -s http://prometheus:9090/api/v1/query?query=ml_scheduler_business_roi

# Alert testing
python scripts/monitoring/test_alerts.py --alert high_cpu_utilization
```

## Certification Program

### Prerequisites
- Kubernetes fundamentals (CKA recommended)
- Python programming (intermediate level)
- Machine learning basics (understanding of ML algorithms)
- Linux command line proficiency

### Certification Process
1. **Self-Study:** Complete relevant track materials (1-2 weeks)
2. **Hands-on Labs:** Practice exercises in development environment (3-5 days)
3. **Practical Assessment:** Live demonstration of skills (4-6 hours)
4. **Written Exam:** Technical knowledge validation (2 hours)
5. **Peer Review:** Code review and technical discussion (1 hour)

### Certification Maintenance
- **Annual Renewal:** Required to maintain certification status
- **Continuing Education:** 20 hours annually of relevant training
- **Practical Validation:** Annual hands-on assessment
- **Peer Teaching:** Contribute to training of new team members

### Resources and Support
- **Technical Mentorship:** Senior team member assignment
- **Office Hours:** Weekly Q&A sessions with ML engineering team
- **Community Forum:** Internal knowledge sharing platform
- **External Training:** Conference attendance and certification support

---

**Training Materials Version:** 1.0  
**Created By:** ML Engineering Team  
**Technical Review:** Senior ML Engineers, Platform Architects  
**Instructional Design:** Training & Development Team  
**Update Schedule:** Quarterly with technology evolution