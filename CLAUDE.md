# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ML-powered Kubernetes scheduler project that replaces the standard Kubernetes scheduler with intelligent placement decisions using machine learning. The system analyzes 30+ days of historical cluster data to optimize pod placement using three ML algorithms:

- **XGBoost Load Predictor**: Predicts future CPU/Memory load (target: 89% CPU accuracy, 86% Memory accuracy)
- **Q-Learning Placement Optimizer**: Optimizes pod placement decisions (target: +34% improvement vs random placement)
- **Isolation Forest Anomaly Detector**: Detects problematic nodes (target: 94% precision, ≤8% false positives)

The project uses Charmed Kubeflow for complete MLOps pipeline with 14-week development timeline targeting 99.7% cluster availability and 65% average CPU utilization.

## Development Environment Setup

### Cluster Environment (HYDATIS)
- **Control Plane**: https://10.110.190.31:6443
- **6-node cluster**: 3 masters (10.110.190.32-34) + 3 workers (10.110.190.35-37)
- **Resources per node**: 8 CPU cores, 16GB RAM
- **Current utilization**: Workers 8-13% CPU, 36-43% Memory
- **Kubeflow already operational**: MLflow, Katib, KServe, Jupyter running

### Access Existing Infrastructure
```bash
# Access Kubeflow dashboard (already deployed)
kubectl port-forward -n kubeflow svc/kubeflow-dashboard 8080:80

# Access MLflow (already running on worker3)
kubectl port-forward -n kubeflow svc/mlflow-server 5000:5000

# Access Jupyter notebooks (operational)
kubectl get notebooks -n kubeflow

# Check Longhorn storage (distributed storage ready)
kubectl get pv,pvc -A

# Monitor cluster metrics
kubectl top nodes
```

### Development Setup
```bash
# Install Python ML dependencies
pip install scikit-learn xgboost torch mlflow feast prometheus-client kubernetes

# Setup Go environment for scheduler plugin
go mod init ml-scheduler-plugin
go get k8s.io/kubernetes@latest

# Connect to existing Feast feature store
feast init --template minimal
feast apply
```

### MLOps Pipeline Commands
```bash
# Run Kubeflow pipeline for model training
kfp run create --experiment-name ml-scheduler --pipeline-id <pipeline-id>

# Model serving with KServe
kubectl apply -f kserve-configs/

# Hyperparameter tuning with Katib
kubectl apply -f katib-experiments/

# Monitor model performance
mlflow ui
```

### Kubernetes Scheduler Plugin Development
```bash
# Build scheduler plugin (Go)
make build-scheduler

# Deploy scheduler plugin
kubectl apply -f scheduler-plugin-config.yaml

# Test scheduler decisions
kubectl apply -f test-pods/

# Monitor scheduling latency
kubectl get pods -o wide --watch
```

### Testing and Validation
```bash
# Run ML model tests
pytest tests/ml_models/

# Test scheduler plugin
go test ./scheduler-plugin/...

# Performance testing
k6 run performance-tests/scheduling-load-test.js

# Business metrics validation
python scripts/validate_business_metrics.py
```

## Architecture Structure

### ML Components
- `src/ml_models/xgboost/`: Load prediction models with feature engineering
- `src/ml_models/qlearning/`: Reinforcement learning placement optimizer  
- `src/ml_models/isolation_forest/`: Anomaly detection for node health
- `src/feature_engineering/`: Feast feature store integration and pipelines
- `src/model_serving/`: KServe serving configurations and custom runtimes

### Kubeflow MLOps Pipeline
- `kubeflow_pipelines/`: End-to-end ML pipeline definitions
- `jupyter_notebooks/data_analysis_exploration/`: EDA and feature development
- `jupyter_notebooks/ml_model_development/`: Model development and experimentation
- `mlflow_configs/`: Experiment tracking and model registry setup
- `katib_experiments/`: Hyperparameter optimization configurations

### Kubernetes Integration
- `scheduler-plugin/`: Go-based Kubernetes scheduler plugin
- `k8s_configs/`: Deployment configurations for ML services
- `monitoring/`: Prometheus metrics and Grafana dashboards
- `kserve-configs/`: Model serving deployment configurations

### Data Pipeline
- `data_collection/`: Prometheus metrics collection and processing
- `feature_store/`: Feast feature definitions and serving layer
- `data_validation/`: Data quality checks and drift detection

## 14-Week Implementation Timeline

### Weeks 1-2: Data Infrastructure & Collection
- Extend Prometheus retention to 30+ days
- Create data collection pipeline (metrics every 30s)
- Setup advanced Jupyter environment for ML development
- Implement data quality monitoring >95% success rate

### Weeks 3-4: Data Analysis & Feature Engineering
- EDA notebooks: temporal patterns, node correlations
- Engineer 50+ features: rolling windows, seasonal patterns, node health
- Implement Feast feature store with <50ms serving latency
- Establish baseline performance metrics

### Weeks 5-7: ML Algorithm Development
- **Week 5**: XGBoost load predictor with MLflow tracking
- **Week 6**: Q-Learning DQN agent with PyTorch  
- **Week 7**: Isolation Forest ensemble for anomaly detection
- Each model: 30+ MLflow experiments, production-ready artifacts

### Weeks 8-9: Pipeline Orchestration & Optimization
- Kubeflow Pipelines for end-to-end ML workflow (<2h execution)
- Katib hyperparameter optimization (330+ total experiments)
- Automated model validation and registry integration

### Weeks 10-11: Production Serving & Plugin Integration
- KServe model serving with auto-scaling (2-10 replicas)
- Go Kubernetes scheduler plugin with ML service integration
- Shadow mode testing and validation (48h+)

### Weeks 12-14: Production Deployment & Monitoring
- Progressive rollout: 10% → 50% → 100% traffic
- Advanced monitoring: drift detection, business metrics
- Continuous learning pipeline automation
- Team knowledge transfer and documentation

## Key Technical Targets

### ML Performance Targets
- XGBoost accuracy: ≥89% CPU prediction, ≥86% Memory prediction
- Q-Learning improvement: ≥+34% vs random placement baseline
- Isolation Forest: ≥94% precision, ≤8% false positive rate
- Scheduling latency: <100ms P99 for placement decisions
- Service availability: ≥99.9% uptime for ML services

### Business Impact Targets (HYDATIS Transformation)
- Cluster CPU utilization: 85% → 65% average (-20% optimization)
- Cluster availability: 95.2% → 99.7% (+4.5% improvement)
- Capacity improvement: 15x concurrent projects capability
- Application performance: +40% latency improvement
- Incident reduction: -80% placement-related failures
- ROI target: 1,428% validated over 12 months

## Development Workflow

### Model Development
1. Use Jupyter notebooks in Kubeflow for experimentation
2. Track all experiments with MLflow
3. Use Katib for hyperparameter optimization
4. Deploy models with KServe for production serving

### Scheduler Plugin Development
1. Develop in Go following Kubernetes scheduler framework
2. Integrate with KServe ML services via HTTP clients
3. Implement fallback to standard scheduler if ML services unavailable
4. Test in shadow mode before production deployment

### Continuous Integration
1. Daily model retraining with new cluster data
2. Drift detection triggers automated retraining
3. A/B testing for new model versions
4. Gradual rollout: 10% → 50% → 100% traffic

## Production Considerations

- Redis caching for frequent scheduling decisions
- Circuit breaker patterns for ML service resilience  
- High availability deployment across multiple master nodes
- Comprehensive monitoring with business and technical dashboards
- Automated remediation for common operational issues