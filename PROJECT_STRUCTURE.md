# ML Scheduler Project Structure

## Project Organization by Development Phases

```
ml-scheduler/
├── src/                                    # Core source code
│   ├── ml_models/                         # ML algorithms implementation
│   │   ├── xgboost/                       # Week 5: Load prediction
│   │   │   ├── model.py                   # XGBoost implementation
│   │   │   ├── feature_engineering.py    # Feature preprocessing
│   │   │   ├── training.py               # Training pipeline
│   │   │   └── serving.py                # Model serving logic
│   │   ├── qlearning/                     # Week 6: Placement optimization
│   │   │   ├── environment.py            # K8s cluster simulation
│   │   │   ├── agent.py                  # DQN agent implementation
│   │   │   ├── training.py               # RL training loop
│   │   │   └── serving.py                # Inference serving
│   │   └── isolation_forest/              # Week 7: Anomaly detection
│   │       ├── ensemble.py               # Ensemble model
│   │       ├── detector.py               # Anomaly detection logic
│   │       ├── training.py               # Model training
│   │       └── serving.py                # Real-time detection
│   ├── feature_engineering/               # Week 4: Feature pipeline
│   │   ├── temporal_features.py          # Time-based features
│   │   ├── node_features.py              # Node characterization
│   │   ├── workload_features.py          # Pod/workload features
│   │   └── feature_store.py              # Feast integration
│   ├── data_collection/                   # Week 1-2: Data pipeline
│   │   ├── prometheus_collector.py       # Metrics collection
│   │   ├── data_processor.py             # Data preprocessing
│   │   ├── storage_manager.py            # Data storage logic
│   │   └── quality_monitor.py            # Data validation
│   └── model_serving/                     # Week 10: KServe integration
│       ├── xgboost_server.py             # XGBoost serving
│       ├── qlearning_server.py           # Q-Learning serving
│       ├── anomaly_server.py             # Anomaly detection serving
│       └── combined_scorer.py            # Aggregated scoring
│
├── scheduler-plugin/                       # Week 11: K8s scheduler plugin
│   ├── cmd/                              # Plugin entry point
│   ├── pkg/                              # Core plugin logic
│   │   ├── framework/                    # Scheduler framework integration
│   │   ├── scoring/                      # ML scoring logic
│   │   ├── caching/                      # Redis caching layer
│   │   └── fallback/                     # Standard scheduler fallback
│   ├── configs/                          # Plugin configuration
│   └── manifests/                        # K8s deployment manifests
│
├── kubeflow_pipelines/                    # Week 8-9: MLOps orchestration
│   ├── components/                       # Pipeline components
│   │   ├── data_collection.py            # Data collection component
│   │   ├── feature_engineering.py       # Feature processing component
│   │   ├── model_training.py             # Training components
│   │   ├── model_validation.py           # Validation component
│   │   └── model_deployment.py           # Deployment component
│   ├── pipelines/                        # Complete pipeline definitions
│   │   ├── training_pipeline.py          # End-to-end training
│   │   ├── inference_pipeline.py         # Inference pipeline
│   │   └── retraining_pipeline.py        # Automated retraining
│   └── experiments/                      # Pipeline experiments
│
├── jupyter_notebooks/                     # Development notebooks by phase
│   ├── data_analysis_exploration/         # Weeks 1-4: Data analysis
│   │   ├── cluster_data_exploration.ipynb # Initial cluster data EDA
│   │   ├── scheduling_pattern_discovery.ipynb # Scheduling pattern analysis
│   │   ├── feature_engineering_development.ipynb # Feature development
│   │   └── baseline_performance_metrics.ipynb # Baseline establishment
│   ├── ml_model_development/              # Weeks 5-7: Model development
│   │   ├── load_predictor_xgboost.ipynb   # XGBoost load prediction
│   │   ├── placement_optimizer_qlearning.ipynb # Q-Learning placement optimization
│   │   └── anomaly_detector_isolation_forest.ipynb # Isolation Forest anomaly detection
│   └── experiments/                      # Ad-hoc experiments
│
├── k8s_configs/                          # Kubernetes configurations
│   ├── monitoring/                       # Week 1: Extended monitoring
│   │   ├── prometheus-retention.yaml     # Extended Prometheus config
│   │   ├── additional-scrape.yaml        # Additional metrics collection
│   │   └── grafana-dashboards.yaml       # ML scheduler dashboards
│   ├── ml_services/                      # Week 10: ML service deployments
│   │   ├── xgboost-service.yaml          # XGBoost deployment
│   │   ├── qlearning-service.yaml        # Q-Learning deployment
│   │   └── anomaly-service.yaml          # Anomaly detection deployment
│   └── scheduler/                        # Week 11: Scheduler configuration
│       ├── scheduler-config.yaml         # Plugin configuration
│       └── scheduler-deployment.yaml     # Plugin deployment
│
├── kserve_configs/                       # Week 10: Model serving
│   ├── xgboost-isvc.yaml               # XGBoost InferenceService
│   ├── qlearning-isvc.yaml             # Q-Learning InferenceService
│   ├── anomaly-isvc.yaml               # Anomaly InferenceService
│   └── serving-runtime.yaml            # Custom serving runtime
│
├── katib_experiments/                    # Week 9: Hyperparameter optimization
│   ├── xgboost-hpo.yaml                # XGBoost tuning experiment
│   ├── qlearning-hpo.yaml              # Q-Learning tuning experiment
│   └── isolation-forest-hpo.yaml       # Isolation Forest tuning
│
├── mlflow_configs/                       # MLflow setup and tracking
│   ├── mlflow-deployment.yaml           # MLflow server config
│   ├── experiment-config.py             # Experiment templates
│   └── model-registry.py               # Model registry setup
│
├── feature_store/                        # Week 4: Feature store setup
│   ├── feast/                           # Feast configuration
│   │   ├── feature_repo/                # Feature definitions
│   │   ├── feature_services.py          # Feature serving logic
│   │   └── feast_config.yaml           # Feast configuration
│   └── feature_definitions/             # Feature schemas
│       ├── node_features.py             # Node-related features
│       ├── temporal_features.py         # Time-based features
│       └── workload_features.py         # Workload features
│
├── data_validation/                      # Week 2: Data quality
│   ├── validators/                       # Data validation logic
│   ├── monitors/                        # Data quality monitoring
│   └── schemas/                         # Data schemas
│
├── monitoring/                           # Week 12-14: Production monitoring
│   ├── dashboards/                      # Grafana dashboards
│   │   ├── business_metrics.json        # Business KPIs
│   │   ├── ml_performance.json          # ML model performance
│   │   └── scheduler_metrics.json       # Scheduler performance
│   ├── alerts/                          # Prometheus alerts
│   │   ├── ml_model_alerts.yaml         # Model performance alerts
│   │   ├── business_alerts.yaml         # Business metric alerts
│   │   └── scheduler_alerts.yaml        # Scheduler alerts
│   └── drift_detection/                 # Week 13: Model drift monitoring
│       ├── statistical_drift.py         # Statistical drift detection
│       ├── performance_drift.py         # Performance drift detection
│       └── concept_drift.py             # Concept drift detection
│
├── tests/                               # Testing infrastructure
│   ├── ml_models/                       # ML model tests
│   │   ├── test_xgboost.py             # XGBoost tests
│   │   ├── test_qlearning.py           # Q-Learning tests
│   │   └── test_isolation_forest.py    # Anomaly detection tests
│   ├── scheduler_plugin/                # Scheduler plugin tests
│   │   ├── unit/                       # Unit tests
│   │   └── integration/                # Integration tests
│   ├── integration/                     # End-to-end tests
│   │   ├── test_pipeline.py            # Pipeline integration tests
│   │   └── test_scheduling.py          # Scheduling integration tests
│   └── performance/                     # Week 12: Performance tests
│       ├── load_tests/                 # Load testing
│       ├── latency_tests/              # Latency testing
│       └── stress_tests/               # Stress testing
│
├── scripts/                             # Utility scripts
│   ├── deployment/                      # Week 11-12: Deployment scripts
│   │   ├── deploy_data_infrastructure.sh # Deploy data infrastructure
│   │   ├── deploy_ml_services.sh        # Deploy ML services
│   │   ├── deploy_scheduler_plugin.sh   # Deploy scheduler plugin
│   │   └── rollback_deployment.sh       # Rollback procedures
│   ├── data_processing/                 # Week 1-2: Data processing
│   │   ├── collect_historical_data.py   # Historical data collection
│   │   ├── process_cluster_metrics.py   # Metrics preprocessing
│   │   └── validate_cluster_data.py     # Data validation
│   └── validation/                      # Week 12: Validation scripts
│       ├── validate_business_metrics.py # Business metrics validation
│       ├── validate_ml_performance.py   # ML performance validation
│       └── generate_reports.py         # Automated reporting
│
├── docs/                               # Documentation
│   ├── architecture/                   # Architecture documentation
│   │   ├── ml_algorithms.md            # ML algorithms design
│   │   ├── scheduler_plugin.md         # Plugin architecture
│   │   └── data_pipeline.md           # Data pipeline design
│   ├── runbooks/                       # Week 14: Operational runbooks
│   │   ├── troubleshooting.md          # Troubleshooting guide
│   │   ├── monitoring.md               # Monitoring procedures
│   │   └── incident_response.md        # Incident response
│   └── api/                           # API documentation
│       ├── ml_services_api.md          # ML services API
│       └── scheduler_api.md            # Scheduler plugin API
│
├── configs/                            # Global configuration
│   ├── development.yaml                # Development environment config
│   ├── production.yaml                 # Production environment config
│   └── feature_flags.yaml             # Feature flag configuration
│
├── .claude_code_prompt                 # Claude Code configuration
├── CLAUDE.md                          # Claude Code documentation
├── weekly_implementation_plan.md       # 14-week timeline
├── ml_scheduler_focused_plan.md       # Original project plan
├── requirements.txt                   # Python dependencies
├── go.mod                            # Go module definition
├── Makefile                          # Build automation
└── README.md                         # Project documentation
```

## Directory Usage by Week

### Weeks 1-2: Data Infrastructure
- `src/data_collection/`: Prometheus collectors and processors
- `k8s_configs/monitoring/`: Extended Prometheus configuration
- `scripts/data_processing/`: Data collection and validation scripts
- `data_validation/`: Data quality monitoring

### Weeks 3-4: Data Analysis & Features
- `jupyter_notebooks/data_analysis_exploration/`: Exploratory data analysis
- `src/feature_engineering/`: Feature pipeline development
- `feature_store/feast/`: Feast feature store setup

### Weeks 5-7: ML Model Development
- `src/ml_models/`: Individual algorithm implementations
- `jupyter_notebooks/ml_model_development/`: Model development notebooks
- `mlflow_configs/`: Experiment tracking configuration

### Weeks 8-9: Pipeline & Optimization
- `kubeflow_pipelines/`: End-to-end ML pipeline
- `katib_experiments/`: Hyperparameter optimization
- `tests/integration/`: Pipeline integration testing

### Weeks 10-11: Serving & Plugin
- `src/model_serving/`: KServe model serving
- `kserve_configs/`: Model serving configurations
- `scheduler-plugin/`: Kubernetes scheduler plugin

### Weeks 12-14: Production & Monitoring
- `monitoring/`: Production monitoring setup
- `scripts/deployment/`: Deployment automation
- `docs/runbooks/`: Operational documentation
- `tests/performance/`: Production validation testing