# ML Scheduler Project Makefile

.PHONY: help setup build test deploy clean

help:
	@echo "ML Scheduler Development Commands"
	@echo "================================="
	@echo "setup          - Setup development environment"
	@echo "build          - Build all components"
	@echo "test           - Run all tests"
	@echo "test-ml        - Run ML model tests"
	@echo "test-plugin    - Run scheduler plugin tests"
	@echo "deploy-dev     - Deploy to development environment"
	@echo "deploy-prod    - Deploy to production environment"
	@echo "clean          - Clean build artifacts"

# Development environment setup
setup:
	pip install -r requirements.txt
	go mod download
	kubectl apply -f k8s_configs/monitoring/
	
# Build scheduler plugin
build-plugin:
	cd scheduler-plugin && go build -o bin/ml-scheduler-plugin ./cmd/
	
# Build ML services
build-ml:
	docker build -t ml-scheduler/xgboost:latest -f src/model_serving/Dockerfile.xgboost .
	docker build -t ml-scheduler/qlearning:latest -f src/model_serving/Dockerfile.qlearning .
	docker build -t ml-scheduler/anomaly:latest -f src/model_serving/Dockerfile.anomaly .

build: build-plugin build-ml

# Testing
test-ml:
	pytest tests/ml_models/ -v --cov=src/ml_models/

test-plugin:
	cd scheduler-plugin && go test ./... -v

test-integration:
	pytest tests/integration/ -v

test-performance:
	k6 run tests/performance/scheduling-load-test.js

test: test-ml test-plugin test-integration

# Development deployment
deploy-dev:
	kubectl apply -f k8s_configs/monitoring/
	kubectl apply -f kserve_configs/
	kubectl apply -f k8s_configs/ml_services/

# Production deployment with progressive rollout
deploy-prod:
	@echo "Starting production deployment..."
	kubectl apply -f k8s_configs/scheduler/scheduler-config.yaml
	@echo "Deploying with 10% traffic..."
	scripts/deployment/progressive_rollout.sh 10
	@echo "Validate metrics before proceeding to 50%"

# Data collection and processing
collect-data:
	python scripts/data_processing/collect_historical_data.py
	python scripts/data_processing/process_metrics.py

# Model training
train-models:
	python src/ml_models/xgboost/training.py
	python src/ml_models/qlearning/training.py
	python src/ml_models/isolation_forest/training.py

# Validation
validate:
	python scripts/validation/validate_business_metrics.py
	python scripts/validation/validate_ml_performance.py

# Clean up
clean:
	rm -rf scheduler-plugin/bin/
	docker system prune -f
	kubectl delete -f k8s_configs/ml_services/ --ignore-not-found=true