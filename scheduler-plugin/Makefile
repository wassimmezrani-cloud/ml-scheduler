# ML Scheduler Plugin Makefile

# Variables
IMAGE_REGISTRY ?= localhost:5000
IMAGE_NAME ?= ml-scheduler-plugin
IMAGE_TAG ?= latest
FULL_IMAGE_NAME = $(IMAGE_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

NAMESPACE ?= kube-system
KUBECTL ?= kubectl

# Go variables
GOCMD = go
GOBUILD = $(GOCMD) build
GOTEST = $(GOCMD) test
GOMOD = $(GOCMD) mod

# Build variables
BINARY_NAME = ml-scheduler
LDFLAGS = -ldflags='-w -s -extldflags "-static"'
BUILD_FLAGS = -a -installsuffix cgo

.PHONY: help build test clean docker-build docker-push deploy undeploy fmt vet mod-tidy

help: ## Display this help message
	@echo "ML Scheduler Plugin Build System"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: fmt vet ## Build the scheduler plugin binary
	@echo "Building ML scheduler plugin..."
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 $(GOBUILD) $(LDFLAGS) $(BUILD_FLAGS) \
		-o bin/$(BINARY_NAME) ./cmd/main.go
	@echo "Build completed: bin/$(BINARY_NAME)"

test: ## Run unit tests
	@echo "Running tests..."
	$(GOTEST) -v -race -coverprofile=coverage.out ./...
	@echo "Tests completed"

test-coverage: test ## Run tests and generate coverage report
	@echo "Generating coverage report..."
	$(GOCMD) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

fmt: ## Format Go code
	@echo "Formatting code..."
	$(GOCMD) fmt ./...

vet: ## Run go vet
	@echo "Running go vet..."
	$(GOCMD) vet ./...

mod-tidy: ## Tidy go modules
	@echo "Tidying go modules..."
	$(GOMOD) tidy

docker-build: build ## Build Docker image
	@echo "Building Docker image: $(FULL_IMAGE_NAME)"
	docker build -t $(FULL_IMAGE_NAME) .
	@echo "Docker image built successfully"

docker-push: docker-build ## Push Docker image to registry
	@echo "Pushing Docker image: $(FULL_IMAGE_NAME)"
	docker push $(FULL_IMAGE_NAME)
	@echo "Docker image pushed successfully"

deploy: docker-push ## Deploy scheduler plugin to Kubernetes
	@echo "Deploying ML scheduler plugin to namespace $(NAMESPACE)..."
	
	# Create namespace if it doesn't exist
	$(KUBECTL) create namespace $(NAMESPACE) --dry-run=client -o yaml | $(KUBECTL) apply -f -
	
	# Apply scheduler deployment
	$(KUBECTL) apply -f manifests/scheduler-deployment.yaml -n $(NAMESPACE)
	
	# Wait for deployment to be ready
	$(KUBECTL) rollout status deployment/ml-scheduler -n $(NAMESPACE) --timeout=300s
	
	@echo "ML scheduler plugin deployed successfully"

undeploy: ## Remove scheduler plugin from Kubernetes
	@echo "Removing ML scheduler plugin from namespace $(NAMESPACE)..."
	$(KUBECTL) delete -f manifests/scheduler-deployment.yaml -n $(NAMESPACE) --ignore-not-found=true
	@echo "ML scheduler plugin removed"

deploy-local: ## Deploy to local Kubernetes (kind/minikube)
	@echo "Deploying to local Kubernetes..."
	$(MAKE) docker-build IMAGE_REGISTRY=localhost:5000
	
	# Load image into kind if available
	@if command -v kind >/dev/null 2>&1; then \
		echo "Loading image into kind cluster..."; \
		kind load docker-image $(FULL_IMAGE_NAME); \
	fi
	
	$(MAKE) deploy NAMESPACE=$(NAMESPACE)

status: ## Check deployment status
	@echo "Checking ML scheduler plugin status..."
	@echo ""
	@echo "=== Deployment Status ==="
	$(KUBECTL) get deployment ml-scheduler -n $(NAMESPACE) -o wide || echo "Deployment not found"
	@echo ""
	@echo "=== Pod Status ==="
	$(KUBECTL) get pods -l app=ml-scheduler -n $(NAMESPACE) -o wide || echo "No pods found"
	@echo ""
	@echo "=== Service Status ==="
	$(KUBECTL) get service ml-scheduler-metrics -n $(NAMESPACE) -o wide || echo "Service not found"
	@echo ""
	@echo "=== Recent Events ==="
	$(KUBECTL) get events -n $(NAMESPACE) --sort-by='.lastTimestamp' | tail -10

logs: ## Show scheduler plugin logs
	@echo "Showing ML scheduler plugin logs..."
	$(KUBECTL) logs -l app=ml-scheduler -n $(NAMESPACE) --tail=100 -f

debug-logs: ## Show detailed debug logs
	@echo "Showing detailed ML scheduler plugin logs..."
	$(KUBECTL) logs -l app=ml-scheduler -n $(NAMESPACE) --tail=500

metrics: ## Show scheduler metrics
	@echo "Fetching scheduler metrics..."
	@POD_NAME=$$($(KUBECTL) get pods -l app=ml-scheduler -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -n "$$POD_NAME" ]; then \
		echo "Metrics from pod: $$POD_NAME"; \
		$(KUBECTL) port-forward pod/$$POD_NAME -n $(NAMESPACE) 10251:10251 & \
		sleep 2; \
		curl -s http://localhost:10251/metrics | grep ml_scheduler || echo "No metrics available"; \
		pkill -f "port-forward.*10251:10251" || true; \
	else \
		echo "No ML scheduler pods found"; \
	fi

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf bin/
	rm -f coverage.out coverage.html
	docker rmi $(FULL_IMAGE_NAME) 2>/dev/null || true
	@echo "Clean completed"

check-deps: ## Check if required tools are installed
	@echo "Checking dependencies..."
	@command -v go >/dev/null 2>&1 || { echo "Go is required but not installed"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed"; exit 1; }
	@command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed"; exit 1; }
	@echo "All dependencies satisfied"

# Development targets
dev-setup: mod-tidy ## Setup development environment
	@echo "Setting up development environment..."
	$(GOCMD) install golang.org/x/tools/cmd/goimports@latest
	$(GOCMD) install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@echo "Development environment ready"

lint: ## Run linter
	@echo "Running linter..."
	golangci-lint run ./...

dev-test: fmt vet lint test ## Run all development checks

# CI/CD targets
ci-build: check-deps mod-tidy dev-test build ## Complete CI build pipeline

ci-deploy: ci-build docker-build docker-push deploy ## Complete CI deployment pipeline

# Utility targets
watch-pods: ## Watch scheduler pods
	$(KUBECTL) get pods -l app=ml-scheduler -n $(NAMESPACE) -w

watch-events: ## Watch Kubernetes events
	$(KUBECTL) get events -n $(NAMESPACE) -w

port-forward-metrics: ## Port forward scheduler metrics
	@POD_NAME=$$($(KUBECTL) get pods -l app=ml-scheduler -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}'); \
	echo "Port forwarding metrics from pod: $$POD_NAME"; \
	$(KUBECTL) port-forward pod/$$POD_NAME -n $(NAMESPACE) 10251:10251

# Default target
all: ci-build ## Build everything