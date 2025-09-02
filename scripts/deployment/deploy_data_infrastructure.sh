#!/bin/bash

# Week 1 Infrastructure Deployment Script for ML Scheduler
# Deploys to existing HYDATIS cluster with proper Longhorn integration

set -e

echo "Deploying ML Scheduler Week 1 Infrastructure to HYDATIS cluster..."

# Check cluster connectivity
echo "Checking cluster connectivity..."
kubectl cluster-info || { echo "Cluster not accessible"; exit 1; }

# Verify Longhorn is operational
echo "Verifying Longhorn storage..."
kubectl get storageclass longhorn || { echo "Longhorn storage class not found"; exit 1; }

# Check existing namespaces
echo "Checking existing namespaces..."
kubectl get namespace hydatis-mlops || kubectl create namespace hydatis-mlops
kubectl get namespace monitoring || { echo "Monitoring namespace required"; exit 1; }

# Label namespace for Kubeflow integration
kubectl label namespace hydatis-mlops \
  katib.kubeflow.org/metrics-collector-injection=enabled \
  serving.kubeflow.org/inferenceservice=enabled \
  pipelines.kubeflow.org/enabled=true \
  --overwrite

echo "Creating Longhorn storage volumes..."
kubectl apply -f k8s_configs/monitoring/longhorn-ml-data-storage.yaml

echo "Waiting for PVCs to be bound..."
kubectl wait --for=condition=Bound pvc/ml-scheduler-historical-data -n hydatis-mlops --timeout=300s
kubectl wait --for=condition=Bound pvc/ml-scheduler-models -n hydatis-mlops --timeout=300s
kubectl wait --for=condition=Bound pvc/ml-scheduler-artifacts -n hydatis-mlops --timeout=300s

echo "Deploying enhanced Prometheus configuration..."
kubectl apply -f k8s_configs/monitoring/ml-scheduler-scrape-configs.yaml

# Note: Prometheus retention update requires careful coordination
echo "WARNING: Prometheus retention update requires manual coordination"
echo "Current Prometheus: monitoring/k8s with 2 replicas"
echo "Review k8s_configs/monitoring/prometheus-extended-retention.yaml before applying"

echo "Deploying ML scheduler data collection job..."
kubectl apply -f k8s_configs/ml_services/ml-data-collector-job.yaml

echo "Deploying enhanced Jupyter notebook environment..."
kubectl apply -f k8s_configs/ml_services/jupyter-ml-scheduler.yaml

echo "Waiting for notebook to be ready..."
kubectl wait --for=condition=Ready pod -l app=ml-scheduler -n hydatis-mlops --timeout=600s

echo "Creating data collection ConfigMap..."
kubectl create configmap ml-scheduler-scripts \
  --from-file=scripts/data_processing/collect_historical_data.py \
  --from-file=src/data_collection/prometheus_collector.py \
  --from-file=src/data_collection/data_processor.py \
  --from-file=src/data_collection/quality_monitor.py \
  -n hydatis-mlops \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Verifying deployment status..."
kubectl get pods -n hydatis-mlops -l app=ml-scheduler
kubectl get pvc -n hydatis-mlops -l app=ml-scheduler
kubectl get jobs,cronjobs -n hydatis-mlops -l app=ml-scheduler

echo "Getting access information..."
echo "Jupyter notebook access:"
kubectl get notebook ml-scheduler-notebook -n hydatis-mlops -o wide

echo "MLflow access:"
echo "MLflow server available at: http://10.110.146.252:5000 (cluster) or http://10.110.190.32:31380 (external)"

echo "Longhorn UI access:"
echo "Longhorn frontend available at: http://10.110.190.81"

echo ""
echo "Week 1 infrastructure deployment completed!"
echo "Next steps:"
echo "1. Access Jupyter notebook for data exploration"
echo "2. Run data collection: kubectl exec -it <notebook-pod> -n hydatis-mlops -- python /app/collect_historical_data.py"
echo "3. Monitor data quality in Longhorn volumes"
echo "4. Review Prometheus metrics collection"