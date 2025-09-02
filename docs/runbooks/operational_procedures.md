# Operational Procedures - HYDATIS ML Scheduler

## Daily Operations

### Morning Health Check (9:00 AM UTC)
```bash
#!/bin/bash
# Daily health check script

echo "=== HYDATIS ML Scheduler Daily Health Check ==="
echo "Date: $(date)"
echo ""

# 1. Cluster Overview
echo "1. CLUSTER STATUS:"
kubectl get nodes
echo ""

# 2. ML Scheduler Status
echo "2. ML SCHEDULER PODS:"
kubectl get pods -n ml-scheduler -o wide
echo ""

# 3. Business Metrics Check
echo "3. BUSINESS METRICS (Last 1 hour):"
python scripts/validation/validate_business_metrics.py --quick-check --period 1h
echo ""

# 4. Performance Overview
echo "4. PERFORMANCE METRICS:"
curl -s http://prometheus:9090/api/v1/query?query=histogram_quantile\(0.99,rate\(ml_scheduler_scheduling_duration_seconds_bucket\[5m\]\)\)*1000 | jq '.data.result[0].value[1]' | awk '{print "Scheduling Latency P99: " $1 "ms"}'
curl -s http://prometheus:9090/api/v1/query?query=rate\(ml_scheduler_scheduling_success_total\[5m\]\)/rate\(ml_scheduler_scheduling_requests_total\[5m\]\)*100 | jq '.data.result[0].value[1]' | awk '{print "Success Rate: " $1 "%"}'
echo ""

# 5. Alert Status
echo "5. ACTIVE ALERTS:"
curl -s http://alertmanager:9093/api/v1/alerts | jq '.data[] | select(.status.state=="firing") | .labels.alertname' | sort | uniq -c
echo ""

# 6. Cache Performance
echo "6. CACHE PERFORMANCE:"
kubectl exec -it deployment/redis -n ml-scheduler -- redis-cli info stats | grep -E "(keyspace_hits|keyspace_misses|hit_rate)"
echo ""

echo "=== Health Check Complete ==="
```

### Evening Business Review (6:00 PM UTC)
```bash
# Generate daily business report
python scripts/validation/validate_business_metrics.py \
  --daily-report \
  --export-format pdf \
  --output-file /tmp/hydatis-daily-$(date +%Y%m%d).pdf

# Email report to stakeholders
curl -X POST "${SMTP_ENDPOINT}/send" \
  -H "Content-Type: application/json" \
  -d '{
    "to": ["ml-team@hydatis.com", "platform-team@hydatis.com"],
    "subject": "HYDATIS Daily Performance Report - $(date +%Y-%m-%d)",
    "attachment": "/tmp/hydatis-daily-$(date +%Y%m%d).pdf"
  }'
```

## Weekly Operations

### Monday: Model Performance Review
```bash
# 1. Check model drift metrics
python src/monitoring/drift_detection.py --report-only

# 2. Review model accuracy trends
curl -s http://prometheus:9090/api/v1/query_range \
  -d 'query=avg(ml_model_accuracy)' \
  -d 'start='$(date -d '7 days ago' --iso-8601) \
  -d 'end='$(date --iso-8601) \
  -d 'step=1h' | jq '.data.result[0].values'

# 3. Validate model serving performance
for model in xgboost-predictor qlearning-optimizer anomaly-detector; do
  echo "Testing $model endpoint..."
  kubectl run test-$model --rm -i --restart=Never --image=curlimages/curl -- \
    curl -s http://$model.ml-scheduler/v1/models/$model:predict \
    -H "Content-Type: application/json" \
    -d '{"instances": [[0.5, 0.8, 0.3]]}'
done
```

### Wednesday: Capacity Planning Review
```bash
# 1. Generate capacity forecast
python src/analytics/predictive_engine.py \
  --horizon 30d \
  --output-format json > /tmp/capacity-forecast.json

# 2. Review scaling recommendations
jq '.forecasts."30d".capacity.recommended_node_count' /tmp/capacity-forecast.json

# 3. Cost optimization opportunities
jq '.forecasts."30d".cost_optimization.optimization_opportunities' /tmp/capacity-forecast.json
```

### Friday: System Maintenance
```bash
# 1. Clean up old data and logs
kubectl delete pods --field-selector=status.phase=Succeeded -n ml-scheduler
kubectl exec -it deployment/redis -n ml-scheduler -- redis-cli FLUSHDB 1  # Clear test DB

# 2. Update model versions if available
python scripts/model_management/check_model_updates.py --auto-update-non-critical

# 3. Backup configuration
kubectl get configmaps -n ml-scheduler -o yaml > backups/configmaps-$(date +%Y%m%d).yaml
kubectl get secrets -n ml-scheduler -o yaml > backups/secrets-$(date +%Y%m%d).yaml
```

## Monthly Operations

### First Monday: Business Review
```bash
# 1. Generate comprehensive monthly report
python scripts/validation/validate_business_metrics.py \
  --monthly-report \
  --include-roi-analysis \
  --include-cost-breakdown \
  --export-format pdf

# 2. ROI calculation verification
python scripts/validation/validate_business_metrics.py \
  --roi-detailed \
  --baseline-period 30d

# 3. Capacity planning for next month
python src/analytics/predictive_engine.py \
  --horizon 30d \
  --include-scaling-plan \
  --output-format yaml > capacity-plan-$(date +%Y%m).yaml
```

### Second Monday: Model Retraining Review
```bash
# 1. Review model performance trends
python scripts/ml_models/performance_analysis.py --period 30d

# 2. Scheduled model retraining
for model in xgboost qlearning isolation_forest; do
  python src/ml_models/$model/training.py --scheduled-retrain --validate-improvement
done

# 3. A/B test new model versions
python src/model_serving/ab_testing.py \
  --start-test \
  --models "xgboost-v2.1,qlearning-v1.8" \
  --traffic-split "10,90" \
  --duration-hours 168
```

### Third Monday: Infrastructure Review
```bash
# 1. Node health assessment
kubectl describe nodes | grep -E "(Conditions|Allocated resources)" > node-health-$(date +%Y%m).txt

# 2. Storage health check
kubectl get pvc -A
kubectl exec -it longhorn-manager-xxxxx -n longhorn-system -- longhorn-manager health

# 3. Network performance validation
kubectl run netperf --rm -i --restart=Never --image=networkstatic/netperf -- \
  netperf -H prometheus.monitoring -t TCP_STREAM
```

### Fourth Monday: Security & Compliance
```bash
# 1. Security scan of container images
for image in $(kubectl get pods -n ml-scheduler -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort | uniq); do
  echo "Scanning $image..."
  trivy image $image --severity HIGH,CRITICAL
done

# 2. RBAC audit
kubectl auth can-i --list --as=system:serviceaccount:ml-scheduler:ml-scheduler-sa

# 3. Configuration drift detection
python scripts/security/detect_config_drift.py --baseline config/production-baseline.yaml
```

## Model Management Procedures

### Model Deployment
```bash
# 1. Validate new model version
python scripts/ml_models/validate_model.py \
  --model-path models/xgboost/v2.1 \
  --validation-dataset data/validation/latest.parquet \
  --performance-baseline 0.85

# 2. Deploy with progressive rollout
python src/deployment/progressive_rollout.py \
  --model-name xgboost-predictor \
  --new-version v2.1 \
  --traffic-phases "10,50,100" \
  --validation-window 30m

# 3. Monitor deployment
kubectl logs -f deployment/progressive-rollout-controller -n ml-scheduler
```

### Model Rollback
```bash
# 1. Immediate rollback (emergency)
kubectl patch inferenceservice xgboost-predictor -n ml-scheduler \
  --type merge -p '{"spec":{"predictor":{"model":{"modelUri":"models:/xgboost/v2.0-stable"}}}}'

# 2. Verify rollback success
python scripts/validation/test_model_endpoints.py --model xgboost-predictor

# 3. Update model registry
python scripts/ml_models/mark_model_version.py \
  --model xgboost \
  --version v2.1 \
  --status failed \
  --reason "performance_degradation"
```

## Performance Optimization

### Cache Optimization
```bash
# 1. Analyze cache hit rates
kubectl exec -it deployment/redis -n ml-scheduler -- redis-cli info stats

# 2. Optimize cache configuration
if [ $(redis-cli -h redis.ml-scheduler info stats | grep hit_rate | cut -d: -f2) -lt 95 ]; then
  echo "Optimizing cache configuration..."
  kubectl patch configmap redis-config -n ml-scheduler \
    --patch '{"data":{"maxmemory-policy":"allkeys-lru","timeout":"300"}}'
fi

# 3. Clear stale cache entries
kubectl exec -it deployment/redis -n ml-scheduler -- \
  redis-cli --scan --pattern "ml:predictions:*" | \
  xargs -I {} redis-cli TTL {} | \
  awk '$1 > 1800 {print $1}' | \
  xargs -I {} redis-cli DEL {}
```

### Scheduler Performance Tuning
```bash
# 1. Analyze scheduling latency distribution
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(ml_scheduler_scheduling_duration_seconds_bucket[1h]))*1000"

# 2. Tune aggressiveness based on current CPU
CURRENT_CPU=$(curl -s "http://prometheus:9090/api/v1/query?query=avg(100-(avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m]))*100))" | jq -r '.data.result[0].value[1]')

if (( $(echo "$CURRENT_CPU < 60" | bc -l) )); then
  # Increase aggressiveness
  kubectl patch configmap ml-scheduler-config -n ml-scheduler \
    --patch '{"data":{"aggressiveness":"0.8","optimization_weight":"0.7"}}'
elif (( $(echo "$CURRENT_CPU > 70" | bc -l) )); then
  # Decrease aggressiveness  
  kubectl patch configmap ml-scheduler-config -n ml-scheduler \
    --patch '{"data":{"aggressiveness":"0.4","optimization_weight":"0.3"}}'
fi
```

## Business Metrics Monitoring

### Real-time Business Dashboard
```bash
# CPU utilization check (target: 65% ±5%)
CPU_CURRENT=$(curl -s "http://prometheus:9090/api/v1/query?query=avg(100-(avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m]))*100))" | jq -r '.data.result[0].value[1]')

# Availability check (target: 99.7%)
AVAILABILITY=$(curl -s "http://prometheus:9090/api/v1/query?query=avg(up{job=\"kubernetes-nodes\"})*100" | jq -r '.data.result[0].value[1]')

# ROI calculation (target: >1400%)
ROI=$(curl -s "http://prometheus:9090/api/v1/query?query=((85-avg(100-(avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m]))*100)))/85*0.3+(avg(up{job=\"kubernetes-nodes\"})-0.952)/0.045)*1200" | jq -r '.data.result[0].value[1]')

echo "HYDATIS Business Metrics:"
echo "CPU Utilization: ${CPU_CURRENT}% (Target: 65% ±5%)"
echo "Availability: ${AVAILABILITY}% (Target: 99.7%)"  
echo "ROI Projection: ${ROI}% (Target: >1400%)"
```

### Alert Validation
```bash
# Check for business-critical alerts
CRITICAL_ALERTS=$(curl -s http://alertmanager:9093/api/v1/alerts | jq '[.data[] | select(.labels.severity=="critical" and .status.state=="firing")] | length')

if [ "$CRITICAL_ALERTS" -gt 0 ]; then
  echo "⚠️  WARNING: $CRITICAL_ALERTS critical alerts active"
  curl -s http://alertmanager:9093/api/v1/alerts | jq '.data[] | select(.labels.severity=="critical" and .status.state=="firing") | .labels.alertname'
else
  echo "✅ No critical alerts active"
fi
```

## Incident Response Procedures

### Incident Classification
```bash
# Automatic incident classification
python scripts/incident_management/classify_incident.py \
  --alert-name "$ALERT_NAME" \
  --severity "$SEVERITY" \
  --affected-components "$COMPONENTS" \
  --business-impact-assessment
```

### Incident Response Workflow
1. **Detection** (Automated via AIOps)
2. **Classification** (P0/P1/P2/P3)
3. **Initial Response** (Automated remediation)
4. **Escalation** (If automated remediation fails)
5. **Resolution** (Manual intervention if needed)
6. **Post-mortem** (Root cause analysis)

### Manual Intervention Procedures

#### CPU Utilization Emergency (>80% sustained)
```bash
# 1. Immediate: Reduce scheduler aggressiveness
kubectl patch configmap ml-scheduler-config -n ml-scheduler \
  --patch '{"data":{"aggressiveness":"0.2","cpu_weight":"0.9"}}'

# 2. Scale critical services down temporarily
kubectl scale deployment ml-scheduler --replicas=2 -n ml-scheduler

# 3. Monitor for improvement
watch 'kubectl top nodes'

# 4. Gradual recovery
sleep 300  # Wait 5 minutes
kubectl scale deployment ml-scheduler --replicas=3 -n ml-scheduler
kubectl patch configmap ml-scheduler-config -n ml-scheduler \
  --patch '{"data":{"aggressiveness":"0.5","cpu_weight":"0.6"}}'
```

#### Availability Emergency (<99.0%)
```bash
# 1. Immediate: Check failed nodes
kubectl get nodes | grep NotReady

# 2. Drain problematic nodes
for node in $(kubectl get nodes | grep NotReady | awk '{print $1}'); do
  kubectl drain $node --ignore-daemonsets --delete-emptydir-data --force
done

# 3. Scale critical services for redundancy
kubectl scale deployment ml-scheduler --replicas=5 -n ml-scheduler
kubectl scale deployment redis --replicas=3 -n ml-scheduler

# 4. Activate all fallback mechanisms
curl -X POST http://aiops-remediation-service:8080/emergency \
  -d '{"action":"activate_all_fallbacks","duration":"1h"}'
```

## Model Management Operations

### Model Retraining
```bash
# 1. Scheduled weekly retraining (Sunday 2:00 AM)
python src/ml_models/xgboost/training.py \
  --training-data-days 14 \
  --validation-split 0.2 \
  --performance-threshold 0.85 \
  --auto-deploy-if-improved

python src/ml_models/qlearning/training.py \
  --training-episodes 10000 \
  --exploration-decay 0.995 \
  --target-improvement 0.02

# 2. Validate new models
python scripts/ml_models/validate_model_performance.py \
  --models "xgboost,qlearning,isolation_forest" \
  --validation-period 24h
```

### A/B Testing Management
```bash
# 1. Start A/B test for new model version
python src/model_serving/ab_testing.py \
  --start-test \
  --model xgboost-predictor \
  --version-a v2.0 \
  --version-b v2.1 \
  --traffic-split 90,10 \
  --duration-hours 72

# 2. Monitor A/B test results
python src/model_serving/ab_testing.py \
  --monitor-test \
  --test-id $TEST_ID \
  --metrics "latency,accuracy,business_impact"

# 3. Conclude A/B test
python src/model_serving/ab_testing.py \
  --conclude-test \
  --test-id $TEST_ID \
  --winner-threshold 0.02
```

## Capacity Management

### Scaling Decisions
```bash
# 1. Check current capacity utilization
kubectl top nodes
kubectl describe nodes | grep -A5 "Allocated resources"

# 2. Predictive capacity analysis
python src/analytics/predictive_engine.py \
  --prediction-type capacity_forecast \
  --horizon 7d \
  --confidence-threshold 0.8

# 3. Execute scaling if recommended
RECOMMENDED_NODES=$(python src/analytics/predictive_engine.py --horizon 7d --output-format json | jq '.forecasts."7d".capacity.recommended_node_count')

if [ "$RECOMMENDED_NODES" -gt 6 ]; then
  echo "Scaling recommendation: $RECOMMENDED_NODES nodes"
  # Implement node scaling procedure (cloud provider specific)
fi
```

### Resource Optimization
```bash
# 1. Identify resource waste
python src/optimization/cost_optimizer.py \
  --analyze-waste \
  --recommendations \
  --export-report

# 2. Optimize pod resource requests
python scripts/optimization/optimize_pod_resources.py \
  --namespace ml-scheduler \
  --dry-run \
  --optimization-target cpu_efficiency

# 3. Apply optimizations (after validation)
python scripts/optimization/optimize_pod_resources.py \
  --namespace ml-scheduler \
  --apply \
  --backup-config
```

## Quality Assurance

### Performance Validation
```bash
# 1. Run performance test suite
python tests/performance/load_tests/scheduler_load_test.py \
  --scenario baseline \
  --duration 600 \
  --target-pods 100

# 2. Latency validation
python tests/performance/latency_tests/scheduling_latency_test.py \
  --samples 500 \
  --p99-target 100

# 3. Business metrics validation
python scripts/validation/validate_business_metrics.py \
  --comprehensive \
  --export-format json \
  --alert-on-deviation
```

### Configuration Validation
```bash
# 1. Validate all configuration files
python scripts/validation/validate_configurations.py \
  --config-dir config/ \
  --check-syntax \
  --check-business-alignment

# 2. Verify RBAC permissions
kubectl auth can-i --list --as=system:serviceaccount:ml-scheduler:ml-scheduler-sa | grep -E "(create|update|patch|delete)"

# 3. Test disaster recovery procedures
./scripts/testing/test_disaster_recovery.sh --dry-run
```

## Maintenance Windows

### Planned Maintenance (Monthly - Second Sunday 2:00 AM UTC)
```bash
# 1. Pre-maintenance validation
./scripts/maintenance/pre_maintenance_check.sh

# 2. Backup current state
./scripts/backup/backup_complete_system.sh

# 3. Maintenance tasks
# - Update container images
# - Apply security patches  
# - Optimize configurations
# - Clean up old data

# 4. Post-maintenance validation
./scripts/maintenance/post_maintenance_validation.sh

# 5. Performance verification
python tests/performance/load_tests/scheduler_load_test.py --scenario post_maintenance
```

### Emergency Maintenance
```bash
# For critical security patches or urgent fixes

# 1. Activate maintenance mode
kubectl patch configmap ml-scheduler-config -n ml-scheduler \
  --patch '{"data":{"maintenance_mode":"true"}}'

# 2. Drain ML scheduler traffic gradually
python src/deployment/progressive_rollout.py \
  --rollout-type maintenance \
  --traffic-reduction "100,50,25,0" \
  --validation-window 5m

# 3. Apply fixes
# [Specific maintenance tasks]

# 4. Gradual traffic restoration
python src/deployment/progressive_rollout.py \
  --rollout-type recovery \
  --traffic-increase "25,50,75,100" \
  --validation-window 10m
```

## Documentation Maintenance

### Weekly Documentation Review
```bash
# 1. Update operational metrics
python scripts/documentation/update_metrics_docs.py \
  --period 7d \
  --update-dashboards

# 2. Review and update runbooks
git diff HEAD~7 docs/runbooks/ | grep -E "^[+-]" | wc -l

# 3. Validate procedure accuracy
./scripts/testing/test_all_procedures.sh --dry-run
```

### Configuration Documentation
```bash
# 1. Generate configuration documentation
python scripts/documentation/generate_config_docs.py \
  --config-dir config/ \
  --output-format markdown

# 2. Update architecture diagrams
python scripts/documentation/generate_architecture_diagrams.py \
  --format svg \
  --include-metrics-flow

# 3. Validate documentation completeness
python scripts/documentation/validate_docs_completeness.py \
  --check-all-procedures \
  --check-all-configs
```

---

**Operational Procedures Version:** 1.0  
**Last Updated:** Week 14 Documentation  
**Review Schedule:** Monthly operational review  
**Owner:** Platform Engineering Team