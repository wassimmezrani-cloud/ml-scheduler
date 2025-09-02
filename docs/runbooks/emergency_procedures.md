# Emergency Procedures - HYDATIS ML Scheduler

## Critical Incident Response

### 🚨 IMMEDIATE ACTIONS (First 5 Minutes)

#### Scheduler Complete Failure
```bash
# 1. IMMEDIATE: Activate default scheduler fallback
kubectl patch deployments --all -n $HYDATIS_NAMESPACE \
  -p '{"spec":{"template":{"spec":{"schedulerName":"default-scheduler"}}}}'

# 2. Scale down ML scheduler to stop interference
kubectl scale deployment ml-scheduler --replicas=0 -n $HYDATIS_NAMESPACE

# 3. Verify pods can be scheduled
kubectl get pods --field-selector=status.phase=Pending
```

#### Critical Business SLA Breach (Availability <99.0%)
```bash
# 1. IMMEDIATE: Check cluster node health
kubectl get nodes
kubectl describe nodes | grep -E "(Ready|MemoryPressure|DiskPressure|PIDPressure)"

# 2. Emergency scale critical services
kubectl scale deployment ml-scheduler --replicas=5 -n $HYDATIS_NAMESPACE
kubectl scale deployment redis --replicas=3 -n $HYDATIS_NAMESPACE

# 3. Activate all fallback modes
curl -X POST http://aiops-remediation-service:8080/emergency \
  -d '{"action":"activate_all_fallbacks","reason":"sla_breach"}'
```

#### Cost Target Exceeded (CPU >75% for >30min)
```bash
# 1. IMMEDIATE: Reduce scheduler aggressiveness
kubectl patch configmap ml-scheduler-config -n $HYDATIS_NAMESPACE \
  --patch '{"data":{"aggressiveness":"0.3"}}'

# 2. Force scheduler restart to apply config
kubectl rollout restart deployment/ml-scheduler -n $HYDATIS_NAMESPACE

# 3. Monitor CPU reduction
watch 'kubectl top nodes'
```

### 📞 ESCALATION CONTACTS

| Severity | Contact | Response Time | Action |
|----------|---------|---------------|--------|
| **P0 - Critical** | SRE On-Call: +1-555-SRE-CALL | 5 minutes | Page immediately |
| **P1 - High** | ML Team Lead: ml-lead@hydatis.com | 15 minutes | Slack + Email |
| **P2 - Medium** | Platform Team: platform@hydatis.com | 1 hour | Email |
| **P3 - Low** | ML Team: ml-team@hydatis.com | 4 hours | Ticket |

### 🔧 EMERGENCY TOOLBOX

#### Quick Diagnostics
```bash
# System health overview
kubectl get pods,nodes,pvc -A | grep -E "(Pending|Failed|NotReady)"

# ML scheduler health
kubectl logs -f deployment/ml-scheduler -n $HYDATIS_NAMESPACE --tail=50

# Business metrics check
curl -s http://prometheus:9090/api/v1/query?query=avg\(100-\(avg\(rate\(node_cpu_seconds_total\{mode=\"idle\"\}\[5m\]\)\)*100\)\)

# Recent scheduling activity
kubectl get events --sort-by='.lastTimestamp' | grep -i schedule | head -10
```

#### Emergency Rollback Commands
```bash
# Rollback to default scheduler (IMMEDIATE)
export EMERGENCY_ROLLBACK=true
./scripts/emergency_rollback.sh

# Rollback specific model
kubectl patch inferenceservice xgboost-predictor -n $HYDATIS_NAMESPACE \
  --type merge -p '{"spec":{"predictor":{"model":{"modelUri":"models:/xgboost/v1.0-stable"}}}}'

# Clear all caches (performance reset)
kubectl exec -it deployment/redis -n $HYDATIS_NAMESPACE -- redis-cli FLUSHALL
```

## Incident Classification

### P0 - Critical (Revenue Impact >$5k/hour)
- **Triggers:** Availability <99.0%, Complete scheduler failure, Data loss
- **Response:** Immediate page, all-hands response
- **Actions:** Emergency fallback, immediate escalation to CTO

### P1 - High (Business Target Miss)
- **Triggers:** CPU >75% sustained, Availability <99.5%, ROI projection <1000%
- **Response:** 15-minute response, ML team engagement
- **Actions:** Automated remediation, business stakeholder notification

### P2 - Medium (Performance Degradation)
- **Triggers:** Latency >150ms, Success rate <98%, Drift detection
- **Response:** 1-hour response, normal business hours
- **Actions:** Standard remediation procedures, monitoring

### P3 - Low (Operational Issues)
- **Triggers:** Cache issues, Non-critical alerts, Monitoring gaps
- **Response:** 4-hour response, ticket-based
- **Actions:** Scheduled maintenance, documentation updates

## Recovery Procedures

### Service Recovery Sequence

#### 1. ML Scheduler Recovery
```bash
# Check dependencies first
kubectl get pods -n $HYDATIS_NAMESPACE | grep -E "(redis|xgboost|qlearning|anomaly)"

# Restart scheduler with health validation
kubectl rollout restart deployment/ml-scheduler -n $HYDATIS_NAMESPACE
kubectl rollout status deployment/ml-scheduler -n $HYDATIS_NAMESPACE --timeout=300s

# Validate scheduling functionality
./scripts/validation/test_scheduling.sh
```

#### 2. ML Model Recovery
```bash
# Restart all model services
for model in xgboost-predictor qlearning-optimizer anomaly-detector; do
  kubectl rollout restart deployment/$model -n $HYDATIS_NAMESPACE
  kubectl rollout status deployment/$model -n $HYDATIS_NAMESPACE --timeout=180s
done

# Validate model endpoints
./scripts/validation/test_model_endpoints.sh
```

#### 3. Cache Recovery
```bash
# Restart Redis with data preservation
kubectl rollout restart deployment/redis -n $HYDATIS_NAMESPACE

# Warm up cache
python scripts/cache/warm_cache.py --priority-features

# Validate cache performance
redis-cli -h redis.ml-scheduler info stats
```

### Data Recovery

#### Metrics Data Recovery
```bash
# Check Prometheus data retention
curl http://prometheus:9090/api/v1/label/__name__/values | jq '.data[]' | grep ml_scheduler

# Restore from backup if needed
kubectl exec -it prometheus-0 -n monitoring -- \
  /bin/sh -c "cd /prometheus && wget https://backup-server/prometheus-backup-$(date +%Y%m%d).tar.gz"
```

#### Model Artifact Recovery
```bash
# Check MLflow model registry
curl http://mlflow:5000/api/2.0/mlflow/registered-models/list

# Restore model from backup version
python scripts/model_management/restore_model.py \
  --model-name xgboost-predictor \
  --version last-known-good
```

## Communication Templates

### P0 Critical Incident Notification
```
SUBJECT: [P0-CRITICAL] HYDATIS ML Scheduler - Availability Impact

INCIDENT: ML Scheduler critical failure
IMPACT: Cluster availability <99.0% - SLA breach
BUSINESS IMPACT: $5,000/hour revenue risk
STATUS: Emergency response activated

ACTIONS TAKEN:
- ✅ Default scheduler fallback activated
- ⏳ Root cause investigation in progress
- ⏳ Emergency remediation team assembled

NEXT UPDATE: 15 minutes

War Room: https://meet.hydatis.com/emergency-response
Incident ID: INC-$(date +%Y%m%d-%H%M%S)
```

### P1 High Priority Notification
```
SUBJECT: [P1-HIGH] HYDATIS ML Scheduler - Business Target Miss

INCIDENT: CPU utilization exceeding 75% for >30 minutes
IMPACT: Cost optimization target at risk
BUSINESS IMPACT: Monthly savings target ($5k) at risk
STATUS: Automated remediation in progress

ACTIONS TAKEN:
- ✅ AIOps automated response triggered
- ✅ Scheduler aggressiveness reduced
- ⏳ Performance monitoring increased

ETA TO RESOLUTION: 1 hour
NEXT UPDATE: 30 minutes
```

## Emergency Contacts & Resources

### Technical Escalation Chain
1. **L1 Support:** SRE On-Call → +1-555-SRE-CALL
2. **L2 Support:** ML Engineering Lead → ml-lead@hydatis.com
3. **L3 Support:** Platform Architect → platform-architect@hydatis.com
4. **L4 Support:** CTO → cto@hydatis.com

### Business Escalation Chain
1. **Business Impact:** Product Manager → pm@hydatis.com
2. **Revenue Impact:** VP Engineering → vp-eng@hydatis.com  
3. **SLA Breach:** CFO → cfo@hydatis.com
4. **Major Incident:** CEO → ceo@hydatis.com

### External Resources
- **Cloud Provider Support:** [Cloud Provider Support Portal]
- **Kubernetes Support:** [K8s Enterprise Support]
- **MLflow Support:** [MLflow Enterprise Support]

### Emergency Access
```bash
# Emergency admin access (break glass)
export KUBECONFIG=/etc/kubernetes/emergency-admin.conf

# Emergency scheduler bypass
kubectl patch deployments --all \
  -p '{"spec":{"template":{"spec":{"schedulerName":"default-scheduler"}}}}'

# Emergency cluster drain (last resort)
for node in $(kubectl get nodes -o name); do
  kubectl drain $node --ignore-daemonsets --delete-emptydir-data --force
done
```

## Post-Incident Procedures

### Immediate Post-Resolution (First Hour)
1. **Verify System Stability**
   ```bash
   # Confirm all metrics within normal ranges
   python scripts/validation/validate_business_metrics.py --quick-check
   
   # Verify no pending pods
   kubectl get pods --all-namespaces --field-selector=status.phase=Pending
   ```

2. **Document Incident Timeline**
   - Record all actions taken with timestamps
   - Capture metrics before/during/after incident
   - Note any manual interventions required

3. **Notify Stakeholders of Resolution**
   ```
   SUBJECT: [RESOLVED] HYDATIS ML Scheduler Incident
   
   INCIDENT: [Original incident description]
   RESOLUTION: [Brief description of fix]
   IMPACT: [Actual business impact]
   ROOT CAUSE: [Preliminary root cause]
   
   SYSTEM STATUS: Fully operational
   BUSINESS METRICS: Restored to targets
   
   POST-MORTEM: Scheduled for [date/time]
   ```

### Within 24 Hours
1. **Conduct Post-Mortem**
   - Root cause analysis
   - Timeline reconstruction
   - Impact assessment
   - Prevention measures

2. **Update Runbooks**
   - Document new procedures learned
   - Update emergency contacts if needed
   - Revise escalation thresholds

3. **Implement Preventive Measures**
   - Add monitoring for gaps identified
   - Update alerting thresholds
   - Enhance automated remediation

### Within 1 Week
1. **Comprehensive Review**
   - Architecture review for resilience gaps
   - Process improvement recommendations
   - Training needs assessment

2. **System Improvements**
   - Implement additional safeguards
   - Enhance monitoring coverage
   - Update documentation

---

**Emergency Procedures Version:** 1.0  
**Last Updated:** Week 14 Documentation  
**Review Schedule:** After every P0/P1 incident  
**Approval:** SRE Team Lead, ML Engineering Lead