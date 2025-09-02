#!/usr/bin/env python3

import asyncio
import logging
import json
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import aiohttp
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BusinessTarget:
    """Business performance target"""
    metric_name: str
    target_value: float
    tolerance_percent: float
    operator: str  # 'gte', 'lte', 'eq', 'range'
    priority: str  # 'critical', 'high', 'medium', 'low'
    description: str

@dataclass
class ValidationResult:
    """Result of business metrics validation"""
    metric_name: str
    current_value: float
    target_value: float
    tolerance_range: Tuple[float, float]
    passed: bool
    score: float
    impact: str
    timestamp: datetime

class BusinessMetricsValidator:
    """
    Validates business metrics against targets for ML scheduler
    Tracks progress toward 65% CPU utilization and 99.7% availability goals
    """
    
    def __init__(self, 
                 prometheus_url: str = "http://prometheus-server:9090",
                 kubeconfig_path: Optional[str] = None):
        """
        Initialize business metrics validator
        
        Args:
            prometheus_url: Prometheus server URL
            kubeconfig_path: Path to kubeconfig (None for in-cluster)
        """
        self.prometheus_url = prometheus_url
        
        # Initialize Kubernetes client
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                config.load_incluster_config()
            
            self.k8s_core_v1 = client.CoreV1Api()
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_custom = client.CustomObjectsApi()
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
        
        # Define business targets for HYDATIS cluster
        self.business_targets = {
            'cpu_utilization': BusinessTarget(
                metric_name='cpu_utilization',
                target_value=65.0,
                tolerance_percent=5.0,  # 61.75% - 68.25%
                operator='range',
                priority='critical',
                description='Cluster CPU utilization target for cost optimization'
            ),
            'memory_utilization': BusinessTarget(
                metric_name='memory_utilization',
                target_value=75.0,
                tolerance_percent=10.0,  # 67.5% - 82.5%
                operator='range',
                priority='high',
                description='Cluster memory utilization for efficiency'
            ),
            'availability': BusinessTarget(
                metric_name='availability',
                target_value=99.7,
                tolerance_percent=0.2,  # 99.5% - 99.9%
                operator='gte',
                priority='critical',
                description='System availability SLA target'
            ),
            'scheduling_latency_p95': BusinessTarget(
                metric_name='scheduling_latency_p95',
                target_value=100.0,  # 100ms
                tolerance_percent=20.0,  # Up to 120ms acceptable
                operator='lte',
                priority='high',
                description='95th percentile scheduling latency'
            ),
            'pod_startup_time_p95': BusinessTarget(
                metric_name='pod_startup_time_p95',
                target_value=30.0,  # 30 seconds
                tolerance_percent=33.0,  # Up to 40 seconds acceptable
                operator='lte',
                priority='medium',
                description='95th percentile pod startup time'
            ),
            'scheduling_success_rate': BusinessTarget(
                metric_name='scheduling_success_rate',
                target_value=99.0,
                tolerance_percent=1.0,  # 98% minimum
                operator='gte',
                priority='critical',
                description='Scheduling success rate'
            ),
            'resource_waste': BusinessTarget(
                metric_name='resource_waste',
                target_value=15.0,  # Max 15% waste
                tolerance_percent=33.0,  # Up to 20% acceptable
                operator='lte',
                priority='medium',
                description='Resource allocation efficiency'
            ),
            'ml_confidence_avg': BusinessTarget(
                metric_name='ml_confidence_avg',
                target_value=80.0,
                tolerance_percent=12.5,  # 70% minimum
                operator='gte',
                priority='high',
                description='Average ML model confidence'
            ),
            'cache_hit_rate': BusinessTarget(
                metric_name='cache_hit_rate',
                target_value=95.0,
                tolerance_percent=5.0,  # 90% minimum
                operator='gte',
                priority='medium',
                description='Cache performance efficiency'
            ),
            'node_balance_score': BusinessTarget(
                metric_name='node_balance_score',
                target_value=85.0,
                tolerance_percent=15.0,  # 72% minimum
                operator='gte',
                priority='medium',
                description='Cluster load balancing effectiveness'
            )
        }
        
        # Prometheus queries for each metric
        self.prometheus_queries = {
            'cpu_utilization': 'avg(100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))',
            'memory_utilization': 'avg((1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100)',
            'availability': 'avg(up{job="kubernetes-nodes"}) * 100',
            'scheduling_latency_p95': 'histogram_quantile(0.95, rate(ml_scheduler_scheduling_duration_seconds_bucket[30m])) * 1000',
            'pod_startup_time_p95': 'histogram_quantile(0.95, rate(kubelet_pod_start_duration_seconds_bucket[30m]))',
            'scheduling_success_rate': 'rate(ml_scheduler_scheduling_success_total[30m]) / rate(ml_scheduler_scheduling_requests_total[30m]) * 100',
            'resource_waste': '100 - avg(cluster_resource_efficiency_percent)',
            'ml_confidence_avg': 'avg(ml_scheduler_ml_confidence_scores)',
            'cache_hit_rate': 'rate(ml_scheduler_cache_hits_total[30m]) / (rate(ml_scheduler_cache_hits_total[30m]) + rate(ml_scheduler_cache_misses_total[30m])) * 100',
            'node_balance_score': 'avg(ml_scheduler_node_balance_score)'
        }
    
    async def validate_all_metrics(self, time_window_minutes: int = 30) -> Dict[str, ValidationResult]:
        """
        Validate all business metrics against targets
        
        Args:
            time_window_minutes: Time window for metric collection
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        logger.info(f"Starting business metrics validation (window: {time_window_minutes}m)")
        
        # Collect current metrics
        current_metrics = await self.collect_current_metrics()
        
        # Validate each target
        for metric_name, target in self.business_targets.items():
            try:
                current_value = current_metrics.get(metric_name, 0.0)
                result = self.validate_metric(target, current_value)
                validation_results[metric_name] = result
                
                # Log result
                status = "✅ PASS" if result.passed else "❌ FAIL"
                logger.info(f"{status} {metric_name}: {current_value:.2f} "
                           f"(target: {target.target_value:.2f}, score: {result.score:.1f})")
                
            except Exception as e:
                logger.error(f"Failed to validate {metric_name}: {e}")
                validation_results[metric_name] = ValidationResult(
                    metric_name=metric_name,
                    current_value=0.0,
                    target_value=target.target_value,
                    tolerance_range=(0.0, 0.0),
                    passed=False,
                    score=0.0,
                    impact="validation_error",
                    timestamp=datetime.utcnow()
                )
        
        return validation_results
    
    async def collect_current_metrics(self) -> Dict[str, float]:
        """Collect current metrics from Prometheus and Kubernetes"""
        metrics = {}
        
        # Collect Prometheus metrics
        for metric_name, query in self.prometheus_queries.items():
            try:
                value = await self.query_prometheus(query)
                metrics[metric_name] = value
            except Exception as e:
                logger.warning(f"Failed to collect {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        # Collect additional Kubernetes metrics
        try:
            k8s_metrics = await self.collect_kubernetes_metrics()
            metrics.update(k8s_metrics)
        except Exception as e:
            logger.warning(f"Failed to collect Kubernetes metrics: {e}")
        
        return metrics
    
    async def collect_kubernetes_metrics(self) -> Dict[str, float]:
        """Collect metrics directly from Kubernetes API"""
        try:
            # Get cluster state
            nodes = self.k8s_core_v1.list_node()
            pods = self.k8s_core_v1.list_pod_for_all_namespaces()
            
            # Calculate node balance
            node_loads = []
            total_nodes = len(nodes.items)
            ready_nodes = 0
            
            for node in nodes.items:
                # Check if node is ready
                is_ready = any(c.status == 'True' and c.type == 'Ready' 
                              for c in node.status.conditions)
                if is_ready:
                    ready_nodes += 1
                
                # Calculate node load (simplified)
                node_pods = [p for p in pods.items 
                           if p.spec.node_name == node.metadata.name and 
                              p.status.phase == 'Running']
                node_loads.append(len(node_pods))
            
            # Calculate balance score (lower std deviation = better balance)
            if node_loads:
                avg_load = sum(node_loads) / len(node_loads)
                variance = sum((load - avg_load) ** 2 for load in node_loads) / len(node_loads)
                std_dev = variance ** 0.5
                balance_score = max(0, 100 - (std_dev / avg_load * 100 if avg_load > 0 else 100))
            else:
                balance_score = 0
            
            # Calculate actual availability
            availability = (ready_nodes / total_nodes * 100) if total_nodes > 0 else 0
            
            return {
                'node_balance_score': balance_score,
                'cluster_availability': availability,
                'total_nodes': float(total_nodes),
                'ready_nodes': float(ready_nodes),
                'total_pods': float(len(pods.items)),
                'running_pods': float(len([p for p in pods.items if p.status.phase == 'Running'])),
                'pending_pods': float(len([p for p in pods.items if p.status.phase == 'Pending']))
            }
            
        except Exception as e:
            logger.error(f"Failed to collect Kubernetes metrics: {e}")
            return {}
    
    def validate_metric(self, target: BusinessTarget, current_value: float) -> ValidationResult:
        """
        Validate a single metric against its target
        
        Args:
            target: Business target configuration
            current_value: Current metric value
            
        Returns:
            Validation result
        """
        try:
            # Calculate tolerance range
            if target.operator == 'range':
                tolerance = target.target_value * target.tolerance_percent / 100
                tolerance_range = (
                    target.target_value - tolerance,
                    target.target_value + tolerance
                )
                passed = tolerance_range[0] <= current_value <= tolerance_range[1]
                
                # Score based on distance from target
                if passed:
                    distance = abs(current_value - target.target_value)
                    max_distance = tolerance
                    score = 100 - (distance / max_distance * 50)  # 50-100 range for passing
                else:
                    # Calculate how far outside tolerance
                    if current_value < tolerance_range[0]:
                        distance = tolerance_range[0] - current_value
                    else:
                        distance = current_value - tolerance_range[1]
                    score = max(0, 50 - (distance / tolerance * 50))  # 0-50 range for failing
                
            elif target.operator == 'gte':
                min_value = target.target_value * (1 - target.tolerance_percent / 100)
                tolerance_range = (min_value, float('inf'))
                passed = current_value >= min_value
                
                if passed:
                    # Higher values are better
                    excess = current_value - target.target_value
                    score = min(100, 75 + (excess / target.target_value * 25))
                else:
                    # Score based on how close to minimum
                    score = max(0, (current_value / min_value) * 75)
                
            elif target.operator == 'lte':
                max_value = target.target_value * (1 + target.tolerance_percent / 100)
                tolerance_range = (0, max_value)
                passed = current_value <= max_value
                
                if passed:
                    # Lower values are better
                    score = min(100, 100 - (current_value / target.target_value * 25))
                else:
                    # Score based on how much over limit
                    excess = current_value - max_value
                    score = max(0, 75 - (excess / target.target_value * 75))
                
            else:  # eq
                tolerance = target.target_value * target.tolerance_percent / 100
                tolerance_range = (
                    target.target_value - tolerance,
                    target.target_value + tolerance
                )
                passed = tolerance_range[0] <= current_value <= tolerance_range[1]
                
                if passed:
                    distance = abs(current_value - target.target_value)
                    score = 100 - (distance / tolerance * 25)
                else:
                    score = 0
            
            # Determine business impact
            if target.priority == 'critical' and not passed:
                impact = 'critical_failure'
            elif target.priority == 'high' and not passed:
                impact = 'performance_degradation'
            elif not passed:
                impact = 'optimization_opportunity'
            else:
                impact = 'target_achieved'
            
            return ValidationResult(
                metric_name=target.metric_name,
                current_value=current_value,
                target_value=target.target_value,
                tolerance_range=tolerance_range,
                passed=passed,
                score=round(score, 2),
                impact=impact,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Validation failed for {target.metric_name}: {e}")
            return ValidationResult(
                metric_name=target.metric_name,
                current_value=current_value,
                target_value=target.target_value,
                tolerance_range=(0.0, 0.0),
                passed=False,
                score=0.0,
                impact='validation_error',
                timestamp=datetime.utcnow()
            )
    
    async def query_prometheus(self, query: str, time_range: str = "30m") -> float:
        """
        Query Prometheus for metric value
        
        Args:
            query: PromQL query
            time_range: Time range for query
            
        Returns:
            Metric value
        """
        try:
            # Simulate Prometheus queries with realistic HYDATIS cluster values
            # In production, implement actual HTTP client
            
            if 'cpu' in query.lower():
                return 67.2  # 67.2% CPU utilization (target: 65%)
            elif 'memory' in query.lower():
                return 74.5  # 74.5% memory utilization
            elif 'availability' in query.lower() or 'up{' in query:
                return 99.8  # 99.8% availability (exceeds 99.7% target)
            elif 'scheduling_duration' in query:
                return 0.045  # 45ms scheduling latency
            elif 'pod_start_duration' in query:
                return 28.5  # 28.5s pod startup time
            elif 'success_total' in query and 'requests_total' in query:
                return 97.8  # 97.8% scheduling success rate
            elif 'efficiency' in query:
                return 87.5  # 87.5% resource efficiency
            elif 'confidence' in query:
                return 82.3  # 82.3% average ML confidence
            elif 'cache_hits' in query:
                return 94.7  # 94.7% cache hit rate
            elif 'balance' in query:
                return 89.2  # 89.2% node balance score
            else:
                return 85.0  # Default value
                
        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return 0.0
    
    async def generate_business_report(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """
        Generate comprehensive business impact report
        
        Args:
            validation_results: Results from metrics validation
            
        Returns:
            Business report with ROI calculations and recommendations
        """
        try:
            # Calculate overall scores
            critical_metrics = [r for r in validation_results.values() 
                              if self.business_targets[r.metric_name].priority == 'critical']
            high_metrics = [r for r in validation_results.values() 
                           if self.business_targets[r.metric_name].priority == 'high']
            
            critical_score = sum(r.score for r in critical_metrics) / len(critical_metrics) if critical_metrics else 100
            high_score = sum(r.score for r in high_metrics) / len(high_metrics) if high_metrics else 100
            overall_score = (critical_score * 0.6 + high_score * 0.4)
            
            # Calculate business impact
            passed_count = sum(1 for r in validation_results.values() if r.passed)
            total_count = len(validation_results)
            pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
            
            # ROI Calculation (based on HYDATIS cluster targets)
            baseline_costs = {
                'infrastructure': 100000,  # Monthly infrastructure cost
                'operations': 50000,       # Monthly operations cost
                'downtime': 25000          # Monthly downtime cost
            }
            
            # Calculate savings from improved efficiency
            cpu_result = validation_results.get('cpu_utilization')
            availability_result = validation_results.get('availability')
            
            infrastructure_savings = 0
            if cpu_result and cpu_result.passed:
                # Better CPU utilization = fewer nodes needed
                efficiency_gain = (85 - cpu_result.current_value) / 85  # Baseline 85% → target 65%
                infrastructure_savings = baseline_costs['infrastructure'] * efficiency_gain * 0.3
            
            downtime_savings = 0
            if availability_result and availability_result.passed:
                # Improved availability = reduced downtime costs
                availability_improvement = (availability_result.current_value - 95.2) / 4.5  # 95.2% → 99.7%
                downtime_savings = baseline_costs['downtime'] * availability_improvement
            
            total_monthly_savings = infrastructure_savings + downtime_savings
            annual_roi = (total_monthly_savings * 12 / 150000) * 100  # ROI based on $150k investment
            
            # Determine deployment readiness
            critical_failures = [r for r in validation_results.values() 
                               if not r.passed and self.business_targets[r.metric_name].priority == 'critical']
            
            deployment_ready = len(critical_failures) == 0 and overall_score >= 80
            
            # Generate recommendations
            recommendations = self.generate_recommendations(validation_results)
            
            report = {
                'validation_summary': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'overall_score': round(overall_score, 2),
                    'pass_rate': round(pass_rate, 2),
                    'metrics_passed': passed_count,
                    'metrics_total': total_count,
                    'critical_score': round(critical_score, 2),
                    'high_priority_score': round(high_score, 2)
                },
                'business_impact': {
                    'deployment_ready': deployment_ready,
                    'estimated_monthly_savings': round(total_monthly_savings, 2),
                    'estimated_annual_roi_percent': round(annual_roi, 2),
                    'infrastructure_savings': round(infrastructure_savings, 2),
                    'downtime_savings': round(downtime_savings, 2),
                    'target_achievement': {
                        'cpu_optimization': cpu_result.passed if cpu_result else False,
                        'availability_improvement': availability_result.passed if availability_result else False
                    }
                },
                'detailed_results': {name: asdict(result) for name, result in validation_results.items()},
                'recommendations': recommendations,
                'next_steps': self.determine_next_steps(validation_results, deployment_ready)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate business report: {e}")
            return {'error': str(e)}
    
    def generate_recommendations(self, validation_results: Dict[str, ValidationResult]) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        # Check CPU utilization
        cpu_result = validation_results.get('cpu_utilization')
        if cpu_result:
            if cpu_result.current_value > 70:
                recommendations.append("CPU utilization high: Consider cluster auto-scaling or workload optimization")
            elif cpu_result.current_value < 60:
                recommendations.append("CPU utilization low: Opportunity to consolidate workloads or reduce cluster size")
        
        # Check availability
        availability_result = validation_results.get('availability')
        if availability_result and not availability_result.passed:
            recommendations.append("Availability below target: Investigate node failures and implement HA improvements")
        
        # Check scheduling performance
        latency_result = validation_results.get('scheduling_latency_p95')
        if latency_result and not latency_result.passed:
            recommendations.append("Scheduling latency high: Optimize ML model inference or increase cache hit rate")
        
        # Check ML confidence
        confidence_result = validation_results.get('ml_confidence_avg')
        if confidence_result and confidence_result.current_value < 75:
            recommendations.append("ML confidence low: Retrain models with recent data or adjust feature engineering")
        
        # Check cache performance
        cache_result = validation_results.get('cache_hit_rate')
        if cache_result and cache_result.current_value < 90:
            recommendations.append("Cache hit rate low: Increase cache TTL or improve cache key generation")
        
        # Overall performance recommendations
        failed_critical = [r for r in validation_results.values() 
                          if not r.passed and self.business_targets[r.metric_name].priority == 'critical']
        
        if failed_critical:
            recommendations.append("Critical metrics failing: Immediate investigation required before production rollout")
        
        if not recommendations:
            recommendations.append("All metrics within targets: Ready for production deployment")
        
        return recommendations
    
    def determine_next_steps(self, validation_results: Dict[str, ValidationResult], 
                           deployment_ready: bool) -> List[str]:
        """Determine next steps based on validation results"""
        next_steps = []
        
        if deployment_ready:
            next_steps.extend([
                "✅ Proceed with 10% canary deployment",
                "Monitor metrics during canary phase",
                "Validate business targets after 30-minute window",
                "Auto-promote to 50% if targets met"
            ])
        else:
            next_steps.extend([
                "❌ Fix critical metric failures before deployment",
                "Re-run validation after fixes",
                "Consider gradual fixes with monitoring"
            ])
            
            # Add specific actions for failed metrics
            failed_critical = [r for r in validation_results.values() 
                             if not r.passed and self.business_targets[r.metric_name].priority == 'critical']
            
            for result in failed_critical:
                if result.metric_name == 'cpu_utilization':
                    next_steps.append("Optimize workload distribution or adjust ML model weights")
                elif result.metric_name == 'availability':
                    next_steps.append("Investigate and fix node failures")
                elif result.metric_name == 'scheduling_success_rate':
                    next_steps.append("Debug scheduler failures and improve fallback logic")
        
        return next_steps
    
    async def continuous_validation(self, rollout_id: str, duration_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Run continuous validation for specified duration
        
        Args:
            rollout_id: Rollout to monitor
            duration_hours: How long to monitor
            
        Returns:
            List of validation reports over time
        """
        reports = []
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        
        logger.info(f"Starting continuous validation for {rollout_id} ({duration_hours}h)")
        
        try:
            while datetime.utcnow() < end_time:
                # Run validation
                validation_results = await self.validate_all_metrics()
                report = await self.generate_business_report(validation_results)
                
                report['validation_id'] = f"{rollout_id}-{len(reports)+1}"
                report['elapsed_hours'] = (datetime.utcnow() - (end_time - timedelta(hours=duration_hours))).total_seconds() / 3600
                
                reports.append(report)
                
                # Check for critical failures
                if not report['business_impact']['deployment_ready']:
                    logger.warning(f"Critical failure detected in continuous validation")
                    break
                
                # Wait before next validation (15 minutes)
                await asyncio.sleep(15 * 60)
            
            logger.info(f"Continuous validation completed: {len(reports)} reports generated")
            return reports
            
        except Exception as e:
            logger.error(f"Continuous validation failed: {e}")
            return reports
    
    def export_validation_report(self, report: Dict[str, Any], 
                                format: str = 'json') -> str:
        """Export validation report in specified format"""
        try:
            if format.lower() == 'json':
                return json.dumps(report, indent=2, default=str)
            elif format.lower() == 'yaml':
                return yaml.dump(report, default_flow_style=False)
            else:
                # Generate human-readable summary
                summary = []
                summary.append("=== ML Scheduler Business Metrics Validation ===")
                summary.append(f"Timestamp: {report['validation_summary']['timestamp']}")
                summary.append(f"Overall Score: {report['validation_summary']['overall_score']:.1f}/100")
                summary.append(f"Pass Rate: {report['validation_summary']['pass_rate']:.1f}%")
                summary.append("")
                
                summary.append("Business Impact:")
                summary.append(f"  Deployment Ready: {'✅ Yes' if report['business_impact']['deployment_ready'] else '❌ No'}")
                summary.append(f"  Estimated Annual ROI: {report['business_impact']['estimated_annual_roi_percent']:.1f}%")
                summary.append(f"  Monthly Savings: ${report['business_impact']['estimated_monthly_savings']:,.2f}")
                summary.append("")
                
                summary.append("Key Metrics:")
                for name, result in report['detailed_results'].items():
                    status = "✅" if result['passed'] else "❌"
                    summary.append(f"  {status} {name}: {result['current_value']:.2f} (target: {result['target_value']:.2f})")
                
                summary.append("")
                summary.append("Recommendations:")
                for rec in report['recommendations']:
                    summary.append(f"  • {rec}")
                
                return "\n".join(summary)
                
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return str(report)


# CLI interface for standalone usage
async def main():
    parser = argparse.ArgumentParser(description='ML Scheduler Business Metrics Validator')
    parser.add_argument('--prometheus-url', default='http://prometheus-server:9090',
                       help='Prometheus server URL')
    parser.add_argument('--kubeconfig', help='Path to kubeconfig file')
    parser.add_argument('--output-format', choices=['json', 'yaml', 'summary'], 
                       default='summary', help='Output format')
    parser.add_argument('--continuous', type=int, metavar='HOURS',
                       help='Run continuous validation for specified hours')
    parser.add_argument('--export-file', help='Export report to file')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = BusinessMetricsValidator(
        prometheus_url=args.prometheus_url,
        kubeconfig_path=args.kubeconfig
    )
    
    try:
        if args.continuous:
            # Run continuous validation
            reports = await validator.continuous_validation(
                rollout_id=f"validation-{int(time.time())}", 
                duration_hours=args.continuous
            )
            
            # Export final report
            final_report = reports[-1] if reports else {}
            output = validator.export_validation_report(final_report, args.output_format)
            
        else:
            # Run single validation
            validation_results = await validator.validate_all_metrics()
            report = await validator.generate_business_report(validation_results)
            output = validator.export_validation_report(report, args.output_format)
        
        # Output results
        if args.export_file:
            with open(args.export_file, 'w') as f:
                f.write(output)
            print(f"Report exported to {args.export_file}")
        else:
            print(output)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))