#!/usr/bin/env python3

import asyncio
import time
import logging
import statistics
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from kubernetes import client, config
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class LatencyTestResult:
    """Result from latency testing"""
    test_name: str
    sample_count: int
    average_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    standard_deviation_ms: float
    cache_hit_rate: float
    ml_service_latency_ms: float
    kubernetes_api_latency_ms: float
    target_met: bool
    timestamp: datetime

class SchedulingLatencyTester:
    """
    Detailed latency testing for ML scheduler components
    Measures end-to-end scheduling latency and component breakdown
    """
    
    def __init__(self, 
                 kubeconfig_path: Optional[str] = None,
                 ml_scorer_url: str = "http://combined-ml-scorer.ml-scheduler.svc.cluster.local:8080"):
        """
        Initialize latency tester
        
        Args:
            kubeconfig_path: Path to kubeconfig file
            ml_scorer_url: URL for ML scoring service
        """
        self.ml_scorer_url = ml_scorer_url
        
        # Initialize Kubernetes client
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                config.load_incluster_config()
            
            self.k8s_core_v1 = client.CoreV1Api()
            self.k8s_apps_v1 = client.AppsV1Api()
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    async def test_end_to_end_latency(self, sample_count: int = 100) -> LatencyTestResult:
        """
        Test end-to-end scheduling latency
        
        Args:
            sample_count: Number of test samples to collect
            
        Returns:
            Latency test results
        """
        logger.info(f"Starting end-to-end latency test ({sample_count} samples)")
        
        latencies = []
        cache_hits = 0
        ml_latencies = []
        k8s_latencies = []
        
        try:
            for i in range(sample_count):
                start_time = time.time()
                
                # Create test pod
                pod_name = f"latency-test-{i}-{int(time.time())}"
                
                # Measure K8s API latency
                k8s_start = time.time()
                pod = await self.create_latency_test_pod(pod_name)
                k8s_latency = (time.time() - k8s_start) * 1000
                k8s_latencies.append(k8s_latency)
                
                # Wait for scheduling and measure total time
                scheduled_time = await self.wait_for_scheduling(pod_name, timeout=30)
                
                if scheduled_time:
                    total_latency = (scheduled_time - start_time) * 1000
                    latencies.append(total_latency)
                    
                    # Test ML scoring latency separately
                    ml_latency = await self.test_ml_scoring_latency(pod)
                    if ml_latency:
                        ml_latencies.append(ml_latency)
                    
                    # Check if result was cached (simplified detection)
                    if ml_latency and ml_latency < 10:  # <10ms likely indicates cache hit
                        cache_hits += 1
                
                # Cleanup
                await self.cleanup_test_pod(pod_name)
                
                # Progress logging
                if (i + 1) % 20 == 0:
                    logger.info(f"Completed {i + 1}/{sample_count} latency tests")
                
                # Brief pause to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            # Calculate statistics
            if latencies:
                result = LatencyTestResult(
                    test_name="end_to_end_latency",
                    sample_count=len(latencies),
                    average_latency_ms=statistics.mean(latencies),
                    median_latency_ms=statistics.median(latencies),
                    p90_latency_ms=np.percentile(latencies, 90),
                    p95_latency_ms=np.percentile(latencies, 95),
                    p99_latency_ms=np.percentile(latencies, 99),
                    min_latency_ms=min(latencies),
                    max_latency_ms=max(latencies),
                    standard_deviation_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
                    cache_hit_rate=(cache_hits / len(latencies) * 100) if latencies else 0,
                    ml_service_latency_ms=statistics.mean(ml_latencies) if ml_latencies else 0,
                    kubernetes_api_latency_ms=statistics.mean(k8s_latencies) if k8s_latencies else 0,
                    target_met=np.percentile(latencies, 99) <= 100,  # P99 < 100ms target
                    timestamp=datetime.utcnow()
                )
                
                logger.info(f"Latency test completed: P95={result.p95_latency_ms:.1f}ms, "
                           f"P99={result.p99_latency_ms:.1f}ms, cache_hit={result.cache_hit_rate:.1f}%")
                
                return result
            else:
                raise Exception("No successful scheduling measurements collected")
                
        except Exception as e:
            logger.error(f"End-to-end latency test failed: {e}")
            raise
    
    async def create_latency_test_pod(self, pod_name: str) -> client.V1Pod:
        """Create a test pod for latency measurement"""
        pod_manifest = client.V1Pod(
            metadata=client.V1ObjectMeta(
                name=pod_name,
                namespace='default',
                labels={
                    'app': 'latency-test',
                    'test-type': 'scheduling-latency'
                }
            ),
            spec=client.V1PodSpec(
                scheduler_name='ml-scheduler',
                containers=[
                    client.V1Container(
                        name='test-container',
                        image='nginx:alpine',
                        resources=client.V1ResourceRequirements(
                            requests={
                                'cpu': '100m',
                                'memory': '128Mi'
                            },
                            limits={
                                'cpu': '200m',
                                'memory': '256Mi'
                            }
                        )
                    )
                ],
                restart_policy='Never',
                termination_grace_period_seconds=1
            )
        )
        
        return self.k8s_core_v1.create_namespaced_pod(
            namespace='default',
            body=pod_manifest
        )
    
    async def wait_for_scheduling(self, pod_name: str, timeout: int = 30) -> Optional[float]:
        """
        Wait for pod to be scheduled and return scheduling time
        
        Args:
            pod_name: Name of pod to monitor
            timeout: Timeout in seconds
            
        Returns:
            Timestamp when pod was scheduled, or None if timeout
        """
        start_wait = time.time()
        
        while time.time() - start_wait < timeout:
            try:
                pod = self.k8s_core_v1.read_namespaced_pod(
                    name=pod_name,
                    namespace='default'
                )
                
                # Check if pod has been scheduled (assigned to a node)
                if pod.spec.node_name:
                    return time.time()
                
                # Check for scheduling failures
                if pod.status.conditions:
                    for condition in pod.status.conditions:
                        if (condition.type == 'PodScheduled' and 
                            condition.status == 'False' and
                            'Unschedulable' in condition.reason):
                            logger.warning(f"Pod {pod_name} unschedulable: {condition.message}")
                            return None
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error checking pod {pod_name}: {e}")
                return None
        
        logger.warning(f"Pod {pod_name} scheduling timeout after {timeout}s")
        return None
    
    async def test_ml_scoring_latency(self, pod: client.V1Pod) -> Optional[float]:
        """
        Test ML scoring service latency directly
        
        Args:
            pod: Pod object for scoring request
            
        Returns:
            ML scoring latency in milliseconds
        """
        try:
            # Prepare scoring request
            scoring_request = {
                'request_id': f"latency-test-{int(time.time())}",
                'pod_spec': {
                    'resources': {
                        'requests': {
                            'cpu': '100m',
                            'memory': '128Mi'
                        }
                    },
                    'priorityClassName': 'normal'
                },
                'node_candidates': [
                    {
                        'name': 'hydatis-worker-1',
                        'allocatable': {
                            'cpu': '4000m',
                            'memory': '8Gi'
                        }
                    }
                ],
                'current_metrics': {
                    'cluster': {
                        'cpu_usage_percent': 65,
                        'memory_usage_percent': 70
                    }
                },
                'user_id': 'latency-test'
            }
            
            # Make request to ML scorer
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ml_scorer_url}/v1/score",
                    json=scoring_request,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        await response.json()
                        latency = (time.time() - start_time) * 1000
                        return latency
                    else:
                        logger.warning(f"ML scoring request failed: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.warning(f"ML scoring latency test failed: {e}")
            return None
    
    async def cleanup_test_pod(self, pod_name: str):
        """Clean up test pod"""
        try:
            self.k8s_core_v1.delete_namespaced_pod(
                name=pod_name,
                namespace='default',
                grace_period_seconds=0
            )
        except Exception as e:
            logger.warning(f"Failed to cleanup pod {pod_name}: {e}")
    
    async def test_component_latencies(self) -> Dict[str, LatencyTestResult]:
        """Test latency of individual components"""
        results = {}
        
        # Test ML scoring service latency
        logger.info("Testing ML scoring service latency...")
        ml_latencies = []
        
        for i in range(50):
            latency = await self.test_direct_ml_scoring()
            if latency:
                ml_latencies.append(latency)
        
        if ml_latencies:
            results['ml_scoring'] = LatencyTestResult(
                test_name="ml_scoring_latency",
                sample_count=len(ml_latencies),
                average_latency_ms=statistics.mean(ml_latencies),
                median_latency_ms=statistics.median(ml_latencies),
                p90_latency_ms=np.percentile(ml_latencies, 90),
                p95_latency_ms=np.percentile(ml_latencies, 95),
                p99_latency_ms=np.percentile(ml_latencies, 99),
                min_latency_ms=min(ml_latencies),
                max_latency_ms=max(ml_latencies),
                standard_deviation_ms=statistics.stdev(ml_latencies) if len(ml_latencies) > 1 else 0,
                cache_hit_rate=0,  # Not applicable for direct testing
                ml_service_latency_ms=statistics.mean(ml_latencies),
                kubernetes_api_latency_ms=0,
                target_met=np.percentile(ml_latencies, 95) <= 30,  # P95 < 30ms target
                timestamp=datetime.utcnow()
            )
        
        # Test Kubernetes API latency
        logger.info("Testing Kubernetes API latency...")
        k8s_latencies = []
        
        for i in range(50):
            latency = await self.test_kubernetes_api_latency()
            if latency:
                k8s_latencies.append(latency)
        
        if k8s_latencies:
            results['kubernetes_api'] = LatencyTestResult(
                test_name="kubernetes_api_latency",
                sample_count=len(k8s_latencies),
                average_latency_ms=statistics.mean(k8s_latencies),
                median_latency_ms=statistics.median(k8s_latencies),
                p90_latency_ms=np.percentile(k8s_latencies, 90),
                p95_latency_ms=np.percentile(k8s_latencies, 95),
                p99_latency_ms=np.percentile(k8s_latencies, 99),
                min_latency_ms=min(k8s_latencies),
                max_latency_ms=max(k8s_latencies),
                standard_deviation_ms=statistics.stdev(k8s_latencies) if len(k8s_latencies) > 1 else 0,
                cache_hit_rate=0,
                ml_service_latency_ms=0,
                kubernetes_api_latency_ms=statistics.mean(k8s_latencies),
                target_met=np.percentile(k8s_latencies, 95) <= 10,  # P95 < 10ms target
                timestamp=datetime.utcnow()
            )
        
        return results
    
    async def test_direct_ml_scoring(self) -> Optional[float]:
        """Test direct ML scoring service latency"""
        try:
            scoring_request = {
                'request_id': f"direct-test-{int(time.time())}",
                'pod_spec': {
                    'resources': {
                        'requests': {'cpu': '100m', 'memory': '128Mi'}
                    }
                },
                'node_candidates': [{
                    'name': 'test-node',
                    'allocatable': {'cpu': '4000m', 'memory': '8Gi'}
                }],
                'current_metrics': {
                    'cluster': {'cpu_usage_percent': 65}
                }
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ml_scorer_url}/v1/score",
                    json=scoring_request,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        await response.json()
                        return (time.time() - start_time) * 1000
                    else:
                        return None
                        
        except Exception as e:
            logger.debug(f"ML scoring test failed: {e}")
            return None
    
    async def test_kubernetes_api_latency(self) -> Optional[float]:
        """Test Kubernetes API latency"""
        try:
            start_time = time.time()
            
            # Simple API call to measure latency
            self.k8s_core_v1.list_node(limit=1)
            
            return (time.time() - start_time) * 1000
            
        except Exception as e:
            logger.debug(f"Kubernetes API test failed: {e}")
            return None
    
    async def test_cache_performance(self, sample_count: int = 100) -> Dict[str, Any]:
        """Test cache performance and hit rates"""
        logger.info(f"Testing cache performance ({sample_count} samples)")
        
        try:
            # Test same request multiple times to measure cache efficiency
            base_request = {
                'request_id': 'cache-test-base',
                'pod_spec': {
                    'resources': {
                        'requests': {'cpu': '200m', 'memory': '256Mi'}
                    }
                },
                'node_candidates': [{
                    'name': 'cache-test-node',
                    'allocatable': {'cpu': '4000m', 'memory': '8Gi'}
                }],
                'current_metrics': {
                    'cluster': {'cpu_usage_percent': 60}
                }
            }
            
            # First request (cache miss)
            first_latency = await self.measure_scoring_latency(base_request)
            
            # Subsequent requests (should be cache hits)
            cache_latencies = []
            for i in range(10):
                latency = await self.measure_scoring_latency(base_request)
                if latency:
                    cache_latencies.append(latency)
            
            # Test different requests (cache misses)
            miss_latencies = []
            for i in range(10):
                varied_request = base_request.copy()
                varied_request['request_id'] = f'cache-test-{i}'
                varied_request['pod_spec']['resources']['requests']['cpu'] = f'{100 + i * 10}m'
                
                latency = await self.measure_scoring_latency(varied_request)
                if latency:
                    miss_latencies.append(latency)
            
            return {
                'first_request_latency_ms': first_latency or 0,
                'avg_cache_hit_latency_ms': statistics.mean(cache_latencies) if cache_latencies else 0,
                'avg_cache_miss_latency_ms': statistics.mean(miss_latencies) if miss_latencies else 0,
                'cache_speedup_factor': (statistics.mean(miss_latencies) / statistics.mean(cache_latencies)) if cache_latencies and miss_latencies else 1,
                'cache_hit_samples': len(cache_latencies),
                'cache_miss_samples': len(miss_latencies)
            }
            
        except Exception as e:
            logger.error(f"Cache performance test failed: {e}")
            return {}
    
    async def measure_scoring_latency(self, request: Dict[str, Any]) -> Optional[float]:
        """Measure latency of a single scoring request"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ml_scorer_url}/v1/score",
                    json=request,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        await response.json()
                        return (time.time() - start_time) * 1000
                    else:
                        return None
                        
        except Exception:
            return None
    
    async def test_concurrent_scheduling(self, concurrent_count: int = 50) -> LatencyTestResult:
        """
        Test scheduling latency under concurrent load
        
        Args:
            concurrent_count: Number of concurrent scheduling requests
            
        Returns:
            Latency test results under load
        """
        logger.info(f"Testing concurrent scheduling latency ({concurrent_count} concurrent)")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_count)
        
        async def single_concurrent_test(test_id: int) -> Optional[float]:
            async with semaphore:
                start_time = time.time()
                pod_name = f"concurrent-test-{test_id}-{int(time.time())}"
                
                try:
                    await self.create_latency_test_pod(pod_name)
                    scheduled_time = await self.wait_for_scheduling(pod_name, timeout=20)
                    
                    if scheduled_time:
                        latency = (scheduled_time - start_time) * 1000
                        await self.cleanup_test_pod(pod_name)
                        return latency
                    else:
                        await self.cleanup_test_pod(pod_name)
                        return None
                        
                except Exception as e:
                    logger.debug(f"Concurrent test {test_id} failed: {e}")
                    await self.cleanup_test_pod(pod_name)
                    return None
        
        # Run concurrent tests
        tasks = [single_concurrent_test(i) for i in range(concurrent_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        latencies = [r for r in results if isinstance(r, float) and r > 0]
        
        if latencies:
            return LatencyTestResult(
                test_name="concurrent_scheduling",
                sample_count=len(latencies),
                average_latency_ms=statistics.mean(latencies),
                median_latency_ms=statistics.median(latencies),
                p90_latency_ms=np.percentile(latencies, 90),
                p95_latency_ms=np.percentile(latencies, 95),
                p99_latency_ms=np.percentile(latencies, 99),
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                standard_deviation_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
                cache_hit_rate=0,
                ml_service_latency_ms=0,
                kubernetes_api_latency_ms=0,
                target_met=np.percentile(latencies, 99) <= 150,  # Higher threshold under load
                timestamp=datetime.utcnow()
            )
        else:
            raise Exception("No successful concurrent scheduling measurements")
    
    def generate_latency_report(self, results: Dict[str, LatencyTestResult], 
                              cache_results: Dict[str, Any]) -> str:
        """Generate comprehensive latency test report"""
        try:
            report = []
            report.append("=" * 80)
            report.append("ML SCHEDULER LATENCY TEST REPORT")
            report.append("=" * 80)
            report.append(f"Test Date: {datetime.utcnow().isoformat()}")
            report.append("")
            
            # Summary
            report.append("LATENCY SUMMARY")
            report.append("-" * 40)
            
            for test_name, result in results.items():
                target_status = "✅ PASS" if result.target_met else "❌ FAIL"
                report.append(f"{test_name.upper()}:")
                report.append(f"  {target_status} P99: {result.p99_latency_ms:.1f}ms")
                report.append(f"  P95: {result.p95_latency_ms:.1f}ms")
                report.append(f"  Average: {result.average_latency_ms:.1f}ms")
                report.append(f"  Samples: {result.sample_count}")
                report.append("")
            
            # Cache Performance
            if cache_results:
                report.append("CACHE PERFORMANCE")
                report.append("-" * 40)
                speedup = cache_results.get('cache_speedup_factor', 1)
                report.append(f"Cache Speedup Factor: {speedup:.1f}x")
                report.append(f"Cache Hit Latency: {cache_results.get('avg_cache_hit_latency_ms', 0):.1f}ms")
                report.append(f"Cache Miss Latency: {cache_results.get('avg_cache_miss_latency_ms', 0):.1f}ms")
                report.append("")
            
            # Performance Targets Assessment
            report.append("PERFORMANCE TARGETS")
            report.append("-" * 40)
            
            end_to_end = results.get('end_to_end_latency')
            if end_to_end:
                report.append(f"End-to-End P99 Target (<100ms): {'✅ PASS' if end_to_end.p99_latency_ms <= 100 else '❌ FAIL'}")
                report.append(f"End-to-End P95 Target (<50ms): {'✅ PASS' if end_to_end.p95_latency_ms <= 50 else '❌ FAIL'}")
            
            ml_scoring = results.get('ml_scoring')
            if ml_scoring:
                report.append(f"ML Scoring P95 Target (<30ms): {'✅ PASS' if ml_scoring.p95_latency_ms <= 30 else '❌ FAIL'}")
            
            concurrent = results.get('concurrent_scheduling')
            if concurrent:
                report.append(f"Concurrent Load P99 Target (<150ms): {'✅ PASS' if concurrent.p99_latency_ms <= 150 else '❌ FAIL'}")
            
            report.append("")
            
            # Recommendations
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            
            recommendations = []
            
            if end_to_end and end_to_end.p99_latency_ms > 100:
                recommendations.append("Optimize end-to-end scheduling pipeline - consider ML model optimization")
            
            if ml_scoring and ml_scoring.p95_latency_ms > 30:
                recommendations.append("Optimize ML scoring service - check model inference time and feature processing")
            
            if cache_results.get('cache_speedup_factor', 1) < 3:
                recommendations.append("Improve cache effectiveness - increase TTL or optimize cache key generation")
            
            if not recommendations:
                recommendations.append("All latency targets met - system ready for production")
            
            for rec in recommendations:
                report.append(f"• {rec}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Failed to generate latency report: {e}")
            return f"Report generation failed: {e}"


# CLI interface
async def main():
    parser = argparse.ArgumentParser(description='ML Scheduler Latency Testing')
    parser.add_argument('--test-type', 
                       choices=['end-to-end', 'components', 'concurrent', 'cache', 'all'],
                       default='all', help='Type of latency test to run')
    parser.add_argument('--samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--concurrent', type=int, default=50, help='Concurrent requests for load test')
    parser.add_argument('--kubeconfig', help='Path to kubeconfig file')
    parser.add_argument('--ml-scorer-url', 
                       default='http://combined-ml-scorer.ml-scheduler.svc.cluster.local:8080',
                       help='ML scorer service URL')
    parser.add_argument('--export-json', help='Export results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = SchedulingLatencyTester(
        kubeconfig_path=args.kubeconfig,
        ml_scorer_url=args.ml_scorer_url
    )
    
    try:
        results = {}
        cache_results = {}
        
        if args.test_type in ['end-to-end', 'all']:
            results['end_to_end_latency'] = await tester.test_end_to_end_latency(args.samples)
        
        if args.test_type in ['components', 'all']:
            component_results = await tester.test_component_latencies()
            results.update(component_results)
        
        if args.test_type in ['concurrent', 'all']:
            results['concurrent_scheduling'] = await tester.test_concurrent_scheduling(args.concurrent)
        
        if args.test_type in ['cache', 'all']:
            cache_results = await tester.test_cache_performance()
        
        # Generate and display report
        report = tester.generate_latency_report(results, cache_results)
        print(report)
        
        # Export results if requested
        if args.export_json:
            export_data = {
                'test_suite': 'ml_scheduler_latency',
                'timestamp': datetime.utcnow().isoformat(),
                'latency_results': {name: asdict(result) for name, result in results.items()},
                'cache_results': cache_results
            }
            
            with open(args.export_json, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Results exported to {args.export_json}")
        
        # Return success if all targets met
        all_targets_met = all(result.target_met for result in results.values())
        return 0 if all_targets_met else 1
        
    except Exception as e:
        logger.error(f"Latency testing failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))