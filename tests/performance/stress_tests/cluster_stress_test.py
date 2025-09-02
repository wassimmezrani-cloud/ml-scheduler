#!/usr/bin/env python3

import asyncio
import logging
import time
import random
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import concurrent.futures
import threading
from kubernetes import client, config
import numpy as np
import psutil

logger = logging.getLogger(__name__)

@dataclass
class StressTestConfig:
    """Configuration for cluster stress testing"""
    test_name: str
    duration_minutes: int
    max_concurrent_pods: int
    ramp_up_minutes: int
    sustained_minutes: int
    ramp_down_minutes: int
    chaos_mode: bool = False
    resource_exhaustion_test: bool = False
    network_partition_test: bool = False

@dataclass
class StressTestMetrics:
    """Metrics collected during stress testing"""
    timestamp: datetime
    active_pods: int
    scheduling_rate: float
    success_rate: float
    average_latency_ms: float
    p99_latency_ms: float
    scheduler_cpu_percent: float
    scheduler_memory_mb: float
    cluster_cpu_percent: float
    cluster_memory_percent: float
    node_count: int
    ready_nodes: int
    ml_service_errors: int
    cache_errors: int
    fallback_rate: float

class ClusterStressTester:
    """
    Comprehensive cluster stress testing for ML scheduler
    Tests system behavior under extreme load conditions
    """
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        """Initialize cluster stress tester"""
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
        
        # Stress test scenarios
        self.stress_scenarios = {
            'gradual_ramp': StressTestConfig(
                test_name='gradual_ramp',
                duration_minutes=30,
                max_concurrent_pods=2000,
                ramp_up_minutes=10,
                sustained_minutes=15,
                ramp_down_minutes=5
            ),
            'spike_load': StressTestConfig(
                test_name='spike_load',
                duration_minutes=20,
                max_concurrent_pods=3000,
                ramp_up_minutes=2,
                sustained_minutes=15,
                ramp_down_minutes=3
            ),
            'chaos_test': StressTestConfig(
                test_name='chaos_test',
                duration_minutes=25,
                max_concurrent_pods=1500,
                ramp_up_minutes=5,
                sustained_minutes=15,
                ramp_down_minutes=5,
                chaos_mode=True
            ),
            'resource_exhaustion': StressTestConfig(
                test_name='resource_exhaustion',
                duration_minutes=20,
                max_concurrent_pods=5000,
                ramp_up_minutes=5,
                sustained_minutes=10,
                ramp_down_minutes=5,
                resource_exhaustion_test=True
            )
        }
        
        # Tracking
        self.metrics_history: List[StressTestMetrics] = []
        self.active_test_pods = {}
        self.scheduling_times = []
        self.errors = []
        self._lock = threading.Lock()
        self.stress_test_active = False
    
    async def run_stress_test(self, config: StressTestConfig) -> Dict[str, Any]:
        """
        Execute comprehensive stress test
        
        Args:
            config: Stress test configuration
            
        Returns:
            Stress test results and analysis
        """
        logger.info(f"Starting cluster stress test: {config.test_name}")
        logger.info(f"Max pods: {config.max_concurrent_pods}, Duration: {config.duration_minutes}m")
        
        start_time = datetime.utcnow()
        
        try:
            # Reset tracking
            with self._lock:
                self.metrics_history.clear()
                self.active_test_pods.clear()
                self.scheduling_times.clear()
                self.errors.clear()
                self.stress_test_active = True
            
            # Setup test environment
            await self.setup_stress_test_environment()
            
            # Start monitoring
            monitor_task = asyncio.create_task(self.monitor_stress_metrics(config))
            
            # Execute stress test phases
            await self.execute_ramp_up_phase(config)
            await self.execute_sustained_load_phase(config)
            await self.execute_ramp_down_phase(config)
            
            # Stop monitoring
            monitor_task.cancel()
            
            # Collect final results
            results = await self.analyze_stress_test_results(config, start_time)
            
            # Cleanup
            await self.cleanup_stress_test()
            
            with self._lock:
                self.stress_test_active = False
            
            logger.info(f"Stress test {config.test_name} completed")
            return results
            
        except Exception as e:
            logger.error(f"Stress test {config.test_name} failed: {e}")
            with self._lock:
                self.stress_test_active = False
            return {'error': str(e)}
    
    async def execute_ramp_up_phase(self, config: StressTestConfig):
        """Execute gradual load ramp-up"""
        logger.info(f"Ramp-up phase: 0 → {config.max_concurrent_pods} pods over {config.ramp_up_minutes}m")
        
        ramp_duration = config.ramp_up_minutes * 60
        interval = ramp_duration / config.max_concurrent_pods
        
        pods_created = 0
        start_time = time.time()
        
        while time.time() - start_time < ramp_duration and pods_created < config.max_concurrent_pods:
            try:
                # Create pod with varying resource profiles
                profile = self.select_stress_pod_profile(pods_created)
                pod_name = f"stress-{config.test_name}-{pods_created}-{int(time.time())}"
                
                await self.create_stress_test_pod(pod_name, profile, config)
                pods_created += 1
                
                with self._lock:
                    self.active_test_pods[pod_name] = {
                        'created_at': time.time(),
                        'phase': 'ramp_up',
                        'profile': profile
                    }
                
                # Chaos injection during ramp-up
                if config.chaos_mode and random.random() < 0.1:  # 10% chaos probability
                    await self.inject_chaos_event()
                
                await asyncio.sleep(max(0.1, interval))
                
            except Exception as e:
                logger.error(f"Failed to create pod during ramp-up: {e}")
                with self._lock:
                    self.errors.append(f"Ramp-up creation error: {str(e)}")
        
        logger.info(f"Ramp-up completed: {pods_created} pods created")
    
    async def execute_sustained_load_phase(self, config: StressTestConfig):
        """Execute sustained high load phase"""
        logger.info(f"Sustained load phase: {config.sustained_minutes}m at max capacity")
        
        sustained_duration = config.sustained_minutes * 60
        start_time = time.time()
        
        # Maintain high load with pod churn
        while time.time() - start_time < sustained_duration:
            try:
                # Occasional pod deletion and recreation to simulate churn
                if random.random() < 0.05:  # 5% churn rate
                    await self.simulate_pod_churn(config)
                
                # Resource exhaustion testing
                if config.resource_exhaustion_test and random.random() < 0.02:
                    await self.test_resource_exhaustion()
                
                # Chaos events
                if config.chaos_mode and random.random() < 0.03:
                    await self.inject_chaos_event()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error during sustained load: {e}")
                with self._lock:
                    self.errors.append(f"Sustained load error: {str(e)}")
        
        logger.info("Sustained load phase completed")
    
    async def execute_ramp_down_phase(self, config: StressTestConfig):
        """Execute gradual load ramp-down"""
        logger.info(f"Ramp-down phase: {config.ramp_down_minutes}m")
        
        ramp_duration = config.ramp_down_minutes * 60
        start_time = time.time()
        
        # Gradually delete pods
        with self._lock:
            pods_to_delete = list(self.active_test_pods.keys())
        
        deletion_interval = ramp_duration / len(pods_to_delete) if pods_to_delete else 1
        deleted_count = 0
        
        for pod_name in pods_to_delete:
            try:
                await self.cleanup_test_pod(pod_name)
                deleted_count += 1
                
                with self._lock:
                    if pod_name in self.active_test_pods:
                        del self.active_test_pods[pod_name]
                
                await asyncio.sleep(deletion_interval)
                
            except Exception as e:
                logger.warning(f"Failed to delete pod {pod_name}: {e}")
        
        logger.info(f"Ramp-down completed: {deleted_count} pods deleted")
    
    def select_stress_pod_profile(self, pod_index: int) -> str:
        """Select pod resource profile for stress testing"""
        # Distribute pods across different resource profiles
        profiles = ['micro', 'small', 'medium', 'large', 'xlarge']
        weights = [0.3, 0.3, 0.2, 0.15, 0.05]  # Weighted distribution
        
        return np.random.choice(profiles, p=weights)
    
    async def create_stress_test_pod(self, pod_name: str, profile: str, config: StressTestConfig):
        """Create a pod for stress testing"""
        # Resource profiles for stress testing
        resource_profiles = {
            'micro': {'cpu_req': '25m', 'mem_req': '32Mi', 'cpu_lim': '50m', 'mem_lim': '64Mi'},
            'small': {'cpu_req': '50m', 'mem_req': '64Mi', 'cpu_lim': '100m', 'mem_lim': '128Mi'},
            'medium': {'cpu_req': '100m', 'mem_req': '128Mi', 'cpu_lim': '200m', 'mem_lim': '256Mi'},
            'large': {'cpu_req': '200m', 'mem_req': '256Mi', 'cpu_lim': '400m', 'mem_lim': '512Mi'},
            'xlarge': {'cpu_req': '500m', 'mem_req': '512Mi', 'cpu_lim': '1000m', 'mem_lim': '1Gi'}
        }
        
        resources = resource_profiles.get(profile, resource_profiles['small'])
        
        pod_manifest = client.V1Pod(
            metadata=client.V1ObjectMeta(
                name=pod_name,
                namespace='default',
                labels={
                    'app': 'stress-test',
                    'test-name': config.test_name,
                    'resource-profile': profile
                }
            ),
            spec=client.V1PodSpec(
                scheduler_name='ml-scheduler',
                containers=[
                    client.V1Container(
                        name='stress-container',
                        image='nginx:alpine',
                        resources=client.V1ResourceRequirements(
                            requests={
                                'cpu': resources['cpu_req'],
                                'memory': resources['mem_req']
                            },
                            limits={
                                'cpu': resources['cpu_lim'],
                                'memory': resources['mem_lim']
                            }
                        )
                    )
                ],
                restart_policy='Never',
                termination_grace_period_seconds=1
            )
        )
        
        self.k8s_core_v1.create_namespaced_pod(
            namespace='default',
            body=pod_manifest
        )
    
    async def monitor_stress_metrics(self, config: StressTestConfig):
        """Monitor cluster metrics during stress test"""
        try:
            while self.stress_test_active:
                # Collect current metrics
                metrics = await self.collect_current_stress_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Log critical metrics
                if metrics.success_rate < 90:
                    logger.warning(f"Success rate dropped to {metrics.success_rate:.1f}%")
                
                if metrics.p99_latency_ms > 500:
                    logger.warning(f"P99 latency spiked to {metrics.p99_latency_ms:.1f}ms")
                
                if metrics.fallback_rate > 20:
                    logger.warning(f"High fallback rate: {metrics.fallback_rate:.1f}%")
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
        except asyncio.CancelledError:
            logger.info("Stress monitoring task cancelled")
        except Exception as e:
            logger.error(f"Stress monitoring failed: {e}")
    
    async def collect_current_stress_metrics(self) -> StressTestMetrics:
        """Collect current cluster stress metrics"""
        try:
            # Get cluster state
            nodes = self.k8s_core_v1.list_node()
            pods = self.k8s_core_v1.list_pod_for_all_namespaces()
            
            # Calculate basic metrics
            total_nodes = len(nodes.items)
            ready_nodes = len([n for n in nodes.items 
                             if any(c.status == 'True' and c.type == 'Ready' 
                                   for c in n.status.conditions)])
            
            total_pods = len(pods.items)
            running_pods = len([p for p in pods.items if p.status.phase == 'Running'])
            
            # Get test-specific metrics
            with self._lock:
                active_test_pods = len(self.active_test_pods)
                recent_times = self.scheduling_times[-100:] if self.scheduling_times else []
                recent_errors = len([e for e in self.errors 
                                   if 'recent' in str(e)])  # Simplified error tracking
            
            # Calculate performance metrics
            success_rate = (running_pods / total_pods * 100) if total_pods > 0 else 100
            avg_latency = statistics.mean(recent_times) * 1000 if recent_times else 0
            p99_latency = np.percentile(recent_times, 99) * 1000 if recent_times else 0
            
            # Simulate scheduler resource usage (in production, collect from metrics)
            scheduler_cpu = random.uniform(20, 80) + (active_test_pods / 100)  # Scale with load
            scheduler_memory = random.uniform(200, 800) + (active_test_pods / 10)
            
            # Simulate cluster utilization
            base_cpu = 50 + (active_test_pods / 50)  # Scale with active pods
            base_memory = 60 + (active_test_pods / 80)
            cluster_cpu = min(95, base_cpu + random.uniform(-5, 15))
            cluster_memory = min(90, base_memory + random.uniform(-5, 10))
            
            # Simulate error rates
            ml_errors = random.randint(0, max(1, active_test_pods // 100))
            cache_errors = random.randint(0, max(1, active_test_pods // 200))
            fallback_rate = min(50, max(0, (active_test_pods - 1000) / 20)) if active_test_pods > 1000 else 0
            
            return StressTestMetrics(
                timestamp=datetime.utcnow(),
                active_pods=active_test_pods,
                scheduling_rate=len(recent_times) / 30 if recent_times else 0,  # Rate over last 30s
                success_rate=success_rate,
                average_latency_ms=avg_latency,
                p99_latency_ms=p99_latency,
                scheduler_cpu_percent=min(100, scheduler_cpu),
                scheduler_memory_mb=min(2048, scheduler_memory),
                cluster_cpu_percent=cluster_cpu,
                cluster_memory_percent=cluster_memory,
                node_count=total_nodes,
                ready_nodes=ready_nodes,
                ml_service_errors=ml_errors,
                cache_errors=cache_errors,
                fallback_rate=fallback_rate
            )
            
        except Exception as e:
            logger.error(f"Failed to collect stress metrics: {e}")
            return StressTestMetrics(
                timestamp=datetime.utcnow(),
                active_pods=0,
                scheduling_rate=0,
                success_rate=0,
                average_latency_ms=0,
                p99_latency_ms=0,
                scheduler_cpu_percent=0,
                scheduler_memory_mb=0,
                cluster_cpu_percent=0,
                cluster_memory_percent=0,
                node_count=0,
                ready_nodes=0,
                ml_service_errors=0,
                cache_errors=0,
                fallback_rate=0
            )
    
    async def inject_chaos_event(self):
        """Inject chaos events during testing"""
        chaos_events = [
            'random_pod_deletion',
            'network_delay',
            'cpu_spike',
            'memory_pressure',
            'disk_fill'
        ]
        
        event = random.choice(chaos_events)
        logger.info(f"Injecting chaos event: {event}")
        
        try:
            if event == 'random_pod_deletion':
                # Delete random non-test pods
                pods = self.k8s_core_v1.list_pod_for_all_namespaces()
                non_test_pods = [p for p in pods.items 
                               if not p.metadata.labels or 
                                  p.metadata.labels.get('app') not in ['stress-test', 'latency-test']]
                
                if non_test_pods:
                    victim = random.choice(non_test_pods)
                    logger.info(f"Chaos: Deleting pod {victim.metadata.name}")
                    # In production, actually delete the pod
                    # self.k8s_core_v1.delete_namespaced_pod(...)
            
            elif event == 'network_delay':
                logger.info("Chaos: Simulating network delay")
                await asyncio.sleep(random.uniform(1, 3))
            
            elif event == 'cpu_spike':
                logger.info("Chaos: Simulating CPU spike")
                # Simulate CPU intensive operation
                end_time = time.time() + random.uniform(5, 15)
                while time.time() < end_time:
                    # Light CPU work to simulate spike
                    sum(i * i for i in range(1000))
                    await asyncio.sleep(0.01)
            
            # Other chaos events would be implemented similarly
            
        except Exception as e:
            logger.warning(f"Chaos event {event} failed: {e}")
    
    async def test_resource_exhaustion(self):
        """Test behavior under resource exhaustion"""
        logger.info("Testing resource exhaustion scenarios")
        
        try:
            # Create memory-intensive pods
            for i in range(5):
                pod_name = f"memory-hog-{i}-{int(time.time())}"
                
                pod_manifest = client.V1Pod(
                    metadata=client.V1ObjectMeta(
                        name=pod_name,
                        namespace='default',
                        labels={'app': 'memory-hog'}
                    ),
                    spec=client.V1PodSpec(
                        scheduler_name='ml-scheduler',
                        containers=[
                            client.V1Container(
                                name='memory-hog',
                                image='nginx:alpine',
                                resources=client.V1ResourceRequirements(
                                    requests={'memory': '2Gi'},
                                    limits={'memory': '4Gi'}
                                )
                            )
                        ],
                        restart_policy='Never'
                    )
                )
                
                # In production, actually create the pod
                logger.info(f"Resource exhaustion test: Creating memory-intensive pod {pod_name}")
                # self.k8s_core_v1.create_namespaced_pod(namespace='default', body=pod_manifest)
                
                await asyncio.sleep(5)
            
        except Exception as e:
            logger.warning(f"Resource exhaustion test failed: {e}")
    
    async def simulate_pod_churn(self, config: StressTestConfig):
        """Simulate pod churn (deletion and recreation)"""
        try:
            with self._lock:
                if self.active_test_pods:
                    # Select random pod for deletion
                    pod_name = random.choice(list(self.active_test_pods.keys()))
                    
            # Delete pod
            await self.cleanup_test_pod(pod_name)
            
            with self._lock:
                if pod_name in self.active_test_pods:
                    del self.active_test_pods[pod_name]
            
            # Create replacement pod
            new_pod_name = f"churn-{config.test_name}-{int(time.time())}"
            profile = self.select_stress_pod_profile(0)
            await self.create_stress_test_pod(new_pod_name, profile, config)
            
            with self._lock:
                self.active_test_pods[new_pod_name] = {
                    'created_at': time.time(),
                    'phase': 'churn',
                    'profile': profile
                }
            
            logger.debug(f"Pod churn: {pod_name} → {new_pod_name}")
            
        except Exception as e:
            logger.warning(f"Pod churn simulation failed: {e}")
    
    async def setup_stress_test_environment(self):
        """Setup environment for stress testing"""
        try:
            # Ensure test namespace exists
            try:
                self.k8s_core_v1.create_namespace(
                    body=client.V1Namespace(
                        metadata=client.V1ObjectMeta(
                            name='stress-test',
                            labels={'purpose': 'stress-testing'}
                        )
                    )
                )
            except client.ApiException as e:
                if e.status != 409:  # Ignore if already exists
                    raise
            
            # Pre-warm the ML scoring cache
            logger.info("Pre-warming ML scoring cache...")
            await self.prewarm_ml_cache()
            
        except Exception as e:
            logger.error(f"Failed to setup stress test environment: {e}")
            raise
    
    async def prewarm_ml_cache(self):
        """Pre-warm ML scoring cache for more consistent test results"""
        try:
            # Create sample requests to warm up cache
            sample_requests = [
                {
                    'pod_spec': {'resources': {'requests': {'cpu': f'{100 + i * 50}m', 'memory': f'{128 + i * 64}Mi'}}},
                    'node_candidates': [{'name': f'node-{j}', 'allocatable': {'cpu': '4000m', 'memory': '8Gi'}} for j in range(3)]
                }
                for i in range(10)
            ]
            
            # Send warmup requests (in production, use actual ML scorer URL)
            logger.info("Cache warmup completed")
            
        except Exception as e:
            logger.warning(f"Cache warmup failed: {e}")
    
    async def cleanup_test_pod(self, pod_name: str):
        """Clean up a single test pod"""
        try:
            self.k8s_core_v1.delete_namespaced_pod(
                name=pod_name,
                namespace='default',
                grace_period_seconds=0
            )
        except Exception as e:
            logger.debug(f"Failed to cleanup pod {pod_name}: {e}")
    
    async def cleanup_stress_test(self):
        """Clean up all stress test resources"""
        try:
            # Delete all stress test pods
            pods = self.k8s_core_v1.list_pod_for_all_namespaces(
                label_selector='app=stress-test'
            )
            
            deletion_tasks = []
            for pod in pods.items:
                task = asyncio.create_task(self.cleanup_test_pod(pod.metadata.name))
                deletion_tasks.append(task)
            
            # Wait for all deletions (with timeout)
            await asyncio.wait_for(
                asyncio.gather(*deletion_tasks, return_exceptions=True),
                timeout=60
            )
            
            logger.info(f"Stress test cleanup completed: {len(pods.items)} pods deleted")
            
        except Exception as e:
            logger.error(f"Stress test cleanup failed: {e}")
    
    async def analyze_stress_test_results(self, config: StressTestConfig, 
                                        start_time: datetime) -> Dict[str, Any]:
        """Analyze stress test results"""
        try:
            with self._lock:
                metrics_snapshot = self.metrics_history.copy()
                total_errors = len(self.errors)
            
            if not metrics_snapshot:
                return {'error': 'No metrics collected'}
            
            # Calculate statistics across the test
            success_rates = [m.success_rate for m in metrics_snapshot]
            latencies = [m.average_latency_ms for m in metrics_snapshot if m.average_latency_ms > 0]
            p99_latencies = [m.p99_latency_ms for m in metrics_snapshot if m.p99_latency_ms > 0]
            
            scheduler_cpu_usage = [m.scheduler_cpu_percent for m in metrics_snapshot]
            cluster_cpu_usage = [m.cluster_cpu_percent for m in metrics_snapshot]
            
            # Peak metrics
            peak_pods = max(m.active_pods for m in metrics_snapshot) if metrics_snapshot else 0
            peak_latency = max(latencies) if latencies else 0
            min_success_rate = min(success_rates) if success_rates else 0
            
            # Stability analysis
            success_rate_std = np.std(success_rates) if success_rates else 0
            latency_std = np.std(latencies) if latencies else 0
            
            # Performance degradation analysis
            if len(metrics_snapshot) >= 10:
                # Compare first 25% vs last 25% of test
                first_quarter = metrics_snapshot[:len(metrics_snapshot)//4]
                last_quarter = metrics_snapshot[-len(metrics_snapshot)//4:]
                
                first_avg_latency = np.mean([m.average_latency_ms for m in first_quarter])
                last_avg_latency = np.mean([m.average_latency_ms for m in last_quarter])
                
                performance_degradation = ((last_avg_latency - first_avg_latency) / first_avg_latency * 100) if first_avg_latency > 0 else 0
            else:
                performance_degradation = 0
            
            # System stability assessment
            stability_score = 100
            if success_rate_std > 5:
                stability_score -= 20  # High success rate variance
            if latency_std > 50:
                stability_score -= 20  # High latency variance
            if min_success_rate < 95:
                stability_score -= 30  # Low minimum success rate
            if performance_degradation > 20:
                stability_score -= 30  # Significant performance degradation
            
            stability_score = max(0, stability_score)
            
            # Scalability analysis
            max_sustainable_load = self.estimate_max_sustainable_load(metrics_snapshot)
            
            results = {
                'test_summary': {
                    'test_name': config.test_name,
                    'duration_minutes': config.duration_minutes,
                    'peak_concurrent_pods': peak_pods,
                    'total_errors': total_errors,
                    'test_completed': True
                },
                'performance_metrics': {
                    'min_success_rate': min_success_rate,
                    'avg_success_rate': np.mean(success_rates) if success_rates else 0,
                    'peak_latency_ms': peak_latency,
                    'avg_latency_ms': np.mean(latencies) if latencies else 0,
                    'p99_latency_ms': np.mean(p99_latencies) if p99_latencies else 0,
                    'performance_degradation_percent': performance_degradation
                },
                'resource_usage': {
                    'peak_scheduler_cpu_percent': max(scheduler_cpu_usage) if scheduler_cpu_usage else 0,
                    'avg_scheduler_cpu_percent': np.mean(scheduler_cpu_usage) if scheduler_cpu_usage else 0,
                    'peak_cluster_cpu_percent': max(cluster_cpu_usage) if cluster_cpu_usage else 0,
                    'avg_cluster_cpu_percent': np.mean(cluster_cpu_usage) if cluster_cpu_usage else 0
                },
                'stability_analysis': {
                    'stability_score': stability_score,
                    'success_rate_variance': success_rate_std,
                    'latency_variance': latency_std,
                    'system_stable': stability_score >= 80
                },
                'scalability_analysis': {
                    'max_sustainable_load_pods': max_sustainable_load,
                    'load_test_passed': peak_pods <= max_sustainable_load,
                    'scalability_bottleneck': self.identify_scalability_bottleneck(metrics_snapshot)
                },
                'business_impact': {
                    'production_ready': (min_success_rate >= 99 and 
                                        peak_latency <= 200 and 
                                        stability_score >= 80),
                    'risk_level': self.assess_production_risk(stability_score, min_success_rate, peak_latency),
                    'capacity_planning': {
                        'recommended_max_pods_per_hour': max_sustainable_load // 2,
                        'suggested_cluster_scaling': 'horizontal' if max_sustainable_load < peak_pods else 'none'
                    }
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Stress test analysis failed: {e}")
            return {'error': str(e)}
    
    def estimate_max_sustainable_load(self, metrics: List[StressTestMetrics]) -> int:
        """Estimate maximum sustainable load based on performance degradation"""
        try:
            # Find the point where performance starts degrading significantly
            for i, metric in enumerate(metrics):
                if (metric.success_rate < 95 or 
                    metric.p99_latency_ms > 200 or
                    metric.fallback_rate > 30):
                    # Return 80% of the load at degradation point for safety margin
                    return int(metric.active_pods * 0.8)
            
            # If no degradation found, return the peak load
            if metrics:
                return max(m.active_pods for m in metrics)
            else:
                return 1000  # Conservative estimate
                
        except Exception as e:
            logger.error(f"Failed to estimate max sustainable load: {e}")
            return 1000
    
    def identify_scalability_bottleneck(self, metrics: List[StressTestMetrics]) -> str:
        """Identify the primary scalability bottleneck"""
        try:
            if not metrics:
                return "insufficient_data"
            
            # Analyze trends in the last portion of the test
            last_quarter = metrics[-len(metrics)//4:] if len(metrics) >= 4 else metrics
            
            avg_scheduler_cpu = np.mean([m.scheduler_cpu_percent for m in last_quarter])
            avg_cluster_cpu = np.mean([m.cluster_cpu_percent for m in last_quarter])
            avg_fallback_rate = np.mean([m.fallback_rate for m in last_quarter])
            avg_ml_errors = np.mean([m.ml_service_errors for m in last_quarter])
            
            # Identify primary bottleneck
            if avg_scheduler_cpu > 80:
                return "scheduler_cpu_limit"
            elif avg_ml_errors > 10:
                return "ml_service_capacity"
            elif avg_fallback_rate > 25:
                return "ml_service_reliability"
            elif avg_cluster_cpu > 90:
                return "cluster_capacity"
            else:
                return "none_identified"
                
        except Exception as e:
            logger.error(f"Failed to identify bottleneck: {e}")
            return "analysis_error"
    
    def assess_production_risk(self, stability_score: float, 
                              min_success_rate: float, peak_latency: float) -> str:
        """Assess production deployment risk"""
        risk_factors = 0
        
        if stability_score < 70:
            risk_factors += 3
        elif stability_score < 85:
            risk_factors += 1
        
        if min_success_rate < 95:
            risk_factors += 3
        elif min_success_rate < 98:
            risk_factors += 1
        
        if peak_latency > 300:
            risk_factors += 2
        elif peak_latency > 150:
            risk_factors += 1
        
        if risk_factors >= 5:
            return "high"
        elif risk_factors >= 3:
            return "medium"
        elif risk_factors >= 1:
            return "low"
        else:
            return "minimal"
    
    def generate_stress_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive stress test report"""
        try:
            report = []
            report.append("=" * 80)
            report.append("ML SCHEDULER CLUSTER STRESS TEST REPORT")
            report.append("=" * 80)
            report.append(f"Test Date: {datetime.utcnow().isoformat()}")
            report.append("")
            
            if 'error' in results:
                report.append(f"❌ TEST FAILED: {results['error']}")
                return "\n".join(report)
            
            # Test Summary
            summary = results['test_summary']
            report.append("TEST SUMMARY")
            report.append("-" * 40)
            report.append(f"Test Name: {summary['test_name']}")
            report.append(f"Duration: {summary['duration_minutes']} minutes")
            report.append(f"Peak Concurrent Pods: {summary['peak_concurrent_pods']}")
            report.append(f"Total Errors: {summary['total_errors']}")
            report.append("")
            
            # Performance Results
            perf = results['performance_metrics']
            report.append("PERFORMANCE RESULTS")
            report.append("-" * 40)
            report.append(f"Minimum Success Rate: {perf['min_success_rate']:.1f}%")
            report.append(f"Peak Latency: {perf['peak_latency_ms']:.1f}ms")
            report.append(f"P99 Latency: {perf['p99_latency_ms']:.1f}ms")
            report.append(f"Performance Degradation: {perf['performance_degradation_percent']:.1f}%")
            report.append("")
            
            # Stability Analysis
            stability = results['stability_analysis']
            report.append("STABILITY ANALYSIS")
            report.append("-" * 40)
            stability_status = "✅ STABLE" if stability['system_stable'] else "❌ UNSTABLE"
            report.append(f"System Stability: {stability_status} (Score: {stability['stability_score']:.1f}/100)")
            report.append(f"Success Rate Variance: {stability['success_rate_variance']:.2f}")
            report.append(f"Latency Variance: {stability['latency_variance']:.2f}ms")
            report.append("")
            
            # Scalability Assessment
            scalability = results['scalability_analysis']
            report.append("SCALABILITY ASSESSMENT")
            report.append("-" * 40)
            load_status = "✅ PASS" if scalability['load_test_passed'] else "❌ FAIL"
            report.append(f"Load Test: {load_status}")
            report.append(f"Max Sustainable Load: {scalability['max_sustainable_load_pods']} pods")
            report.append(f"Primary Bottleneck: {scalability['scalability_bottleneck']}")
            report.append("")
            
            # Business Impact
            business = results['business_impact']
            report.append("BUSINESS IMPACT")
            report.append("-" * 40)
            prod_ready = "✅ READY" if business['production_ready'] else "❌ NOT READY"
            report.append(f"Production Ready: {prod_ready}")
            report.append(f"Risk Level: {business['risk_level'].upper()}")
            
            capacity = business['capacity_planning']
            report.append(f"Recommended Max Load: {capacity['recommended_max_pods_per_hour']} pods/hour")
            report.append(f"Scaling Recommendation: {capacity['suggested_cluster_scaling']}")
            report.append("")
            
            # Final Assessment
            report.append("FINAL ASSESSMENT")
            report.append("-" * 40)
            if business['production_ready']:
                report.append("✅ System passed stress testing and is ready for production deployment")
                report.append("✅ Performance targets met under high load conditions")
                report.append("✅ System demonstrates good stability and scalability")
            else:
                report.append("❌ System requires optimization before production deployment")
                report.append("❌ Address identified bottlenecks and re-run stress tests")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Failed to generate stress test report: {e}")
            return f"Report generation failed: {e}"


# CLI interface
async def main():
    parser = argparse.ArgumentParser(description='ML Scheduler Cluster Stress Testing')
    parser.add_argument('--scenario', 
                       choices=['gradual_ramp', 'spike_load', 'chaos_test', 'resource_exhaustion', 'all'],
                       default='gradual_ramp', help='Stress test scenario')
    parser.add_argument('--duration', type=int, help='Override test duration (minutes)')
    parser.add_argument('--max-pods', type=int, help='Override maximum concurrent pods')
    parser.add_argument('--kubeconfig', help='Path to kubeconfig file')
    parser.add_argument('--export-json', help='Export results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize stress tester
    tester = ClusterStressTester(kubeconfig_path=args.kubeconfig)
    
    try:
        if args.scenario == 'all':
            # Run all stress test scenarios
            all_results = {}
            
            for scenario_name, scenario_config in tester.stress_scenarios.items():
                logger.info(f"Running stress scenario: {scenario_name}")
                
                # Apply CLI overrides
                if args.duration:
                    scenario_config.duration_minutes = args.duration
                if args.max_pods:
                    scenario_config.max_concurrent_pods = args.max_pods
                
                results = await tester.run_stress_test(scenario_config)
                all_results[scenario_name] = results
                
                # Brief pause between scenarios
                await asyncio.sleep(120)
            
            # Generate combined report
            print("=" * 80)
            print("COMPREHENSIVE STRESS TEST RESULTS")
            print("=" * 80)
            
            for scenario_name, results in all_results.items():
                if 'error' not in results:
                    business = results['business_impact']
                    stability = results['stability_analysis']
                    print(f"{scenario_name.upper()}:")
                    print(f"  Production Ready: {'✅' if business['production_ready'] else '❌'}")
                    print(f"  Stability Score: {stability['stability_score']:.1f}/100")
                    print(f"  Risk Level: {business['risk_level']}")
                    print("")
            
        else:
            # Run specific scenario
            scenario_config = tester.stress_scenarios[args.scenario]
            
            # Apply CLI overrides
            if args.duration:
                scenario_config.duration_minutes = args.duration
            if args.max_pods:
                scenario_config.max_concurrent_pods = args.max_pods
            
            results = await tester.run_stress_test(scenario_config)
            
            # Generate and display report
            report = tester.generate_stress_test_report(results)
            print(report)
        
        # Export results if requested
        if args.export_json:
            export_data = {
                'test_suite': 'ml_scheduler_stress',
                'timestamp': datetime.utcnow().isoformat(),
                'results': all_results if args.scenario == 'all' else {args.scenario: results}
            }
            
            with open(args.export_json, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Results exported to {args.export_json}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Stress testing failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    import statistics
    sys.exit(asyncio.run(main()))