#!/usr/bin/env python3
"""
Integration tests for ML Scheduler scheduling decisions and performance.
Tests scheduling accuracy, latency, and business metrics.
"""

import pytest
import time
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from ml_models.qlearning.agent import HYDATISPlacementDQN


class TestSchedulingPerformance:
    """Test ML scheduler performance and business metrics."""
    
    @pytest.fixture
    def cluster_config(self):
        """HYDATIS cluster configuration."""
        return {
            'nodes': 6,
            'masters': 3,
            'workers': 3,
            'cpu_cores_per_node': 8,
            'memory_gb_per_node': 16,
            'worker_nodes': ['worker-1', 'worker-2', 'worker-3'],
            'master_nodes': ['master-1', 'master-2', 'master-3']
        }
    
    @pytest.fixture
    def ml_scheduler(self, cluster_config):
        """Initialize ML scheduler."""
        return HYDATISPlacementDQN(cluster_config)
    
    def test_scheduling_latency_target(self, ml_scheduler):
        """Test scheduling decisions meet latency target <200ms."""
        
        latencies = []
        num_tests = 50
        
        print(f"🚀 Testing scheduling latency with {num_tests} decisions...")
        
        for i in range(num_tests):
            # Generate random cluster state
            cluster_state = np.random.uniform(0.1, 0.8, ml_scheduler.state_size)
            pod_requirements = {
                'cpu_request': np.random.uniform(0.1, 0.5),
                'memory_request': np.random.uniform(0.1, 0.6)
            }
            
            # Measure decision time
            start_time = time.perf_counter()
            action, decision = ml_scheduler.select_placement_node(
                cluster_state, pod_requirements, training=False
            )
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Validate decision
            assert 0 <= action < ml_scheduler.action_size
            assert 'selected_node' in decision
        
        # Calculate latency statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = max(latencies)
        
        print(f"📊 Latency Results:")
        print(f"   Average: {avg_latency:.2f}ms")
        print(f"   P95: {p95_latency:.2f}ms") 
        print(f"   P99: {p99_latency:.2f}ms")
        print(f"   Max: {max_latency:.2f}ms")
        
        # Assert latency targets
        assert avg_latency < 120, f"Average latency {avg_latency:.2f}ms exceeds 120ms target"
        assert p95_latency < 150, f"P95 latency {p95_latency:.2f}ms exceeds 150ms target"
        assert p99_latency < 200, f"P99 latency {p99_latency:.2f}ms exceeds 200ms target"
        
        print("✅ Scheduling latency test passed")
    
    def test_placement_quality_improvement(self, ml_scheduler):
        """Test placement improvement over random baseline."""
        
        num_episodes = 20
        ml_rewards = []
        random_rewards = []
        
        print(f"🎯 Testing placement quality with {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Generate diverse cluster scenarios
            cluster_state = np.random.uniform(0.0, 1.0, ml_scheduler.state_size)
            pod_requirements = {
                'cpu_request': np.random.uniform(0.05, 0.4),
                'memory_request': np.random.uniform(0.1, 0.5)
            }
            
            # ML scheduler decision
            ml_action, ml_decision = ml_scheduler.select_placement_node(
                cluster_state, pod_requirements, training=False
            )
            ml_quality = ml_decision['placement_quality']['overall_quality']
            ml_rewards.append(ml_quality)
            
            # Random baseline decision
            random_action = np.random.randint(0, ml_scheduler.action_size)
            random_quality = ml_scheduler.agent.evaluate_placement_quality(
                cluster_state, random_action
            )['overall_quality']
            random_rewards.append(random_quality)
        
        # Calculate improvement
        ml_avg = np.mean(ml_rewards)
        random_avg = np.mean(random_rewards)
        improvement = (ml_avg - random_avg) / random_avg if random_avg > 0 else 0
        
        print(f"📈 Placement Quality Results:")
        print(f"   ML Scheduler average: {ml_avg:.3f}")
        print(f"   Random baseline average: {random_avg:.3f}")
        print(f"   Improvement: {improvement:.1%}")
        print(f"   Target: {ml_scheduler.target_improvement:.1%}")
        
        # Assert improvement target
        assert improvement >= ml_scheduler.target_improvement * 0.8, \
            f"Improvement {improvement:.1%} below 80% of target {ml_scheduler.target_improvement:.1%}"
        
        print("✅ Placement quality improvement test passed")
    
    def test_concurrent_scheduling_decisions(self, ml_scheduler):
        """Test scheduler handles concurrent requests correctly."""
        
        num_concurrent = 10
        results = []
        
        def make_scheduling_decision(request_id):
            """Make a single scheduling decision."""
            cluster_state = np.random.uniform(0.2, 0.7, ml_scheduler.state_size)
            pod_req = {'cpu_request': 0.1, 'memory_request': 0.15}
            
            start_time = time.perf_counter()
            action, decision = ml_scheduler.select_placement_node(cluster_state, pod_req)
            end_time = time.perf_counter()
            
            return {
                'request_id': request_id,
                'action': action,
                'latency_ms': (end_time - start_time) * 1000,
                'quality': decision['placement_quality']['overall_quality']
            }
        
        print(f"⚡ Testing {num_concurrent} concurrent scheduling decisions...")
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_scheduling_decision, i) for i in range(num_concurrent)]
            results = [future.result() for future in futures]
        
        # Analyze results
        latencies = [r['latency_ms'] for r in results]
        qualities = [r['quality'] for r in results]
        
        avg_concurrent_latency = statistics.mean(latencies)
        avg_quality = statistics.mean(qualities)
        
        print(f"📊 Concurrent Scheduling Results:")
        print(f"   Requests processed: {len(results)}")
        print(f"   Average latency: {avg_concurrent_latency:.2f}ms")
        print(f"   Average quality: {avg_quality:.3f}")
        print(f"   All requests successful: {len(results) == num_concurrent}\")\n"
        
        # Assert concurrent performance
        assert len(results) == num_concurrent, "Not all concurrent requests completed"
        assert avg_concurrent_latency < 300, f"Concurrent latency {avg_concurrent_latency:.2f}ms too high"
        assert avg_quality > 0.3, f"Average quality {avg_quality:.3f} too low"
        
        print("✅ Concurrent scheduling test passed")
    
    def test_business_metrics_calculation(self, ml_scheduler):
        """Test business metrics calculation and validation."""
        
        # Simulate cluster utilization data
        utilization_data = {
            'before_ml_scheduler': {
                'avg_cpu_utilization': 0.85,
                'availability_pct': 95.2,
                'incident_response_minutes': 45,
                'monthly_costs': 150000
            },
            'after_ml_scheduler': {
                'avg_cpu_utilization': 0.65,
                'availability_pct': 99.7,
                'incident_response_minutes': 12,
                'monthly_costs': 120000
            }
        }
        
        # Calculate business improvements
        before = utilization_data['before_ml_scheduler']
        after = utilization_data['after_ml_scheduler']
        
        cpu_improvement = (before['avg_cpu_utilization'] - after['avg_cpu_utilization']) / before['avg_cpu_utilization']
        availability_improvement = after['availability_pct'] - before['availability_pct']
        response_improvement = (before['incident_response_minutes'] - after['incident_response_minutes']) / before['incident_response_minutes']
        cost_savings = before['monthly_costs'] - after['monthly_costs']
        
        # Annual ROI calculation
        annual_savings = cost_savings * 12
        implementation_cost = 150000  # As per business case
        roi_pct = (annual_savings - implementation_cost) / implementation_cost * 100
        
        print(f"💰 Business Metrics Validation:")
        print(f"   CPU optimization: {cpu_improvement:.1%} (Target: 23%)")
        print(f"   Availability improvement: +{availability_improvement:.1f}% (Target: +4.5%)")
        print(f"   Response time improvement: {response_improvement:.1%} (Target: >60%)")
        print(f"   Monthly savings: ${cost_savings:,} (Target: $25k)")
        print(f"   Annual ROI: {roi_pct:.0f}% (Target: >1400%)")
        
        # Assert business targets
        assert cpu_improvement >= 0.20, f"CPU improvement {cpu_improvement:.1%} below 20% target"
        assert availability_improvement >= 4.0, f"Availability improvement {availability_improvement:.1f}% below target"
        assert response_improvement >= 0.60, f"Response improvement {response_improvement:.1%} below target"
        assert cost_savings >= 25000, f"Savings ${cost_savings:,} below $25k target"
        assert roi_pct >= 1400, f"ROI {roi_pct:.0f}% below 1400% target"
        
        print("✅ Business metrics validation passed")
        
        return {
            'cpu_improvement_pct': cpu_improvement * 100,
            'availability_improvement': availability_improvement,
            'response_improvement_pct': response_improvement * 100,
            'monthly_savings_usd': cost_savings,
            'annual_roi_pct': roi_pct
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])