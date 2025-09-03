#!/usr/bin/env python3
"""
Prometheus metrics collector for ML scheduler training data.
Collects node and cluster metrics every 30 seconds for historical analysis.
"""

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
from prometheus_client.parser import text_string_to_metric_families
import yaml
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrometheusCollector:
    """Collects metrics from Prometheus for ML scheduler training."""
    
    def __init__(self, prometheus_url: str = "http://10.110.190.83:9090"):
        self.prometheus_url = prometheus_url
        self.session = requests.Session()
        self.session.timeout = 30
        
    def query_range(self, query: str, start_time: datetime, end_time: datetime, step: str = "30s") -> pd.DataFrame:
        """Query Prometheus for range data."""
        url = f"{self.prometheus_url}/api/v1/query_range"
        params = {
            'query': query,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': step
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'success':
                logger.error(f"Prometheus query failed: {data}")
                return pd.DataFrame()
                
            return self._parse_prometheus_response(data['data']['result'])
            
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")
            return pd.DataFrame()
    
    def _parse_prometheus_response(self, result: List[Dict]) -> pd.DataFrame:
        """Parse Prometheus response into DataFrame."""
        records = []
        
        for series in result:
            labels = series['metric']
            for timestamp, value in series['values']:
                record = {
                    'timestamp': pd.to_datetime(float(timestamp), unit='s'),
                    'value': float(value) if value != 'NaN' else np.nan,
                    **labels
                }
                records.append(record)
        
        return pd.DataFrame(records)
    
    def collect_node_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive node metrics for ML training."""
        
        queries = {
            'cpu_usage': 'rate(node_cpu_seconds_total{mode!="idle"}[5m])',
            'memory_usage': '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes',
            'memory_available': 'node_memory_MemAvailable_bytes',
            'memory_total': 'node_memory_MemTotal_bytes',
            'load_1m': 'node_load1',
            'load_5m': 'node_load5', 
            'load_15m': 'node_load15',
            'disk_usage': '(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes',
            'network_rx_bytes': 'rate(node_network_receive_bytes_total[5m])',
            'network_tx_bytes': 'rate(node_network_transmit_bytes_total[5m])',
            'disk_read_bytes': 'rate(node_disk_read_bytes_total[5m])',
            'disk_write_bytes': 'rate(node_disk_written_bytes_total[5m])'
        }
        
        metrics_data = {}
        
        for metric_name, query in queries.items():
            logger.info(f"Collecting {metric_name} metrics...")
            df = self.query_range(query, start_time, end_time)
            if not df.empty:
                metrics_data[metric_name] = df
            else:
                logger.warning(f"No data collected for {metric_name}")
        
        return metrics_data
    
    def collect_scheduler_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """Collect scheduler performance metrics."""
        
        queries = {
            'scheduling_duration': 'histogram_quantile(0.95, rate(scheduler_scheduling_duration_seconds_bucket[5m]))',
            'pending_pods': 'scheduler_pending_pods',
            'scheduling_attempts': 'rate(scheduler_pod_scheduling_attempts_total[5m])',
            'scheduling_errors': 'rate(scheduler_pod_scheduling_errors_total[5m])'
        }
        
        metrics_data = {}
        
        for metric_name, query in queries.items():
            logger.info(f"Collecting scheduler {metric_name} metrics...")
            df = self.query_range(query, start_time, end_time)
            if not df.empty:
                metrics_data[metric_name] = df
        
        return metrics_data
    
    def collect_pod_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """Collect pod placement and performance metrics."""
        
        queries = {
            'pod_cpu_usage': 'rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m])',
            'pod_memory_usage': 'container_memory_working_set_bytes{container!="POD",container!=""}',
            'pod_restarts': 'increase(kube_pod_container_status_restarts_total[1h])',
            'pod_status': 'kube_pod_status_phase'
        }
        
        metrics_data = {}
        
        for metric_name, query in queries.items():
            logger.info(f"Collecting pod {metric_name} metrics...")
            df = self.query_range(query, start_time, end_time)
            if not df.empty:
                metrics_data[metric_name] = df
        
        return metrics_data


def main():
    """Main data collection function."""
    collector = PrometheusCollector()
    
    # Collect last 7 days of data (Week 1 target) - corrected for current date
    end_time = datetime(2024, 9, 3, 16, 57)
    start_time = end_time - timedelta(days=7)
    
    logger.info(f"Collecting data from {start_time} to {end_time}")
    
    # Collect all metrics
    node_metrics = collector.collect_node_metrics(start_time, end_time)
    scheduler_metrics = collector.collect_scheduler_metrics(start_time, end_time)
    pod_metrics = collector.collect_pod_metrics(start_time, end_time)
    
    # Save collected data to local directory
    output_dir = os.environ.get('DATA_OUTPUT_DIR', './data/collected_metrics')
    os.makedirs(output_dir, exist_ok=True)
    
    for category, metrics in [
        ("node", node_metrics),
        ("scheduler", scheduler_metrics), 
        ("pod", pod_metrics)
    ]:
        for metric_name, df in metrics.items():
            filename = f"{output_dir}/{category}_{metric_name}_{end_time.strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(df)} records to {filename}")
    
    logger.info("Data collection completed successfully")


if __name__ == "__main__":
    main()