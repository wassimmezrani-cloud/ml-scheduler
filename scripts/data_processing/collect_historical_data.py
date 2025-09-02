#!/usr/bin/env python3
"""
Script to collect and process historical data for ML scheduler training.
Orchestrates the complete data collection and processing pipeline.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data_collection.prometheus_collector import PrometheusCollector
from data_collection.data_processor import MLSchedulerDataProcessor
from data_collection.quality_monitor import DataQualityMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main orchestration function for historical data collection."""
    
    logger.info("Starting ML Scheduler historical data collection pipeline")
    
    try:
        # Step 1: Collect raw metrics from Prometheus
        logger.info("Step 1: Collecting raw metrics from Prometheus...")
        collector = PrometheusCollector(prometheus_url="http://10.110.190.83:9090")
        
        # Collect last 30 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        
        # Collect all types of metrics
        node_metrics = collector.collect_node_metrics(start_time, end_time)
        scheduler_metrics = collector.collect_scheduler_metrics(start_time, end_time)
        pod_metrics = collector.collect_pod_metrics(start_time, end_time)
        
        if not node_metrics:
            logger.error("Failed to collect node metrics")
            return False
        
        logger.info(f"Collected {len(node_metrics)} node metric types")
        
        # Step 2: Process raw data into ML-ready format
        logger.info("Step 2: Processing data into ML format...")
        processor = MLSchedulerDataProcessor()
        dataset_path = processor.process_full_dataset(days_back=30)
        
        if not dataset_path:
            logger.error("Failed to process dataset")
            return False
        
        logger.info(f"Processed dataset saved: {dataset_path}")
        
        # Step 3: Validate data quality
        logger.info("Step 3: Validating data quality...")
        monitor = DataQualityMonitor()
        
        # Load processed dataset for validation
        df = pd.read_parquet(dataset_path)
        quality_metrics = monitor.generate_quality_report(df)
        quality_checks = monitor.check_quality_thresholds(quality_metrics)
        
        # Check if quality meets requirements
        if quality_metrics.overall_score >= 95.0:
            logger.info(f"Data quality validation PASSED: {quality_metrics.overall_score:.2f}%")
        else:
            logger.warning(f"Data quality validation FAILED: {quality_metrics.overall_score:.2f}%")
            failed_checks = [check for check, passed in quality_checks.items() if not passed]
            monitor.send_quality_alert(quality_metrics, failed_checks)
        
        # Step 4: Generate collection summary
        summary = {
            'collection_timestamp': datetime.now().isoformat(),
            'data_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'days': 30
            },
            'dataset_info': {
                'path': dataset_path,
                'records': len(df),
                'features': len(df.columns),
                'nodes': len(df['node'].unique()) if 'node' in df.columns else 0
            },
            'quality_metrics': quality_metrics.to_dict(),
            'pipeline_status': 'SUCCESS' if quality_metrics.overall_score >= 95.0 else 'WARNING'
        }
        
        # Save summary to Longhorn storage
        summary_path = Path("/data/ml_scheduler_longhorn/processed") / f"hydatis_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Collection summary saved: {summary_path}")
        logger.info("Historical data collection pipeline completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)