#!/usr/bin/env python3
"""
Data quality monitoring system for ML scheduler.
Validates data completeness, accuracy, and timeliness for ML training pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import yaml
import json
from pathlib import Path
import requests
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality metrics structure."""
    completeness_pct: float
    timeliness_pct: float
    accuracy_pct: float
    consistency_pct: float
    overall_score: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'completeness_pct': self.completeness_pct,
            'timeliness_pct': self.timeliness_pct,
            'accuracy_pct': self.accuracy_pct,
            'consistency_pct': self.consistency_pct,
            'overall_score': self.overall_score,
            'timestamp': self.timestamp.isoformat()
        }


class DataQualityMonitor:
    """Monitors data quality for ML scheduler training pipeline."""
    
    def __init__(self, data_dir: str = "/tmp/ml_scheduler_data"):
        self.data_dir = Path(data_dir)
        self.quality_reports_dir = self.data_dir / "quality_reports"
        self.quality_reports_dir.mkdir(exist_ok=True)
        
        # Quality thresholds
        self.thresholds = {
            'completeness_min': 95.0,
            'timeliness_min': 98.0,
            'accuracy_min': 95.0,
            'consistency_min': 90.0,
            'overall_min': 95.0
        }
    
    def validate_completeness(self, df: pd.DataFrame) -> float:
        """Validate data completeness (no missing values)."""
        if df.empty:
            return 0.0
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        logger.info(f"Data completeness: {completeness:.2f}% ({missing_cells} missing out of {total_cells} cells)")
        return completeness
    
    def validate_timeliness(self, df: pd.DataFrame, expected_interval_seconds: int = 30) -> float:
        """Validate data timeliness (regular intervals)."""
        if df.empty or 'timestamp' not in df.columns:
            return 0.0
        
        # Check if data collection is within expected intervals
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dt.total_seconds()
        
        # Allow 10% tolerance on interval timing
        tolerance = expected_interval_seconds * 0.1
        valid_intervals = ((time_diffs >= expected_interval_seconds - tolerance) & 
                          (time_diffs <= expected_interval_seconds + tolerance)).sum()
        
        total_intervals = len(time_diffs) - 1  # Exclude first NaN
        timeliness = (valid_intervals / total_intervals) * 100 if total_intervals > 0 else 0
        
        logger.info(f"Data timeliness: {timeliness:.2f}% ({valid_intervals}/{total_intervals} valid intervals)")
        return timeliness
    
    def validate_accuracy(self, df: pd.DataFrame) -> float:
        """Validate data accuracy (reasonable value ranges)."""
        if df.empty:
            return 0.0
        
        accuracy_checks = []
        
        # CPU usage should be between 0 and 1
        if 'cpu_usage_rate' in df.columns:
            cpu_valid = ((df['cpu_usage_rate'] >= 0) & (df['cpu_usage_rate'] <= 1)).mean() * 100
            accuracy_checks.append(cpu_valid)
            logger.info(f"CPU usage accuracy: {cpu_valid:.2f}%")
        
        # Memory usage should be between 0 and 1
        if 'memory_usage' in df.columns:
            memory_valid = ((df['memory_usage'] >= 0) & (df['memory_usage'] <= 1)).mean() * 100
            accuracy_checks.append(memory_valid)
            logger.info(f"Memory usage accuracy: {memory_valid:.2f}%")
        
        # Load should be non-negative and reasonable (< 50 for 8-core machines)
        if 'load_1m' in df.columns:
            load_valid = ((df['load_1m'] >= 0) & (df['load_1m'] <= 50)).mean() * 100
            accuracy_checks.append(load_valid)
            logger.info(f"Load accuracy: {load_valid:.2f}%")
        
        overall_accuracy = np.mean(accuracy_checks) if accuracy_checks else 0
        logger.info(f"Overall data accuracy: {overall_accuracy:.2f}%")
        return overall_accuracy
    
    def validate_consistency(self, df: pd.DataFrame) -> float:
        """Validate data consistency across nodes."""
        if df.empty or 'node' not in df.columns:
            return 0.0
        
        consistency_checks = []
        
        # Check that all nodes have similar data coverage
        node_counts = df['node'].value_counts()
        expected_count = node_counts.median()
        consistency = (node_counts / expected_count).clip(upper=1.0).mean() * 100
        consistency_checks.append(consistency)
        
        # Check temporal consistency (no large gaps per node)
        temporal_consistency_scores = []
        for node in df['node'].unique():
            node_data = df[df['node'] == node].sort_values('timestamp')
            if len(node_data) > 1:
                time_diffs = node_data['timestamp'].diff().dt.total_seconds()
                # Allow up to 5 minutes gaps (10x normal interval)
                valid_gaps = (time_diffs <= 300).mean() * 100
                temporal_consistency_scores.append(valid_gaps)
        
        if temporal_consistency_scores:
            temporal_consistency = np.mean(temporal_consistency_scores)
            consistency_checks.append(temporal_consistency)
        
        overall_consistency = np.mean(consistency_checks) if consistency_checks else 0
        logger.info(f"Data consistency: {overall_consistency:.2f}%")
        return overall_consistency
    
    def generate_quality_report(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Generate comprehensive data quality report."""
        
        logger.info("Generating data quality report...")
        
        completeness = self.validate_completeness(df)
        timeliness = self.validate_timeliness(df)
        accuracy = self.validate_accuracy(df)
        consistency = self.validate_consistency(df)
        
        # Calculate overall score (weighted average)
        weights = {
            'completeness': 0.3,
            'timeliness': 0.2,
            'accuracy': 0.3,
            'consistency': 0.2
        }
        
        overall_score = (
            completeness * weights['completeness'] +
            timeliness * weights['timeliness'] +
            accuracy * weights['accuracy'] +
            consistency * weights['consistency']
        )
        
        quality_metrics = DataQualityMetrics(
            completeness_pct=completeness,
            timeliness_pct=timeliness,
            accuracy_pct=accuracy,
            consistency_pct=consistency,
            overall_score=overall_score,
            timestamp=datetime.now()
        )
        
        # Save quality report
        report_path = self.quality_reports_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(quality_metrics.to_dict(), f, indent=2)
        
        logger.info(f"Quality report saved: {report_path}")
        return quality_metrics
    
    def check_quality_thresholds(self, quality_metrics: DataQualityMetrics) -> Dict[str, bool]:
        """Check if quality metrics meet minimum thresholds."""
        
        checks = {
            'completeness_ok': quality_metrics.completeness_pct >= self.thresholds['completeness_min'],
            'timeliness_ok': quality_metrics.timeliness_pct >= self.thresholds['timeliness_min'],
            'accuracy_ok': quality_metrics.accuracy_pct >= self.thresholds['accuracy_min'],
            'consistency_ok': quality_metrics.consistency_pct >= self.thresholds['consistency_min'],
            'overall_ok': quality_metrics.overall_score >= self.thresholds['overall_min']
        }
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        logger.info(f"Quality checks passed: {passed_checks}/{total_checks}")
        
        for check_name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"{check_name}: {status}")
        
        return checks
    
    def send_quality_alert(self, quality_metrics: DataQualityMetrics, failed_checks: List[str]):
        """Send quality alert if thresholds not met."""
        
        alert_data = {
            'timestamp': quality_metrics.timestamp.isoformat(),
            'overall_score': quality_metrics.overall_score,
            'failed_checks': failed_checks,
            'metrics': quality_metrics.to_dict()
        }
        
        # In production, this would send to alerting system
        alert_path = self.quality_reports_dir / f"quality_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(alert_path, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        logger.warning(f"Data quality alert generated: {alert_path}")
        logger.warning(f"Failed checks: {', '.join(failed_checks)}")


def main():
    """Main data quality monitoring function."""
    
    monitor = DataQualityMonitor()
    
    # Load latest processed data for validation
    data_dir = Path("/tmp/ml_scheduler_data/processed")
    if not data_dir.exists():
        logger.error("No processed data directory found")
        return
    
    parquet_files = list(data_dir.glob("ml_scheduler_training_data_*.parquet"))
    if not parquet_files:
        logger.error("No processed data files found")
        return
    
    # Load latest dataset
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Validating data quality for: {latest_file}")
    
    df = pd.read_parquet(latest_file)
    
    # Generate quality report
    quality_metrics = monitor.generate_quality_report(df)
    
    # Check thresholds
    quality_checks = monitor.check_quality_thresholds(quality_metrics)
    
    # Send alerts if quality issues detected
    failed_checks = [check for check, passed in quality_checks.items() if not passed]
    if failed_checks:
        monitor.send_quality_alert(quality_metrics, failed_checks)
    
    logger.info(f"Data quality monitoring completed. Overall score: {quality_metrics.overall_score:.2f}%")


if __name__ == "__main__":
    main()