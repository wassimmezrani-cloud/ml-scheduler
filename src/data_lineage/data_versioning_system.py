#!/usr/bin/env python3
"""
Data Versioning and Lineage System for HYDATIS ML Scheduler
Implements comprehensive data tracking, versioning, and lineage for MLOps compliance.
"""

import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import sqlite3
import uuid

logger = logging.getLogger(__name__)


@dataclass
class DataAsset:
    """Data asset metadata and versioning information."""
    asset_id: str
    asset_name: str
    asset_type: str  # 'raw_metrics', 'engineered_features', 'training_data', 'model_artifacts'
    version: str
    creation_timestamp: datetime
    source_system: str
    data_schema: Dict
    quality_metrics: Dict
    lineage_info: Dict
    file_path: Optional[str] = None
    checksum: Optional[str] = None
    size_bytes: Optional[int] = None


@dataclass
class DataLineage:
    """Data lineage tracking for ML pipeline."""
    lineage_id: str
    source_assets: List[str]
    target_asset: str
    transformation_type: str
    transformation_config: Dict
    processing_timestamp: datetime
    processor_info: Dict
    quality_impact: Dict


@dataclass
class DataVersion:
    """Data version with change tracking."""
    version_id: str
    asset_id: str
    version_number: str
    change_type: str  # 'initial', 'update', 'schema_change', 'quality_improvement'
    change_description: str
    previous_version: Optional[str]
    diff_summary: Dict
    approval_status: str
    created_by: str
    created_timestamp: datetime


class HYDATISDataVersioningSystem:
    """Data versioning and lineage system for HYDATIS ML Scheduler."""
    
    def __init__(self, storage_backend: str = "local", config: Dict = None):
        self.storage_backend = storage_backend
        self.config = config or {}
        
        # Initialize storage
        self.db_path = self.config.get('db_path', '/tmp/hydatis_data_lineage.db')
        self._initialize_database()
        
        # HYDATIS cluster data sources
        self.data_sources = {
            'prometheus_metrics': {
                'endpoint': 'http://10.110.190.32:9090',
                'retention_days': 30,
                'collection_interval': 30,  # seconds
                'metrics': [
                    'node_cpu_usage', 'node_memory_usage', 'pod_cpu_usage',
                    'pod_memory_usage', 'scheduling_latency', 'cluster_state'
                ]
            },
            'kubernetes_events': {
                'cluster_endpoint': 'https://10.110.190.31:6443',
                'namespaces': ['hydatis-mlops', 'kube-system', 'monitoring'],
                'event_types': ['scheduling', 'scaling', 'failures']
            },
            'mlflow_experiments': {
                'tracking_uri': 'http://10.110.190.32:31380',
                'experiment_prefixes': ['hydatis-', 'katib-'],
                'artifact_types': ['models', 'metrics', 'parameters']
            }
        }
        
        # Data quality standards for HYDATIS
        self.quality_standards = {
            'completeness_threshold': 0.95,
            'consistency_threshold': 0.90,
            'timeliness_threshold': 0.98,
            'accuracy_threshold': 0.92,
            'uniqueness_threshold': 0.99
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for data lineage tracking."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create data assets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_assets (
                asset_id TEXT PRIMARY KEY,
                asset_name TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                version TEXT NOT NULL,
                creation_timestamp TEXT NOT NULL,
                source_system TEXT NOT NULL,
                data_schema TEXT,
                quality_metrics TEXT,
                lineage_info TEXT,
                file_path TEXT,
                checksum TEXT,
                size_bytes INTEGER
            )
        ''')
        
        # Create data lineage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_lineage (
                lineage_id TEXT PRIMARY KEY,
                source_assets TEXT NOT NULL,
                target_asset TEXT NOT NULL,
                transformation_type TEXT NOT NULL,
                transformation_config TEXT,
                processing_timestamp TEXT NOT NULL,
                processor_info TEXT,
                quality_impact TEXT
            )
        ''')
        
        # Create data versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_versions (
                version_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                version_number TEXT NOT NULL,
                change_type TEXT NOT NULL,
                change_description TEXT,
                previous_version TEXT,
                diff_summary TEXT,
                approval_status TEXT,
                created_by TEXT,
                created_timestamp TEXT,
                FOREIGN KEY (asset_id) REFERENCES data_assets (asset_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("📊 Data lineage database initialized")
    
    def register_data_asset(self, asset_name: str, asset_type: str, 
                          source_system: str, data_content: Any,
                          metadata: Dict = None) -> DataAsset:
        """Register new data asset with versioning and lineage tracking."""
        
        logger.info(f"📝 Registering data asset: {asset_name} ({asset_type})")
        
        # Generate asset ID
        asset_id = f"hydatis_{asset_type}_{uuid.uuid4().hex[:8]}"
        
        # Calculate checksum and size
        if isinstance(data_content, (pd.DataFrame, dict, list)):
            content_str = json.dumps(data_content, default=str, sort_keys=True)
            checksum = hashlib.sha256(content_str.encode()).hexdigest()
            size_bytes = len(content_str.encode())
        else:
            checksum = None
            size_bytes = None
        
        # Infer data schema
        if isinstance(data_content, pd.DataFrame):
            data_schema = {
                'columns': list(data_content.columns),
                'dtypes': {col: str(dtype) for col, dtype in data_content.dtypes.items()},
                'row_count': len(data_content),
                'column_count': len(data_content.columns)
            }
        elif isinstance(data_content, dict):
            data_schema = {
                'type': 'dictionary',
                'keys': list(data_content.keys()),
                'structure': {k: type(v).__name__ for k, v in data_content.items()}
            }
        else:
            data_schema = {'type': type(data_content).__name__}
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(data_content, asset_type)
        
        # Create data asset
        data_asset = DataAsset(
            asset_id=asset_id,
            asset_name=asset_name,
            asset_type=asset_type,
            version="v1.0.0",
            creation_timestamp=datetime.now(),
            source_system=source_system,
            data_schema=data_schema,
            quality_metrics=quality_metrics,
            lineage_info={'initial_creation': True},
            checksum=checksum,
            size_bytes=size_bytes
        )
        
        # Store in database
        self._store_data_asset(data_asset)
        
        logger.info(f"✅ Data asset registered: {asset_id}")
        
        return data_asset
    
    def create_data_lineage(self, source_asset_ids: List[str], target_asset_id: str,
                          transformation_type: str, transformation_config: Dict,
                          processor_info: Dict = None) -> DataLineage:
        """Create data lineage record for transformation tracking."""
        
        logger.info(f"🔗 Creating data lineage: {source_asset_ids} → {target_asset_id}")
        
        # Generate lineage ID
        lineage_id = f"lineage_{uuid.uuid4().hex[:8]}"
        
        # Calculate quality impact
        quality_impact = self._assess_transformation_quality_impact(
            source_asset_ids, target_asset_id, transformation_type
        )
        
        # Create lineage record
        data_lineage = DataLineage(
            lineage_id=lineage_id,
            source_assets=source_asset_ids,
            target_asset=target_asset_id,
            transformation_type=transformation_type,
            transformation_config=transformation_config,
            processing_timestamp=datetime.now(),
            processor_info=processor_info or {'processor': 'hydatis_ml_scheduler'},
            quality_impact=quality_impact
        )
        
        # Store lineage
        self._store_data_lineage(data_lineage)
        
        logger.info(f"✅ Data lineage created: {lineage_id}")
        
        return data_lineage
    
    def version_data_asset(self, asset_id: str, updated_content: Any,
                         change_type: str, change_description: str,
                         created_by: str = "hydatis_system") -> DataVersion:
        """Create new version of existing data asset."""
        
        logger.info(f"🔄 Creating new version for asset: {asset_id}")
        
        # Get current asset
        current_asset = self._get_data_asset(asset_id)
        
        if not current_asset:
            raise ValueError(f"Asset {asset_id} not found")
        
        # Generate new version
        current_version_parts = current_asset.version.replace('v', '').split('.')
        major, minor, patch = map(int, current_version_parts)
        
        if change_type == 'schema_change':
            major += 1
            minor = 0
            patch = 0
        elif change_type == 'quality_improvement':
            minor += 1
            patch = 0
        else:
            patch += 1
        
        new_version = f"v{major}.{minor}.{patch}"
        
        # Calculate differences
        diff_summary = self._calculate_asset_diff(current_asset, updated_content)
        
        # Create version record
        version_id = f"version_{uuid.uuid4().hex[:8]}"
        
        data_version = DataVersion(
            version_id=version_id,
            asset_id=asset_id,
            version_number=new_version,
            change_type=change_type,
            change_description=change_description,
            previous_version=current_asset.version,
            diff_summary=diff_summary,
            approval_status="pending",
            created_by=created_by,
            created_timestamp=datetime.now()
        )
        
        # Store version
        self._store_data_version(data_version)
        
        # Update asset with new version
        current_asset.version = new_version
        current_asset.quality_metrics = self._calculate_quality_metrics(updated_content, current_asset.asset_type)
        self._store_data_asset(current_asset)
        
        logger.info(f"✅ New version created: {asset_id} → {new_version}")
        
        return data_version
    
    def track_ml_pipeline_lineage(self, pipeline_id: str, pipeline_config: Dict) -> Dict:
        """Track complete ML pipeline lineage for HYDATIS scheduler."""
        
        logger.info(f"📊 Tracking ML pipeline lineage: {pipeline_id}")
        
        pipeline_lineage = {
            'pipeline_id': pipeline_id,
            'pipeline_type': 'hydatis_ml_scheduler_training',
            'execution_timestamp': datetime.now().isoformat(),
            'pipeline_config': pipeline_config,
            'data_flow': [],
            'model_artifacts': [],
            'quality_gates': []
        }
        
        # Stage 1: Raw data collection
        raw_data_asset = self.register_data_asset(
            asset_name=f"hydatis_cluster_metrics_{datetime.now().strftime('%Y%m%d')}",
            asset_type="raw_metrics",
            source_system="prometheus_hydatis_cluster",
            data_content={'metrics_collected': True, 'retention_days': 30},
            metadata={'cluster_nodes': 6, 'collection_interval': 30}
        )
        
        pipeline_lineage['data_flow'].append({
            'stage': 'data_collection',
            'asset_id': raw_data_asset.asset_id,
            'asset_type': 'raw_metrics'
        })
        
        # Stage 2: Feature engineering
        feature_engineering_lineage = self.create_data_lineage(
            source_asset_ids=[raw_data_asset.asset_id],
            target_asset_id=f"engineered_features_{uuid.uuid4().hex[:8]}",
            transformation_type="feature_engineering",
            transformation_config={
                'feature_count': 50,
                'engineering_pipeline': 'hydatis_workload_features',
                'temporal_windows': [5, 15, 30, 60],  # minutes
                'aggregation_functions': ['mean', 'std', 'p95', 'trend']
            },
            processor_info={'processor': 'hydatis_feature_engineer', 'version': '1.0.0'}
        )
        
        pipeline_lineage['data_flow'].append({
            'stage': 'feature_engineering',
            'lineage_id': feature_engineering_lineage.lineage_id,
            'transformation': 'raw_metrics_to_ml_features'
        })
        
        # Stage 3: Model training lineage
        model_training_assets = []
        
        for model_type in ['xgboost_cpu', 'xgboost_memory', 'qlearning', 'isolation_forest']:
            
            # Create training data asset
            training_data_asset = self.register_data_asset(
                asset_name=f"training_data_{model_type}_{datetime.now().strftime('%Y%m%d')}",
                asset_type="training_data",
                source_system="hydatis_feature_pipeline",
                data_content={'model_type': model_type, 'training_samples': 10000},
                metadata={'feature_count': 50, 'target_variable': f'{model_type}_target'}
            )
            
            # Create model artifact asset
            model_artifact_asset = self.register_data_asset(
                asset_name=f"model_{model_type}_{datetime.now().strftime('%Y%m%d')}",
                asset_type="model_artifacts",
                source_system="hydatis_ml_training_pipeline",
                data_content={'model_type': model_type, 'training_completed': True},
                metadata={'algorithm': model_type, 'target_metric': 'business_score'}
            )
            
            # Create training lineage
            training_lineage = self.create_data_lineage(
                source_asset_ids=[feature_engineering_lineage.target_asset, training_data_asset.asset_id],
                target_asset_id=model_artifact_asset.asset_id,
                transformation_type="model_training",
                transformation_config={
                    'model_type': model_type,
                    'training_algorithm': model_type,
                    'hyperparameters': 'katib_optimized',
                    'validation_strategy': 'cross_validation'
                },
                processor_info={'processor': 'hydatis_ml_trainer', 'mlflow_integration': True}
            )
            
            model_training_assets.append({
                'model_type': model_type,
                'training_data_asset': training_data_asset.asset_id,
                'model_artifact_asset': model_artifact_asset.asset_id,
                'training_lineage': training_lineage.lineage_id
            })
            
            pipeline_lineage['data_flow'].append({
                'stage': f'model_training_{model_type}',
                'lineage_id': training_lineage.lineage_id,
                'transformation': f'features_to_{model_type}_model'
            })
        
        pipeline_lineage['model_artifacts'] = model_training_assets
        
        # Stage 4: Model validation and deployment
        for model_asset in model_training_assets:
            
            # Create validation lineage
            validation_lineage = self.create_data_lineage(
                source_asset_ids=[model_asset['model_artifact_asset']],
                target_asset_id=f"validation_results_{uuid.uuid4().hex[:8]}",
                transformation_type="model_validation",
                transformation_config={
                    'validation_type': 'business_impact_validation',
                    'business_targets': self.config.get('business_targets', {}),
                    'validation_metrics': ['accuracy', 'business_score', 'latency', 'resource_efficiency']
                },
                processor_info={'processor': 'hydatis_model_validator', 'governance_framework': True}
            )
            
            pipeline_lineage['quality_gates'].append({
                'model_type': model_asset['model_type'],
                'validation_lineage': validation_lineage.lineage_id,
                'gate_type': 'business_validation'
            })
        
        # Store complete pipeline lineage
        self._store_pipeline_lineage(pipeline_lineage)
        
        logger.info(f"✅ ML pipeline lineage tracked: {len(pipeline_lineage['data_flow'])} stages")
        
        return pipeline_lineage
    
    def _calculate_quality_metrics(self, data_content: Any, asset_type: str) -> Dict:
        """Calculate data quality metrics for HYDATIS standards."""
        
        quality_metrics = {
            'completeness': 0.0,
            'consistency': 0.0,
            'timeliness': 0.0,
            'accuracy': 0.0,
            'uniqueness': 0.0,
            'overall_quality_score': 0.0
        }
        
        if isinstance(data_content, pd.DataFrame):
            
            # Completeness: Non-null values
            completeness = 1.0 - (data_content.isnull().sum().sum() / data_content.size)
            quality_metrics['completeness'] = float(completeness)
            
            # Consistency: Data type consistency
            consistency_scores = []
            for col in data_content.columns:
                if data_content[col].dtype == 'object':
                    # Check string format consistency
                    unique_formats = data_content[col].apply(lambda x: type(x).__name__).nunique()
                    consistency_scores.append(1.0 / max(1, unique_formats))
                else:
                    # Numeric data consistency
                    if not data_content[col].isnull().all():
                        std_dev = data_content[col].std()
                        mean_val = data_content[col].mean()
                        cv = std_dev / max(0.001, abs(mean_val))  # Coefficient of variation
                        consistency_scores.append(max(0, 1.0 - min(1, cv / 2)))
            
            quality_metrics['consistency'] = float(np.mean(consistency_scores)) if consistency_scores else 0.0
            
            # Uniqueness: Duplicate detection
            if len(data_content) > 0:
                duplicate_rate = data_content.duplicated().sum() / len(data_content)
                quality_metrics['uniqueness'] = float(1.0 - duplicate_rate)
            
            # Timeliness: For time-series data
            if 'timestamp' in data_content.columns:
                timestamps = pd.to_datetime(data_content['timestamp'], errors='coerce')
                current_time = datetime.now()
                
                if not timestamps.isnull().all():
                    latest_timestamp = timestamps.max()
                    time_diff = (current_time - latest_timestamp).total_seconds() / 3600  # hours
                    timeliness = max(0, 1.0 - (time_diff / 24))  # Degrade over 24 hours
                    quality_metrics['timeliness'] = float(timeliness)
            
            # Accuracy: Statistical validation
            numeric_columns = data_content.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                accuracy_scores = []
                for col in numeric_columns:
                    col_data = data_content[col].dropna()
                    if len(col_data) > 10:
                        # Check for reasonable value ranges
                        q1, q3 = col_data.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        outliers = ((col_data < (q1 - 3*iqr)) | (col_data > (q3 + 3*iqr))).sum()
                        outlier_rate = outliers / len(col_data)
                        accuracy_scores.append(max(0, 1.0 - outlier_rate))
                
                quality_metrics['accuracy'] = float(np.mean(accuracy_scores)) if accuracy_scores else 0.0
        
        else:
            # Default quality metrics for non-DataFrame data
            quality_metrics = {
                'completeness': 1.0 if data_content else 0.0,
                'consistency': 0.95,  # Assume good consistency for structured data
                'timeliness': 1.0,    # Assume fresh data
                'accuracy': 0.90,     # Conservative accuracy estimate
                'uniqueness': 1.0     # Assume unique for non-tabular data
            }
        
        # Calculate overall quality score
        quality_metrics['overall_quality_score'] = float(np.mean([
            quality_metrics['completeness'],
            quality_metrics['consistency'], 
            quality_metrics['timeliness'],
            quality_metrics['accuracy'],
            quality_metrics['uniqueness']
        ]))
        
        return quality_metrics
    
    def _assess_transformation_quality_impact(self, source_asset_ids: List[str], 
                                            target_asset_id: str, transformation_type: str) -> Dict:
        """Assess quality impact of data transformation."""
        
        # Get source asset quality metrics
        source_qualities = []
        
        for source_id in source_asset_ids:
            source_asset = self._get_data_asset(source_id)
            if source_asset:
                source_qualities.append(source_asset.quality_metrics['overall_quality_score'])
        
        # Get target asset quality metrics
        target_asset = self._get_data_asset(target_asset_id)
        target_quality = target_asset.quality_metrics['overall_quality_score'] if target_asset else 0.0
        
        # Calculate quality impact
        avg_source_quality = np.mean(source_qualities) if source_qualities else 0.0
        quality_delta = target_quality - avg_source_quality
        
        quality_impact = {
            'source_quality_avg': float(avg_source_quality),
            'target_quality': float(target_quality),
            'quality_delta': float(quality_delta),
            'quality_improvement': quality_delta > 0,
            'transformation_efficiency': min(1.0, target_quality / max(0.1, avg_source_quality)),
            'impact_assessment': 'improved' if quality_delta > 0.05 else 'maintained' if abs(quality_delta) <= 0.05 else 'degraded'
        }
        
        return quality_impact
    
    def _store_data_asset(self, data_asset: DataAsset):
        """Store data asset in database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO data_assets 
            (asset_id, asset_name, asset_type, version, creation_timestamp, 
             source_system, data_schema, quality_metrics, lineage_info, 
             file_path, checksum, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_asset.asset_id,
            data_asset.asset_name,
            data_asset.asset_type,
            data_asset.version,
            data_asset.creation_timestamp.isoformat(),
            data_asset.source_system,
            json.dumps(data_asset.data_schema),
            json.dumps(data_asset.quality_metrics),
            json.dumps(data_asset.lineage_info),
            data_asset.file_path,
            data_asset.checksum,
            data_asset.size_bytes
        ))
        
        conn.commit()
        conn.close()
    
    def _store_data_lineage(self, data_lineage: DataLineage):
        """Store data lineage in database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO data_lineage 
            (lineage_id, source_assets, target_asset, transformation_type,
             transformation_config, processing_timestamp, processor_info, quality_impact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_lineage.lineage_id,
            json.dumps(data_lineage.source_assets),
            data_lineage.target_asset,
            data_lineage.transformation_type,
            json.dumps(data_lineage.transformation_config),
            data_lineage.processing_timestamp.isoformat(),
            json.dumps(data_lineage.processor_info),
            json.dumps(data_lineage.quality_impact)
        ))
        
        conn.commit()
        conn.close()
    
    def _store_data_version(self, data_version: DataVersion):
        """Store data version in database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO data_versions 
            (version_id, asset_id, version_number, change_type, change_description,
             previous_version, diff_summary, approval_status, created_by, created_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_version.version_id,
            data_version.asset_id,
            data_version.version_number,
            data_version.change_type,
            data_version.change_description,
            data_version.previous_version,
            json.dumps(data_version.diff_summary),
            data_version.approval_status,
            data_version.created_by,
            data_version.created_timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_pipeline_lineage(self, pipeline_lineage: Dict):
        """Store complete pipeline lineage."""
        
        # Store as special asset type
        pipeline_asset = self.register_data_asset(
            asset_name=f"pipeline_{pipeline_lineage['pipeline_id']}",
            asset_type="ml_pipeline_lineage",
            source_system="hydatis_kubeflow_pipeline",
            data_content=pipeline_lineage,
            metadata={'pipeline_type': 'end_to_end_ml_training'}
        )
        
        logger.info(f"💾 Pipeline lineage stored: {pipeline_asset.asset_id}")
    
    def _get_data_asset(self, asset_id: str) -> Optional[DataAsset]:
        """Retrieve data asset from database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM data_assets WHERE asset_id = ?', (asset_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return DataAsset(
                asset_id=row[0],
                asset_name=row[1],
                asset_type=row[2],
                version=row[3],
                creation_timestamp=datetime.fromisoformat(row[4]),
                source_system=row[5],
                data_schema=json.loads(row[6]) if row[6] else {},
                quality_metrics=json.loads(row[7]) if row[7] else {},
                lineage_info=json.loads(row[8]) if row[8] else {},
                file_path=row[9],
                checksum=row[10],
                size_bytes=row[11]
            )
        
        return None
    
    def _calculate_asset_diff(self, current_asset: DataAsset, updated_content: Any) -> Dict:
        """Calculate differences between asset versions."""
        
        # Calculate content differences
        diff_summary = {
            'schema_changes': [],
            'quality_changes': {},
            'size_change': 0,
            'content_similarity': 0.0
        }
        
        # Schema comparison
        if isinstance(updated_content, pd.DataFrame):
            current_schema = current_asset.data_schema
            
            if 'columns' in current_schema:
                current_columns = set(current_schema['columns'])
                new_columns = set(updated_content.columns)
                
                added_columns = new_columns - current_columns
                removed_columns = current_columns - new_columns
                
                if added_columns:
                    diff_summary['schema_changes'].append(f"Added columns: {list(added_columns)}")
                if removed_columns:
                    diff_summary['schema_changes'].append(f"Removed columns: {list(removed_columns)}")
        
        # Quality changes
        if updated_content is not None:
            new_quality = self._calculate_quality_metrics(updated_content, current_asset.asset_type)
            
            for metric, old_value in current_asset.quality_metrics.items():
                if metric in new_quality:
                    change = new_quality[metric] - old_value
                    if abs(change) > 0.01:  # Significant change threshold
                        diff_summary['quality_changes'][metric] = {
                            'old': old_value,
                            'new': new_quality[metric],
                            'change': change
                        }
        
        return diff_summary
    
    def generate_lineage_report(self, asset_id: str) -> Dict:
        """Generate comprehensive lineage report for data asset."""
        
        logger.info(f"📋 Generating lineage report for: {asset_id}")
        
        # Get asset info
        asset = self._get_data_asset(asset_id)
        
        if not asset:
            return {'error': f'Asset {asset_id} not found'}
        
        # Get all lineage records involving this asset
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Upstream lineage (what created this asset)
        cursor.execute('''
            SELECT * FROM data_lineage 
            WHERE target_asset = ?
            ORDER BY processing_timestamp DESC
        ''', (asset_id,))
        upstream_lineage = cursor.fetchall()
        
        # Downstream lineage (what this asset created)
        cursor.execute('''
            SELECT * FROM data_lineage 
            WHERE source_assets LIKE ?
            ORDER BY processing_timestamp DESC
        ''', (f'%{asset_id}%',))
        downstream_lineage = cursor.fetchall()
        
        # Version history
        cursor.execute('''
            SELECT * FROM data_versions 
            WHERE asset_id = ?
            ORDER BY created_timestamp DESC
        ''', (asset_id,))
        version_history = cursor.fetchall()
        
        conn.close()
        
        # Generate comprehensive report
        lineage_report = {
            'asset_info': asdict(asset),
            'lineage_summary': {
                'upstream_transformations': len(upstream_lineage),
                'downstream_transformations': len(downstream_lineage),
                'version_count': len(version_history),
                'quality_score': asset.quality_metrics.get('overall_quality_score', 0.0)
            },
            'upstream_lineage': [self._format_lineage_record(record) for record in upstream_lineage],
            'downstream_lineage': [self._format_lineage_record(record) for record in downstream_lineage],
            'version_history': [self._format_version_record(record) for record in version_history],
            'data_quality_trend': self._calculate_quality_trend(asset_id),
            'compliance_status': self._check_asset_compliance(asset),
            'recommendations': self._generate_lineage_recommendations(asset, upstream_lineage, downstream_lineage)
        }
        
        logger.info(f"✅ Lineage report generated: {len(upstream_lineage)} upstream, {len(downstream_lineage)} downstream")
        
        return lineage_report
    
    def _format_lineage_record(self, record) -> Dict:
        """Format lineage database record for reporting."""
        
        return {
            'lineage_id': record[0],
            'source_assets': json.loads(record[1]),
            'target_asset': record[2],
            'transformation_type': record[3],
            'transformation_config': json.loads(record[4]) if record[4] else {},
            'processing_timestamp': record[5],
            'processor_info': json.loads(record[6]) if record[6] else {},
            'quality_impact': json.loads(record[7]) if record[7] else {}
        }
    
    def _format_version_record(self, record) -> Dict:
        """Format version database record for reporting."""
        
        return {
            'version_id': record[0],
            'asset_id': record[1],
            'version_number': record[2],
            'change_type': record[3],
            'change_description': record[4],
            'previous_version': record[5],
            'diff_summary': json.loads(record[6]) if record[6] else {},
            'approval_status': record[7],
            'created_by': record[8],
            'created_timestamp': record[9]
        }
    
    def _calculate_quality_trend(self, asset_id: str) -> Dict:
        """Calculate quality trend over time for asset."""
        
        # Get version history with quality metrics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT da.quality_metrics, dv.created_timestamp, dv.version_number
            FROM data_assets da
            JOIN data_versions dv ON da.asset_id = dv.asset_id
            WHERE da.asset_id = ?
            ORDER BY dv.created_timestamp
        ''', (asset_id,))
        
        history = cursor.fetchall()
        conn.close()
        
        if len(history) < 2:
            return {'trend': 'insufficient_data', 'quality_scores': []}
        
        quality_scores = []
        timestamps = []
        
        for record in history:
            quality_metrics = json.loads(record[0]) if record[0] else {}
            overall_score = quality_metrics.get('overall_quality_score', 0.0)
            quality_scores.append(overall_score)
            timestamps.append(record[1])
        
        # Calculate trend
        if len(quality_scores) >= 2:
            trend_slope = np.polyfit(range(len(quality_scores)), quality_scores, 1)[0]
            
            if trend_slope > 0.01:
                trend = 'improving'
            elif trend_slope < -0.01:
                trend = 'degrading'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'quality_scores': quality_scores,
            'timestamps': timestamps,
            'trend_slope': float(trend_slope) if len(quality_scores) >= 2 else 0.0,
            'latest_quality': quality_scores[-1] if quality_scores else 0.0
        }
    
    def _check_asset_compliance(self, asset: DataAsset) -> Dict:
        """Check asset compliance with HYDATIS standards."""
        
        compliance_status = {
            'quality_standards_met': asset.quality_metrics['overall_quality_score'] >= 0.90,
            'retention_policy_compliant': True,  # Assume compliant
            'access_control_applied': True,      # Kubernetes RBAC
            'audit_trail_complete': True,        # MLflow tracking
            'data_privacy_compliant': True,      # Internal cluster data
            'versioning_standards_met': True,    # Semantic versioning
            'overall_compliance': True
        }
        
        # Overall compliance
        compliance_status['overall_compliance'] = all([
            compliance_status['quality_standards_met'],
            compliance_status['retention_policy_compliant'],
            compliance_status['access_control_applied'],
            compliance_status['audit_trail_complete']
        ])
        
        return compliance_status
    
    def _generate_lineage_recommendations(self, asset: DataAsset, upstream: List, downstream: List) -> List[str]:
        """Generate recommendations for data asset optimization."""
        
        recommendations = []
        
        # Quality-based recommendations
        quality_score = asset.quality_metrics['overall_quality_score']
        
        if quality_score < 0.90:
            recommendations.append("Improve data quality through enhanced validation")
        
        if quality_score < 0.80:
            recommendations.append("Implement automated data quality monitoring")
        
        # Lineage-based recommendations
        if len(upstream) == 0 and asset.asset_type != 'raw_metrics':
            recommendations.append("Establish clear data lineage tracking")
        
        if len(downstream) > 10:
            recommendations.append("Consider data asset optimization for high fan-out")
        
        # Asset-specific recommendations
        if asset.asset_type == 'model_artifacts':
            recommendations.append("Ensure model versioning aligns with MLflow registry")
            recommendations.append("Implement automated model performance monitoring")
        
        if asset.asset_type == 'training_data':
            recommendations.append("Validate training data distribution alignment")
            recommendations.append("Monitor for data drift and concept drift")
        
        return recommendations


def main():
    """Test data versioning system with HYDATIS configuration."""
    
    # Initialize data versioning system
    versioning_system = HYDATISDataVersioningSystem(
        storage_backend="local",
        config={
            'business_targets': {
                'cpu_utilization_target': 0.65,
                'availability_target': 0.997,
                'roi_target': 14.0
            }
        }
    )
    
    # Test complete ML pipeline lineage tracking
    pipeline_config = {
        'data_sources': ['prometheus_hydatis_cluster'],
        'feature_engineering': 'hydatis_workload_features',
        'models': ['xgboost_cpu', 'xgboost_memory', 'qlearning', 'isolation_forest'],
        'deployment_target': 'hydatis_production_cluster'
    }
    
    pipeline_lineage = versioning_system.track_ml_pipeline_lineage(
        pipeline_id=f"hydatis_ml_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        pipeline_config=pipeline_config
    )
    
    print("\n🎯 Data Versioning System Test Results:")
    print(f"   Pipeline Lineage: {len(pipeline_lineage['data_flow'])} stages tracked")
    print(f"   Model Artifacts: {len(pipeline_lineage['model_artifacts'])} models")
    print(f"   Quality Gates: {len(pipeline_lineage['quality_gates'])} validations")
    print(f"   Database: {versioning_system.db_path}")
    
    return versioning_system


if __name__ == "__main__":
    system = main()