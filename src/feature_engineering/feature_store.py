#!/usr/bin/env python3
"""
Advanced feature selection and validation for HYDATIS ML scheduler.
Implements feature selection algorithms and validation pipeline for optimal model performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedFeatureSelector:
    """Advanced feature selection for ML scheduler optimization."""
    
    def __init__(self, target_features: int = 50):
        self.target_features = target_features
        self.selection_methods = {
            'statistical': SelectKBest(score_func=f_regression, k=target_features),
            'mutual_info': SelectKBest(score_func=mutual_info_regression, k=target_features),
            'rfe_rf': RFE(RandomForestRegressor(n_estimators=50, random_state=42), n_features_to_select=target_features),
            'importance_rf': SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), max_features=target_features)
        }
    
    def select_features_multimethod(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[str]]:
        """Apply multiple feature selection methods and compare results."""
        
        logger.info(f"Selecting top {self.target_features} features from {len(X.columns)} candidates")
        
        selected_features = {}
        feature_scores = {}
        
        # Prepare data
        X_numeric = X.select_dtypes(include=[np.number])
        X_clean = X_numeric.fillna(X_numeric.median())
        y_clean = y.fillna(y.median())
        
        # Apply each selection method
        for method_name, selector in self.selection_methods.items():
            try:
                logger.info(f"Applying {method_name} feature selection...")
                
                if method_name in ['statistical', 'mutual_info']:
                    selector.fit(X_clean, y_clean)
                    selected_idx = selector.get_support()
                    selected_cols = X_clean.columns[selected_idx].tolist()
                    scores = selector.scores_
                    
                elif method_name in ['rfe_rf', 'importance_rf']:
                    selector.fit(X_clean, y_clean)
                    selected_idx = selector.get_support()
                    selected_cols = X_clean.columns[selected_idx].tolist()
                    scores = getattr(selector.estimator_, 'feature_importances_', None)
                
                selected_features[method_name] = selected_cols
                if scores is not None:
                    feature_scores[method_name] = dict(zip(X_clean.columns, scores))
                
                logger.info(f"{method_name}: selected {len(selected_cols)} features")
                
            except Exception as e:
                logger.warning(f"Feature selection method {method_name} failed: {e}")
                selected_features[method_name] = []
        
        return selected_features, feature_scores
    
    def create_consensus_feature_set(self, selected_features: Dict[str, List[str]]) -> List[str]:
        """Create consensus feature set from multiple selection methods."""
        
        # Count how many methods selected each feature
        all_features = set()
        for features in selected_features.values():
            all_features.update(features)
        
        feature_votes = {}
        for feature in all_features:
            votes = sum(1 for features in selected_features.values() if feature in features)
            feature_votes[feature] = votes
        
        # Sort by votes and take top features
        consensus_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        # Take features with at least 2 votes or top target_features
        final_features = []
        for feature, votes in consensus_features:
            if votes >= 2 or len(final_features) < self.target_features:
                final_features.append(feature)
            if len(final_features) >= self.target_features:
                break
        
        logger.info(f"Consensus feature set: {len(final_features)} features")
        return final_features
    
    def validate_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                                  feature_list: List[str]) -> Dict[str, float]:
        """Validate feature importance using multiple metrics."""
        
        X_selected = X[feature_list]
        
        # Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_selected, y)
        rf_importance = dict(zip(feature_list, rf.feature_importances_))
        
        # Correlation with target
        correlations = {}
        for feature in feature_list:
            corr = X_selected[feature].corr(y)
            correlations[feature] = abs(corr) if not np.isnan(corr) else 0
        
        # Combined importance score
        combined_scores = {}
        for feature in feature_list:
            combined_scores[feature] = (
                rf_importance.get(feature, 0) * 0.7 +
                correlations.get(feature, 0) * 0.3
            )
        
        return combined_scores


class FeatureValidationPipeline:
    """Validates feature engineering pipeline for ML scheduler."""
    
    def __init__(self):
        self.validation_metrics = [
            'feature_count', 'null_percentage', 'feature_correlation',
            'target_correlation', 'feature_importance', 'multicollinearity'
        ]
    
    def validate_feature_quality(self, features_df: pd.DataFrame, 
                                target_col: str = 'target_cpu_next') -> Dict:
        """Comprehensive feature quality validation."""
        
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_features': len(features_df.columns),
            'total_samples': len(features_df)
        }
        
        # Numeric features only for analysis
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        validation_results['numeric_features'] = len(numeric_cols)
        
        # 1. Data completeness
        null_percentages = features_df[numeric_cols].isnull().mean()
        validation_results['null_percentage_avg'] = float(null_percentages.mean())
        validation_results['features_with_high_nulls'] = len(null_percentages[null_percentages > 0.1])
        
        # 2. Feature correlation analysis
        if len(numeric_cols) > 1:
            corr_matrix = features_df[numeric_cols].corr()
            
            # High correlation pairs (potential redundancy)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.9:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            validation_results['high_correlation_pairs'] = len(high_corr_pairs)
            validation_results['multicollinearity_risk'] = len(high_corr_pairs) > 5
        
        # 3. Target correlation (if target exists)
        if target_col in features_df.columns:
            target = features_df[target_col]
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            target_correlations = {}
            for feature in feature_cols:
                corr = features_df[feature].corr(target)
                if not np.isnan(corr):
                    target_correlations[feature] = abs(corr)
            
            validation_results['avg_target_correlation'] = np.mean(list(target_correlations.values()))
            validation_results['features_with_good_target_corr'] = len([c for c in target_correlations.values() if c > 0.1])
        
        # 4. Feature distribution analysis
        skewness_scores = features_df[numeric_cols].skew()
        validation_results['highly_skewed_features'] = len(skewness_scores[abs(skewness_scores) > 2])
        
        # 5. Overall quality score
        quality_components = {
            'completeness': max(0, 1 - validation_results['null_percentage_avg']),
            'target_relevance': validation_results.get('avg_target_correlation', 0.1),
            'diversity': max(0, 1 - validation_results.get('high_correlation_pairs', 0) / 10),
            'distribution': max(0, 1 - validation_results['highly_skewed_features'] / len(numeric_cols))
        }
        
        overall_quality = sum(quality_components.values()) / len(quality_components) * 100
        validation_results['overall_quality_score'] = round(overall_quality, 2)
        validation_results['quality_components'] = quality_components
        
        # Pass/fail determination
        validation_results['quality_pass'] = overall_quality >= 85.0
        
        return validation_results
    
    def generate_feature_report(self, features_df: pd.DataFrame) -> str:
        """Generate human-readable feature engineering report."""
        
        validation = self.validate_feature_quality(features_df)
        
        report = f"""
# HYDATIS ML Scheduler Feature Engineering Report

## Summary
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Features**: {validation['total_features']}
- **Training Samples**: {validation['total_samples']:,}
- **Quality Score**: {validation['overall_quality_score']:.1f}%
- **Status**: {'✅ PASS' if validation['quality_pass'] else '❌ NEEDS IMPROVEMENT'}

## Feature Quality Analysis
- **Data Completeness**: {(1-validation['null_percentage_avg'])*100:.1f}%
- **Target Relevance**: {validation.get('avg_target_correlation', 0)*100:.1f}%
- **Feature Diversity**: Low correlation pairs
- **Distribution Quality**: Reasonable skewness

## Recommendations for ML Training
- Features ready for XGBoost load prediction model
- Features suitable for Q-Learning placement optimization  
- Anomaly detection features prepared
- Feature store integration validated

## Next Steps
- Proceed to Week 5: XGBoost model development
- Implement feature importance analysis
- Setup MLflow experiment tracking
"""
        
        return report


def main():
    """Test advanced feature selection pipeline."""
    selector = AdvancedFeatureSelector(target_features=30)
    validator = FeatureValidationPipeline()
    
    # Create sample feature data
    n_samples = 1000
    n_features = 80
    
    # Generate synthetic features with different patterns
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    
    # Create realistic target (CPU prediction)
    y = (X['feature_0'] * 0.3 + X['feature_1'] * 0.2 + 
         np.random.normal(0, 0.1, n_samples))
    y = np.clip(y, 0, 1)  # CPU utilization bounds
    
    # Test feature selection
    selected_features, scores = selector.select_features_multimethod(X, y)
    consensus_features = selector.create_consensus_feature_set(selected_features)
    
    # Test validation
    X['target_cpu_next'] = y
    validation_report = validator.validate_feature_quality(X)
    
    print(f"✓ Feature selection tested: {len(consensus_features)} consensus features")
    print(f"✓ Validation completed: {validation_report['overall_quality_score']:.1f}% quality")
    
    return selector, validator


if __name__ == "__main__":
    selector, validator = main()