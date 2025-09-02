#!/usr/bin/env python3
"""
Model ensemble coordination for HYDATIS ML Scheduler.
Orchestrates decision-making across XGBoost, Q-Learning, and Isolation Forest models.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DecisionConfidence(Enum):
    """Decision confidence levels."""
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2


class SchedulingAction(Enum):
    """Possible scheduling actions."""
    SCHEDULE = "schedule"
    DEFER = "defer"
    RESCHEDULE = "reschedule"
    REJECT = "reject"


@dataclass
class ModelPrediction:
    """Structured prediction from individual ML model."""
    model_type: str
    prediction: Dict[str, Any]
    confidence: float
    latency_ms: float
    error: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class EnsembleDecision:
    """Final ensemble decision for pod scheduling."""
    action: SchedulingAction
    selected_node: str
    confidence: DecisionConfidence
    reasoning: List[str]
    risk_factors: List[str]
    alternative_nodes: List[Dict[str, Any]]
    model_contributions: Dict[str, float]
    decision_metadata: Dict[str, Any]


class HYDATISModelEnsemble:
    """Advanced ensemble coordinator for HYDATIS ML Scheduler models."""
    
    def __init__(self):
        self.model_weights = {
            'xgboost_load_predictor': 0.35,
            'qlearning_placement': 0.45,
            'isolation_forest_anomaly': 0.20
        }
        
        self.decision_thresholds = {
            'high_confidence_threshold': 0.8,
            'medium_confidence_threshold': 0.6,
            'low_confidence_threshold': 0.4,
            'anomaly_risk_threshold': 0.7,
            'capacity_warning_threshold': 0.8,
            'load_imbalance_threshold': 0.3
        }
        
        self.ensemble_config = {
            'voting_strategy': 'weighted_confidence',
            'consensus_requirement': 0.6,
            'conflict_resolution': 'highest_confidence',
            'anomaly_override': True,
            'load_prediction_weight_adjustment': True,
            'placement_optimization_priority': True
        }
        
        self.decision_history = []
        self.ensemble_performance = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'high_confidence_decisions': 0,
            'anomaly_overrides': 0,
            'consensus_failures': 0,
            'model_agreement_rates': {
                'load_placement_agreement': 0.0,
                'placement_anomaly_agreement': 0.0,
                'load_anomaly_agreement': 0.0
            }
        }
    
    def coordinate_scheduling_decision(self, 
                                     load_prediction: ModelPrediction,
                                     placement_optimization: ModelPrediction,
                                     anomaly_detection: ModelPrediction,
                                     pod_spec: Dict[str, Any]) -> EnsembleDecision:
        """Coordinate final scheduling decision using ensemble methods."""
        
        coordination_start = time.time()
        
        self.ensemble_performance['total_decisions'] += 1
        
        valid_predictions = [pred for pred in [load_prediction, placement_optimization, anomaly_detection] 
                           if pred.error is None]
        
        if len(valid_predictions) == 0:
            return self._generate_fallback_decision(pod_spec, "All ML models unavailable")
        
        conflict_analysis = self._analyze_model_conflicts(valid_predictions)
        
        risk_assessment = self._assess_scheduling_risks(valid_predictions, pod_spec)
        
        if anomaly_detection.error is None and self.ensemble_config['anomaly_override']:
            anomaly_override = self._check_anomaly_override(anomaly_detection, risk_assessment)
            if anomaly_override['should_override']:
                self.ensemble_performance['anomaly_overrides'] += 1
                return self._generate_anomaly_override_decision(anomaly_override, pod_spec)
        
        consensus_result = self._achieve_model_consensus(valid_predictions, conflict_analysis)
        
        if not consensus_result['consensus_reached']:
            self.ensemble_performance['consensus_failures'] += 1
            return self._resolve_ensemble_conflict(valid_predictions, pod_spec, conflict_analysis)
        
        final_decision = self._synthesize_ensemble_decision(
            consensus_result, 
            valid_predictions, 
            risk_assessment, 
            pod_spec
        )
        
        decision_confidence = self._calculate_ensemble_confidence(valid_predictions, consensus_result, risk_assessment)
        
        if decision_confidence >= DecisionConfidence.HIGH.value:
            self.ensemble_performance['high_confidence_decisions'] += 1
        
        self.ensemble_performance['successful_decisions'] += 1
        
        ensemble_decision = EnsembleDecision(
            action=SchedulingAction(final_decision['action']),
            selected_node=final_decision['selected_node'],
            confidence=self._map_confidence_level(decision_confidence),
            reasoning=final_decision['reasoning'],
            risk_factors=risk_assessment['identified_risks'],
            alternative_nodes=final_decision.get('alternative_nodes', []),
            model_contributions=self._calculate_model_contributions(valid_predictions),
            decision_metadata={
                'coordination_latency_ms': round((time.time() - coordination_start) * 1000, 2),
                'models_consulted': [pred.model_type for pred in valid_predictions],
                'consensus_achieved': consensus_result['consensus_reached'],
                'conflict_resolution_used': not consensus_result['consensus_reached'],
                'risk_level': risk_assessment['overall_risk_level'],
                'decision_timestamp': datetime.now().isoformat()
            }
        )
        
        self._update_agreement_rates(valid_predictions)
        
        self.decision_history.append(asdict(ensemble_decision))
        
        logger.info(f"Ensemble decision: {ensemble_decision.action.value} node {ensemble_decision.selected_node} "
                   f"(confidence: {ensemble_decision.confidence.name})")
        
        return ensemble_decision
    
    def _analyze_model_conflicts(self, predictions: List[ModelPrediction]) -> Dict[str, Any]:
        """Analyze conflicts between model predictions."""
        
        conflicts = {
            'node_recommendations': [],
            'confidence_variance': 0.0,
            'prediction_disagreement': False,
            'major_conflicts': []
        }
        
        for pred in predictions:
            if pred.model_type == 'qlearning_placement' and 'selected_node' in pred.prediction:
                conflicts['node_recommendations'].append({
                    'model': pred.model_type,
                    'node': pred.prediction['selected_node'],
                    'confidence': pred.confidence
                })
            elif pred.model_type == 'xgboost_load_predictor' and 'scheduling_recommendations' in pred.prediction:
                preferred_nodes = pred.prediction['scheduling_recommendations'].get('preferred_nodes', [])
                if preferred_nodes:
                    conflicts['node_recommendations'].append({
                        'model': pred.model_type,
                        'node': preferred_nodes[0],
                        'confidence': pred.confidence
                    })
        
        if len(conflicts['node_recommendations']) > 1:
            nodes = [rec['node'] for rec in conflicts['node_recommendations']]
            if len(set(nodes)) > 1:
                conflicts['prediction_disagreement'] = True
                conflicts['major_conflicts'].append('node_selection_disagreement')
        
        confidences = [pred.confidence for pred in predictions]
        conflicts['confidence_variance'] = np.var(confidences) if len(confidences) > 1 else 0.0
        
        if conflicts['confidence_variance'] > 0.1:
            conflicts['major_conflicts'].append('confidence_variance_high')
        
        return conflicts
    
    def _assess_scheduling_risks(self, predictions: List[ModelPrediction], 
                               pod_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Assess scheduling risks based on all model inputs."""
        
        risks = {
            'identified_risks': [],
            'risk_scores': {},
            'overall_risk_level': 'low'
        }
        
        for pred in predictions:
            if pred.model_type == 'isolation_forest_anomaly' and 'cluster_health_assessment' in pred.prediction:
                cluster_health = pred.prediction['cluster_health_assessment'].get('overall_health', 'healthy')
                
                if cluster_health == 'critical':
                    risks['identified_risks'].append('Critical cluster anomalies detected')
                    risks['risk_scores']['anomaly_risk'] = 1.0
                elif cluster_health == 'degraded':
                    risks['identified_risks'].append('Cluster degradation detected')
                    risks['risk_scores']['anomaly_risk'] = 0.7
                elif cluster_health == 'warning':
                    risks['identified_risks'].append('Cluster warning indicators')
                    risks['risk_scores']['anomaly_risk'] = 0.4
                else:
                    risks['risk_scores']['anomaly_risk'] = 0.1
            
            elif pred.model_type == 'xgboost_load_predictor' and 'cluster_summary' in pred.prediction:
                cluster_summary = pred.prediction['cluster_summary']
                
                avg_cpu = cluster_summary.get('avg_cpu_prediction', 0)
                avg_memory = cluster_summary.get('avg_memory_prediction', 0)
                
                if avg_cpu > 0.85 or avg_memory > 0.85:
                    risks['identified_risks'].append('High predicted cluster utilization')
                    risks['risk_scores']['capacity_risk'] = 0.8
                elif avg_cpu > 0.7 or avg_memory > 0.7:
                    risks['identified_risks'].append('Moderate predicted cluster utilization')
                    risks['risk_scores']['capacity_risk'] = 0.5
                else:
                    risks['risk_scores']['capacity_risk'] = 0.2
        
        pod_resources = pod_spec.get('resources', {})
        cpu_request = pod_resources.get('cpu_request', 0)
        memory_request = pod_resources.get('memory_request', 0)
        
        if cpu_request > 0.5 or memory_request > 0.5:
            risks['identified_risks'].append('High resource requirements')
            risks['risk_scores']['resource_risk'] = 0.6
        elif cpu_request > 0.3 or memory_request > 0.3:
            risks['risk_scores']['resource_risk'] = 0.3
        else:
            risks['risk_scores']['resource_risk'] = 0.1
        
        if risks['risk_scores']:
            overall_risk_score = max(risks['risk_scores'].values())
            
            if overall_risk_score > 0.8:
                risks['overall_risk_level'] = 'high'
            elif overall_risk_score > 0.5:
                risks['overall_risk_level'] = 'medium'
            else:
                risks['overall_risk_level'] = 'low'
        
        return risks
    
    def _check_anomaly_override(self, anomaly_prediction: ModelPrediction, 
                              risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Check if anomaly detection should override other decisions."""
        
        cluster_health = anomaly_prediction.prediction.get('cluster_health_assessment', {}).get('overall_health', 'healthy')
        active_critical = anomaly_prediction.prediction.get('active_issues', {}).get('critical_active', 0)
        
        should_override = False
        override_reason = ""
        
        if cluster_health == 'critical':
            should_override = True
            override_reason = "Critical cluster health detected"
        elif active_critical > 0:
            should_override = True
            override_reason = f"{active_critical} critical anomalies active"
        elif risk_assessment['overall_risk_level'] == 'high' and cluster_health == 'degraded':
            should_override = True
            override_reason = "High risk combined with cluster degradation"
        
        return {
            'should_override': should_override,
            'override_reason': override_reason,
            'cluster_health': cluster_health,
            'critical_anomalies': active_critical
        }
    
    def _generate_anomaly_override_decision(self, override_info: Dict[str, Any], 
                                          pod_spec: Dict[str, Any]) -> EnsembleDecision:
        """Generate decision when anomaly detection overrides other models."""
        
        return EnsembleDecision(
            action=SchedulingAction.DEFER,
            selected_node="none",
            confidence=DecisionConfidence.HIGH,
            reasoning=[
                f"Anomaly override: {override_info['override_reason']}",
                "Deferring scheduling until cluster health improves",
                f"Cluster health status: {override_info['cluster_health']}"
            ],
            risk_factors=[
                override_info['override_reason'],
                "Cluster stability compromised"
            ],
            alternative_nodes=[],
            model_contributions={'isolation_forest_anomaly': 1.0},
            decision_metadata={
                'override_triggered': True,
                'override_reason': override_info['override_reason'],
                'cluster_health': override_info['cluster_health'],
                'pod_name': pod_spec.get('metadata', {}).get('name', 'unknown')
            }
        )
    
    def _achieve_model_consensus(self, predictions: List[ModelPrediction], 
                               conflict_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to achieve consensus between model predictions."""
        
        if conflict_analysis['prediction_disagreement']:
            return {'consensus_reached': False, 'reason': 'node_selection_conflict'}
        
        consensus_score = 0.0
        consensus_factors = []
        
        confidence_agreement = 1.0 - conflict_analysis['confidence_variance']
        if confidence_agreement > 0.7:
            consensus_score += 0.4
            consensus_factors.append('confidence_alignment')
        
        node_recommendations = conflict_analysis['node_recommendations']
        if len(node_recommendations) > 1:
            nodes = [rec['node'] for rec in node_recommendations]
            if len(set(nodes)) == 1:
                consensus_score += 0.6
                consensus_factors.append('node_agreement')
        
        consensus_reached = consensus_score >= self.ensemble_config['consensus_requirement']
        
        consensus_result = {
            'consensus_reached': consensus_reached,
            'consensus_score': consensus_score,
            'consensus_factors': consensus_factors,
            'agreed_node': node_recommendations[0]['node'] if node_recommendations and consensus_reached else None
        }
        
        return consensus_result
    
    def _resolve_ensemble_conflict(self, predictions: List[ModelPrediction],
                                 pod_spec: Dict[str, Any],
                                 conflict_analysis: Dict[str, Any]) -> EnsembleDecision:
        """Resolve conflicts when models disagree."""
        
        conflict_resolution_strategy = self.ensemble_config['conflict_resolution']
        
        if conflict_resolution_strategy == 'highest_confidence':
            best_prediction = max(predictions, key=lambda p: p.confidence)
            
            selected_node = self._extract_node_recommendation(best_prediction)
            
            return EnsembleDecision(
                action=SchedulingAction.SCHEDULE,
                selected_node=selected_node,
                confidence=self._map_confidence_level(best_prediction.confidence),
                reasoning=[
                    f"Conflict resolved using highest confidence model: {best_prediction.model_type}",
                    f"Selected based on {best_prediction.confidence:.3f} confidence score"
                ],
                risk_factors=[
                    "Model disagreement detected",
                    "Decision based on single best model"
                ],
                alternative_nodes=self._get_alternative_recommendations(predictions, selected_node),
                model_contributions=self._calculate_conflict_contributions(predictions, best_prediction),
                decision_metadata={
                    'conflict_resolution': True,
                    'resolution_strategy': conflict_resolution_strategy,
                    'winning_model': best_prediction.model_type,
                    'conflict_factors': conflict_analysis['major_conflicts']
                }
            )
        
        elif conflict_resolution_strategy == 'weighted_voting':
            return self._weighted_voting_resolution(predictions, pod_spec, conflict_analysis)
        
        else:
            return self._generate_fallback_decision(pod_spec, "Unknown conflict resolution strategy")
    
    def _weighted_voting_resolution(self, predictions: List[ModelPrediction],
                                  pod_spec: Dict[str, Any],
                                  conflict_analysis: Dict[str, Any]) -> EnsembleDecision:
        """Resolve conflicts using weighted voting."""
        
        node_votes = {}
        total_weight = 0.0
        
        for pred in predictions:
            model_weight = self.model_weights.get(pred.model_type, 0.33)
            confidence_weight = pred.confidence
            
            effective_weight = model_weight * confidence_weight
            total_weight += effective_weight
            
            recommended_node = self._extract_node_recommendation(pred)
            
            if recommended_node:
                node_votes[recommended_node] = node_votes.get(recommended_node, 0) + effective_weight
        
        if not node_votes:
            return self._generate_fallback_decision(pod_spec, "No valid node recommendations")
        
        winning_node = max(node_votes.items(), key=lambda x: x[1])
        winning_confidence = winning_node[1] / total_weight if total_weight > 0 else 0
        
        return EnsembleDecision(
            action=SchedulingAction.SCHEDULE,
            selected_node=winning_node[0],
            confidence=self._map_confidence_level(winning_confidence),
            reasoning=[
                f"Weighted voting selected {winning_node[0]}",
                f"Voting confidence: {winning_confidence:.3f}",
                f"Models contributing: {len(predictions)}"
            ],
            risk_factors=["Model conflict resolved through voting"],
            alternative_nodes=[
                {'node': node, 'vote_weight': weight/total_weight} 
                for node, weight in sorted(node_votes.items(), key=lambda x: x[1], reverse=True)[1:3]
            ],
            model_contributions=self._calculate_weighted_contributions(predictions, node_votes, winning_node[0]),
            decision_metadata={
                'weighted_voting': True,
                'total_vote_weight': total_weight,
                'node_vote_distribution': node_votes
            }
        )
    
    def _synthesize_ensemble_decision(self, consensus_result: Dict[str, Any],
                                    predictions: List[ModelPrediction],
                                    risk_assessment: Dict[str, Any],
                                    pod_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final decision when consensus is achieved."""
        
        agreed_node = consensus_result['agreed_node']
        consensus_score = consensus_result['consensus_score']
        
        reasoning = [
            f"Model consensus achieved (score: {consensus_score:.3f})",
            f"Agreed recommendation: {agreed_node}",
            f"Consensus factors: {', '.join(consensus_result['consensus_factors'])}"
        ]
        
        for pred in predictions:
            if pred.model_type == 'xgboost_load_predictor':
                capacity = pred.prediction.get('cluster_summary', {}).get('avg_cpu_prediction', 0)
                if capacity < 0.7:
                    reasoning.append("Load predictor confirms adequate capacity")
                else:
                    reasoning.append("Load predictor indicates high utilization")
            
            elif pred.model_type == 'qlearning_placement':
                quality = pred.prediction.get('placement_reasoning', {}).get('quality_score', 0)
                if quality > 0.8:
                    reasoning.append("Q-Learning confirms high placement quality")
        
        alternative_nodes = []
        for pred in predictions:
            if pred.model_type == 'xgboost_load_predictor':
                preferred_nodes = pred.prediction.get('scheduling_recommendations', {}).get('preferred_nodes', [])
                for node in preferred_nodes[:2]:
                    if node != agreed_node:
                        alternative_nodes.append({
                            'node': node,
                            'reason': 'Load predictor alternative',
                            'source_model': 'xgboost_load_predictor'
                        })
        
        decision = {
            'action': 'schedule',
            'selected_node': agreed_node,
            'reasoning': reasoning,
            'alternative_nodes': alternative_nodes[:3]
        }
        
        return decision
    
    def _extract_node_recommendation(self, prediction: ModelPrediction) -> Optional[str]:
        """Extract node recommendation from model prediction."""
        
        if prediction.model_type == 'qlearning_placement':
            return prediction.prediction.get('selected_node')
        
        elif prediction.model_type == 'xgboost_load_predictor':
            scheduling_recs = prediction.prediction.get('scheduling_recommendations', {})
            preferred_nodes = scheduling_recs.get('preferred_nodes', [])
            return preferred_nodes[0] if preferred_nodes else None
        
        return None
    
    def _calculate_ensemble_confidence(self, predictions: List[ModelPrediction],
                                     consensus_result: Dict[str, Any],
                                     risk_assessment: Dict[str, Any]) -> float:
        """Calculate overall ensemble confidence."""
        
        base_confidence = np.mean([pred.confidence for pred in predictions])
        
        consensus_bonus = consensus_result['consensus_score'] * 0.2 if consensus_result['consensus_reached'] else 0
        
        risk_penalty = 0.0
        if risk_assessment['overall_risk_level'] == 'high':
            risk_penalty = 0.3
        elif risk_assessment['overall_risk_level'] == 'medium':
            risk_penalty = 0.15
        
        model_coverage_bonus = (len(predictions) / 3.0) * 0.1
        
        ensemble_confidence = base_confidence + consensus_bonus + model_coverage_bonus - risk_penalty
        
        return max(0.1, min(1.0, ensemble_confidence))
    
    def _map_confidence_level(self, confidence_score: float) -> DecisionConfidence:
        """Map confidence score to enum level."""
        
        if confidence_score >= DecisionConfidence.VERY_HIGH.value:
            return DecisionConfidence.VERY_HIGH
        elif confidence_score >= DecisionConfidence.HIGH.value:
            return DecisionConfidence.HIGH
        elif confidence_score >= DecisionConfidence.MEDIUM.value:
            return DecisionConfidence.MEDIUM
        elif confidence_score >= DecisionConfidence.LOW.value:
            return DecisionConfidence.LOW
        else:
            return DecisionConfidence.VERY_LOW
    
    def _calculate_model_contributions(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """Calculate each model's contribution to final decision."""
        
        contributions = {}
        total_confidence = sum(pred.confidence for pred in predictions)
        
        for pred in predictions:
            model_weight = self.model_weights.get(pred.model_type, 0.33)
            confidence_weight = pred.confidence / total_confidence if total_confidence > 0 else 0.33
            
            contribution = (model_weight + confidence_weight) / 2.0
            contributions[pred.model_type] = contribution
        
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {k: v / total_contribution for k, v in contributions.items()}
        
        return contributions
    
    def _calculate_weighted_contributions(self, predictions: List[ModelPrediction],
                                        node_votes: Dict[str, float],
                                        winning_node: str) -> Dict[str, float]:
        """Calculate model contributions for weighted voting resolution."""
        
        contributions = {}
        
        for pred in predictions:
            recommended_node = self._extract_node_recommendation(pred)
            
            if recommended_node == winning_node:
                model_weight = self.model_weights.get(pred.model_type, 0.33)
                contributions[pred.model_type] = model_weight * pred.confidence
            else:
                contributions[pred.model_type] = 0.1
        
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        
        return contributions
    
    def _calculate_conflict_contributions(self, predictions: List[ModelPrediction],
                                        winning_prediction: ModelPrediction) -> Dict[str, float]:
        """Calculate contributions when using highest confidence resolution."""
        
        contributions = {}
        
        for pred in predictions:
            if pred == winning_prediction:
                contributions[pred.model_type] = 0.8
            else:
                contributions[pred.model_type] = 0.1
        
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        
        return contributions
    
    def _get_alternative_recommendations(self, predictions: List[ModelPrediction],
                                       selected_node: str) -> List[Dict[str, Any]]:
        """Get alternative node recommendations from other models."""
        
        alternatives = []
        
        for pred in predictions:
            recommended_node = self._extract_node_recommendation(pred)
            
            if recommended_node and recommended_node != selected_node:
                alternatives.append({
                    'node': recommended_node,
                    'source_model': pred.model_type,
                    'confidence': pred.confidence,
                    'reason': f"Alternative from {pred.model_type}"
                })
        
        return sorted(alternatives, key=lambda x: x['confidence'], reverse=True)[:2]
    
    def _update_agreement_rates(self, predictions: List[ModelPrediction]):
        """Update model agreement tracking."""
        
        model_nodes = {}
        for pred in predictions:
            node = self._extract_node_recommendation(pred)
            if node:
                model_nodes[pred.model_type] = node
        
        if len(model_nodes) >= 2:
            models = list(model_nodes.keys())
            
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    model1, model2 = models[i], models[j]
                    
                    agreement_key = f"{model1}_{model2}_agreement"
                    if agreement_key not in self.ensemble_performance['model_agreement_rates']:
                        self.ensemble_performance['model_agreement_rates'][agreement_key] = 0.0
                    
                    if model_nodes[model1] == model_nodes[model2]:
                        current_rate = self.ensemble_performance['model_agreement_rates'][agreement_key]
                        decisions_count = self.ensemble_performance['total_decisions']
                        
                        new_rate = (current_rate * (decisions_count - 1) + 1.0) / decisions_count
                        self.ensemble_performance['model_agreement_rates'][agreement_key] = new_rate
    
    def _generate_fallback_decision(self, pod_spec: Dict[str, Any], reason: str) -> EnsembleDecision:
        """Generate fallback decision when ensemble coordination fails."""
        
        return EnsembleDecision(
            action=SchedulingAction.SCHEDULE,
            selected_node='worker-1',
            confidence=DecisionConfidence.LOW,
            reasoning=[
                f"Fallback decision: {reason}",
                "Using default node selection strategy",
                "ML ensemble coordination unavailable"
            ],
            risk_factors=[
                reason,
                "No ML optimization available",
                "Using basic fallback strategy"
            ],
            alternative_nodes=[
                {'node': 'worker-2', 'reason': 'Fallback alternative 1'},
                {'node': 'worker-3', 'reason': 'Fallback alternative 2'}
            ],
            model_contributions={},
            decision_metadata={
                'fallback_mode': True,
                'fallback_reason': reason,
                'pod_name': pod_spec.get('metadata', {}).get('name', 'unknown')
            }
        )
    
    def get_ensemble_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble performance metrics."""
        
        success_rate = (self.ensemble_performance['successful_decisions'] / 
                       max(self.ensemble_performance['total_decisions'], 1))
        
        high_confidence_rate = (self.ensemble_performance['high_confidence_decisions'] / 
                              max(self.ensemble_performance['total_decisions'], 1))
        
        consensus_rate = 1.0 - (self.ensemble_performance['consensus_failures'] / 
                              max(self.ensemble_performance['total_decisions'], 1))
        
        metrics = {
            'ensemble_statistics': {
                'total_decisions': self.ensemble_performance['total_decisions'],
                'success_rate': success_rate,
                'high_confidence_rate': high_confidence_rate,
                'consensus_achievement_rate': consensus_rate,
                'anomaly_override_rate': (self.ensemble_performance['anomaly_overrides'] / 
                                        max(self.ensemble_performance['total_decisions'], 1))
            },
            'model_coordination': {
                'model_weights': self.model_weights,
                'agreement_rates': self.ensemble_performance['model_agreement_rates'],
                'decision_thresholds': self.decision_thresholds
            },
            'ensemble_effectiveness': {
                'decision_quality_score': (success_rate + high_confidence_rate + consensus_rate) / 3.0,
                'coordination_efficiency': 1.0 - (self.ensemble_performance['consensus_failures'] / 
                                                max(self.ensemble_performance['total_decisions'], 1)),
                'ml_utilization_rate': self._calculate_ml_utilization_rate()
            },
            'recent_decisions': self.decision_history[-10:] if self.decision_history else [],
            'metrics_timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def _calculate_ml_utilization_rate(self) -> float:
        """Calculate how effectively ML models are being utilized."""
        
        if self.ensemble_performance['total_decisions'] == 0:
            return 0.0
        
        successful_ml_decisions = self.ensemble_performance['successful_decisions']
        fallback_decisions = (self.ensemble_performance['total_decisions'] - 
                            self.ensemble_performance['successful_decisions'])
        
        utilization_rate = successful_ml_decisions / self.ensemble_performance['total_decisions']
        
        return utilization_rate
    
    def optimize_ensemble_weights(self, performance_feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize ensemble model weights based on performance feedback."""
        
        if len(performance_feedback) < 10:
            return {'error': 'Insufficient feedback data for optimization'}
        
        model_performance_scores = {
            'xgboost_load_predictor': [],
            'qlearning_placement': [],
            'isolation_forest_anomaly': []
        }
        
        for feedback in performance_feedback:
            decision_quality = feedback.get('decision_quality_score', 0.5)
            
            for model_type in model_performance_scores.keys():
                contribution = feedback.get('model_contributions', {}).get(model_type, 0)
                
                if contribution > 0:
                    model_score = decision_quality * contribution
                    model_performance_scores[model_type].append(model_score)
        
        new_weights = {}
        total_weight = 0.0
        
        for model_type, scores in model_performance_scores.items():
            if scores:
                avg_performance = np.mean(scores)
                stability = 1.0 - np.std(scores) if len(scores) > 1 else 1.0
                
                weight = avg_performance * stability
                new_weights[model_type] = weight
                total_weight += weight
        
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in new_weights.items()}
            
            weight_changes = {}
            for model_type in self.model_weights.keys():
                old_weight = self.model_weights[model_type]
                new_weight = normalized_weights.get(model_type, old_weight)
                
                weight_changes[model_type] = {
                    'old_weight': old_weight,
                    'new_weight': new_weight,
                    'change': new_weight - old_weight
                }
            
            self.model_weights.update(normalized_weights)
            
            optimization_result = {
                'optimization_successful': True,
                'weight_updates': weight_changes,
                'performance_samples': len(performance_feedback),
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            logger.info("Ensemble weights optimized based on performance feedback")
            
            return optimization_result
        
        return {'error': 'Unable to calculate new weights from feedback'}


def main():
    """Main ensemble coordinator demonstration."""
    
    print("HYDATIS Model Ensemble Coordinator - Week 8")
    print("Coordinating XGBoost + Q-Learning + Isolation Forest")
    
    ensemble = HYDATISModelEnsemble()
    
    print("Ensemble Configuration:")
    print(f"  Model Weights: {ensemble.model_weights}")
    print(f"  Voting Strategy: {ensemble.ensemble_config['voting_strategy']}")
    print(f"  Consensus Requirement: {ensemble.ensemble_config['consensus_requirement']:.1%}")
    print(f"  Conflict Resolution: {ensemble.ensemble_config['conflict_resolution']}")
    print(f"  Anomaly Override: {'✅ ENABLED' if ensemble.ensemble_config['anomaly_override'] else '❌ DISABLED'}")
    
    sample_predictions = [
        ModelPrediction(
            model_type='qlearning_placement',
            prediction={'selected_node': 'worker-2', 'placement_reasoning': {'quality_score': 0.89}},
            confidence=0.85,
            latency_ms=12.3
        ),
        ModelPrediction(
            model_type='xgboost_load_predictor', 
            prediction={'scheduling_recommendations': {'preferred_nodes': ['worker-2', 'worker-1']}},
            confidence=0.78,
            latency_ms=8.7
        ),
        ModelPrediction(
            model_type='isolation_forest_anomaly',
            prediction={'cluster_health_assessment': {'overall_health': 'healthy'}},
            confidence=0.92,
            latency_ms=15.1
        )
    ]
    
    sample_pod = {
        'metadata': {'name': 'nginx-deployment', 'namespace': 'production'},
        'resources': {'cpu_request': 0.2, 'memory_request': 0.3}
    }
    
    decision = ensemble.coordinate_scheduling_decision(
        sample_predictions[1], sample_predictions[0], sample_predictions[2], sample_pod
    )
    
    print(f"Sample Decision:")
    print(f"  Action: {decision.action.value}")
    print(f"  Selected Node: {decision.selected_node}")
    print(f"  Confidence: {decision.confidence.name}")
    print(f"  Models Used: {len(decision.model_contributions)}")
    
    return ensemble


if __name__ == "__main__":
    coordinator = main()