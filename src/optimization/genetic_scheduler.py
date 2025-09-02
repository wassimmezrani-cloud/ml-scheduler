#!/usr/bin/env python3

import asyncio
import json
import logging
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import copy

import requests
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

class OptimizationObjective(Enum):
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_AVAILABILITY = "maximize_availability"

@dataclass
class NodeResource:
    name: str
    cpu_capacity: float
    memory_capacity: float
    cpu_allocated: float
    memory_allocated: float
    network_bandwidth: float
    storage_iops: float
    cost_per_hour: float
    reliability_score: float
    current_pods: int
    max_pods: int

@dataclass
class PodRequirement:
    name: str
    cpu_request: float
    memory_request: float
    cpu_limit: float
    memory_limit: float
    priority_class: int
    affinity_requirements: Dict[str, Any]
    anti_affinity_requirements: Dict[str, Any]
    network_requirements: float
    storage_requirements: float
    sla_requirements: Dict[str, float]

@dataclass
class SchedulingChromosome:
    pod_assignments: Dict[str, str]
    fitness_score: float
    objective_scores: Dict[str, float]
    constraint_violations: int
    generation: int

class GeneticSchedulerOptimizer:
    def __init__(self, prometheus_endpoint: str = "http://10.110.190.83:9090"):
        self.logger = self._setup_logging()
        self.prometheus_endpoint = prometheus_endpoint
        
        self.ga_parameters = {
            'population_size': 100,
            'generations': 50,
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'elite_percentage': 0.1,
            'tournament_size': 5,
            'convergence_threshold': 1e-6,
            'max_stagnant_generations': 10
        }
        
        self.optimization_weights = {
            OptimizationObjective.MINIMIZE_LATENCY: 0.25,
            OptimizationObjective.MAXIMIZE_THROUGHPUT: 0.20,
            OptimizationObjective.BALANCE_LOAD: 0.25,
            OptimizationObjective.MINIMIZE_COST: 0.15,
            OptimizationObjective.MAXIMIZE_AVAILABILITY: 0.15
        }
        
        self.constraint_penalties = {
            'resource_overflow': 1000.0,
            'affinity_violation': 500.0,
            'sla_violation': 750.0,
            'capacity_violation': 600.0
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for genetic scheduler optimizer."""
        logger = logging.getLogger('genetic_scheduler')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    async def optimize_cluster_scheduling(self, 
                                        pods: List[PodRequirement], 
                                        nodes: List[NodeResource],
                                        objectives: List[OptimizationObjective] = None) -> Dict[str, Any]:
        """Optimize cluster scheduling using genetic algorithm."""
        
        if objectives is None:
            objectives = list(self.optimization_weights.keys())
        
        self.logger.info(f"Starting genetic optimization for {len(pods)} pods on {len(nodes)} nodes")
        
        start_time = datetime.now()
        
        try:
            population = self._initialize_population(pods, nodes)
            
            best_chromosome = None
            stagnant_generations = 0
            generation_stats = []
            
            for generation in range(self.ga_parameters['generations']):
                generation_start = datetime.now()
                
                fitness_scores = await self._evaluate_population_fitness(
                    population, pods, nodes, objectives
                )
                
                for i, chromosome in enumerate(population):
                    chromosome.fitness_score = fitness_scores[i]
                    chromosome.generation = generation
                
                population.sort(key=lambda x: x.fitness_score, reverse=True)
                
                current_best = population[0]
                if best_chromosome is None or current_best.fitness_score > best_chromosome.fitness_score:
                    if best_chromosome and abs(current_best.fitness_score - best_chromosome.fitness_score) < self.ga_parameters['convergence_threshold']:
                        stagnant_generations += 1
                    else:
                        stagnant_generations = 0
                    
                    best_chromosome = copy.deepcopy(current_best)
                else:
                    stagnant_generations += 1
                
                generation_stats.append({
                    'generation': generation,
                    'best_fitness': current_best.fitness_score,
                    'avg_fitness': sum(c.fitness_score for c in population) / len(population),
                    'constraint_violations': current_best.constraint_violations,
                    'execution_time': (datetime.now() - generation_start).total_seconds()
                })
                
                if stagnant_generations >= self.ga_parameters['max_stagnant_generations']:
                    self.logger.info(f"Converged after {generation} generations")
                    break
                
                if generation < self.ga_parameters['generations'] - 1:
                    population = self._evolve_population(population, pods, nodes)
                
                if generation % 10 == 0:
                    self.logger.info(f"Generation {generation}: Best fitness = {current_best.fitness_score:.4f}")
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            optimization_result = {
                'status': 'success',
                'best_assignment': best_chromosome.pod_assignments,
                'fitness_score': best_chromosome.fitness_score,
                'objective_scores': best_chromosome.objective_scores,
                'constraint_violations': best_chromosome.constraint_violations,
                'generations_executed': len(generation_stats),
                'total_execution_time': total_time,
                'convergence_stats': generation_stats,
                'resource_utilization': self._calculate_resource_utilization(
                    best_chromosome.pod_assignments, pods, nodes
                )
            }
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Genetic optimization failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }

    def _initialize_population(self, pods: List[PodRequirement], nodes: List[NodeResource]) -> List[SchedulingChromosome]:
        """Initialize random population of scheduling solutions."""
        population = []
        
        for _ in range(self.ga_parameters['population_size']):
            chromosome = SchedulingChromosome(
                pod_assignments={},
                fitness_score=0.0,
                objective_scores={},
                constraint_violations=0,
                generation=0
            )
            
            for pod in pods:
                available_nodes = [n for n in nodes if self._can_schedule_pod(pod, n)]
                if available_nodes:
                    selected_node = random.choice(available_nodes)
                    chromosome.pod_assignments[pod.name] = selected_node.name
                else:
                    chromosome.pod_assignments[pod.name] = random.choice(nodes).name
            
            population.append(chromosome)
        
        return population

    def _can_schedule_pod(self, pod: PodRequirement, node: NodeResource) -> bool:
        """Check if pod can be scheduled on node considering basic constraints."""
        cpu_available = node.cpu_capacity - node.cpu_allocated
        memory_available = node.memory_capacity - node.memory_allocated
        
        return (pod.cpu_request <= cpu_available and 
                pod.memory_request <= memory_available and
                node.current_pods < node.max_pods)

    async def _evaluate_population_fitness(self, 
                                         population: List[SchedulingChromosome],
                                         pods: List[PodRequirement], 
                                         nodes: List[NodeResource],
                                         objectives: List[OptimizationObjective]) -> List[float]:
        """Evaluate fitness scores for entire population."""
        
        fitness_tasks = []
        for chromosome in population:
            task = self._evaluate_chromosome_fitness(chromosome, pods, nodes, objectives)
            fitness_tasks.append(task)
        
        fitness_scores = await asyncio.gather(*fitness_tasks)
        return fitness_scores

    async def _evaluate_chromosome_fitness(self, 
                                         chromosome: SchedulingChromosome,
                                         pods: List[PodRequirement], 
                                         nodes: List[NodeResource],
                                         objectives: List[OptimizationObjective]) -> float:
        """Evaluate fitness score for single chromosome."""
        
        objective_scores = {}
        constraint_violations = 0
        
        node_utilization = self._calculate_node_utilization(chromosome.pod_assignments, pods, nodes)
        
        for objective in objectives:
            if objective == OptimizationObjective.MINIMIZE_LATENCY:
                objective_scores[objective.value] = self._calculate_latency_score(
                    chromosome.pod_assignments, pods, nodes
                )
            elif objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
                objective_scores[objective.value] = self._calculate_throughput_score(
                    node_utilization
                )
            elif objective == OptimizationObjective.BALANCE_LOAD:
                objective_scores[objective.value] = self._calculate_load_balance_score(
                    node_utilization
                )
            elif objective == OptimizationObjective.MINIMIZE_COST:
                objective_scores[objective.value] = self._calculate_cost_score(
                    chromosome.pod_assignments, pods, nodes
                )
            elif objective == OptimizationObjective.MAXIMIZE_AVAILABILITY:
                objective_scores[objective.value] = self._calculate_availability_score(
                    chromosome.pod_assignments, pods, nodes
                )
        
        constraint_violations += self._count_constraint_violations(
            chromosome.pod_assignments, pods, nodes
        )
        
        weighted_score = sum(
            objective_scores[obj.value] * self.optimization_weights[obj]
            for obj in objectives
        )
        
        penalty = constraint_violations * sum(self.constraint_penalties.values()) / len(self.constraint_penalties)
        
        fitness_score = max(0.0, weighted_score - penalty)
        
        chromosome.objective_scores = objective_scores
        chromosome.constraint_violations = constraint_violations
        
        return fitness_score

    def _calculate_node_utilization(self, 
                                  assignments: Dict[str, str], 
                                  pods: List[PodRequirement], 
                                  nodes: List[NodeResource]) -> Dict[str, Dict[str, float]]:
        """Calculate resource utilization for each node."""
        
        node_utilization = {}
        
        for node in nodes:
            node_utilization[node.name] = {
                'cpu_used': node.cpu_allocated,
                'memory_used': node.memory_allocated,
                'cpu_capacity': node.cpu_capacity,
                'memory_capacity': node.memory_capacity,
                'pod_count': node.current_pods
            }
        
        for pod in pods:
            assigned_node = assignments.get(pod.name)
            if assigned_node and assigned_node in node_utilization:
                node_utilization[assigned_node]['cpu_used'] += pod.cpu_request
                node_utilization[assigned_node]['memory_used'] += pod.memory_request
                node_utilization[assigned_node]['pod_count'] += 1
        
        for node_name, util in node_utilization.items():
            util['cpu_utilization'] = util['cpu_used'] / util['cpu_capacity']
            util['memory_utilization'] = util['memory_used'] / util['memory_capacity']
        
        return node_utilization

    def _calculate_latency_score(self, assignments: Dict[str, str], pods: List[PodRequirement], nodes: List[NodeResource]) -> float:
        """Calculate latency optimization score (higher is better)."""
        total_latency = 0.0
        
        node_lookup = {node.name: node for node in nodes}
        
        for pod in pods:
            assigned_node_name = assignments.get(pod.name)
            if assigned_node_name and assigned_node_name in node_lookup:
                node = node_lookup[assigned_node_name]
                
                base_latency = 10.0
                utilization_penalty = (node.cpu_allocated / node.cpu_capacity) * 20
                network_latency = max(1.0, pod.network_requirements / node.network_bandwidth * 5)
                
                pod_latency = base_latency + utilization_penalty + network_latency
                total_latency += pod_latency
        
        avg_latency = total_latency / len(pods) if pods else 0
        
        return max(0.0, 1.0 - (avg_latency / 100.0))

    def _calculate_throughput_score(self, node_utilization: Dict[str, Dict[str, float]]) -> float:
        """Calculate throughput optimization score (higher is better)."""
        total_throughput = 0.0
        
        for node_name, util in node_utilization.items():
            cpu_efficiency = min(1.0, util['cpu_utilization'])
            memory_efficiency = min(1.0, util['memory_utilization'])
            
            if util['cpu_utilization'] > 0.9 or util['memory_utilization'] > 0.9:
                efficiency_penalty = 0.5
            else:
                efficiency_penalty = 0.0
            
            node_throughput = (cpu_efficiency + memory_efficiency) / 2.0 - efficiency_penalty
            total_throughput += max(0.0, node_throughput)
        
        return total_throughput / len(node_utilization) if node_utilization else 0.0

    def _calculate_load_balance_score(self, node_utilization: Dict[str, Dict[str, float]]) -> float:
        """Calculate load balance score (higher is better)."""
        if not node_utilization:
            return 0.0
        
        cpu_utilizations = [util['cpu_utilization'] for util in node_utilization.values()]
        memory_utilizations = [util['memory_utilization'] for util in node_utilization.values()]
        
        cpu_std = np.std(cpu_utilizations)
        memory_std = np.std(memory_utilizations)
        
        cpu_balance_score = max(0.0, 1.0 - cpu_std)
        memory_balance_score = max(0.0, 1.0 - memory_std)
        
        return (cpu_balance_score + memory_balance_score) / 2.0

    def _calculate_cost_score(self, assignments: Dict[str, str], pods: List[PodRequirement], nodes: List[NodeResource]) -> float:
        """Calculate cost optimization score (higher is better)."""
        total_cost = 0.0
        node_lookup = {node.name: node for node in nodes}
        
        for pod in pods:
            assigned_node_name = assignments.get(pod.name)
            if assigned_node_name and assigned_node_name in node_lookup:
                node = node_lookup[assigned_node_name]
                
                resource_fraction = (pod.cpu_request / node.cpu_capacity + 
                                   pod.memory_request / node.memory_capacity) / 2.0
                
                pod_cost = node.cost_per_hour * resource_fraction
                total_cost += pod_cost
        
        max_possible_cost = sum(node.cost_per_hour for node in nodes)
        
        return max(0.0, 1.0 - (total_cost / max_possible_cost)) if max_possible_cost > 0 else 0.0

    def _calculate_availability_score(self, assignments: Dict[str, str], pods: List[PodRequirement], nodes: List[NodeResource]) -> float:
        """Calculate availability optimization score (higher is better)."""
        total_availability = 0.0
        node_lookup = {node.name: node for node in nodes}
        
        for pod in pods:
            assigned_node_name = assignments.get(pod.name)
            if assigned_node_name and assigned_node_name in node_lookup:
                node = node_lookup[assigned_node_name]
                
                base_availability = node.reliability_score
                
                overload_penalty = 0.0
                if node.cpu_allocated / node.cpu_capacity > 0.8:
                    overload_penalty += 0.2
                if node.memory_allocated / node.memory_capacity > 0.8:
                    overload_penalty += 0.2
                
                pod_availability = max(0.0, base_availability - overload_penalty)
                total_availability += pod_availability
        
        return total_availability / len(pods) if pods else 0.0

    def _count_constraint_violations(self, assignments: Dict[str, str], pods: List[PodRequirement], nodes: List[NodeResource]) -> int:
        """Count constraint violations in assignment."""
        violations = 0
        node_lookup = {node.name: node for node in nodes}
        node_usage = {node.name: {'cpu': node.cpu_allocated, 'memory': node.memory_allocated, 'pods': node.current_pods} for node in nodes}
        
        for pod in pods:
            assigned_node_name = assignments.get(pod.name)
            if assigned_node_name and assigned_node_name in node_lookup:
                node = node_lookup[assigned_node_name]
                
                node_usage[assigned_node_name]['cpu'] += pod.cpu_request
                node_usage[assigned_node_name]['memory'] += pod.memory_request
                node_usage[assigned_node_name]['pods'] += 1
                
                if node_usage[assigned_node_name]['cpu'] > node.cpu_capacity:
                    violations += 1
                
                if node_usage[assigned_node_name]['memory'] > node.memory_capacity:
                    violations += 1
                
                if node_usage[assigned_node_name]['pods'] > node.max_pods:
                    violations += 1
                
                if pod.affinity_requirements:
                    if not self._check_affinity_constraints(pod, node, assignments, pods):
                        violations += 1
        
        return violations

    def _check_affinity_constraints(self, pod: PodRequirement, node: NodeResource, assignments: Dict[str, str], all_pods: List[PodRequirement]) -> bool:
        """Check if pod affinity constraints are satisfied."""
        
        if 'required_during_scheduling' in pod.affinity_requirements:
            required_labels = pod.affinity_requirements['required_during_scheduling']
            
            for label_key, label_value in required_labels.items():
                if not hasattr(node, label_key) or getattr(node, label_key) != label_value:
                    return False
        
        if 'preferred_during_scheduling' in pod.affinity_requirements:
            pass
        
        if pod.anti_affinity_requirements:
            anti_affinity_labels = pod.anti_affinity_requirements.get('labels', {})
            
            for other_pod in all_pods:
                if other_pod.name != pod.name and assignments.get(other_pod.name) == node.name:
                    for label_key, label_value in anti_affinity_labels.items():
                        if hasattr(other_pod, label_key) and getattr(other_pod, label_key) == label_value:
                            return False
        
        return True

    def _evolve_population(self, population: List[SchedulingChromosome], pods: List[PodRequirement], nodes: List[NodeResource]) -> List[SchedulingChromosome]:
        """Evolve population using genetic operators."""
        
        elite_count = int(len(population) * self.ga_parameters['elite_percentage'])
        elite_chromosomes = population[:elite_count]
        
        new_population = elite_chromosomes[:]
        
        while len(new_population) < self.ga_parameters['population_size']:
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            if random.random() < self.ga_parameters['crossover_rate']:
                child1, child2 = self._crossover(parent1, parent2, pods, nodes)
                
                if random.random() < self.ga_parameters['mutation_rate']:
                    child1 = self._mutate(child1, pods, nodes)
                if random.random() < self.ga_parameters['mutation_rate']:
                    child2 = self._mutate(child2, pods, nodes)
                
                new_population.extend([child1, child2])
            else:
                new_population.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])
        
        return new_population[:self.ga_parameters['population_size']]

    def _tournament_selection(self, population: List[SchedulingChromosome]) -> SchedulingChromosome:
        """Select parent using tournament selection."""
        tournament = random.sample(population, min(self.ga_parameters['tournament_size'], len(population)))
        return max(tournament, key=lambda x: x.fitness_score)

    def _crossover(self, parent1: SchedulingChromosome, parent2: SchedulingChromosome, 
                   pods: List[PodRequirement], nodes: List[NodeResource]) -> Tuple[SchedulingChromosome, SchedulingChromosome]:
        """Perform crossover between two parent chromosomes."""
        
        child1 = SchedulingChromosome(
            pod_assignments={},
            fitness_score=0.0,
            objective_scores={},
            constraint_violations=0,
            generation=0
        )
        
        child2 = SchedulingChromosome(
            pod_assignments={},
            fitness_score=0.0,
            objective_scores={},
            constraint_violations=0,
            generation=0
        )
        
        crossover_point = random.randint(1, len(pods) - 1)
        pod_names = list(parent1.pod_assignments.keys())
        
        for i, pod_name in enumerate(pod_names):
            if i < crossover_point:
                child1.pod_assignments[pod_name] = parent1.pod_assignments[pod_name]
                child2.pod_assignments[pod_name] = parent2.pod_assignments[pod_name]
            else:
                child1.pod_assignments[pod_name] = parent2.pod_assignments[pod_name]
                child2.pod_assignments[pod_name] = parent1.pod_assignments[pod_name]
        
        return child1, child2

    def _mutate(self, chromosome: SchedulingChromosome, pods: List[PodRequirement], nodes: List[NodeResource]) -> SchedulingChromosome:
        """Apply mutation to chromosome."""
        
        mutated = copy.deepcopy(chromosome)
        
        mutation_count = max(1, int(len(pods) * 0.1))
        pods_to_mutate = random.sample(list(mutated.pod_assignments.keys()), mutation_count)
        
        for pod_name in pods_to_mutate:
            available_nodes = [node.name for node in nodes]
            new_node = random.choice(available_nodes)
            mutated.pod_assignments[pod_name] = new_node
        
        return mutated

    def _calculate_resource_utilization(self, assignments: Dict[str, str], pods: List[PodRequirement], nodes: List[NodeResource]) -> Dict[str, Any]:
        """Calculate detailed resource utilization metrics."""
        
        node_utilization = self._calculate_node_utilization(assignments, pods, nodes)
        
        total_cpu_used = sum(util['cpu_used'] for util in node_utilization.values())
        total_memory_used = sum(util['memory_used'] for util in node_utilization.values())
        total_cpu_capacity = sum(util['cpu_capacity'] for util in node_utilization.values())
        total_memory_capacity = sum(util['memory_capacity'] for util in node_utilization.values())
        
        utilization_stats = {
            'cluster_cpu_utilization': total_cpu_used / total_cpu_capacity if total_cpu_capacity > 0 else 0,
            'cluster_memory_utilization': total_memory_used / total_memory_capacity if total_memory_capacity > 0 else 0,
            'node_utilizations': node_utilization,
            'load_distribution': {
                'cpu_std_dev': np.std([util['cpu_utilization'] for util in node_utilization.values()]),
                'memory_std_dev': np.std([util['memory_utilization'] for util in node_utilization.values()])
            }
        }
        
        return utilization_stats

    async def optimize_with_differential_evolution(self, 
                                                 pods: List[PodRequirement], 
                                                 nodes: List[NodeResource]) -> Dict[str, Any]:
        """Alternative optimization using differential evolution."""
        
        self.logger.info("Starting differential evolution optimization")
        
        def objective_function(assignment_vector):
            assignments = self._decode_assignment_vector(assignment_vector, pods, nodes)
            
            node_utilization = self._calculate_node_utilization(assignments, pods, nodes)
            constraint_violations = self._count_constraint_violations(assignments, pods, nodes)
            
            latency_score = self._calculate_latency_score(assignments, pods, nodes)
            balance_score = self._calculate_load_balance_score(node_utilization)
            
            fitness = 0.5 * latency_score + 0.5 * balance_score
            penalty = constraint_violations * 0.1
            
            return -(fitness - penalty)
        
        bounds = [(0, len(nodes) - 1) for _ in pods]
        
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=100,
            popsize=20,
            atol=1e-6,
            seed=42
        )
        
        optimal_assignments = self._decode_assignment_vector(result.x, pods, nodes)
        
        return {
            'status': 'success',
            'assignment': optimal_assignments,
            'optimization_score': -result.fun,
            'iterations': result.nit,
            'function_evaluations': result.nfev
        }

    def _decode_assignment_vector(self, vector: np.ndarray, pods: List[PodRequirement], nodes: List[NodeResource]) -> Dict[str, str]:
        """Decode optimization vector to pod assignments."""
        assignments = {}
        
        for i, pod in enumerate(pods):
            node_index = int(round(vector[i])) % len(nodes)
            assignments[pod.name] = nodes[node_index].name
        
        return assignments

    async def adaptive_parameter_tuning(self, 
                                      pods: List[PodRequirement], 
                                      nodes: List[NodeResource],
                                      performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adaptively tune GA parameters based on performance history."""
        
        if len(performance_history) < 5:
            return {'status': 'insufficient_data', 'current_parameters': self.ga_parameters}
        
        recent_performance = performance_history[-5:]
        
        avg_convergence_time = np.mean([p['generations_executed'] for p in recent_performance])
        avg_fitness_improvement = np.mean([
            p['convergence_stats'][-1]['best_fitness'] - p['convergence_stats'][0]['best_fitness']
            for p in recent_performance if p['convergence_stats']
        ])
        
        new_parameters = copy.deepcopy(self.ga_parameters)
        
        if avg_convergence_time > 40:
            new_parameters['population_size'] = min(150, int(self.ga_parameters['population_size'] * 1.2))
            new_parameters['mutation_rate'] = min(0.25, self.ga_parameters['mutation_rate'] * 1.1)
        
        if avg_fitness_improvement < 0.1:
            new_parameters['mutation_rate'] = min(0.3, self.ga_parameters['mutation_rate'] * 1.3)
            new_parameters['crossover_rate'] = max(0.6, self.ga_parameters['crossover_rate'] * 0.9)
        
        if avg_convergence_time < 20 and avg_fitness_improvement > 0.3:
            new_parameters['population_size'] = max(50, int(self.ga_parameters['population_size'] * 0.9))
        
        parameter_changes = {
            key: {'old': self.ga_parameters[key], 'new': new_parameters[key]}
            for key in self.ga_parameters
            if self.ga_parameters[key] != new_parameters[key]
        }
        
        self.ga_parameters.update(new_parameters)
        
        return {
            'status': 'success',
            'parameter_changes': parameter_changes,
            'tuning_rationale': {
                'avg_convergence_time': avg_convergence_time,
                'avg_fitness_improvement': avg_fitness_improvement,
                'performance_samples': len(recent_performance)
            }
        }

    async def multi_objective_pareto_optimization(self, 
                                                pods: List[PodRequirement], 
                                                nodes: List[NodeResource]) -> Dict[str, Any]:
        """Find Pareto-optimal solutions for multi-objective optimization."""
        
        self.logger.info("Starting Pareto-optimal multi-objective optimization")
        
        objectives = list(self.optimization_weights.keys())
        population = self._initialize_population(pods, nodes)
        
        pareto_solutions = []
        
        for generation in range(self.ga_parameters['generations']):
            objective_scores_matrix = []
            
            for chromosome in population:
                fitness_score = await self._evaluate_chromosome_fitness(
                    chromosome, pods, nodes, objectives
                )
                
                objective_scores = [chromosome.objective_scores[obj.value] for obj in objectives]
                objective_scores_matrix.append(objective_scores)
            
            pareto_indices = self._find_pareto_optimal_solutions(objective_scores_matrix)
            
            generation_pareto = [population[i] for i in pareto_indices]
            pareto_solutions.extend(generation_pareto)
            
            if generation < self.ga_parameters['generations'] - 1:
                population = self._evolve_population(population, pods, nodes)
        
        final_pareto = self._filter_dominated_solutions(pareto_solutions, objectives)
        
        return {
            'status': 'success',
            'pareto_solutions': [
                {
                    'assignment': sol.pod_assignments,
                    'objective_scores': sol.objective_scores,
                    'fitness_score': sol.fitness_score
                } for sol in final_pareto
            ],
            'solution_count': len(final_pareto),
            'objectives_optimized': [obj.value for obj in objectives]
        }

    def _find_pareto_optimal_solutions(self, objective_scores: List[List[float]]) -> List[int]:
        """Find Pareto-optimal solutions from objective scores."""
        pareto_indices = []
        
        for i, scores_i in enumerate(objective_scores):
            is_dominated = False
            
            for j, scores_j in enumerate(objective_scores):
                if i != j and self._dominates(scores_j, scores_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return pareto_indices

    def _dominates(self, scores_a: List[float], scores_b: List[float]) -> bool:
        """Check if solution A dominates solution B."""
        return (all(a >= b for a, b in zip(scores_a, scores_b)) and 
                any(a > b for a, b in zip(scores_a, scores_b)))

    def _filter_dominated_solutions(self, solutions: List[SchedulingChromosome], objectives: List[OptimizationObjective]) -> List[SchedulingChromosome]:
        """Filter out dominated solutions from Pareto set."""
        if not solutions:
            return []
        
        objective_scores = [
            [sol.objective_scores[obj.value] for obj in objectives]
            for sol in solutions
        ]
        
        pareto_indices = self._find_pareto_optimal_solutions(objective_scores)
        return [solutions[i] for i in pareto_indices]

async def main():
    """Main genetic scheduler optimization entry point."""
    optimizer = GeneticSchedulerOptimizer()
    
    sample_pods = [
        PodRequirement(
            name="web-frontend-1",
            cpu_request=500,
            memory_request=1024,
            cpu_limit=1000,
            memory_limit=2048,
            priority_class=1,
            affinity_requirements={},
            anti_affinity_requirements={},
            network_requirements=100,
            storage_requirements=0,
            sla_requirements={'availability': 0.99}
        ),
        PodRequirement(
            name="ml-training-job",
            cpu_request=2000,
            memory_request=4096,
            cpu_limit=4000,
            memory_limit=8192,
            priority_class=2,
            affinity_requirements={},
            anti_affinity_requirements={},
            network_requirements=1000,
            storage_requirements=10000,
            sla_requirements={'completion_time': 3600}
        )
    ]
    
    sample_nodes = [
        NodeResource(
            name="hydatis-worker-1",
            cpu_capacity=4000,
            memory_capacity=8192,
            cpu_allocated=1000,
            memory_allocated=2048,
            network_bandwidth=10000,
            storage_iops=5000,
            cost_per_hour=0.15,
            reliability_score=0.95,
            current_pods=5,
            max_pods=30
        ),
        NodeResource(
            name="hydatis-worker-2",
            cpu_capacity=4000,
            memory_capacity=8192,
            cpu_allocated=800,
            memory_allocated=1536,
            network_bandwidth=10000,
            storage_iops=5000,
            cost_per_hour=0.15,
            reliability_score=0.97,
            current_pods=3,
            max_pods=30
        )
    ]
    
    try:
        result = await optimizer.optimize_cluster_scheduling(sample_pods, sample_nodes)
        
        with open('/tmp/genetic_optimization_result.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Genetic optimization completed: {result['status']}")
        print(f"Best fitness score: {result.get('fitness_score', 'N/A')}")
        print("Results saved to /tmp/genetic_optimization_result.json")
        
    except Exception as e:
        optimizer.logger.error(f"Genetic optimization failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())