"""Fitness evaluator — bridges the GP engine with the simulator.

Runs one or more simulation instances with a candidate GP individual
and returns a scalar fitness value (lower is better).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from config.schema import ClusterConfig, DynamicsConfig, FitnessWeights, WorkloadConfig
from gp.interface import IGeneticEngine
from metrics.collector import MetricsCollector, SchedulingMetrics
from models.pod import Pod
from scheduling.gp_strategy import GPSchedulingStrategy
from simulator.engine import SimulationEngine
from workload.generator import IWorkloadGenerator

logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """Evaluates a GP individual by simulating it on training instances.

    Supports two modes controlled by ``dynamic_instances``:

    **Static** (default): A fixed list of training instances is used
    throughout the entire evolutionary run.

    **Dynamic**: Training instances are regenerated at the start of
    every generation via ``rotate_instances(generation)``.  All
    individuals within the same generation still share the same
    instances (fair comparison), but different generations see
    different workloads, which prevents overfitting.
    """

    def __init__(
        self,
        gp_engine: IGeneticEngine,
        training_instances: List[List[Pod]],
        cluster_config: ClusterConfig,
        fitness_weights: FitnessWeights,
        schedule_interval: float = 1.0,
        max_pending_retries: int = 3,
        *,
        dynamic_instances: bool = False,
        workload_generator: Optional[IWorkloadGenerator] = None,
        workload_config: Optional[WorkloadConfig] = None,
        base_seed: int = 42,
        num_instances: Optional[int] = None,
        dynamics_config: Optional[DynamicsConfig] = None,
    ) -> None:
        self._gp_engine = gp_engine
        self._training_instances = training_instances
        self._cluster_config = cluster_config
        self._weights = fitness_weights
        self._schedule_interval = schedule_interval
        self._max_pending_retries = max_pending_retries

        # Dynamic instance support
        self._dynamic = dynamic_instances
        self._generator = workload_generator
        self._workload_config = workload_config
        self._base_seed = base_seed
        self._num_instances = num_instances or len(training_instances)

        # Node failure dynamics
        self._dynamics_config = dynamics_config

        if self._dynamic and (self._generator is None or self._workload_config is None):
            raise ValueError(
                "dynamic_instances=True requires workload_generator and workload_config"
            )

    def rotate_instances(self, generation: int) -> None:
        """Regenerate training instances for the given generation.

        Each instance uses seed = base_seed + generation * num_instances + i,
        ensuring deterministic but unique workloads per generation.
        """
        if not self._dynamic:
            return

        new_instances: List[List[Pod]] = []
        for i in range(self._num_instances):
            seed = self._base_seed + generation * self._num_instances + i
            pods = self._generator.generate(self._workload_config, seed=seed)
            new_instances.append(pods)
        self._training_instances = new_instances
        logger.debug(
            "Rotated training instances for generation %d (%d instances)",
            generation, self._num_instances,
        )

    def __call__(self, individual: Any) -> float:
        """Fitness function signature expected by IGeneticEngine.train().

        Returns a scalar fitness (lower is better).
        """
        scores: List[float] = []

        for idx, instance_pods in enumerate(self._training_instances):
            metrics = self._evaluate_single(individual, instance_pods, instance_index=idx)
            score = self._compute_score(metrics)
            scores.append(score)

        return sum(scores) / len(scores) if scores else float("inf")

    def evaluate_objectives(self, individual: Any) -> Tuple[float, float, float]:
        """Multi-objective fitness: return (wait_time, resource_waste, rejection_rate).

        Each objective is averaged across all training instances.
        Lower is better for all three.
        """
        all_w: List[float] = []
        all_r: List[float] = []
        all_f: List[float] = []

        for idx, instance_pods in enumerate(self._training_instances):
            m = self._evaluate_single(individual, instance_pods, instance_index=idx)
            all_w.append(m.avg_wait_time)
            all_r.append(1.0 - m.avg_cpu_utilization if m.avg_cpu_utilization > 0 else 1.0)
            all_f.append((m.rejected_pods / m.total_pods) if m.total_pods > 0 else 1.0)

        n = len(all_w) or 1
        return (sum(all_w) / n, sum(all_r) / n, sum(all_f) / n)

    def evaluate_on_instances(
        self,
        individual: Any,
        instances: List[List[Pod]],
    ) -> List[SchedulingMetrics]:
        """Run evaluation on arbitrary instances (e.g. test set).

        Returns per-instance SchedulingMetrics.
        """
        results = []
        for idx, instance_pods in enumerate(instances):
            metrics = self._evaluate_single(individual, instance_pods, instance_index=idx)
            results.append(metrics)
        return results

    # ── Private ──────────────────────────────────────────────────────

    def _evaluate_single(
        self,
        individual: Any,
        pods: List[Pod],
        instance_index: int = 0,
    ) -> SchedulingMetrics:
        """Run one simulation instance and return its metrics."""
        # Deep-copy pods so each simulation starts fresh
        fresh_pods = [self._copy_pod(p) for p in pods]

        strategy = GPSchedulingStrategy(self._gp_engine, individual)
        failure_seed = self._base_seed + instance_index * 7919
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=self._cluster_config,
            schedule_interval=self._schedule_interval,
            max_pending_retries=self._max_pending_retries,
            dynamics_config=self._dynamics_config,
            failure_seed=failure_seed,
        )
        engine.build_cluster()
        engine.load_workload(fresh_pods)
        engine.run()

        return engine.collector.get_metrics()

    def _compute_score(self, m: SchedulingMetrics) -> float:
        """Combined fitness:  α·W + β·R + γ·F

        W = avg wait time (normalised by total pods)
        R = 1 - avg_cpu_utilization  (waste = unused capacity)
        F = rejection rate
        """
        w = m.avg_wait_time
        r = 1.0 - m.avg_cpu_utilization if m.avg_cpu_utilization > 0 else 1.0
        f = (m.rejected_pods / m.total_pods) if m.total_pods > 0 else 1.0

        return (
            self._weights.alpha_wait_time * w
            + self._weights.beta_resource_waste * r
            + self._weights.gamma_failed_pods * f
        )

    @staticmethod
    def _copy_pod(pod: Pod) -> Pod:
        """Create a fresh copy of a pod (reset mutable state)."""
        return Pod(
            pod_id=pod.pod_id,
            cpu_request=pod.cpu_request,
            mem_request=pod.mem_request,
            gpu_request=pod.gpu_request,
            priority=pod.priority,
            qos_class=pod.qos_class,
            arrival_time=pod.arrival_time,
            duration=pod.duration,
            namespace=pod.namespace,
            tolerations=pod.tolerations,
            node_selector=pod.node_selector,
            cpu_limit=pod.cpu_limit,
            mem_limit=pod.mem_limit,
            anti_affinity_key=pod.anti_affinity_key,
            workload_type=pod.workload_type,
            replica_group=pod.replica_group,
        )
