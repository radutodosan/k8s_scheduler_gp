"""Fitness evaluator — bridges the GP engine with the simulator.

Runs one or more simulation instances with a candidate GP individual
and returns a scalar quality score in [0, 1] (higher is better,
1.0 = perfect scheduling: zero wait, full utilisation, no rejections).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Optional, Tuple

import numpy as np

from config.schema import ClusterConfig, DynamicsConfig, FitnessWeights, WorkloadConfig
from gp.interface import IGeneticEngine
from metrics.collector import MetricsCollector, SchedulingMetrics
from models.pod import Pod
from scheduling.gp_strategy import GPSchedulingStrategy
from simulator.engine import SimulationEngine
from workload.generator import IWorkloadGenerator

logger = logging.getLogger(__name__)


# ── Shared quality helper ──────────────────────────────────────────────

def compute_quality_score(metrics: SchedulingMetrics, weights: FitnessWeights) -> float:
    """Compute normalized scheduling quality in [0, 1] (higher is better).

    quality = 1 - (alpha * W + beta * R + gamma * F + delta * E + epsilon * P + eta * C + zeta * A)
        W      = wait_time / (wait_time + 1)
        R      = 1 - mean(avg_cpu_utilization, avg_mem_utilization)
        F      = rejected_pods / total_pods
        E      = (evicted_pods / total_pods) normalized as x/(x+1)
        P      = (preemption_count / total_pods) normalized as x/(x+1)
        C      = churn_rate normalized as x/(x+1)
        A      = avg_scheduling_attempts / (avg_scheduling_attempts + 1)
    """
    w = max(0.0, metrics.avg_wait_time)
    w_norm = w / (w + 1.0)

    util_cpu = max(0.0, min(1.0, metrics.avg_cpu_utilization))
    util_mem = max(0.0, min(1.0, metrics.avg_mem_utilization))
    util_mean = (util_cpu + util_mem) / 2.0
    r = 1.0 - util_mean
    r = max(0.0, min(1.0, r))

    f = (metrics.rejected_pods / metrics.total_pods) if metrics.total_pods > 0 else 1.0
    f = max(0.0, min(1.0, f))

    # Eviction/preemption/churn can naturally exceed 1.0 in dynamic scenarios
    # (same pod may be churned multiple times). Use x/(x+1) instead of hard
    # clipping so the penalty remains bounded while preserving ordering.
    # Use a slightly steeper saturating curve for instability-related metrics
    # so high churn/preemption policies are penalized earlier.
    e_raw = (metrics.evicted_pods / metrics.total_pods) if metrics.total_pods > 0 else 0.0
    e_nonneg = max(0.0, e_raw)
    e = e_nonneg / (e_nonneg + 0.75)

    p_raw = (metrics.preemption_count / metrics.total_pods) if metrics.total_pods > 0 else 0.0
    p_nonneg = max(0.0, p_raw)
    p = p_nonneg / (p_nonneg + 0.5)

    # Churn rate: instability metric (evicted + preempted pods / scheduled pods)
    c_raw = max(0.0, metrics.churn_rate)  # already computed in SchedulingMetrics
    c = c_raw / (c_raw + 0.5)

    a = max(0.0, metrics.avg_scheduling_attempts)
    a_norm = a / (a + 1.0)

    # GPU waste: fraction of GPU capacity left idle. Only meaningful when the
    # workload actually requests GPUs; returns 0 otherwise to avoid penalising
    # non-GPU experiments for not using hardware they don't have.
    gpu_samples = getattr(metrics, "gpu_util_samples", [])
    if gpu_samples:
        gpu_util = sum(gpu_samples) / len(gpu_samples)
        g = max(0.0, min(1.0, 1.0 - gpu_util))
    else:
        g = 0.0

    # Cost waste: placement-based cost per completed pod (run_time × node.cost_per_hour).
    # Reference 0.005 €/pod gives k ∈ [0.59, 0.92] for the g_batchp cost-extreme cluster
    # (cheap 0.5 €/h → 0.0072 €/pod, expensive 4.0 €/h → 0.058 €/pod, avg pod duration 52 s).
    # LA spreads proportionally → k ≈ 0.80; GP preferring cheap nodes → k ≈ 0.59.
    cost_per_pod = getattr(metrics, "cost_per_pod", 0.0)
    k = cost_per_pod / (cost_per_pod + 0.005) if cost_per_pod > 0 else 0.0

    raw_cost = (
        weights.alpha_wait_time * w_norm
        + weights.beta_resource_waste * r
        + weights.gamma_failed_pods * f
        + getattr(weights, "delta_evicted_pods", 0.0) * e
        + getattr(weights, "epsilon_preemptions", 0.0) * p
        + getattr(weights, "eta_churn", 0.0) * c
        + getattr(weights, "zeta_scheduling_attempts", 0.0) * a_norm
        + getattr(weights, "theta_gpu_waste", 0.0) * g
        + getattr(weights, "iota_cost", 0.0) * k
    )
    return max(0.0, min(1.0, 1.0 - raw_cost))


# ── Top-level worker for multiprocessing (must be picklable) ─────────

def _evaluate_worker(args: tuple) -> float:
    """Evaluate a single training instance in a worker process.

    Reconstructs a minimal DEAP GP individual from its expression string,
    runs a simulation, and returns the scalar fitness score.
    """
    (
        expr_str, pods, cluster_config, weights, schedule_interval,
        max_pending_retries, dynamics_config, base_seed, instance_index,
        engine_name, terminal_names,
    ) = args

    from gp.deap_engine import DeapGeneticEngine
    from gp.primitives import TERMINAL_NAMES

    # Rebuild a lightweight GP engine in this worker
    engine = DeapGeneticEngine()
    t_names = terminal_names or list(TERMINAL_NAMES)
    engine.setup(terminal_names=t_names)

    # Parse the expression string back into a DEAP individual
    from deap import gp as deap_gp, creator
    individual = deap_gp.PrimitiveTree.from_string(expr_str, engine._pset)

    strategy = GPSchedulingStrategy(engine, individual)
    failure_seed = base_seed + instance_index * 7919
    sim = SimulationEngine(
        strategy=strategy,
        cluster_config=cluster_config,
        schedule_interval=schedule_interval,
        max_pending_retries=max_pending_retries,
        dynamics_config=dynamics_config,
        failure_seed=failure_seed,
    )
    sim.build_cluster()
    sim.load_workload(pods)
    sim.run()

    m = sim.collector.get_metrics()

    return compute_quality_score(m, weights)


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
        n_workers: int = 1,
        aggregation_mode: str = "mean",
        std_penalty: float = 0.0,
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

        # Parallel evaluation — pool is created lazily and reused across all calls
        self._n_workers = min(n_workers, os.cpu_count() or 1)
        self._pool: Optional[ProcessPoolExecutor] = None
        self._aggregation_mode = aggregation_mode
        self._std_penalty = std_penalty

        if self._aggregation_mode not in ("mean", "mean_minus_std"):
            raise ValueError(
                "aggregation_mode must be 'mean' or 'mean_minus_std'"
            )
        if self._std_penalty < 0.0:
            raise ValueError("std_penalty must be >= 0")

        if self._dynamic and (self._generator is None or self._workload_config is None):
            raise ValueError(
                "dynamic_instances=True requires workload_generator and workload_config"
            )

    def shutdown(self) -> None:
        """Shut down the worker pool (call after training completes)."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def __del__(self) -> None:
        self.shutdown()

    def _get_pool(self) -> ProcessPoolExecutor:
        """Return the persistent worker pool, creating it on first use."""
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self._n_workers)
        return self._pool

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

        Returns a scalar quality score in [0, 1] (higher is better).
        When n_workers > 1, training instances are evaluated in parallel
        using a process pool.
        """
        if self._n_workers > 1 and len(self._training_instances) > 1:
            return self._evaluate_parallel(individual)

        scores: List[float] = []
        for idx, instance_pods in enumerate(self._training_instances):
            metrics = self._evaluate_single(individual, instance_pods, instance_index=idx)
            score = self._compute_score(metrics)
            scores.append(score)

        return self._aggregate_scores(scores)

    def _evaluate_parallel(self, individual: Any) -> float:
        """Evaluate training instances across multiple processes."""
        expr_str = str(individual)
        args = [
            (
                expr_str,
                [self._copy_pod(p) for p in instance_pods],
                self._cluster_config,
                self._weights,
                self._schedule_interval,
                self._max_pending_retries,
                self._dynamics_config,
                self._base_seed,
                idx,
                self._gp_engine.name,
                list(self._gp_engine._terminal_names)
                if hasattr(self._gp_engine, "_terminal_names")
                else None,
            )
            for idx, instance_pods in enumerate(self._training_instances)
        ]

        scores = list(self._get_pool().map(_evaluate_worker, args))

        return self._aggregate_scores(scores)

    def evaluate_objectives(self, individual: Any) -> Tuple[float, float, float]:
        """Multi-objective quality: return (scheduling_quality, utilization, acceptance_rate).

        Each objective is averaged across all training instances.
        Higher is better for all three.
        """
        all_w: List[float] = []
        all_r: List[float] = []
        all_f: List[float] = []

        for idx, instance_pods in enumerate(self._training_instances):
            m = self._evaluate_single(individual, instance_pods, instance_index=idx)
            # Flip each component so higher = better:
            #   scheduling_quality = 1 - W_norm
            #   utilization        = avg_cpu_utilization
            #   acceptance_rate    = 1 - rejection_rate
            w = max(0.0, m.avg_wait_time)
            w_norm = w / (w + 1.0)
            all_w.append(1.0 - w_norm)
            util = max(0.0, min(1.0, m.avg_cpu_utilization))
            all_r.append(util)
            f = (m.rejected_pods / m.total_pods) if m.total_pods > 0 else 1.0
            all_f.append(1.0 - max(0.0, min(1.0, f)))

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
        """Combined quality score:  1 - (α·W_norm + β·R + γ·F)

        W_norm = avg_wait_time / (avg_wait_time + 1)  — normalised wait, in [0,1)
        R      = 1 - avg_cpu_utilization              — resource waste, in [0,1]
        F      = rejected_pods / total_pods           — rejection rate, in [0,1]

        Each component is clamped to [0, 1]; since α+β+γ≈1 the weighted sum
        (the "cost") is in [0, 1].  Subtracting from 1 gives a *quality* score
        in [0, 1] where 1.0 = perfect (no wait, full utilisation, no rejections)
        and 0.0 = worst.  Higher is better.
        """
        return compute_quality_score(m, self._weights)

    def _aggregate_scores(self, scores: List[float]) -> float:
        """Aggregate per-instance scores into a single scalar fitness."""
        if not scores:
            return 0.0
        mean = float(sum(scores) / len(scores))
        if self._aggregation_mode == "mean":
            return mean
        std = float(np.std(scores))
        return mean - (self._std_penalty * std)

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
