"""Experiment sweep runner for dissertation Chapter 5.

Defines and executes systematic experiments comparing:
  - DEAP GP vs 7 baseline scheduling strategies
  - Problem scale (small / medium / large)
  - Fitness weight sensitivity (α / β / γ)
  - GP parameters (population, generations)
  - Dynamics (with / without node failures)

Each experiment runs the full pipeline: generate → train → evaluate
on test instances, including all baseline strategies.

Usage:
    py run_experiments.py                    # Run all experiments
    py run_experiments.py --quick            # Quick mode (small params)
    py run_experiments.py --group engine     # Run only one group
    py run_experiments.py --list             # List all experiments
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.schema import (
    ClusterConfig,
    DynamicsConfig,
    ExperimentConfig,
    FitnessWeights,
    GPConfig,
    NodeConfig,
    NodeHeterogeneityConfig,
    WorkloadConfig,
)
from gp.deap_engine import DeapGeneticEngine
from gp.fitness import FitnessEvaluator, compute_quality_score
from metrics.reporter import MetricsReporter
from metrics.resource_monitor import ResourceMonitor
from models.pod import Pod
from scheduling.balanced_allocation import BalancedAllocationStrategy
from scheduling.bin_packing import BinPackingStrategy
from scheduling.first_fit import FirstFitStrategy
from scheduling.gp_strategy import GPSchedulingStrategy
from scheduling.least_allocated import LeastAllocatedStrategy
from scheduling.most_allocated import MostAllocatedStrategy
from scheduling.random_strategy import RandomStrategy
from scheduling.round_robin import RoundRobinStrategy
from scheduling.strategy import ISchedulingStrategy
from simulator.engine import SimulationEngine
from workload.poisson_generator import PoissonWorkloadGenerator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Experiment definition
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentDef:
    """A single experiment to run."""

    name: str
    group: str
    config: ExperimentConfig
    description: str = ""


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    name: str
    group: str
    training_time: float
    best_fitness: float
    best_expression: str
    convergence_log: List[Dict[str, Any]]
    reporter: MetricsReporter
    gp_resource_monitor: Optional[ResourceMonitor] = None


# ═══════════════════════════════════════════════════════════════════════
# Experiment catalogue
# ═══════════════════════════════════════════════════════════════════════


# Baseline-derived seed expressions for GP warm-start.
# These encode known-good heuristics so GP starts at baseline level and evolves upward.
# All terminals must be present in CORE_TERMINALS_14 to work with the default terminal set.
SEED_EXPRESSIONS_BASELINE: List[str] = [
    # Best static baselines — GP starts here and can only improve
    "neg(add(NODE_CPU_UTIL, NODE_MEM_UTIL))",            # ~LeastAllocated (strongest baseline)
    "add(NODE_CPU_FREE_AFTER, NODE_MEM_FREE_AFTER)",      # look-ahead headroom (better than LA)
    "neg(NODE_OVERCOMMIT_RATIO)",                         # stability: avoid OOM risk
    "RESOURCE_FIT",                                       # compound CPU+MEM fit score
    # Best evolved rules from overnight runs
    "if_positive(NODE_CPU_UTIL, min(NODE_MEM_FREE_AFTER, NODE_CPU_FREE_AFTER), NODE_MEM_FREE_AFTER)",
    "sub(NODE_MEM_AVAIL, NODE_POD_COUNT)",                # best in b3_large
    # CORE_20 signals
    "protected_div(RESOURCE_FIT, NODE_COST)",             # cost-aware: good fit on cheap nodes
    "neg(add(add(NODE_CPU_UTIL, NODE_MEM_UTIL), NODE_OVERCOMMIT_RATIO))",  # combined risk
    "neg(NODE_COST)",                                     # cost-minimizing: prefer cheapest node
]
# Removed from seeds: ~BinPacking (consistently bottom-3 baseline),
# ~RoundRobin (2nd weakest), duplicate neg(NODE_OVERCOMMIT_RATIO),
# simple single-terminal NODE_MEM_FREE_AFTER / NODE_CPU_FREE_AFTER
# (subsumed by the combined headroom seed above).

# CORE_TERMINALS_20 — active terminal set after iterative refinement.
# Evolution: CORE_14 (original) → CORE_17 (+wait/taint/pressure) →
#            CORE_21 (+mem_req/mem_util/affinity_conflict/cost) →
#            CORE_20 (removed NODE_AFFINITY_CONFLICT: confirmed dead terminal —
#                     feasible_nodes() hard-filters anti-affinity before GP scoring,
#                     so the terminal is always 0.0 for every node GP evaluates).
CORE_TERMINALS_20: List[str] = [
    # ── Node resource signals (kube-scheduler: LeastAllocated / MostAllocated) ──
    "NODE_OVERCOMMIT_RATIO",    # #1  — OOM risk (limits vs capacity)
    "NODE_MEM_FREE_AFTER",      # #2  — look-ahead MEM headroom after pod placement
    "NODE_CPU_FREE_AFTER",      # #3  — look-ahead CPU headroom after pod placement
    "NODE_MEM_AVAIL",           # #4  — absolute MEM available
    "NODE_CPU_AVAIL",           # #5  — absolute CPU available
    "NODE_CPU_UTIL",            # #6  — current CPU utilization ratio
    "NODE_MEM_UTIL",            # #7  — current MEM utilization ratio
    "NODE_POD_COUNT",           # #8  — pod density on node
    "NODE_TAINT_COUNT",         # #9  — taints (problematic/repairing nodes)
    "NODE_COST",                # #10 — cost/hour (cost-aware scheduling on het. cluster)
    # ── Pod demand signals (kube-scheduler: resource requests) ───────────────
    "POD_CPU_REQ",              # #11 — CPU demand
    "POD_MEM_REQ",              # #12 — MEM demand
    "POD_QOS",                  # #13 — QoS class (BestEffort/Burstable/Guaranteed)
    "POD_WAIT_TIME",            # #14 — time in queue (responsiveness / SLA signal)
    # ── Cluster-level signals (kube-scheduler: cluster utilization metrics) ──
    "CLUSTER_CPU_UTIL",         # #15 — cluster CPU pressure
    "CLUSTER_MEM_UTIL",         # #16 — cluster MEM pressure
    "CLUSTER_HEALTHY_RATIO",    # #17 — fraction of healthy nodes
    "PENDING_PRESSURE",         # #18 — queue saturation (pending / capacity ratio)
    # ── Compound / K8s-specific ──────────────────────────────────────────────
    "RESOURCE_FIT",             # #19 — geometric mean of CPU+MEM headroom after placement
    "REPLICA_GROUP_COLOCATED",  # #20 — replicas from same group already on this node
]

# Backward-compatible aliases.
CORE_TERMINALS_21 = CORE_TERMINALS_20
CORE_TERMINALS_14 = CORE_TERMINALS_20
CORE_TERMINALS_17 = CORE_TERMINALS_20

# Extended set for AI/GPU workloads — adds GPU signals on top of CORE_20.
CORE_TERMINALS_GPU: List[str] = CORE_TERMINALS_20 + [
    "NODE_GPU_AVAIL",           # GPU headroom on node
    "CLUSTER_GPU_UTIL",         # cluster-wide GPU pressure
    "POD_GPU_REQ",              # pod GPU demand
]


def _base_config(
    *,
    name: str = "experiment",
    seed: int = 42,
    n_train: int = 5,
    n_val: int = 0,
    n_test: int = 5,
    nodes: int = 5,
    cpu: float = 8.0,
    mem: float = 16384.0,
    pods: int = 100,
    engine: str = "deap",
    pop: int = 100,
    gen: int = 30,
    depth: int = 8,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    failure_mode: str = "off",
    failure_rate: int = 1,
    multi_objective: bool = False,
    profile: str = "",
    dynamic_instances: bool = True,
    gpu: float = 0.0,
    parsimony: float = 0.0002,
    tournament: int = 3,
    validation_hof: int = 0,
    terminal_mandatory: Optional[List[str]] = None,
    all_terminals: bool = False,
    fitness_aggregation: str = "mean",
    std_penalty: float = 0.0,
    n_restarts: int = 1,
    n_workers: int = 1,
    heterogeneous: bool = True,
    het_tiers: int = 3,
    het_cpu_range: Optional[List[float]] = None,
    het_mem_range: Optional[List[float]] = None,
    het_cost_range: Optional[List[float]] = None,
    cluster_override: Optional[Any] = None,
    theta: float = 0.0,
    iota: float = 0.0,
) -> ExperimentConfig:
    """Build an ExperimentConfig from simplified parameters.

    By default uses CORE_TERMINALS_14 to keep the GP search space focused.
    Pass all_terminals=True to use the full 31-terminal set.
    """
    effective_terminals = [] if all_terminals else (terminal_mandatory or CORE_TERMINALS_14)
    het_config = NodeHeterogeneityConfig(
        enabled=heterogeneous,
        tiers=het_tiers,
        cpu_range=het_cpu_range or [cpu * 0.5, cpu * 2.0],
        mem_range=het_mem_range or [mem * 0.5, mem * 2.0],
        cost_range=het_cost_range or [0.5, 2.5],
    )
    return ExperimentConfig(
        name=name,
        seed=seed,
        num_training_instances=n_train,
        num_validation_instances=n_val,
        num_test_instances=n_test,
        dynamic_instances=dynamic_instances,
        output_dir=f"tmp/results/experiments/{name}",
        output_format="csv",
        cluster=cluster_override if cluster_override is not None else ClusterConfig(
            node_templates=[NodeConfig(count=nodes, cpu_capacity=cpu, mem_capacity=mem, gpu_capacity=gpu)],
            heterogeneity=het_config,
        ),
        workload=WorkloadConfig(total_pods=pods, profile=profile),
        gp=GPConfig(
            engine=engine,
            population_size=pop,
            n_generations=gen,
            max_tree_depth=depth,
            multi_objective=multi_objective,
            parsimony_coefficient=parsimony,
            tournament_size=tournament,
            validation_hof_size=validation_hof,
            terminal_mandatory=effective_terminals,
            fitness_aggregation=fitness_aggregation,
            fitness_std_penalty=std_penalty,
            n_restarts=n_restarts,
            n_workers=n_workers,
        ),
        fitness=FitnessWeights(
            alpha_wait_time=alpha,
            beta_resource_waste=beta,
            gamma_failed_pods=gamma,
            delta_evicted_pods=0.0,
            epsilon_preemptions=0.0,
            eta_churn=0.0,
            zeta_scheduling_attempts=0.0,
            theta_gpu_waste=theta,
            iota_cost=iota,
        ),
        dynamics=DynamicsConfig(
            failure_mode=failure_mode,
            failure_rate=failure_rate,
        ),
    )


def define_experiments(
    quick: bool = False,
    medium: bool = False,
    overnight: bool = False,
    day: bool = False,
) -> List[ExperimentDef]:
    """Define all dissertation experiments.

    Args:
        quick:     Very small parameters for syntax/smoke validation (~0.3s/exp).
        medium:    Balanced parameters (~45s/exp, ~1.3h for 5 seeds). Rapid iteration.
        day:       Medium scale + overnight quality extras (~100s/exp, ~3h for 5 seeds).
                   All 21 experiments, 5 seeds, recommended for quick validation.
        overnight: Deep exploration (~312s/exp, ~10.7h for 5 seeds). Run before sleep.
                   pop=120, gen=50, n_train=5, large cluster (pods ≤ 250, nodes ≤ 12).
        (default): Standard full preset (~96s/exp, ~2.8h for 5 seeds).
    """
    if quick:
        pop, gen, pods_s, pods_m, pods_l = 20, 5, 15, 30, 60
        n_train, n_test = 2, 2
        nodes_s, nodes_m, nodes_l = 2, 3, 5
    elif medium:
        pop, gen, pods_s, pods_m, pods_l = 80, 30, 50, 100, 150
        n_train, n_test = 3, 5
        nodes_s, nodes_m, nodes_l = 3, 5, 8
    elif day:
        pop, gen, pods_s, pods_m, pods_l = 80, 30, 50, 100, 150
        n_train, n_test = 3, 5
        nodes_s, nodes_m, nodes_l = 3, 5, 8
    elif overnight:
        pop, gen, pods_s, pods_m, pods_l = 120, 50, 60, 120, 250
        n_train, n_test = 5, 5
        nodes_s, nodes_m, nodes_l = 4, 6, 12
    else:
        pop, gen, pods_s, pods_m, pods_l = 100, 30, 50, 100, 200
        n_train, n_test = 5, 5
        nodes_s, nodes_m, nodes_l = 3, 5, 10

    # Overnight/day extras: robust fitness, validation champion selection, stable tournament
    _fit_agg    = "mean_minus_std" if (overnight or day) else "mean"
    _std_pen    = 0.15 if overnight else (0.08 if day else 0.0)
    _n_val      = 3 if overnight else (2 if day else 0)
    _vhof       = 5 if overnight else (3 if day else 0)
    _tourn      = 5 if (overnight or day) else 3
    _n_restarts = 1   # diversity reinsertion makes single runs robust; n_restarts=2 was too slow
    _n_workers  = 3 if overnight else (2 if day else 1)  # parallel training-instance eval

    def _cfg(**kw):
        """Shorthand: merges overnight extras unless overridden by caller."""
        kw.setdefault("fitness_aggregation", _fit_agg)
        kw.setdefault("std_penalty", _std_pen)
        kw.setdefault("n_val", _n_val)
        kw.setdefault("validation_hof", _vhof)
        kw.setdefault("tournament", _tourn)
        kw.setdefault("n_restarts", _n_restarts)
        kw.setdefault("n_workers", _n_workers)
        return _base_config(**kw)

    experiments: List[ExperimentDef] = []

    # ── Group A: Primary DEAP experiment ─────────────────────────
    experiments.append(ExperimentDef(
        name="a1_deap_medium",
        group="engine",
        description="DEAP engine — medium scenario (primary reference experiment)",
        config=_cfg(
            name="a1_deap_medium", engine="deap",
            pods=pods_m, nodes=nodes_m, pop=pop, gen=gen,
            n_train=n_train, n_val=3, n_test=n_test,
            validation_hof=5, tournament=5,
        ),
    ))

    # ── Group B: Scale Sensitivity ───────────────────────────────
    experiments.append(ExperimentDef(
        name="b1_small",
        group="scale",
        description=f"Small scale: {pods_s} pods, {nodes_s} nodes",
        config=_cfg(
            name="b1_small", pods=pods_s, nodes=nodes_s,
            pop=pop, gen=gen, n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="b2_medium",
        group="scale",
        description=f"Medium scale: {pods_m} pods, {nodes_m} nodes",
        config=_cfg(
            name="b2_medium", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, n_train=n_train, n_val=3, n_test=n_test,
            validation_hof=5, tournament=5,
        ),
    ))
    experiments.append(ExperimentDef(
        name="b3_large",
        group="scale",
        description=f"Large scale: {pods_l} pods, {nodes_l} nodes (dynamic instances for GP generalization)",
        config=_cfg(
            name="b3_large", pods=pods_l, nodes=nodes_l,
            pop=pop, gen=gen, n_train=n_train, n_test=n_test,
            dynamic_instances=True,
        ),
    ))
    # ── Group C: Fitness Weight Sensitivity ──────────────────────
    experiments.append(ExperimentDef(
        name="c1_balanced",
        group="fitness_weights",
        description="Balanced weights: alpha=0.33, beta=0.33, gamma=0.34",
        config=_cfg(
            name="c1_balanced", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, alpha=0.33, beta=0.33, gamma=0.34,
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="c2_wait_focused",
        group="fitness_weights",
        description="Wait-time focused: alpha=0.7, beta=0.15, gamma=0.15",
        config=_cfg(
            name="c2_wait_focused", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, alpha=0.7, beta=0.15, gamma=0.15,
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="c3_resource_focused",
        group="fitness_weights",
        description="Resource-efficiency focused: alpha=0.15, beta=0.7, gamma=0.15",
        config=_cfg(
            name="c3_resource_focused", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, alpha=0.15, beta=0.7, gamma=0.15,
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="c4_reliability_focused",
        group="fitness_weights",
        description="Reliability focused: alpha=0.15, beta=0.15, gamma=0.7",
        config=_cfg(
            name="c4_reliability_focused", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, alpha=0.15, beta=0.15, gamma=0.7,
            n_train=n_train, n_test=n_test,
        ),
    ))

    # ── Group D: GP Parameters ───────────────────────────────────
    d_pop_small = max(pop // 2, 10)
    d_pop_large = pop * 3 if not quick else pop * 2
    d_gen_long = gen * 2

    experiments.append(ExperimentDef(
        name="d1_small_pop",
        group="gp_params",
        description=f"Small population: {d_pop_small}",
        config=_cfg(
            name="d1_small_pop", pods=pods_m, nodes=nodes_m,
            pop=d_pop_small, gen=gen, n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="d2_large_pop",
        group="gp_params",
        description=f"Large population: {d_pop_large}",
        config=_cfg(
            name="d2_large_pop", pods=pods_m, nodes=nodes_m,
            pop=d_pop_large, gen=gen, n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="d3_more_generations",
        group="gp_params",
        description=f"More generations: {d_gen_long}",
        config=_cfg(
            name="d3_more_generations", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=d_gen_long, n_train=n_train, n_test=n_test,
        ),
    ))

    # ── Group E: Dynamics (Node Failures) ────────────────────────
    experiments.append(ExperimentDef(
        name="e1_no_failures",
        group="dynamics",
        description="No node failures (static cluster)",
        config=_cfg(
            name="e1_no_failures", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, failure_mode="off",
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="e2_reschedule",
        group="dynamics",
        description="Node failures — reschedule mode (rate=2, 20%)",
        config=_cfg(
            name="e2_reschedule", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, failure_mode="reschedule", failure_rate=2,
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="e3_kill",
        group="dynamics",
        description="Node failures — kill mode (rate=2, 20%)",
        config=_cfg(
            name="e3_kill", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, failure_mode="kill", failure_rate=2,
            n_train=n_train, n_test=n_test,
        ),
    ))

    # ── Group G: Cross-Profile Comparison ────────────────────────
    # ai_training uses a larger cluster: 8 nodes × 32GB × 4 GPU each,
    # because its pods request 2-8GB RAM and 1-2 GPUs — the default
    # 5×16GB cluster fills up immediately and all schedulers tie on
    # rejection rate, making differentiation impossible.
    profile_cluster_overrides = {
        "ai_training": {"nodes": 8, "mem": 32768.0, "gpu": 4.0},
    }
    # ai_training cluster: fixed 4+4 for all presets.
    # 3+5 overnight was tried (2026_06_15) but REJECTED — GPU scarcity too extreme,
    # noise dominates and even Random beats GP occasionally. 4+4 in 2026_06_03 gave δ=0.261.
    _gpu_count, _cpu_count = 4, 4
    _aitrai_cluster = ClusterConfig(
        node_templates=[
            NodeConfig(count=_gpu_count, cpu_capacity=16.0, mem_capacity=32768.0,
                       gpu_capacity=4.0, cost_per_hour=3.0,
                       labels={"type": "gpu-node"}),
            NodeConfig(count=_cpu_count, cpu_capacity=8.0,  mem_capacity=16384.0,
                       gpu_capacity=0.0, cost_per_hour=1.0,
                       labels={"type": "cpu-node"}),
        ],
        heterogeneity=NodeHeterogeneityConfig(enabled=False),
    )
    # g_batchp cluster: explicit cost tiers to create GP leverage on cost-aware placement.
    # All nodes same capacity (16 CPU / 32 GB) so resource fit is equal — only cost differs.
    # 6 cheap (0.5 €/h) + 2 expensive (4.0 €/h) = 8× cost ratio, 8 nodes total.
    # Cluster is sized generously (128 CPU) to keep rejection rate low (<30%) so the
    # fitness is dominated by cost efficiency, not rejection rate.
    # LA places pods proportionally → avg node cost ≈ 1.375 €/h.
    # GP can learn neg(NODE_COST) → prefer cheap nodes → avg cost ≈ 0.5 €/h.
    _batchp_cluster = ClusterConfig(
        node_templates=[
            NodeConfig(count=6, cpu_capacity=16.0, mem_capacity=32768.0,
                       cost_per_hour=0.5, labels={"tier": "cheap"}),
            NodeConfig(count=2, cpu_capacity=16.0, mem_capacity=32768.0,
                       cost_per_hour=4.0, labels={"tier": "expensive"}),
        ],
        heterogeneity=NodeHeterogeneityConfig(enabled=False),
    )
    profile_cluster_overrides_v2 = {
        "ai_training":      {"cluster": _aitrai_cluster},
        "batch_processing": {"cluster": _batchp_cluster},
    }
    # Per-profile fitness weights: default α=0.4/β=0.3/γ=0.3 is tuned for
    # mixed workloads. GPU-heavy and batch profiles need different emphasis:
    # - ai_training: wait time is dominated by GPU scarcity (structural, not
    #   scheduler-driven) → reduce alpha, raise beta so GP learns GPU placement.
    # - batch_processing: constant-rate arrival, cost-extreme cluster.
    #   iota=0.35 (dominant): GP rewarded for preferring cheap nodes (8× cost ratio).
    #   Low alpha/beta/gamma: constant rate = all schedulers queue alike, bin-packing
    #   advantage is minimal, rejection rate is similar across strategies.
    #   parsimony=0.001: combats severe bloat observed in seed_789 (size_avg=62).
    # Per-profile fitness overrides. Keys map to _base_config parameters.
    # Weights must sum to 1.0 across alpha+beta+gamma+iota.
    # ai_training: reduce wait (GPU scarcity dominates, not scheduler);
    #              raise beta so GP learns efficient GPU placement.
    _profile_fitness: Dict[str, Dict[str, float]] = {
        "ai_training":     {"alpha": 0.2, "beta": 0.45, "gamma": 0.35},
        "batch_processing": {"alpha": 0.10, "beta": 0.30, "gamma": 0.25,
                             "iota": 0.35, "parsimony": 0.001},
    }
    for profile_name in ["web_serving", "ai_training", "ci_cd",
                         "batch_processing", "microservices"]:
        short = profile_name.replace("_", "")[:6]
        cluster_ov = profile_cluster_overrides_v2.get(profile_name, {}).get("cluster")
        simple_ov = profile_cluster_overrides.get(profile_name, {})
        fw = _profile_fitness.get(profile_name, {})
        experiments.append(ExperimentDef(
            name=f"g_{short}",
            group="profile",
            description=f"Profile: {profile_name}",
            config=_cfg(
                name=f"g_{short}",
                pods=pods_m,
                nodes=simple_ov.get("nodes", nodes_m),
                mem=simple_ov.get("mem", 16384.0),
                gpu=simple_ov.get("gpu", 0.0),
                pop=pop, gen=gen, profile=profile_name,
                n_train=n_train, n_test=n_test,
                terminal_mandatory=CORE_TERMINALS_GPU if profile_name == "ai_training" else None,
                **({"cluster_override": cluster_ov} if cluster_ov else {}),
                **fw,
            ),
        ))

    # ── Group H: Blind-spot experiments ─────────────────────────
    # Workloads and clusters designed to expose LeastAllocated's structural weaknesses.

    # h1_asymmetric: cluster cu noduri CPU-heavy și MEM-heavy.
    # LA scorează nodurile prin media (CPU_util + MEM_util)/2, care tratează identic
    # un nod de compute la 90% CPU/10% MEM și un nod balanced la 50/50.
    # GP cu NODE_CPU_FREE_AFTER, NODE_MEM_FREE_AFTER, POD_CPU_REQ, POD_MEM_REQ
    # poate învăța să potrivească profilul podului la specialitatea nodului.
    _asymmetric_cluster = ClusterConfig(
        node_templates=[
            NodeConfig(count=3, cpu_capacity=16.0, mem_capacity=8192.0,
                       cost_per_hour=2.0, labels={"type": "compute"}),
            NodeConfig(count=3, cpu_capacity=4.0,  mem_capacity=32768.0,
                       cost_per_hour=2.0, labels={"type": "memory"}),
            NodeConfig(count=2, cpu_capacity=8.0,  mem_capacity=16384.0,
                       cost_per_hour=1.5, labels={"type": "balanced"}),
        ],
        heterogeneity=NodeHeterogeneityConfig(enabled=False),
    )
    experiments.append(ExperimentDef(
        name="h1_asymmetric",
        group="blindspot",
        description="Asymmetric cluster (compute/memory/balanced) — LA's averaged utilization fails to match pod resource profiles",
        config=_cfg(
            name="h1_asymmetric", pods=pods_m, pop=pop, gen=gen,
            n_train=n_train, n_test=n_test,
            alpha=0.25, beta=0.55, gamma=0.20,
            cluster_override=_asymmetric_cluster,
        ),
    ))

    # h2_spike: profil spike cu burst ×15 și 50% probabilitate.
    # LA plasează uniform indiferent de presiunea cozii — umple nodurile constant,
    # lăsând puțin headroom pentru val. GP cu PENDING_PRESSURE și NODE_CPU_FREE_AFTER
    # poate învăța să bin-packeze mai agresiv înainte de val, eliberând noduri pentru spike.
    experiments.append(ExperimentDef(
        name="h2_spike",
        group="blindspot",
        description="Spike workload (×15 burst, 50% prob) — LA uniform spreading fails to absorb spikes; GP learns headroom preservation",
        config=_cfg(
            name="h2_spike", pods=pods_m, pop=pop, gen=gen,
            n_train=n_train, n_test=n_test,
            profile="spike",
            alpha=0.30, beta=0.30, gamma=0.40,
        ),
    ))

    return experiments


# ═══════════════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════════════


def run_single_experiment(exp: ExperimentDef) -> ExperimentResult:
    """Execute one full experiment (train GP + evaluate GP + baselines)."""
    cfg = exp.config
    log = logging.getLogger(f"exp.{exp.name}")
    log.info("=" * 60)
    log.info("EXPERIMENT: %s (%s)", exp.name, exp.description)
    log.info("=" * 60)

    # ── Generate workload ────────────────────────────────────────
    generator = PoissonWorkloadGenerator()

    training_instances: List[List[Pod]] = [
        generator.generate(cfg.workload, seed=cfg.seed + i)
        for i in range(cfg.num_training_instances)
    ]
    validation_instances: List[List[Pod]] = [
        generator.generate(
            cfg.workload,
            seed=cfg.seed + cfg.num_training_instances + i,
        )
        for i in range(cfg.num_validation_instances)
    ]
    test_seed_start = cfg.seed + cfg.num_training_instances + cfg.num_validation_instances
    test_instances: List[List[Pod]] = [
        generator.generate(cfg.workload, seed=test_seed_start + i)
        for i in range(cfg.num_test_instances)
    ]

    # ── Setup GP engine ──────────────────────────────────────────
    dynamics = cfg.dynamics if cfg.dynamics.enabled else None
    selected_terminals = cfg.gp.selected_terminals()

    gp_engine = DeapGeneticEngine()
    gp_engine.setup(
        terminal_names=selected_terminals,
        population_size=cfg.gp.population_size,
        n_generations=cfg.gp.n_generations,
        tournament_size=cfg.gp.tournament_size,
        crossover_prob=cfg.gp.crossover_prob,
        mutation_prob=cfg.gp.mutation_prob,
        max_tree_depth=cfg.gp.max_tree_depth,
        elitism_ratio=cfg.gp.elitism_ratio,
        parsimony_coefficient=cfg.gp.parsimony_coefficient,
        multi_objective=cfg.gp.multi_objective,
        seed_expressions=SEED_EXPRESSIONS_BASELINE,
        n_restarts=cfg.gp.n_restarts,
    )

    # ── Build fitness evaluator ──────────────────────────────────
    fitness_evaluator = FitnessEvaluator(
        gp_engine=gp_engine,
        training_instances=training_instances,
        cluster_config=cfg.cluster,
        fitness_weights=cfg.fitness,
        dynamics_config=dynamics,
        base_seed=cfg.seed,
        n_workers=cfg.gp.n_workers,
        aggregation_mode=cfg.gp.fitness_aggregation,
        std_penalty=cfg.gp.fitness_std_penalty,
    )

    # ── Train ────────────────────────────────────────────────────
    log.info(
        "Training: engine=%s  pop=%d  gen=%d  pods=%d  nodes=%d",
        cfg.gp.engine, cfg.gp.population_size, cfg.gp.n_generations,
        cfg.workload.total_pods,
        sum(t.count for t in cfg.cluster.effective_templates()),
    )
    t_start = time.perf_counter()
    fitness_fn = fitness_evaluator.evaluate_objectives if cfg.gp.multi_objective else fitness_evaluator
    gp_result = gp_engine.train(fitness_function=fitness_fn, seed=cfg.seed)
    training_time = time.perf_counter() - t_start
    fitness_evaluator.shutdown()  # release worker pool after training

    if (
        cfg.gp.engine == "deap"
        and validation_instances
        and cfg.gp.validation_hof_size > 0
        and gp_result.hall_of_fame
    ):
        _select_deap_champion_on_validation(
            gp_engine=gp_engine,
            gp_result=gp_result,
            validation_instances=validation_instances,
            cfg=cfg,
            dynamics=dynamics,
            hof_size=cfg.gp.validation_hof_size,
        )

    log.info("Done in %.1fs — quality=%.6f", training_time, gp_result.best_fitness)
    log.info("Rule: %s", gp_result.best_expression)

    # ── Evaluate GP on test set ──────────────────────────────────
    reporter = MetricsReporter()
    gp_strategy = GPSchedulingStrategy(gp_engine, gp_result.best_individual)
    gp_resource_monitor = None

    for i, instance_pods in enumerate(test_instances):
        instance_seed = test_seed_start + i
        fresh_pods = [FitnessEvaluator._copy_pod(p) for p in instance_pods]
        sim = SimulationEngine(
            strategy=gp_strategy,
            cluster_config=cfg.cluster,
            dynamics_config=dynamics,
            failure_seed=cfg.seed + i * 7919,
        )
        sim.build_cluster()
        sim.load_workload(fresh_pods)
        sim.run()
        reporter.add_run(
            strategy_name=f"GP({gp_engine.name})",
            instance_id=f"test-{i}",
            seed=instance_seed,
            metrics=sim.collector.get_metrics(),
            quality_score=compute_quality_score(sim.collector.get_metrics(), cfg.fitness),
        )
        if i == 0:
            gp_resource_monitor = sim.resource_monitor

    # ── Evaluate baselines ───────────────────────────────────────
    baselines: List[ISchedulingStrategy] = [
        RandomStrategy(seed=cfg.seed),
        RoundRobinStrategy(),
        FirstFitStrategy(),
        LeastAllocatedStrategy(),
        MostAllocatedStrategy(),
        BalancedAllocationStrategy(),
        BinPackingStrategy(),
    ]
    for strategy in baselines:
        for i, instance_pods in enumerate(test_instances):
            instance_seed = test_seed_start + i
            fresh_pods = [FitnessEvaluator._copy_pod(p) for p in instance_pods]
            sim = SimulationEngine(
                strategy=strategy,
                cluster_config=cfg.cluster,
                dynamics_config=dynamics,
                failure_seed=cfg.seed + i * 7919,
            )
            sim.build_cluster()
            sim.load_workload(fresh_pods)
            sim.run()
            reporter.add_run(
                strategy_name=strategy.name,
                instance_id=f"test-{i}",
                seed=instance_seed,
                metrics=sim.collector.get_metrics(),
                quality_score=compute_quality_score(sim.collector.get_metrics(), cfg.fitness),
            )

    return ExperimentResult(
        name=exp.name,
        group=exp.group,
        training_time=training_time,
        best_fitness=gp_result.best_fitness,
        best_expression=gp_result.best_expression,
        convergence_log=gp_result.log,
        reporter=reporter,
        gp_resource_monitor=gp_resource_monitor,
    )


def _select_deap_champion_on_validation(
    *,
    gp_engine: DeapGeneticEngine,
    gp_result: Any,
    validation_instances: List[List[Pod]],
    cfg: ExperimentConfig,
    dynamics: Optional[DynamicsConfig],
    hof_size: int,
) -> None:
    """Pick the deployment champion from DEAP Hall-of-Fame using validation workloads."""
    candidates = list(gp_result.hall_of_fame)[:max(1, hof_size)]
    evaluator = FitnessEvaluator(
        gp_engine=gp_engine,
        training_instances=[],
        cluster_config=cfg.cluster,
        fitness_weights=cfg.fitness,
        dynamics_config=dynamics,
        base_seed=cfg.seed + 100_000,
        aggregation_mode=cfg.gp.fitness_aggregation,
        std_penalty=cfg.gp.fitness_std_penalty,
    )

    best_idx = 0
    best_robust = float("-inf")
    for idx, candidate in enumerate(candidates):
        metrics_list = evaluator.evaluate_on_instances(candidate, validation_instances)
        scores = [compute_quality_score(m, cfg.fitness) for m in metrics_list]
        if not scores:
            continue
        mean_score = float(sum(scores) / len(scores))
        std_score = float(np.std(scores))
        robust = (
            mean_score - cfg.gp.fitness_std_penalty * std_score
            if cfg.gp.fitness_aggregation == "mean_minus_std"
            else mean_score
        )
        if robust > best_robust:
            best_idx = idx
            best_robust = robust

    selected = candidates[best_idx]
    gp_result.best_individual = selected
    gp_result.best_expression = gp_engine.get_expression_string(selected)


def run_experiments(
    experiments: List[ExperimentDef],
    output_dir: Path,
) -> List[ExperimentResult]:
    """Run all experiments and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[ExperimentResult] = []

    for idx, exp in enumerate(experiments, 1):
        logger.info(
            "[%d/%d] Starting experiment: %s", idx, len(experiments), exp.name,
        )
        result = run_single_experiment(exp)
        results.append(result)

        # Save per-experiment CSV
        exp_dir = output_dir / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        result.reporter.export_csv(exp_dir / "results.csv")

        # Save convergence log
        with open(exp_dir / "convergence.json", "w", encoding="utf-8") as f:
            json.dump(result.convergence_log, f, indent=2, default=str)

        # Save experiment metadata
        meta = {
            "name": result.name,
            "group": result.group,
            "training_time_s": round(result.training_time, 2),
            "best_fitness": round(result.best_fitness, 6),
            "best_expression": result.best_expression,
            "engine": exp.config.gp.engine,
            "population_size": exp.config.gp.population_size,
            "n_workers": exp.config.gp.n_workers,
            "fitness_aggregation": exp.config.gp.fitness_aggregation,
            "fitness_std_penalty": exp.config.gp.fitness_std_penalty,
            "num_validation_instances": exp.config.num_validation_instances,
            "validation_hof_size": exp.config.gp.validation_hof_size,
            "n_generations": exp.config.gp.n_generations,
            "total_pods": exp.config.workload.total_pods,
            "node_count": sum(t.count for t in exp.config.cluster.node_templates),
            "alpha": exp.config.fitness.alpha_wait_time,
            "beta": exp.config.fitness.beta_resource_waste,
            "gamma": exp.config.fitness.gamma_failed_pods,
            "failure_mode": exp.config.dynamics.failure_mode,
            "failure_rate": exp.config.dynamics.failure_rate,
        }
        with open(exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save full experiment configuration (used by UI to display per-run config)
        with open(exp_dir / "experiment_config.json", "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(exp.config), f, indent=2, default=str)

        # Save GP resource timeline (first test instance)
        if result.gp_resource_monitor is not None:
            result.gp_resource_monitor.export_json(
                exp_dir / "resource_timeline.json"
            )

        logger.info(
            "[%d/%d] Completed %s in %.1fs (quality=%.4f)",
            idx, len(experiments), exp.name,
            result.training_time, result.best_fitness,
        )

    # ── Save combined results ────────────────────────────────────
    _save_combined_csv(results, output_dir / "combined_results.csv")
    _save_summary(results, output_dir / "experiment_summary.txt")

    return results


def _save_combined_csv(results: List[ExperimentResult], path: Path) -> None:
    """Merge all per-experiment CSVs into one file with an experiment column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    for r in results:
        for record in r.reporter._records:
            row = dict(record)
            row["experiment"] = r.name
            row["group"] = r.group
            all_rows.append(row)

    if not all_rows:
        return

    import csv

    fieldnames = [
        "experiment", "group", "strategy", "instance_id", "seed",
        "quality_score",
        "total_pods", "scheduled_pods", "completed_pods", "rejected_pods",
        "scheduling_success_rate", "avg_wait_time",
        "avg_cpu_utilization", "avg_mem_utilization",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def _save_summary(results: List[ExperimentResult], path: Path) -> None:
    """Save a human-readable summary of all experiments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "EXPERIMENT SWEEP SUMMARY",
        "=" * 80,
        "",
    ]
    for r in results:
        lines.append(f"Experiment: {r.name} [{r.group}]")
        lines.append(f"  Training time: {r.training_time:.1f}s")
        lines.append(f"  Best quality:  {r.best_fitness:.6f}")
        lines.append(f"  Best rule:     {r.best_expression}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# Multi-seed evaluation
# ═══════════════════════════════════════════════════════════════════════


def run_experiments_multi_seed(
    experiments: List[ExperimentDef],
    seeds: List[int],
    output_dir: Path,
) -> None:
    """Run every experiment for each seed in *seeds* and aggregate results.

    For each (experiment, seed) pair the full pipeline is executed with the
    seed substituted into the config.  All per-run quality scores are
    accumulated into a flat CSV (multiseed_results.csv) and a human-readable
    summary table (multiseed_summary.txt) with median ± IQR per strategy.

    This is the primary evaluation path for robust, anti-cherry-picking
    comparisons between GP and baseline strategies.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    total = len(experiments) * len(seeds)
    run_idx = 0

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=== SEED %d (1 of %d seeds) ===", seed, len(seeds))

        for exp in experiments:
            run_idx += 1
            logger.info(
                "[%d/%d] experiment=%s seed=%d",
                run_idx, total, exp.name, seed,
            )

            seeded_config = replace(exp.config, seed=seed)
            seeded_exp = ExperimentDef(
                name=exp.name,
                group=exp.group,
                config=seeded_config,
                description=exp.description,
            )

            result = run_single_experiment(seeded_exp)

            exp_seed_dir = seed_dir / exp.name
            exp_seed_dir.mkdir(parents=True, exist_ok=True)
            result.reporter.export_csv(exp_seed_dir / "results.csv")
            with open(exp_seed_dir / "convergence.json", "w", encoding="utf-8") as f:
                json.dump(result.convergence_log, f, indent=2, default=str)
            with open(exp_seed_dir / "gp_rule.json", "w", encoding="utf-8") as f:
                json.dump({
                    "best_expression": result.best_expression,
                    "best_fitness": round(result.best_fitness, 6),
                    "training_time_s": round(result.training_time, 2),
                    "seed": seed,
                    "experiment": exp.name,
                    "group": exp.group,
                }, f, indent=2)

            # Save full config for UI display (same for all seeds, seed number differs)
            with open(exp_seed_dir / "experiment_config.json", "w", encoding="utf-8") as f:
                json.dump(dataclasses.asdict(seeded_config), f, indent=2, default=str)

            for record in result.reporter._records:
                row = dict(record)
                row["experiment"] = result.name
                row["group"] = result.group
                row["run_seed"] = seed
                all_rows.append(row)

            logger.info(
                "[%d/%d] Done: experiment=%s seed=%d quality=%.4f time=%.1fs",
                run_idx, total, exp.name, seed,
                result.best_fitness, result.training_time,
            )

    _save_multiseed_csv(all_rows, output_dir / "multiseed_results.csv")
    _save_multiseed_summary(all_rows, output_dir / "multiseed_summary.txt", seeds=seeds)
    logger.info("Multi-seed results saved to %s", output_dir)


def _save_multiseed_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Save flat CSV with a run_seed column for every (seed, experiment, strategy, instance) row."""
    if not rows:
        return
    import csv
    fieldnames = [
        "run_seed", "experiment", "group", "strategy", "instance_id", "seed",
        "quality_score",
        "total_pods", "scheduled_pods", "completed_pods", "rejected_pods",
        "scheduling_success_rate", "avg_wait_time",
        "avg_cpu_utilization", "avg_mem_utilization",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_multiseed_summary(
    rows: List[Dict[str, Any]],
    path: Path,
    seeds: List[int],
) -> None:
    """Aggregate quality scores across seeds and print median ± IQR table."""
    from collections import defaultdict

    scores_by_exp_strategy: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        exp = row.get("experiment", "?")
        strategy = row.get("strategy", "?")
        qs = row.get("quality_score")
        if qs is not None:
            scores_by_exp_strategy[exp][strategy].append(float(qs))

    lines = [
        "MULTI-SEED EXPERIMENT SUMMARY",
        f"Seeds: {seeds}  (n_seeds={len(seeds)})",
        "=" * 92,
        "",
    ]

    for exp_name in sorted(scores_by_exp_strategy):
        by_strategy = scores_by_exp_strategy[exp_name]
        lines.append(f"Experiment: {exp_name}")
        lines.append(
            f"  {'Strategy':<30} {'Median':>8} {'IQR':>8} {'Mean':>8} {'Std':>8} {'N':>4}"
        )
        lines.append("  " + "-" * 70)

        sorted_strats = sorted(
            by_strategy.items(),
            key=lambda kv: -float(np.median(kv[1])),
        )
        for strategy, q_scores in sorted_strats:
            arr = np.array(q_scores, dtype=float)
            q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            lines.append(
                f"  {strategy:<30} {float(np.median(arr)):>8.4f} {q75 - q25:>8.4f}"
                f" {float(arr.mean()):>8.4f} {float(arr.std()):>8.4f} {len(arr):>4}"
            )
        lines.append("")

    summary_text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("\n" + summary_text)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K8s GP Scheduler - Experiment Sweep Runner",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Very small parameters for smoke/syntax validation (~0.3s per experiment)",
    )
    parser.add_argument(
        "--medium", action="store_true",
        help="Balanced parameters: pop=80, gen=30, pods=100 (~45s/exp, recommended for --seeds runs)",
    )
    parser.add_argument(
        "--day", action="store_true",
        help="Medium scale + overnight quality extras: ~100s/exp, ~3h for 21 exp x 5 seeds. "
             "Recommended for quick validation with all experiments.",
    )
    parser.add_argument(
        "--overnight", action="store_true",
        help="Deep exploration: pop=120, gen=50, n_train=5, large cluster (~312s/exp, ~10.7h for 5 seeds)",
    )
    parser.add_argument(
        "--group", type=str, default=None,
        help="Run only experiments from this group "
             "(engine, scale, fitness_weights, gp_params, dynamics)",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Run a single experiment by name",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all experiments and exit",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds for multi-seed evaluation "
             "(e.g. --seeds 42,123,456,789,1337). Each experiment is repeated "
             "for every seed; results are aggregated into multiseed_results.csv.",
    )
    parser.add_argument(
        "--output", type=str, default="tmp/results/experiments",
        help="Output directory for results (default: tmp/results/experiments)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    flags = [args.quick, args.medium, getattr(args, 'day', False), getattr(args, 'overnight', False)]
    if sum(flags) > 1:
        print("Error: --quick, --medium, --day, --overnight are mutually exclusive.")
        return

    experiments = define_experiments(
        quick=args.quick,
        medium=args.medium,
        day=getattr(args, 'day', False),
        overnight=getattr(args, 'overnight', False),
    )

    # ── List mode ────────────────────────────────────────────────
    if args.list:
        print(f"\n{'Name':<25} {'Group':<18} Description")
        print("-" * 75)
        for exp in experiments:
            print(f"{exp.name:<25} {exp.group:<18} {exp.description}")
        print(f"\nTotal: {len(experiments)} experiments")
        return

    # ── Filter ───────────────────────────────────────────────────
    if args.experiment:
        experiments = [e for e in experiments if e.name == args.experiment]
        if not experiments:
            print(f"Error: experiment '{args.experiment}' not found. Use --list.")
            return
    elif args.group:
        experiments = [e for e in experiments if e.group == args.group]
        if not experiments:
            print(f"Error: group '{args.group}' not found. Use --list.")
            return

    # ── Parse seeds ──────────────────────────────────────────────
    seeds: Optional[List[int]] = None
    if args.seeds:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        except ValueError:
            print("Error: --seeds must be comma-separated integers (e.g. --seeds 42,123,456)")
            return
        if len(seeds) < 2:
            print("Error: --seeds requires at least 2 seeds for meaningful multi-seed evaluation.")
            return

    # ── Run ──────────────────────────────────────────────────────
    mode = "QUICK" if args.quick else ("MEDIUM" if args.medium else "FULL")
    seed_label = f"  Seeds: {seeds}" if seeds else ""
    print(f"\n{'=' * 60}")
    print(f"  K8s GP Scheduler - Experiment Sweep ({mode})")
    print(f"  Running {len(experiments)} experiment(s){(' x ' + str(len(seeds)) + ' seeds') if seeds else ''}")
    if seed_label:
        print(seed_label)
    print(f"{'=' * 60}\n")

    _overnight = getattr(args, 'overnight', False)
    _day = getattr(args, 'day', False)
    preset = "quick" if args.quick else ("medium" if args.medium else ("day" if _day else ("overnight" if _overnight else "full")))
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = Path(args.output) / f"{ts}_{preset}"
    t_total_start = time.perf_counter()

    if seeds:
        run_experiments_multi_seed(experiments, seeds, output_dir)
        t_total = time.perf_counter() - t_total_start
        print(f"\nTotal time: {t_total:.1f}s")
        print(f"Results saved to: {output_dir}")
        print(f"\nRun analysis: py analysis.py --input {output_dir} --multiseed")
    else:
        results = run_experiments(experiments, output_dir)
        t_total = time.perf_counter() - t_total_start

        # ── Print summary ────────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print("  RESULTS SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Experiment':<25} {'Engine':<8} {'Quality':>8} {'Time':>8}")
        print("-" * 55)
        for r in results:
            eng = [e for e in define_experiments(args.quick, args.medium) if e.name == r.name]
            engine_name = eng[0].config.gp.engine if eng else "?"
            print(f"{r.name:<25} {engine_name:<8} {r.best_fitness:>8.4f} {r.training_time:>7.1f}s")
        print("-" * 55)
        print(f"Total time: {t_total:.1f}s")
        print(f"Results saved to: {output_dir}")
        print(f"\nRun analysis: py analysis.py --input {output_dir}")


if __name__ == "__main__":
    main()
