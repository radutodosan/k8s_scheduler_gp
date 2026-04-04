"""Experiment sweep runner for dissertation Chapter 5.

Defines and executes systematic experiments comparing:
  - GP engines (DEAP vs gplearn)
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
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.schema import (
    ClusterConfig,
    DynamicsConfig,
    ExperimentConfig,
    FitnessWeights,
    GPConfig,
    NodeConfig,
    WorkloadConfig,
)
from gp.deap_engine import DeapGeneticEngine
from gp.fitness import FitnessEvaluator
from gp.gplearn_engine import GplearnEngine
from gp.primitives import TERMINAL_NAMES
from metrics.reporter import MetricsReporter
from metrics.resource_monitor import ResourceMonitor
from models.pod import Pod
from scheduling.balanced_allocation import BalancedAllocationStrategy
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


def _base_config(
    *,
    name: str = "experiment",
    seed: int = 42,
    n_train: int = 5,
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
) -> ExperimentConfig:
    """Build an ExperimentConfig from simplified parameters."""
    return ExperimentConfig(
        name=name,
        seed=seed,
        num_training_instances=n_train,
        num_test_instances=n_test,
        output_dir=f"tmp/results/experiments/{name}",
        output_format="csv",
        cluster=ClusterConfig(
            node_templates=[NodeConfig(count=nodes, cpu_capacity=cpu, mem_capacity=mem)]
        ),
        workload=WorkloadConfig(total_pods=pods, profile=profile),
        gp=GPConfig(
            engine=engine,
            population_size=pop,
            n_generations=gen,
            max_tree_depth=depth,
            multi_objective=multi_objective,
        ),
        fitness=FitnessWeights(
            alpha_wait_time=alpha,
            beta_resource_waste=beta,
            gamma_failed_pods=gamma,
        ),
        dynamics=DynamicsConfig(
            failure_mode=failure_mode,
            failure_rate=failure_rate,
        ),
    )


def define_experiments(quick: bool = False) -> List[ExperimentDef]:
    """Define all dissertation experiments.

    Args:
        quick: If True, use smaller parameters for fast validation.
    """
    # Scale factors for quick mode
    if quick:
        pop, gen, pods_s, pods_m, pods_l = 20, 5, 15, 30, 60
        n_train, n_test = 2, 2
        nodes_s, nodes_m, nodes_l = 2, 3, 5
    else:
        pop, gen, pods_s, pods_m, pods_l = 100, 30, 50, 100, 200
        n_train, n_test = 5, 5
        nodes_s, nodes_m, nodes_l = 3, 5, 10

    experiments: List[ExperimentDef] = []

    # ── Group A: Engine Comparison (DEAP vs gplearn) ─────────────
    experiments.append(ExperimentDef(
        name="a1_deap_medium",
        group="engine",
        description="DEAP engine — medium scenario",
        config=_base_config(
            name="a1_deap_medium", engine="deap",
            pods=pods_m, nodes=nodes_m, pop=pop, gen=gen,
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="a2_gplearn_medium",
        group="engine",
        description="gplearn engine — medium scenario (same params as a1)",
        config=_base_config(
            name="a2_gplearn_medium", engine="gplearn",
            pods=pods_m, nodes=nodes_m, pop=pop, gen=gen,
            n_train=n_train, n_test=n_test,
        ),
    ))

    # ── Group B: Scale Sensitivity ───────────────────────────────
    experiments.append(ExperimentDef(
        name="b1_small",
        group="scale",
        description=f"Small scale: {pods_s} pods, {nodes_s} nodes",
        config=_base_config(
            name="b1_small", pods=pods_s, nodes=nodes_s,
            pop=pop, gen=gen, n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="b2_medium",
        group="scale",
        description=f"Medium scale: {pods_m} pods, {nodes_m} nodes",
        config=_base_config(
            name="b2_medium", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="b3_large",
        group="scale",
        description=f"Large scale: {pods_l} pods, {nodes_l} nodes",
        config=_base_config(
            name="b3_large", pods=pods_l, nodes=nodes_l,
            pop=pop, gen=gen, n_train=n_train, n_test=n_test,
        ),
    ))

    # ── Group C: Fitness Weight Sensitivity ──────────────────────
    experiments.append(ExperimentDef(
        name="c1_balanced",
        group="fitness_weights",
        description="Balanced weights: α=0.33, β=0.33, γ=0.34",
        config=_base_config(
            name="c1_balanced", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, alpha=0.33, beta=0.33, gamma=0.34,
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="c2_wait_focused",
        group="fitness_weights",
        description="Wait-time focused: α=0.7, β=0.15, γ=0.15",
        config=_base_config(
            name="c2_wait_focused", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, alpha=0.7, beta=0.15, gamma=0.15,
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="c3_resource_focused",
        group="fitness_weights",
        description="Resource-efficiency focused: α=0.15, β=0.7, γ=0.15",
        config=_base_config(
            name="c3_resource_focused", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, alpha=0.15, beta=0.7, gamma=0.15,
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="c4_reliability_focused",
        group="fitness_weights",
        description="Reliability focused: α=0.15, β=0.15, γ=0.7",
        config=_base_config(
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
        config=_base_config(
            name="d1_small_pop", pods=pods_m, nodes=nodes_m,
            pop=d_pop_small, gen=gen, n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="d2_large_pop",
        group="gp_params",
        description=f"Large population: {d_pop_large}",
        config=_base_config(
            name="d2_large_pop", pods=pods_m, nodes=nodes_m,
            pop=d_pop_large, gen=gen, n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="d3_more_generations",
        group="gp_params",
        description=f"More generations: {d_gen_long}",
        config=_base_config(
            name="d3_more_generations", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=d_gen_long, n_train=n_train, n_test=n_test,
        ),
    ))

    # ── Group E: Dynamics (Node Failures) ────────────────────────
    experiments.append(ExperimentDef(
        name="e1_no_failures",
        group="dynamics",
        description="No node failures (static cluster)",
        config=_base_config(
            name="e1_no_failures", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, failure_mode="off",
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="e2_reschedule",
        group="dynamics",
        description="Node failures — reschedule mode (rate=2, 20%)",
        config=_base_config(
            name="e2_reschedule", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, failure_mode="reschedule", failure_rate=2,
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="e3_kill",
        group="dynamics",
        description="Node failures — kill mode (rate=2, 20%)",
        config=_base_config(
            name="e3_kill", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, failure_mode="kill", failure_rate=2,
            n_train=n_train, n_test=n_test,
        ),
    ))

    # ── Group F: NSGA-II Multi-Objective ─────────────────────────
    experiments.append(ExperimentDef(
        name="f1_single_objective",
        group="nsga2",
        description="Standard single-objective fitness (α·W + β·R + γ·F)",
        config=_base_config(
            name="f1_single_objective", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, multi_objective=False,
            n_train=n_train, n_test=n_test,
        ),
    ))
    experiments.append(ExperimentDef(
        name="f2_nsga2",
        group="nsga2",
        description="NSGA-II 3-objective (wait, waste, reject)",
        config=_base_config(
            name="f2_nsga2", pods=pods_m, nodes=nodes_m,
            pop=pop, gen=gen, multi_objective=True,
            n_train=n_train, n_test=n_test,
        ),
    ))

    # ── Group G: Cross-Profile Comparison ────────────────────────
    for profile_name in ["web_serving", "ai_training", "ci_cd",
                         "batch_processing", "microservices"]:
        short = profile_name.replace("_", "")[:6]
        gpu_cap = 4.0 if profile_name == "ai_training" else 0.0
        experiments.append(ExperimentDef(
            name=f"g_{short}",
            group="profile",
            description=f"Profile: {profile_name}",
            config=_base_config(
                name=f"g_{short}", pods=pods_m, nodes=nodes_m,
                pop=pop, gen=gen, profile=profile_name,
                n_train=n_train, n_test=n_test,
            ),
        ))
        # Add GPU capacity for ai_training nodes
        if gpu_cap > 0:
            experiments[-1].config.cluster.node_templates[0].gpu_capacity = gpu_cap

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
    test_instances: List[List[Pod]] = [
        generator.generate(cfg.workload, seed=cfg.seed + cfg.num_training_instances + i)
        for i in range(cfg.num_test_instances)
    ]

    # ── Setup GP engine ──────────────────────────────────────────
    dynamics = cfg.dynamics if cfg.dynamics.enabled else None

    if cfg.gp.engine == "gplearn":
        gp_engine = GplearnEngine()
        gp_engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=cfg.gp.population_size,
            n_generations=cfg.gp.n_generations,
            tournament_size=cfg.gp.tournament_size,
            crossover_prob=cfg.gp.crossover_prob,
            mutation_prob=cfg.gp.mutation_prob,
            max_tree_depth=cfg.gp.max_tree_depth,
            parsimony_coefficient=cfg.gp.parsimony_coefficient,
            training_instances=training_instances,
            cluster_config=cfg.cluster,
            dynamics_config=dynamics,
            reference_strategies=[
                LeastAllocatedStrategy,
                MostAllocatedStrategy,
                BalancedAllocationStrategy,
            ],
        )
    else:
        gp_engine = DeapGeneticEngine()
        gp_engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=cfg.gp.population_size,
            n_generations=cfg.gp.n_generations,
            tournament_size=cfg.gp.tournament_size,
            crossover_prob=cfg.gp.crossover_prob,
            mutation_prob=cfg.gp.mutation_prob,
            max_tree_depth=cfg.gp.max_tree_depth,
            elitism_ratio=cfg.gp.elitism_ratio,
            parsimony_coefficient=cfg.gp.parsimony_coefficient,
            multi_objective=cfg.gp.multi_objective,
        )

    # ── Build fitness evaluator ──────────────────────────────────
    fitness_evaluator = FitnessEvaluator(
        gp_engine=gp_engine,
        training_instances=training_instances,
        cluster_config=cfg.cluster,
        fitness_weights=cfg.fitness,
        dynamics_config=dynamics,
        base_seed=cfg.seed,
    )

    # ── Train ────────────────────────────────────────────────────
    log.info(
        "Training: engine=%s  pop=%d  gen=%d  pods=%d  nodes=%d",
        cfg.gp.engine, cfg.gp.population_size, cfg.gp.n_generations,
        cfg.workload.total_pods,
        sum(t.count for t in cfg.cluster.node_templates),
    )
    t_start = time.perf_counter()
    fitness_fn = fitness_evaluator.evaluate_objectives if cfg.gp.multi_objective else fitness_evaluator
    gp_result = gp_engine.train(fitness_function=fitness_fn, seed=cfg.seed)
    training_time = time.perf_counter() - t_start

    log.info("Done in %.1fs — fitness=%.6f", training_time, gp_result.best_fitness)
    log.info("Rule: %s", gp_result.best_expression)

    # ── Evaluate GP on test set ──────────────────────────────────
    reporter = MetricsReporter()
    gp_strategy = GPSchedulingStrategy(gp_engine, gp_result.best_individual)
    gp_resource_monitor = None

    for i, instance_pods in enumerate(test_instances):
        instance_seed = cfg.seed + cfg.num_training_instances + i
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
    ]
    for strategy in baselines:
        for i, instance_pods in enumerate(test_instances):
            instance_seed = cfg.seed + cfg.num_training_instances + i
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

        # Save GP resource timeline (first test instance)
        if result.gp_resource_monitor is not None:
            result.gp_resource_monitor.export_json(
                exp_dir / "resource_timeline.json"
            )

        logger.info(
            "[%d/%d] Completed %s in %.1fs (fitness=%.4f)",
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
        lines.append(f"  Best fitness:  {r.best_fitness:.6f}")
        lines.append(f"  Best rule:     {r.best_expression}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K8s GP Scheduler — Experiment Sweep Runner",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use small parameters for fast validation",
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

    experiments = define_experiments(quick=args.quick)

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

    # ── Run ──────────────────────────────────────────────────────
    mode = "QUICK" if args.quick else "FULL"
    print(f"\n{'=' * 60}")
    print(f"  K8s GP Scheduler — Experiment Sweep ({mode})")
    print(f"  Running {len(experiments)} experiment(s)")
    print(f"{'=' * 60}\n")

    ts = datetime.now().strftime("%Y_%m_%d_%H%M")
    output_dir = Path(args.output) / f"{ts}_results"
    t_total_start = time.perf_counter()

    results = run_experiments(experiments, output_dir)

    t_total = time.perf_counter() - t_total_start

    # ── Print summary ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Experiment':<25} {'Engine':<8} {'Fitness':>8} {'Time':>8}")
    print("-" * 55)
    for r in results:
        eng = [e for e in define_experiments(args.quick) if e.name == r.name]
        engine_name = eng[0].config.gp.engine if eng else "?"
        print(f"{r.name:<25} {engine_name:<8} {r.best_fitness:>8.4f} {r.training_time:>7.1f}s")
    print("-" * 55)
    print(f"Total time: {t_total:.1f}s")
    print(f"Results saved to: {output_dir}")
    print(f"\nRun analysis: py analysis.py --input {output_dir}")


if __name__ == "__main__":
    main()
