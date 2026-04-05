"""K8s GP Scheduler — Experiment Runner.

Entry point that orchestrates the full pipeline:
  1. Load configuration
  2. Generate training and test workload instances
  3. Train the GP engine (evolve scheduling rules)
  4. Evaluate the best rule on test instances
  5. Export results
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from config.schema import ClusterConfig, ExperimentConfig, NodeConfig
from gp.deap_engine import DeapGeneticEngine
from gp.fitness import FitnessEvaluator
from gp.primitives import TERMINAL_NAMES
from metrics.collector import SchedulingMetrics
from metrics.reporter import MetricsReporter
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
from visualization.gantt import plot_gantt_from_engine, save_gantt
from visualization.resource_plots import (
    plot_cluster_utilization,
    plot_cluster_comparison,
    plot_free_resources,
    plot_utilization_variance,
    plot_wait_time_distribution,
    save_resource_plot,
)
from visualization.gp_tree import plot_gp_tree, plot_pareto_front, save_gp_tree
from workload.poisson_generator import PoissonWorkloadGenerator
from generate_dataset import PRESETS, load_dataset


def main() -> None:
    args = _parse_args()

    # ── Logging ──────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("main")

    # ── Configuration ────────────────────────────────────────────────
    if args.config:
        log.info("Loading config from %s", args.config)
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        log.info("Using default configuration")
        cfg = ExperimentConfig()

    # ── Timestamped output directory ──────────────────────────────
    ts = datetime.now().strftime("%Y_%m_%d_%H%M")
    cfg.output_dir = str(Path("tmp") / "results" / "runs" / ts)

    log.info("Experiment: %s  seed=%d", cfg.name, cfg.seed)

    # ── Load or generate workload instances ──────────────────────────
    if args.dataset == "dynamic":
        log.info("Loading dynamic dataset from disk...")
        meta, training_instances, test_instances = load_dataset()
        # Override config with dataset metadata
        preset = PRESETS[meta["size"]]
        cfg.num_training_instances = meta["num_training_instances"]
        cfg.num_test_instances = meta["num_test_instances"]
        cfg.workload.total_pods = meta["total_pods"]
        cfg.cluster = ClusterConfig(
            node_templates=[NodeConfig(
                count=meta["nodes"],
                cpu_capacity=meta["cpu_capacity"],
                mem_capacity=meta["mem_capacity"],
            )]
        )
        cfg.gp.population_size = meta["gp_population_size"]
        cfg.gp.n_generations = meta["gp_n_generations"]
        log.info(
            "  Loaded %s dataset: %d train + %d test instances × %d pods",
            meta["size"], len(training_instances), len(test_instances),
            meta["total_pods"],
        )
    else:
        # Apply --size preset if provided
        if args.size:
            preset = PRESETS[args.size]
            cfg.workload.total_pods = preset["total_pods"]
            cfg.cluster = ClusterConfig(
                node_templates=[NodeConfig(
                    count=preset["nodes"],
                    cpu_capacity=preset["cpu_capacity"],
                    mem_capacity=preset["mem_capacity"],
                )]
            )
            cfg.num_training_instances = preset["num_training_instances"]
            cfg.num_test_instances = preset["num_test_instances"]
            cfg.gp.population_size = preset["gp_population_size"]
            cfg.gp.n_generations = preset["gp_n_generations"]
            log.info("Using %s preset: %d pods, %d nodes",
                     args.size, preset["total_pods"], preset["nodes"])

        generator = PoissonWorkloadGenerator()
        log.info("Generating %d training instances...", cfg.num_training_instances)
        training_instances: List[List[Pod]] = []
        for i in range(cfg.num_training_instances):
            instance_seed = cfg.seed + i
            pods = generator.generate(cfg.workload, seed=instance_seed)
            training_instances.append(pods)
            log.info(
                "  Training instance %d: %d pods, seed=%d",
                i, len(pods), instance_seed,
            )

        log.info("Generating %d test instances...", cfg.num_test_instances)
        test_instances: List[List[Pod]] = []
        for i in range(cfg.num_test_instances):
            instance_seed = cfg.seed + cfg.num_training_instances + i
            pods = generator.generate(cfg.workload, seed=instance_seed)
            test_instances.append(pods)
            log.info(
                "  Test instance %d: %d pods, seed=%d",
                i, len(pods), instance_seed,
            )

    # ── Setup GP engine ──────────────────────────────────────────────
    log.info("Setting up GP engine: %s", cfg.gp.engine)

    dynamics = cfg.dynamics if cfg.dynamics.enabled else None

    if cfg.gp.engine == "gplearn":
        from gp.gplearn_engine import GplearnEngine
        from scheduling.balanced_allocation import BalancedAllocationStrategy
        from scheduling.least_allocated import LeastAllocatedStrategy
        from scheduling.most_allocated import MostAllocatedStrategy
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

    # ── Build fitness evaluator ──────────────────────────────────────
    fitness_evaluator = FitnessEvaluator(
        gp_engine=gp_engine,
        training_instances=training_instances,
        cluster_config=cfg.cluster,
        fitness_weights=cfg.fitness,
        dynamic_instances=cfg.dynamic_instances,
        workload_generator=generator if cfg.dynamic_instances else None,
        workload_config=cfg.workload if cfg.dynamic_instances else None,
        base_seed=cfg.seed,
        num_instances=cfg.num_training_instances,
        dynamics_config=dynamics,
    )

    if cfg.dynamic_instances:
        log.info(
            "Dynamic training instances ENABLED — "
            "instances will be regenerated each generation"
        )

    if cfg.dynamics.enabled:
        log.info(
            "Node failures ENABLED — mode=%s, rate=%d, recovery=[%.1f, %.1f]",
            cfg.dynamics.failure_mode,
            cfg.dynamics.failure_rate,
            cfg.dynamics.recovery_time_min,
            cfg.dynamics.recovery_time_max,
        )

    # ── Train ────────────────────────────────────────────────────────
    log.info(
        "Starting GP training: pop=%d, gen=%d",
        cfg.gp.population_size,
        cfg.gp.n_generations,
    )
    t_start = time.perf_counter()
    fitness_fn = fitness_evaluator.evaluate_objectives if cfg.gp.multi_objective else fitness_evaluator
    gp_result = gp_engine.train(
        fitness_function=fitness_fn,
        seed=cfg.seed,
    )
    t_elapsed = time.perf_counter() - t_start

    log.info("Training complete in %.1f seconds", t_elapsed)
    log.info("Best fitness: %.6f", gp_result.best_fitness)
    log.info("Best expression: %s", gp_result.best_expression)

    # Print convergence log
    log.info("Convergence log:")
    for entry in gp_result.log:
        log.info(
            "  gen=%s  nevals=%s  min=%.4f  avg=%.4f  std=%.4f",
            entry.get("gen", "?"),
            entry.get("nevals", "?"),
            entry.get("min", 0.0),
            entry.get("avg", 0.0),
            entry.get("std", 0.0),
        )

    # ── Evaluate on test set ─────────────────────────────────────────
    log.info("Evaluating best individual on %d test instances...", len(test_instances))
    reporter = MetricsReporter()
    gantt_dir = Path(cfg.output_dir) / "gantt" if args.gantt else None
    resource_monitors = {}  # {strategy_name: ResourceMonitor}
    strategy_wait_times = {}  # {strategy_name: List[float]}

    # GP evaluation on test set
    gp_strategy = GPSchedulingStrategy(gp_engine, gp_result.best_individual)
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

        m = sim.collector.get_metrics()
        reporter.add_run(
            strategy_name=f"GP({gp_engine.name})",
            instance_id=f"test-{i}",
            seed=instance_seed,
            metrics=m,
        )
        strategy_wait_times.setdefault(f"GP({gp_engine.name})", []).extend(m.per_pod_wait_times)
        log.info(
            "  test-%d: success=%.1f%%  wait=%.2f  cpu=%.1f%%  rej=%d",
            i,
            m.scheduling_success_rate * 100,
            m.avg_wait_time,
            m.avg_cpu_utilization * 100,
            m.rejected_pods,
        )

        # Gantt for first test instance
        if gantt_dir and i == 0:
            fig = plot_gantt_from_engine(
                sim, title=f"GP({gp_engine.name}) — test-0",
            )
            save_gantt(fig, gantt_dir / f"gantt_GP_{gp_engine.name}.png")
        # Resource timeline for first test instance
        if i == 0:
            resource_monitors[f"GP({gp_engine.name})"] = sim.resource_monitor

    # ── Evaluate baselines on test set ─────────────────────────────────
    baselines: list[ISchedulingStrategy] = [
        RandomStrategy(seed=cfg.seed),
        RoundRobinStrategy(),
        FirstFitStrategy(),
        LeastAllocatedStrategy(),
        MostAllocatedStrategy(),
        BalancedAllocationStrategy(),
        BinPackingStrategy(),
    ]

    for strategy in baselines:
        log.info("Evaluating baseline: %s", strategy.name)
        for i, instance_pods in enumerate(test_instances):
            instance_seed = cfg.seed + cfg.num_training_instances + i
            fresh_pods = [FitnessEvaluator._copy_pod(p) for p in instance_pods]

            engine = SimulationEngine(
                strategy=strategy,
                cluster_config=cfg.cluster,
                dynamics_config=dynamics,
                failure_seed=cfg.seed + i * 7919,
            )
            engine.build_cluster()
            engine.load_workload(fresh_pods)
            engine.run()

            m = engine.collector.get_metrics()
            reporter.add_run(
                strategy_name=strategy.name,
                instance_id=f"test-{i}",
                seed=instance_seed,
                metrics=m,
            )
            strategy_wait_times.setdefault(strategy.name, []).extend(m.per_pod_wait_times)
            log.info(
                "  %s test-%d: success=%.1f%%  wait=%.2f  cpu=%.1f%%  rej=%d",
                strategy.name, i,
                m.scheduling_success_rate * 100,
                m.avg_wait_time,
                m.avg_cpu_utilization * 100,
                m.rejected_pods,
            )

            # Gantt for first test instance
            if gantt_dir and i == 0:
                fig = plot_gantt_from_engine(
                    engine, title=f"{strategy.name} — test-0",
                )
                save_gantt(fig, gantt_dir / f"gantt_{strategy.name}.png")
            # Resource timeline for first test instance
            if i == 0:
                resource_monitors[strategy.name] = engine.resource_monitor

    # ── Export results ───────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output_format == "csv":
        csv_path = output_dir / f"{cfg.name}_results.csv"
        reporter.export_csv(csv_path)
        log.info("Results exported to %s", csv_path)
    elif cfg.output_format == "json":
        json_path = output_dir / f"{cfg.name}_results.json"
        reporter.export_json(json_path)
        log.info("Results exported to %s", json_path)

    # ── Save GP evolved rule ──────────────────────────────────────
    rule_path = output_dir / "gp_evolved_rule.txt"
    rule_path.write_text(
        f"Best GP Evolved Rule\n"
        f"{'=' * 60}\n\n"
        f"Expression:\n  {gp_result.best_expression}\n\n"
        f"Fitness (lower is better): {gp_result.best_fitness:.6f}\n"
        f"Training time: {t_elapsed:.1f}s\n"
        f"Generations: {cfg.gp.n_generations}\n"
        f"Population: {cfg.gp.population_size}\n",
        encoding="utf-8",
    )
    log.info("GP rule saved to %s", rule_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(reporter.summary_table())
    print("=" * 70)
    print(f"\nBest GP Rule: {gp_result.best_expression}")
    print(f"Best Fitness: {gp_result.best_fitness:.6f}")
    print(f"Training Time: {t_elapsed:.1f}s")

    # ── Resource timeline export & plots ──────────────────────────────
    resource_dir = output_dir / "resource_timelines"
    for name, mon in resource_monitors.items():
        safe = name.replace("(", "_").replace(")", "").replace(" ", "_")
        mon.export_json(resource_dir / f"{safe}_timeline.json")
        fig = plot_cluster_utilization(mon, title=f"{name} — Cluster Utilization")
        save_resource_plot(fig, resource_dir / f"{safe}_cluster_util.png")
    if len(resource_monitors) > 1:
        fig = plot_cluster_comparison(resource_monitors)
        save_resource_plot(fig, resource_dir / "strategy_comparison.png")
    log.info("Resource timelines saved to %s", resource_dir)

    # ── Additional visualization plots ────────────────────────────────
    viz_dir = output_dir / "visualizations"
    for name, wt in strategy_wait_times.items():
        safe = name.replace("(", "_").replace(")", "").replace(" ", "_")
        fig = plot_wait_time_distribution(wt, title=f"{name} — Wait-Time Distribution")
        save_resource_plot(fig, viz_dir / f"{safe}_wait_dist.png")
    for name, mon in resource_monitors.items():
        safe = name.replace("(", "_").replace(")", "").replace(" ", "_")
        fig = plot_free_resources(mon, title=f"{name} — Free Resources")
        save_resource_plot(fig, viz_dir / f"{safe}_free_resources.png")
        fig = plot_utilization_variance(mon, title=f"{name} — CPU Util Variance")
        save_resource_plot(fig, viz_dir / f"{safe}_cpu_variance.png")
    log.info("Additional visualizations saved to %s", viz_dir)

    # ── GP tree visualization ─────────────────────────────────────────
    from statistical import simplify_expression
    simplified = simplify_expression(gp_result.best_expression)
    fig = plot_gp_tree(
        gp_result.best_individual,
        title="Best Evolved GP Scheduling Rule",
        simplified_expr=simplified if simplified != gp_result.best_expression else None,
    )
    save_gp_tree(fig, viz_dir / "gp_tree_best.png")
    log.info("GP tree visualization saved to %s", viz_dir / "gp_tree_best.png")

    # ── Pareto front plot (NSGA-II only) ──────────────────────────────
    if gp_result.pareto_front:
        fig = plot_pareto_front(gp_result.pareto_front)
        save_resource_plot(fig, viz_dir / "pareto_front.png")
        log.info(
            "Pareto front (%d individuals) saved to %s",
            len(gp_result.pareto_front), viz_dir / "pareto_front.png",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K8s GP Scheduler — Experiment Runner",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML experiment configuration (default: built-in defaults)",
    )
    parser.add_argument(
        "--dataset",
        choices=["dynamic", "standard"],
        default="standard",
        help="Data source: 'dynamic' loads from tmp/data/dynamic_dataset/, "
             "'standard' generates on-the-fly (default: standard)",
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default=None,
        help="Workload size preset (overrides config). "
             "small: ~1-2 min, medium: ~5 min, large: >5 min",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--gantt",
        action="store_true",
        help="Generate Gantt charts for each strategy on the first test instance",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
