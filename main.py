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
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.schema import ClusterConfig, ExperimentConfig, NodeConfig
from gp.deap_engine import DeapGeneticEngine
from gp.fitness import FitnessEvaluator, compute_quality_score
from statistical import simplify_expression
from metrics.reporter import MetricsReporter
from metrics.resource_monitor import ResourceMonitor, ResourceSnapshot
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


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default_config.yaml"


def main() -> None:
    args = _parse_args()
    generator = PoissonWorkloadGenerator()

    # ── Logging ──────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("main")
    
    # Suppress verbose simulator debug logging (keep only high-level info)
    logging.getLogger("simulator").setLevel(logging.WARNING)
    logging.getLogger("metrics").setLevel(logging.WARNING)
    
    # If verbose, enable debug for main and GP modules only
    if args.verbose:
        log.setLevel(logging.DEBUG)
        logging.getLogger("gp").setLevel(logging.DEBUG)

    # ── Configuration ────────────────────────────────────────────────
    if args.config:
        config_path = Path(args.config)
        log.info("Loading config from %s", config_path)
        cfg = ExperimentConfig.from_yaml(config_path)
    else:
        if DEFAULT_CONFIG_PATH.exists():
            log.info("Loading default config from %s", DEFAULT_CONFIG_PATH)
            cfg = ExperimentConfig.from_yaml(DEFAULT_CONFIG_PATH)
        else:
            log.info("Default config file not found; using built-in configuration")
            cfg = ExperimentConfig()

    # ── Timestamped output directory ──────────────────────────────
    ts = datetime.now().strftime("%Y_%m_%d_%H%M")
    cfg.output_dir = str(Path("tmp") / "results" / "runs" / ts)

    log.info("Experiment: %s  seed=%d", cfg.name, cfg.seed)

    # ── Load or generate workload instances ──────────────────────────
    validation_instances: List[List[Pod]] = []
    test_seed_start = cfg.seed + cfg.num_training_instances + cfg.num_validation_instances
    if args.dataset == "dynamic":
        log.info("Loading dynamic dataset from disk...")
        meta, training_instances, test_instances = load_dataset()
        # Override config with dataset metadata
        preset = PRESETS[meta["size"]]
        cfg.num_training_instances = meta["num_training_instances"]
        cfg.num_test_instances = meta["num_test_instances"]
        cfg.num_validation_instances = 0
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

        if cfg.num_validation_instances > 0:
            log.info("Generating %d validation instances...", cfg.num_validation_instances)
            validation_instances = []
            for i in range(cfg.num_validation_instances):
                instance_seed = cfg.seed + cfg.num_training_instances + i
                pods = generator.generate(cfg.workload, seed=instance_seed)
                validation_instances.append(pods)
                log.info(
                    "  Validation instance %d: %d pods, seed=%d",
                    i, len(pods), instance_seed,
                )

        test_seed_start = cfg.seed + cfg.num_training_instances + cfg.num_validation_instances

        log.info("Generating %d test instances...", cfg.num_test_instances)
        test_instances: List[List[Pod]] = []
        for i in range(cfg.num_test_instances):
            instance_seed = test_seed_start + i
            pods = generator.generate(cfg.workload, seed=instance_seed)
            test_instances.append(pods)
            log.info(
                "  Test instance %d: %d pods, seed=%d",
                i, len(pods), instance_seed,
            )

    dynamics = cfg.dynamics if cfg.dynamics.enabled else None

    # ── Train GP engine (DEAP) ───────────────────────────────────────
    trained_runs: list[dict] = []

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

    selected_terminals = cfg.gp.selected_terminals()
    log.info(
        "Active GP terminals: %d (mandatory=%d, optional=%d)",
        len(selected_terminals),
        len(cfg.gp.terminal_mandatory),
        len(cfg.gp.terminal_optional_enabled),
    )

    for engine_name in ["deap"]:
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
        )
        use_mo = cfg.gp.multi_objective

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
            n_workers=cfg.gp.n_workers,
            aggregation_mode=cfg.gp.fitness_aggregation,
            std_penalty=cfg.gp.fitness_std_penalty,
        )

        objective_mode = "multi-objective" if use_mo else "single-objective"
        log.info(
            "Starting GP training [%s]: pop=%d, gen=%d, %s",
            engine_name,
            cfg.gp.population_size,
            cfg.gp.n_generations,
            objective_mode,
        )
        t_start = time.perf_counter()
        fitness_fn = fitness_evaluator.evaluate_objectives if use_mo else fitness_evaluator
        gp_result = gp_engine.train(
            fitness_function=fitness_fn,
            seed=cfg.seed,
        )
        t_elapsed = time.perf_counter() - t_start

        validation_selection = None
        if (
            engine_name == "deap"
            and validation_instances
            and cfg.gp.validation_hof_size > 0
            and gp_result.hall_of_fame
        ):
            validation_selection = _select_deap_champion_on_validation(
                gp_engine=gp_engine,
                gp_result=gp_result,
                validation_instances=validation_instances,
                cfg=cfg,
                dynamics=dynamics,
                hof_size=cfg.gp.validation_hof_size,
            )
            log.info(
                "Validation champion selected [%s]: rank=%d robust=%.6f mean=%.6f std=%.6f",
                engine_name,
                validation_selection["selected_rank"],
                validation_selection["selected_robust_score"],
                validation_selection["selected_mean_score"],
                validation_selection["selected_std"],
            )

        log.info("Training complete [%s] in %.1f seconds", engine_name, t_elapsed)
        log.info("Best quality [%s]: %.6f", engine_name, gp_result.best_fitness)
        log.info("Best expression [%s]: %s", engine_name, gp_result.best_expression)

        # ── Export convergence.json ──────────────────────────────────
        convergence_path = Path(cfg.output_dir) / f"convergence_{engine_name}.json"
        gp_result.export_convergence_json(convergence_path)
        log.info("Convergence log exported to %s", convergence_path)

        trained_runs.append(
            {
                "engine_name": engine_name,
                "engine": gp_engine,
                "result": gp_result,
                "train_time_s": t_elapsed,
                "validation_selection": validation_selection,
            }
        )

    # ── Evaluate on test set ─────────────────────────────────────────
    log.info("Evaluating best individual on %d test instances...", len(test_instances))
    reporter = MetricsReporter()
    gantt_dir = Path(cfg.output_dir) / "gantt" if args.gantt else None
    resource_monitors: dict[str, list[ResourceMonitor]] = {}
    strategy_wait_times = {}  # {strategy_name: List[float]}

    # GP evaluation on test set (single champion per engine)
    for run in trained_runs:
        gp_engine = run["engine"]
        gp_result = run["result"]
        gp_name = f"GP({gp_engine.name})"

        # Build list of candidates: best only
        candidates_to_eval = [
            (gp_result.best_individual, f"{gp_name}", "best"),
        ]

        for candidate, strategy_label, candidate_type in candidates_to_eval:
            log.info(
                "Evaluating %s on %d test instances (%s)...",
                strategy_label, len(test_instances), candidate_type
            )
            gp_strategy = GPSchedulingStrategy(gp_engine, candidate)
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

                m = sim.collector.get_metrics()
                reporter.add_run(
                    strategy_name=strategy_label,
                    instance_id=f"test-{i}",
                    seed=instance_seed,
                    metrics=m,
                    quality_score=compute_quality_score(m, cfg.fitness),
                )
                strategy_wait_times.setdefault(strategy_label, []).extend(m.per_pod_wait_times)
                log.info(
                    "  %s test-%d: success=%.1f%%  wait=%.2f  cpu=%.1f%%  rej=%d churn=%.4f",
                    strategy_label,
                    i,
                    m.scheduling_success_rate * 100,
                    m.avg_wait_time,
                    m.avg_cpu_utilization * 100,
                    m.rejected_pods,
                    m.churn_rate,
                )

                if gantt_dir and i == 0:
                    fig = plot_gantt_from_engine(
                        sim, title=f"{strategy_label} — test-0",
                    )
                    save_gantt(fig, gantt_dir / f"gantt_{strategy_label.replace('(', '_').replace(')', '')}.png")
                resource_monitors.setdefault(strategy_label, []).append(sim.resource_monitor)

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
            instance_seed = test_seed_start + i
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
                quality_score=compute_quality_score(m, cfg.fitness),
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
            resource_monitors.setdefault(strategy.name, []).append(engine.resource_monitor)

    # Aggregate resource timelines over all test instances per strategy.
    # This avoids comparing strategies based on a single potentially noisy instance.
    aggregated_monitors: dict[str, ResourceMonitor] = {
        name: _aggregate_monitors(monitors)
        for name, monitors in resource_monitors.items()
        if monitors
    }

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

    # ── Save GP evolved rule(s) ───────────────────────────────────
    primary_run = next(
        (r for r in trained_runs if r["engine_name"] == cfg.gp.engine),
        trained_runs[0],
    )

    # Compute per-engine mean quality on TEST instances for transparency.
    records = getattr(reporter, "_records", [])
    test_quality_by_engine: dict[str, float] = {}
    for run in trained_runs:
        gp_engine = run["engine"]
        strat = f"GP({gp_engine.name})"
        vals = [
            float(r.get("quality_score", 0.0))
            for r in records
            if r.get("strategy") == strat and r.get("quality_score") is not None
        ]
        if vals:
            test_quality_by_engine[gp_engine.name] = sum(vals) / len(vals)

    for run in trained_runs:
        gp_engine = run["engine"]
        gp_result = run["result"]
        train_time = run["train_time_s"]
        engine_name = gp_engine.name
        test_q = test_quality_by_engine.get(engine_name)
        validation_sel = run.get("validation_selection")
        validation_line = ""
        if validation_sel is not None:
            validation_line = (
                "Validation champion selection:\n"
                f"  rank in HOF: {validation_sel['selected_rank']} of {validation_sel['hof_evaluated']}\n"
                f"  robust score: {validation_sel['selected_robust_score']:.6f}\n"
                f"  mean score: {validation_sel['selected_mean_score']:.6f}\n"
                f"  std: {validation_sel['selected_std']:.6f}\n"
            )
        test_q_line = (
            f"Mean test quality (from table below): {test_q:.6f}\n"
            if test_q is not None
            else "Mean test quality (from table below): n/a\n"
        )
        aggregation_line = (
            f"Fitness aggregation: {cfg.gp.fitness_aggregation} (lambda={cfg.gp.fitness_std_penalty:.3f})\n"
            if cfg.gp.fitness_aggregation == "mean_minus_std"
            else f"Fitness aggregation: {cfg.gp.fitness_aggregation}\n"
        )
        rule_text = (
            f"Best GP Evolved Rule ({engine_name})\n"
            f"{'=' * 60}\n\n"
            f"Expression:\n  {gp_result.best_expression}\n\n"
            f"Training quality (higher is better): {gp_result.best_fitness:.6f}\n"
            f"(computed during training over training instances)\n"
            f"Validation instances: {cfg.num_validation_instances}\n"
            f"{aggregation_line}"
            f"{validation_line}"
            f"{test_q_line}"
            f"Training time: {train_time:.1f}s\n"
            f"Generations: {cfg.gp.n_generations}\n"
            f"Population: {cfg.gp.population_size}\n"
        )
        rule_path = output_dir / f"gp_evolved_rule_{engine_name}.txt"
        rule_path.write_text(rule_text, encoding="utf-8")
        log.info("GP rule saved to %s", rule_path)

    # Backward-compatible single-file alias for the primary configured engine
    primary_alias_path = output_dir / "gp_evolved_rule.txt"
    primary_alias_path.write_text(
        (output_dir / f"gp_evolved_rule_{primary_run['engine'].name}.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(reporter.summary_table())
    print("=" * 70)

    # ── Churn vs Quality Analysis ─────────────────────────────────
    log.info("Generating churn vs quality analysis...")
    _export_churn_analysis(
        records=getattr(reporter, "_records", []),
        output_dir=output_dir,
        cfg=cfg,
    )

    print("\nGP ENGINES")
    for run in trained_runs:
        gp_engine = run["engine"]
        gp_result = run["result"]
        test_q = test_quality_by_engine.get(gp_engine.name)
        test_q_str = f", test_mean_quality={test_q:.6f}" if test_q is not None else ""
        validation_sel = run.get("validation_selection")
        val_str = (
            f", val_robust={validation_sel['selected_robust_score']:.6f}"
            if validation_sel is not None
            else ""
        )
        print(
            f"- {gp_engine.name}: training_quality={gp_result.best_fitness:.6f}"
            f"{val_str}{test_q_str}, train={run['train_time_s']:.1f}s"
        )
    print(f"\nBest GP Rule ({primary_run['engine'].name}): {primary_run['result'].best_expression}")
    print(f"Best Quality Score ({primary_run['engine'].name}): {primary_run['result'].best_fitness:.6f}")

    # ── Resource timeline export & plots ──────────────────────────────
    resource_dir = output_dir / "resource_timelines"
    for name, mon in aggregated_monitors.items():
        safe = name.replace("(", "_").replace(")", "").replace(" ", "_")
        mon.export_json(resource_dir / f"{safe}_timeline.json")
        fig = plot_cluster_utilization(mon, title=f"{name} — Cluster Utilization (mean over test instances)")
        save_resource_plot(fig, resource_dir / f"{safe}_cluster_util.png")
    if len(aggregated_monitors) > 1:
        fig = plot_cluster_comparison(aggregated_monitors)
        save_resource_plot(fig, resource_dir / "strategy_comparison.png")
    log.info("Resource timelines saved to %s", resource_dir)

    # ── Additional visualization plots ────────────────────────────────
    viz_dir = output_dir / "visualizations"
    for name, wt in strategy_wait_times.items():
        safe = name.replace("(", "_").replace(")", "").replace(" ", "_")
        fig = plot_wait_time_distribution(wt, title=f"{name} — Wait-Time Distribution")
        save_resource_plot(fig, viz_dir / f"{safe}_wait_dist.png")
    for name, mon in aggregated_monitors.items():
        safe = name.replace("(", "_").replace(")", "").replace(" ", "_")
        fig = plot_free_resources(mon, title=f"{name} — Free Resources (mean over test instances)")
        save_resource_plot(fig, viz_dir / f"{safe}_free_resources.png")
        fig = plot_utilization_variance(mon, title=f"{name} — CPU Util Variance (mean over test instances)")
        save_resource_plot(fig, viz_dir / f"{safe}_cpu_variance.png")
    log.info("Additional visualizations saved to %s", viz_dir)

    # ── GP tree visualization ─────────────────────────────────────────
    deap_run = next((r for r in trained_runs if r["engine_name"] == "deap"), None)
    if deap_run is not None:
        deap_result = deap_run["result"]
        simplified = simplify_expression(deap_result.best_expression)
        fig = plot_gp_tree(
            deap_result.best_individual,
            title="Best Evolved GP Scheduling Rule (deap)",
            simplified_expr=simplified if simplified != deap_result.best_expression else None,
        )
        save_gp_tree(fig, viz_dir / "gp_tree_best.png")
        log.info("GP tree visualization saved to %s", viz_dir / "gp_tree_best.png")
    else:
        log.info("Skipping GP tree visualization (available only for deap engine)")

    # ── Pareto front plot (NSGA-II only) ──────────────────────────────
    if deap_run is not None and deap_run["result"].pareto_front:
        fig = plot_pareto_front(deap_run["result"].pareto_front)
        save_resource_plot(fig, viz_dir / "pareto_front.png")
        log.info(
            "Pareto front (%d individuals) saved to %s",
            len(deap_run["result"].pareto_front), viz_dir / "pareto_front.png",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K8s GP Scheduler — Experiment Runner",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML experiment configuration (default: config/default_config.yaml)",
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


def _aggregate_monitors(monitors: list[ResourceMonitor]) -> ResourceMonitor:
    """Average multiple resource timelines on a normalized time axis."""
    valid = [m for m in monitors if m.get_timestamps()]
    if not valid:
        return ResourceMonitor()
    if len(valid) == 1:
        return valid[0]

    point_count = max(30, int(sum(len(m.get_timestamps()) for m in valid) / len(valid)))
    point_count = min(point_count, 240)
    grid = np.linspace(0.0, 1.0, point_count)

    cpu_series = []
    mem_series = []
    pending_series = []
    cpu_free_series = []
    mem_free_series = []
    variance_series = []
    completed_series = []
    durations = []

    for mon in valid:
        ts = np.asarray(mon.get_timestamps(), dtype=float)
        if ts.size == 0:
            continue
        duration = float(ts[-1] - ts[0]) if ts.size > 1 else 0.0
        durations.append(duration)
        denom = duration if duration > 1e-12 else 1.0
        t_norm = (ts - ts[0]) / denom

        def interp(values: list[float]) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            if arr.size == 1:
                return np.full_like(grid, arr[0], dtype=float)
            return np.interp(grid, t_norm, arr)

        cpu_series.append(interp(mon.get_cluster_cpu_series()))
        mem_series.append(interp(mon.get_cluster_mem_series()))
        pending_series.append(interp([float(v) for v in mon.get_pending_series()]))
        cpu_free_series.append(interp(mon.get_cluster_cpu_free_series()))
        mem_free_series.append(interp(mon.get_cluster_mem_free_series()))
        variance_series.append(interp(mon.get_cpu_util_variance_series()))
        completed_series.append(interp([float(v) for v in mon.get_completed_series()]))

    avg_duration = float(np.mean(durations)) if durations else 0.0
    out = ResourceMonitor()

    cpu_mean = np.mean(np.vstack(cpu_series), axis=0)
    mem_mean = np.mean(np.vstack(mem_series), axis=0)
    pending_mean = np.mean(np.vstack(pending_series), axis=0)
    cpu_free_mean = np.mean(np.vstack(cpu_free_series), axis=0)
    mem_free_mean = np.mean(np.vstack(mem_free_series), axis=0)
    variance_mean = np.mean(np.vstack(variance_series), axis=0)
    completed_mean = np.mean(np.vstack(completed_series), axis=0)

    for idx, f in enumerate(grid):
        out.snapshots.append(
            ResourceSnapshot(
                timestamp=float(f * avg_duration),
                cluster_cpu_util=float(cpu_mean[idx]),
                cluster_mem_util=float(mem_mean[idx]),
                cluster_cpu_free=float(cpu_free_mean[idx]),
                cluster_mem_free=float(mem_free_mean[idx]),
                cpu_util_variance=float(variance_mean[idx]),
                pending_count=max(0, int(round(pending_mean[idx]))),
                completed_count=max(0, int(round(completed_mean[idx]))),
            )
        )

    return out


def _export_churn_analysis(
    *,
    records: List[Dict[str, Any]],
    output_dir: Path,
    cfg: ExperimentConfig,
) -> None:
    """Export churn vs quality trade-off analysis."""
    if not records:
        return
    
    # Group by strategy
    by_strategy: dict[str, list[dict]] = {}
    for rec in records:
        strat = rec.get("strategy", "unknown")
        if strat not in by_strategy:
            by_strategy[strat] = []
        by_strategy[strat].append(rec)
    
    # Compute per-strategy stats
    analysis = {}
    for strat, strat_records in by_strategy.items():
        churn_rates = [
            float(r.get("churn_rate", 0.0))
            for r in strat_records if "churn_rate" in r
        ]
        quality_scores = [
            float(r.get("quality_score", 0.0))
            for r in strat_records if "quality_score" in r
        ]
        
        if churn_rates and quality_scores:
            analysis[strat] = {
                "mean_churn": float(np.mean(churn_rates)),
                "std_churn": float(np.std(churn_rates)),
                "mean_quality": float(np.mean(quality_scores)),
                "std_quality": float(np.std(quality_scores)),
                "num_runs": len(strat_records),
            }
    
    # Export to JSON
    analysis_path = output_dir / "churn_quality_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2)
    
    # Log summary
    if analysis:
        print("\n" + "=" * 70)
        print("CHURN vs QUALITY ANALYSIS")
        print("=" * 70)
        print(f"{'Strategy':<30} {'Mean Churn':>12} {'Mean Quality':>14}")
        print("-" * 70)
        for strat in sorted(analysis.keys()):
            stats = analysis[strat]
            print(
                f"{strat:<30} {stats['mean_churn']:>12.4f} {stats['mean_quality']:>14.6f}"
            )
        print("=" * 70)


def _select_deap_champion_on_validation(
    *,
    gp_engine: DeapGeneticEngine,
    gp_result: Any,
    validation_instances: List[List[Pod]],
    cfg: ExperimentConfig,
    dynamics: Optional[Any],
    hof_size: int,
) -> Dict[str, float]:
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
    best_mean = 0.0
    best_std = 0.0

    for idx, candidate in enumerate(candidates):
        metrics_list = evaluator.evaluate_on_instances(candidate, validation_instances)
        scores = [compute_quality_score(m, cfg.fitness) for m in metrics_list]
        if not scores:
            continue
        mean_score = float(sum(scores) / len(scores))
        std_score = float(np.std(scores))
        if cfg.gp.fitness_aggregation == "mean_minus_std":
            robust = mean_score - cfg.gp.fitness_std_penalty * std_score
        else:
            robust = mean_score
        if robust > best_robust:
            best_idx = idx
            best_robust = robust
            best_mean = mean_score
            best_std = std_score

    selected = candidates[best_idx]
    gp_result.best_individual = selected
    gp_result.best_expression = gp_engine.get_expression_string(selected)

    return {
        "selected_rank": best_idx + 1,
        "hof_evaluated": len(candidates),
        "selected_robust_score": best_robust,
        "selected_mean_score": best_mean,
        "selected_std": best_std,
    }


if __name__ == "__main__":
    main()
