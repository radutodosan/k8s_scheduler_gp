"""Tests for experiment framework — run_experiments.py and analysis.py."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.schema import (
    ClusterConfig,
    DynamicsConfig,
    ExperimentConfig,
    FitnessWeights,
    GPConfig,
    NodeConfig,
    WorkloadConfig,
)
from metrics.reporter import MetricsReporter
from run_experiments import (
    ExperimentDef,
    ExperimentResult,
    _base_config,
    _save_combined_csv,
    _save_summary,
    define_experiments,
)


# ═══════════════════════════════════════════════════════════════════════
# _base_config
# ═══════════════════════════════════════════════════════════════════════


class TestBaseConfig:
    def test_returns_experiment_config(self):
        cfg = _base_config(name="t1")
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name == "t1"

    def test_default_values(self):
        cfg = _base_config()
        assert cfg.seed == 42
        assert cfg.gp.engine == "deap"
        assert cfg.gp.population_size == 100
        assert cfg.gp.n_generations == 30
        assert cfg.workload.total_pods == 100
        assert cfg.fitness.alpha_wait_time == 0.4

    def test_custom_gp_params(self):
        cfg = _base_config(engine="gplearn", pop=200, gen=50, depth=10)
        assert cfg.gp.engine == "gplearn"
        assert cfg.gp.population_size == 200
        assert cfg.gp.n_generations == 50
        assert cfg.gp.max_tree_depth == 10

    def test_custom_fitness_weights(self):
        cfg = _base_config(alpha=0.7, beta=0.15, gamma=0.15)
        assert cfg.fitness.alpha_wait_time == 0.7
        assert cfg.fitness.beta_resource_waste == 0.15
        assert cfg.fitness.gamma_failed_pods == 0.15

    def test_custom_cluster(self):
        cfg = _base_config(nodes=10, cpu=16.0, mem=32768.0)
        assert len(cfg.cluster.node_templates) == 1
        t = cfg.cluster.node_templates[0]
        assert t.count == 10
        assert t.cpu_capacity == 16.0
        assert t.mem_capacity == 32768.0

    def test_dynamics_defaults(self):
        cfg = _base_config()
        assert cfg.dynamics.failure_mode == "off"
        assert cfg.dynamics.enabled is False

    def test_dynamics_with_failures(self):
        cfg = _base_config(failure_mode="reschedule", failure_rate=2)
        assert cfg.dynamics.failure_mode == "reschedule"
        assert cfg.dynamics.failure_rate == 2
        assert cfg.dynamics.enabled is True

    def test_output_dir_matches_name(self):
        cfg = _base_config(name="exp_test")
        assert "exp_test" in cfg.output_dir


# ═══════════════════════════════════════════════════════════════════════
# define_experiments
# ═══════════════════════════════════════════════════════════════════════


class TestDefineExperiments:
    def test_returns_list_of_experiment_defs(self):
        exps = define_experiments()
        assert isinstance(exps, list)
        assert all(isinstance(e, ExperimentDef) for e in exps)
        assert len(exps) == 22

    def test_all_names_unique(self):
        exps = define_experiments()
        names = [e.name for e in exps]
        assert len(names) == len(set(names))

    def test_groups(self):
        exps = define_experiments()
        groups = {e.group for e in exps}
        assert groups == {"engine", "scale", "fitness_weights", "gp_params", "dynamics", "nsga2", "profile"}

    def test_engine_group_has_deap_and_gplearn(self):
        exps = define_experiments()
        engine_exps = [e for e in exps if e.group == "engine"]
        engines = {e.config.gp.engine for e in engine_exps}
        assert engines == {"deap", "gplearn"}

    def test_scale_group_has_three_sizes(self):
        exps = define_experiments()
        scale_exps = [e for e in exps if e.group == "scale"]
        assert len(scale_exps) == 3
        pod_counts = sorted(e.config.workload.total_pods for e in scale_exps)
        assert pod_counts[0] < pod_counts[1] < pod_counts[2]

    def test_fitness_weights_group_has_four_configs(self):
        exps = define_experiments()
        fw_exps = [e for e in exps if e.group == "fitness_weights"]
        assert len(fw_exps) == 4

    def test_dynamics_group_has_three_modes(self):
        exps = define_experiments()
        dyn_exps = [e for e in exps if e.group == "dynamics"]
        assert len(dyn_exps) == 3
        modes = {e.config.dynamics.failure_mode for e in dyn_exps}
        assert modes == {"off", "reschedule", "kill"}

    def test_quick_mode_smaller_params(self):
        full = define_experiments(quick=False)
        quick = define_experiments(quick=True)

        for qe, fe in zip(quick, full):
            assert qe.config.gp.population_size <= fe.config.gp.population_size
            assert qe.config.gp.n_generations <= fe.config.gp.n_generations
            assert qe.config.workload.total_pods <= fe.config.workload.total_pods

    def test_quick_mode_training_instances(self):
        quick = define_experiments(quick=True)
        for e in quick:
            assert e.config.num_training_instances == 2
            assert e.config.num_test_instances == 2

    def test_all_experiments_have_descriptions(self):
        exps = define_experiments()
        for e in exps:
            assert e.description, f"{e.name} has no description"


# ═══════════════════════════════════════════════════════════════════════
# ExperimentResult & saving helpers
# ═══════════════════════════════════════════════════════════════════════


def _dummy_reporter() -> MetricsReporter:
    """Build a MetricsReporter with a few dummy records."""
    rpt = MetricsReporter()
    from metrics.collector import SchedulingMetrics

    rpt.add_run(
        strategy_name="GP(deap)",
        instance_id="test-0",
        seed=42,
        metrics=SchedulingMetrics(
            total_pods=20, scheduled_pods=18, completed_pods=18,
            rejected_pods=2, total_wait_time=9.0,
            cpu_util_samples=[0.5], mem_util_samples=[0.4],
        ),
    )
    rpt.add_run(
        strategy_name="Random",
        instance_id="test-0",
        seed=42,
        metrics=SchedulingMetrics(
            total_pods=20, scheduled_pods=14, completed_pods=14,
            rejected_pods=6, total_wait_time=14.0,
            cpu_util_samples=[0.3], mem_util_samples=[0.25],
        ),
    )
    return rpt


def _dummy_result(name="exp1", group="test") -> ExperimentResult:
    return ExperimentResult(
        name=name,
        group=group,
        training_time=1.23,
        best_fitness=0.456,
        best_expression="add(POD_CPU, NODE_FREE_MEM)",
        convergence_log=[
            {"gen": 0, "min": 0.9, "avg": 1.2, "max": 1.5},
            {"gen": 1, "min": 0.7, "avg": 1.0, "max": 1.3},
        ],
        reporter=_dummy_reporter(),
    )


class TestSaveCombinedCSV:
    def test_creates_csv_file(self, tmp_path):
        path = tmp_path / "combined.csv"
        _save_combined_csv([_dummy_result()], path)
        assert path.exists()

    def test_csv_has_experiment_column(self, tmp_path):
        path = tmp_path / "combined.csv"
        _save_combined_csv([_dummy_result(name="myexp", group="mygrp")], path)
        df = pd.read_csv(path)
        assert "experiment" in df.columns
        assert "group" in df.columns
        assert (df["experiment"] == "myexp").all()
        assert (df["group"] == "mygrp").all()

    def test_csv_has_all_columns(self, tmp_path):
        path = tmp_path / "combined.csv"
        _save_combined_csv([_dummy_result()], path)
        df = pd.read_csv(path)
        expected = {
            "experiment", "group", "strategy", "instance_id", "seed",
            "total_pods", "scheduled_pods", "completed_pods", "rejected_pods",
            "scheduling_success_rate", "avg_wait_time",
            "avg_cpu_utilization", "avg_mem_utilization",
        }
        assert expected.issubset(set(df.columns))

    def test_multiple_experiments(self, tmp_path):
        path = tmp_path / "combined.csv"
        results = [
            _dummy_result(name="e1", group="g1"),
            _dummy_result(name="e2", group="g2"),
        ]
        _save_combined_csv(results, path)
        df = pd.read_csv(path)
        assert df["experiment"].nunique() == 2

    def test_empty_results(self, tmp_path):
        path = tmp_path / "combined.csv"
        _save_combined_csv([], path)
        assert not path.exists()


class TestSaveSummary:
    def test_creates_summary_file(self, tmp_path):
        path = tmp_path / "summary.txt"
        _save_summary([_dummy_result()], path)
        assert path.exists()

    def test_summary_contains_experiment_name(self, tmp_path):
        path = tmp_path / "summary.txt"
        _save_summary([_dummy_result(name="myexp")], path)
        text = path.read_text(encoding="utf-8")
        assert "myexp" in text

    def test_summary_contains_fitness(self, tmp_path):
        path = tmp_path / "summary.txt"
        r = _dummy_result()
        _save_summary([r], path)
        text = path.read_text(encoding="utf-8")
        assert "0.456" in text


# ═══════════════════════════════════════════════════════════════════════
# Analysis module
# ═══════════════════════════════════════════════════════════════════════


class TestAnalysis:
    """Tests for analysis.py functions."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"experiment": "e1", "group": "g1", "strategy": "GP(deap)",
             "instance_id": "t-0", "seed": 42, "total_pods": 20,
             "scheduled_pods": 18, "completed_pods": 18, "rejected_pods": 2,
             "scheduling_success_rate": 0.9, "avg_wait_time": 0.5,
             "avg_cpu_utilization": 0.6, "avg_mem_utilization": 0.4},
            {"experiment": "e1", "group": "g1", "strategy": "Random",
             "instance_id": "t-0", "seed": 42, "total_pods": 20,
             "scheduled_pods": 14, "completed_pods": 14, "rejected_pods": 6,
             "scheduling_success_rate": 0.7, "avg_wait_time": 1.0,
             "avg_cpu_utilization": 0.3, "avg_mem_utilization": 0.25},
            {"experiment": "e2", "group": "g2", "strategy": "GP(deap)",
             "instance_id": "t-0", "seed": 42, "total_pods": 50,
             "scheduled_pods": 45, "completed_pods": 45, "rejected_pods": 5,
             "scheduling_success_rate": 0.9, "avg_wait_time": 0.6,
             "avg_cpu_utilization": 0.55, "avg_mem_utilization": 0.45},
        ])

    @pytest.fixture
    def sample_convergence_dir(self, tmp_path) -> Path:
        exp_dir = tmp_path / "e1"
        exp_dir.mkdir()
        log = [
            {"gen": 0, "min": 0.8, "avg": 1.0, "max": 1.2},
            {"gen": 1, "min": 0.6, "avg": 0.8, "max": 1.0},
        ]
        with open(exp_dir / "convergence.json", "w") as f:
            json.dump(log, f)
        return tmp_path

    def test_load_combined_results(self, tmp_path, sample_df):
        from analysis import load_combined_results
        path = tmp_path / "combined.csv"
        sample_df.to_csv(path, index=False)
        loaded = load_combined_results(path)
        assert len(loaded) == len(sample_df)
        assert list(loaded.columns) == list(sample_df.columns)

    def test_load_convergence_logs(self, sample_convergence_dir):
        from analysis import load_convergence_logs
        logs = load_convergence_logs(sample_convergence_dir)
        assert "e1" in logs
        assert len(logs["e1"]) == 2

    def test_strategy_comparison_table(self, sample_df):
        from analysis import strategy_comparison_table
        table = strategy_comparison_table(sample_df, experiment="e1")
        assert "GP(deap)" in table.index
        assert "Random" in table.index
        assert table.loc["GP(deap)", "sched_rate"] == pytest.approx(0.9)

    def test_strategy_comparison_by_group(self, sample_df):
        from analysis import strategy_comparison_table
        table = strategy_comparison_table(sample_df, group="g1")
        assert len(table) == 2  # GP(deap) and Random in g1

    def test_cross_experiment_table(self, sample_df):
        from analysis import cross_experiment_table
        table = cross_experiment_table(sample_df)
        assert len(table) == 2  # GP(deap) in e1 and e2
        assert "experiment" in table.columns
        assert "sched_rate" in table.columns

    def test_cross_experiment_table_empty(self):
        from analysis import cross_experiment_table
        df = pd.DataFrame([{
            "experiment": "e1", "group": "g1", "strategy": "Random",
            "scheduling_success_rate": 0.7, "avg_wait_time": 1.0,
            "avg_cpu_utilization": 0.3, "rejected_pods": 5,
            "instance_id": "t-0",
        }])
        table = cross_experiment_table(df)
        assert table.empty

    def test_format_comparison_text(self, sample_df):
        from analysis import strategy_comparison_table, format_comparison_text
        table = strategy_comparison_table(sample_df, experiment="e1")
        text = format_comparison_text(table, title="Test")
        assert "Test" in text
        assert "GP(deap)" in text
        assert "Random" in text

    def test_compute_statistics(self, sample_df):
        from analysis import compute_statistics
        stats = compute_statistics(sample_df)
        assert "scheduling_success_rate_mean" in stats.columns
        assert "avg_wait_time_std" in stats.columns

    def test_plot_convergence(self, sample_convergence_dir, tmp_path):
        from analysis import load_convergence_logs, plot_convergence
        logs = load_convergence_logs(sample_convergence_dir)
        out = tmp_path / "conv.png"
        plot_convergence(logs, out)
        assert out.exists()

    def test_plot_strategy_boxes(self, sample_df, tmp_path):
        from analysis import plot_strategy_boxes
        out = tmp_path / "box.png"
        plot_strategy_boxes(sample_df, "scheduling_success_rate", out, experiment="e1")
        assert out.exists()

    def test_generate_report(self, tmp_path, sample_df):
        from analysis import generate_report

        # Set up input directory mimicking experiment output
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        sample_df.to_csv(input_dir / "combined_results.csv", index=False)

        exp1_dir = input_dir / "e1"
        exp1_dir.mkdir()
        log = [{"gen": 0, "min": 0.8, "avg": 1.0, "max": 1.2}]
        with open(exp1_dir / "convergence.json", "w") as f:
            json.dump(log, f)
        meta = {"name": "e1", "engine": "deap", "best_fitness": 0.5,
                "training_time_s": 1.0, "total_pods": 20, "node_count": 3}
        with open(exp1_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

        output_dir = tmp_path / "output"
        generate_report(input_dir, output_dir, plots=True)

        assert (output_dir / "analysis_report.txt").exists()
        report = (output_dir / "analysis_report.txt").read_text()
        assert "EXPERIMENT ANALYSIS REPORT" in report

    def test_generate_report_no_combined_csv(self, tmp_path, capsys):
        from analysis import generate_report
        empty_input = tmp_path / "empty"
        empty_input.mkdir()
        generate_report(empty_input, tmp_path / "out")
        captured = capsys.readouterr()
        assert "not found" in captured.out
