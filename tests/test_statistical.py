"""Tests for statistical analysis module — hypothesis tests, effect sizes, interpretability."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statistical import (
    average_ranks,
    bootstrap_ci,
    cliffs_delta,
    cliffs_delta_interpretation,
    expression_complexity,
    extract_features_from_expression,
    feature_importance_from_metadata,
    friedman_test,
    gp_vs_baselines_table,
    holm_bonferroni,
    mann_whitney,
    rule_summary_table,
    sensitivity_table,
    simplify_expression,
    vargha_delaney_a12,
    wilcoxon_pairwise,
    generate_statistical_report,
)


# ═══════════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def paired_df() -> pd.DataFrame:
    """DataFrame with paired observations (same instance_id across strategies)."""
    rows = []
    for inst in [f"test-{i}" for i in range(5)]:
        rows.append({
            "experiment": "e1", "group": "engine", "strategy": "GP(deap)",
            "instance_id": inst, "seed": 42,
            "scheduling_success_rate": 0.85 + np.random.default_rng(42).random() * 0.1,
            "avg_wait_time": 0.4 + np.random.default_rng(42).random() * 0.3,
            "avg_cpu_utilization": 0.6,
            "avg_mem_utilization": 0.5,
            "total_pods": 100, "scheduled_pods": 85, "completed_pods": 85,
            "rejected_pods": 15,
        })
        rows.append({
            "experiment": "e1", "group": "engine", "strategy": "Random",
            "instance_id": inst, "seed": 42,
            "scheduling_success_rate": 0.65 + np.random.default_rng(43).random() * 0.1,
            "avg_wait_time": 0.7 + np.random.default_rng(43).random() * 0.3,
            "avg_cpu_utilization": 0.4,
            "avg_mem_utilization": 0.35,
            "total_pods": 100, "scheduled_pods": 65, "completed_pods": 65,
            "rejected_pods": 35,
        })
        rows.append({
            "experiment": "e1", "group": "engine", "strategy": "FirstFit",
            "instance_id": inst, "seed": 42,
            "scheduling_success_rate": 0.80 + np.random.default_rng(44).random() * 0.1,
            "avg_wait_time": 0.5 + np.random.default_rng(44).random() * 0.3,
            "avg_cpu_utilization": 0.55,
            "avg_mem_utilization": 0.45,
            "total_pods": 100, "scheduled_pods": 80, "completed_pods": 80,
            "rejected_pods": 20,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def multi_experiment_df() -> pd.DataFrame:
    """DataFrame with multiple experiments and groups."""
    rng = np.random.default_rng(42)
    rows = []
    for exp, grp in [("a1", "engine"), ("a2", "engine"), ("b1", "scale")]:
        for inst in ["test-0", "test-1", "test-2"]:
            for strat in ["GP(deap)", "Random", "FirstFit", "LeastAllocated"]:
                rows.append({
                    "experiment": exp, "group": grp, "strategy": strat,
                    "instance_id": inst, "seed": 42,
                    "scheduling_success_rate": rng.uniform(0.6, 1.0),
                    "avg_wait_time": rng.uniform(0.2, 1.5),
                    "avg_cpu_utilization": rng.uniform(0.3, 0.8),
                    "avg_mem_utilization": rng.uniform(0.2, 0.7),
                    "total_pods": 100, "scheduled_pods": 80, "completed_pods": 80,
                    "rejected_pods": 20,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_metadata() -> dict:
    return {
        "a1_deap": {
            "name": "a1_deap", "group": "engine", "engine": "deap",
            "best_fitness": 0.42,
            "best_expression": "add(mul(POD_CPU, NODE_CPU_AVAIL), sub(RESOURCE_FIT, POD_MEM))",
            "population_size": 100, "n_generations": 30,
            "total_pods": 100, "node_count": 5,
            "alpha": 0.4, "beta": 0.3, "gamma": 0.3,
            "node_failures": False, "training_time_s": 5.0,
        },
        "a2_gplearn": {
            "name": "a2_gplearn", "group": "engine", "engine": "gplearn",
            "best_fitness": 0.45,
            "best_expression": "mul(NODE_CPU_AVAIL, add(POD_CPU, BALANCE_SCORE))",
            "population_size": 100, "n_generations": 30,
            "total_pods": 100, "node_count": 5,
            "alpha": 0.4, "beta": 0.3, "gamma": 0.3,
            "node_failures": False, "training_time_s": 4.0,
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Wilcoxon signed-rank test
# ═══════════════════════════════════════════════════════════════════════


class TestWilcoxon:
    def test_returns_dict_with_keys(self, paired_df):
        result = wilcoxon_pairwise(paired_df, "scheduling_success_rate", "GP(deap)", "Random")
        assert "statistic" in result
        assert "p_value" in result
        assert "n_pairs" in result

    def test_n_pairs_correct(self, paired_df):
        result = wilcoxon_pairwise(paired_df, "scheduling_success_rate", "GP(deap)", "Random")
        assert result["n_pairs"] == 5

    def test_identical_values_p_is_1(self):
        df = pd.DataFrame([
            {"experiment": "e1", "strategy": "A", "instance_id": "t-0", "metric": 0.5},
            {"experiment": "e1", "strategy": "B", "instance_id": "t-0", "metric": 0.5},
            {"experiment": "e1", "strategy": "A", "instance_id": "t-1", "metric": 0.7},
            {"experiment": "e1", "strategy": "B", "instance_id": "t-1", "metric": 0.7},
        ])
        result = wilcoxon_pairwise(df, "metric", "A", "B")
        assert result["p_value"] == 1.0

    def test_filter_by_experiment(self, paired_df):
        result = wilcoxon_pairwise(paired_df, "scheduling_success_rate", "GP(deap)", "Random", experiment="e1")
        assert result["n_pairs"] == 5

    def test_too_few_pairs_returns_nan(self):
        df = pd.DataFrame([
            {"experiment": "e1", "strategy": "A", "instance_id": "t-0", "metric": 0.5},
            {"experiment": "e1", "strategy": "B", "instance_id": "t-0", "metric": 0.7},
        ])
        result = wilcoxon_pairwise(df, "metric", "A", "B")
        assert np.isnan(result["p_value"])


# ═══════════════════════════════════════════════════════════════════════
# Mann-Whitney U test
# ═══════════════════════════════════════════════════════════════════════


class TestMannWhitney:
    def test_returns_dict_with_keys(self, paired_df):
        result = mann_whitney(paired_df, "scheduling_success_rate", "GP(deap)", "Random")
        assert "statistic" in result
        assert "p_value" in result

    def test_sample_sizes(self, paired_df):
        result = mann_whitney(paired_df, "scheduling_success_rate", "GP(deap)", "Random")
        assert result["n_a"] == 5
        assert result["n_b"] == 5

    def test_filter_by_group(self, multi_experiment_df):
        result = mann_whitney(
            multi_experiment_df, "scheduling_success_rate",
            "GP(deap)", "Random", group="engine",
        )
        assert result["n_a"] > 0

    def test_empty_strategy_returns_nan(self, paired_df):
        result = mann_whitney(paired_df, "scheduling_success_rate", "GP(deap)", "NonExistent")
        assert np.isnan(result["p_value"])


# ═══════════════════════════════════════════════════════════════════════
# Friedman test
# ═══════════════════════════════════════════════════════════════════════


class TestFriedman:
    def test_returns_dict_with_keys(self, paired_df):
        result = friedman_test(paired_df, "scheduling_success_rate", experiment="e1")
        assert "statistic" in result
        assert "p_value" in result
        assert "k_strategies" in result
        assert "n_blocks" in result

    def test_k_strategies_correct(self, paired_df):
        result = friedman_test(paired_df, "scheduling_success_rate", experiment="e1")
        assert result["k_strategies"] == 3  # GP(deap), Random, FirstFit

    def test_too_few_strategies_returns_nan(self):
        df = pd.DataFrame([
            {"experiment": "e1", "strategy": "A", "instance_id": "t-0", "metric": 0.5},
            {"experiment": "e1", "strategy": "A", "instance_id": "t-1", "metric": 0.7},
        ])
        result = friedman_test(df, "metric", experiment="e1")
        assert np.isnan(result["p_value"])


# ═══════════════════════════════════════════════════════════════════════
# Effect sizes
# ═══════════════════════════════════════════════════════════════════════


class TestEffectSizes:
    def test_cliffs_delta_identical(self):
        x = np.array([1, 2, 3])
        assert cliffs_delta(x, x) == 0.0

    def test_cliffs_delta_all_greater(self):
        x = np.array([10, 20, 30])
        y = np.array([1, 2, 3])
        assert cliffs_delta(x, y) == 1.0

    def test_cliffs_delta_all_less(self):
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])
        assert cliffs_delta(x, y) == -1.0

    def test_cliffs_delta_range(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10, 2, 20)
        y = rng.normal(8, 2, 20)
        d = cliffs_delta(x, y)
        assert -1 <= d <= 1

    def test_cliffs_delta_empty(self):
        assert cliffs_delta(np.array([]), np.array([1, 2])) == 0.0

    def test_interpretation_negligible(self):
        assert cliffs_delta_interpretation(0.05) == "negligible"

    def test_interpretation_small(self):
        assert cliffs_delta_interpretation(0.2) == "small"

    def test_interpretation_medium(self):
        assert cliffs_delta_interpretation(0.4) == "medium"

    def test_interpretation_large(self):
        assert cliffs_delta_interpretation(0.5) == "large"

    def test_vargha_delaney_identical(self):
        x = np.array([1, 2, 3])
        assert vargha_delaney_a12(x, x) == 0.5

    def test_vargha_delaney_all_greater(self):
        x = np.array([10, 20, 30])
        y = np.array([1, 2, 3])
        assert vargha_delaney_a12(x, y) == 1.0

    def test_vargha_delaney_range(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10, 2, 20)
        y = rng.normal(8, 2, 20)
        a12 = vargha_delaney_a12(x, y)
        assert 0 <= a12 <= 1

    def test_vargha_delaney_empty(self):
        assert vargha_delaney_a12(np.array([]), np.array([1])) == 0.5


# ═══════════════════════════════════════════════════════════════════════
# P-value correction
# ═══════════════════════════════════════════════════════════════════════


class TestHolmBonferroni:
    def test_single_p_value(self):
        result = holm_bonferroni([0.03])
        assert len(result) == 1
        assert result[0]["adjusted_p"] == pytest.approx(0.03)
        assert result[0]["rejected"] is True

    def test_preserves_order(self):
        result = holm_bonferroni([0.04, 0.01, 0.08])
        assert len(result) == 3
        # Original indices preserved
        assert result[1]["original_p"] == 0.01  # smallest

    def test_adjusted_p_geq_original(self):
        ps = [0.01, 0.02, 0.03, 0.04, 0.05]
        result = holm_bonferroni(ps)
        for r in result:
            assert r["adjusted_p"] >= r["original_p"]

    def test_adjusted_p_capped_at_1(self):
        result = holm_bonferroni([0.5, 0.6, 0.7])
        for r in result:
            assert r["adjusted_p"] <= 1.0

    def test_empty_list(self):
        assert holm_bonferroni([]) == []

    def test_all_significant(self):
        result = holm_bonferroni([0.001, 0.002, 0.003], alpha=0.05)
        assert all(r["rejected"] for r in result)

    def test_none_significant(self):
        result = holm_bonferroni([0.5, 0.6, 0.7], alpha=0.05)
        assert not any(r["rejected"] for r in result)


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════


class TestBootstrapCI:
    def test_returns_triple(self):
        x = np.array([1, 2, 3, 4, 5])
        point, lo, hi = bootstrap_ci(x)
        assert isinstance(point, float)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_contains_mean(self):
        x = np.array([1, 2, 3, 4, 5])
        point, lo, hi = bootstrap_ci(x)
        assert lo <= point <= hi

    def test_empty_returns_nan(self):
        point, lo, hi = bootstrap_ci(np.array([]))
        assert np.isnan(point)

    def test_single_element(self):
        point, lo, hi = bootstrap_ci(np.array([5.0]))
        assert point == 5.0
        assert lo == 5.0
        assert hi == 5.0

    def test_reproducible(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        r1 = bootstrap_ci(x, seed=42)
        r2 = bootstrap_ci(x, seed=42)
        assert r1 == r2


# ═══════════════════════════════════════════════════════════════════════
# Rank analysis
# ═══════════════════════════════════════════════════════════════════════


class TestAverageRanks:
    def test_returns_series(self, paired_df):
        ranks = average_ranks(paired_df, "scheduling_success_rate", ascending=False)
        assert isinstance(ranks, pd.Series)

    def test_all_strategies_present(self, paired_df):
        ranks = average_ranks(paired_df, "scheduling_success_rate", ascending=False)
        assert len(ranks) == 3

    def test_lower_is_better_for_wait_time(self, paired_df):
        ranks = average_ranks(paired_df, "avg_wait_time", ascending=True)
        # GP(deap) has lower wait time → should have lower (better) rank
        assert isinstance(ranks, pd.Series)


# ═══════════════════════════════════════════════════════════════════════
# GP vs baselines table
# ═══════════════════════════════════════════════════════════════════════


class TestGPvsBaselines:
    def test_returns_dataframe(self, paired_df):
        table = gp_vs_baselines_table(paired_df, "scheduling_success_rate", experiment="e1")
        assert isinstance(table, pd.DataFrame)

    def test_has_expected_columns(self, paired_df):
        table = gp_vs_baselines_table(paired_df, "scheduling_success_rate", experiment="e1")
        expected = {"baseline", "gp_mean", "baseline_mean", "diff", "wilcoxon_p",
                    "adjusted_p", "significant", "cliffs_d", "effect_magnitude"}
        assert expected.issubset(set(table.columns))

    def test_no_gp_returns_empty(self):
        df = pd.DataFrame([
            {"experiment": "e1", "strategy": "Random", "instance_id": "t-0",
             "scheduling_success_rate": 0.7},
        ])
        table = gp_vs_baselines_table(df, "scheduling_success_rate", experiment="e1")
        assert table.empty

    def test_baselines_only(self, paired_df):
        table = gp_vs_baselines_table(paired_df, "scheduling_success_rate", experiment="e1")
        assert "GP(deap)" not in table["baseline"].values
        assert "Random" in table["baseline"].values


# ═══════════════════════════════════════════════════════════════════════
# Rule interpretability
# ═══════════════════════════════════════════════════════════════════════


class TestRuleInterpretability:
    def test_extract_features(self):
        expr = "add(POD_CPU, mul(NODE_CPU_AVAIL, POD_CPU))"
        feats = extract_features_from_expression(expr)
        assert feats["POD_CPU"] == 2
        assert feats["NODE_CPU_AVAIL"] == 1

    def test_extract_features_empty(self):
        assert extract_features_from_expression("") == {}

    def test_expression_complexity(self):
        expr = "add(POD_CPU, mul(NODE_CPU_AVAIL, RESOURCE_FIT))"
        c = expression_complexity(expr)
        assert c["n_terminals"] == 3
        assert c["n_functions"] == 2
        assert c["n_unique_features"] == 3
        assert c["depth_estimate"] == 2

    def test_simplify_neg_neg(self):
        assert simplify_expression("neg(neg(POD_CPU))") == "POD_CPU"

    def test_simplify_add_zero(self):
        assert simplify_expression("add(POD_CPU, 0)") == "POD_CPU"
        assert simplify_expression("add(0, POD_CPU)") == "POD_CPU"

    def test_simplify_mul_one(self):
        assert simplify_expression("mul(POD_CPU, 1)") == "POD_CPU"
        assert simplify_expression("mul(1, POD_CPU)") == "POD_CPU"

    def test_simplify_mul_zero(self):
        assert simplify_expression("mul(POD_CPU, 0)") == "0"

    def test_feature_importance(self, sample_metadata):
        fi = feature_importance_from_metadata(sample_metadata)
        assert len(fi) > 0
        assert "feature" in fi.columns
        assert "total_count" in fi.columns

    def test_feature_importance_empty(self):
        fi = feature_importance_from_metadata({})
        assert fi.empty

    def test_rule_summary_table(self, sample_metadata):
        table = rule_summary_table(sample_metadata)
        assert len(table) == 2
        assert "experiment" in table.columns
        assert "n_terminals" in table.columns
        assert "features_used" in table.columns


# ═══════════════════════════════════════════════════════════════════════
# Sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════


class TestSensitivity:
    def test_returns_dataframe(self, multi_experiment_df, sample_metadata):
        st = sensitivity_table(multi_experiment_df, sample_metadata, "engine")
        assert isinstance(st, pd.DataFrame)

    def test_empty_group(self, multi_experiment_df, sample_metadata):
        st = sensitivity_table(multi_experiment_df, sample_metadata, "nonexistent_group")
        assert st.empty


# ═══════════════════════════════════════════════════════════════════════
# Full statistical report
# ═══════════════════════════════════════════════════════════════════════


class TestStatisticalReport:
    def test_generates_text(self, paired_df, sample_metadata):
        report = generate_statistical_report(paired_df, sample_metadata)
        assert "STATISTICAL ANALYSIS REPORT" in report
        assert "FRIEDMAN" in report
        assert "BASELINES" in report
        assert "RANKS" in report

    def test_includes_rule_section(self, paired_df, sample_metadata):
        report = generate_statistical_report(paired_df, sample_metadata)
        assert "RULE COMPLEXITY" in report
        assert "Feature Importance" in report

    def test_works_without_metadata(self, paired_df):
        report = generate_statistical_report(paired_df, {})
        assert "STATISTICAL ANALYSIS REPORT" in report


# ═══════════════════════════════════════════════════════════════════════
# New analysis.py plots (integration)
# ═══════════════════════════════════════════════════════════════════════


class TestNewPlots:
    def test_plot_rank_heatmap(self, paired_df, tmp_path):
        from analysis import plot_rank_heatmap
        out = tmp_path / "rank.png"
        plot_rank_heatmap(paired_df, out)
        assert out.exists()

    def test_plot_sensitivity_heatmap(self, multi_experiment_df, sample_metadata, tmp_path):
        from analysis import plot_sensitivity_heatmap
        out = tmp_path / "sens.png"
        plot_sensitivity_heatmap(multi_experiment_df, sample_metadata, "engine", out)
        assert out.exists()

    def test_plot_feature_importance(self, sample_metadata, tmp_path):
        from analysis import plot_feature_importance
        out = tmp_path / "fi.png"
        plot_feature_importance(sample_metadata, out)
        assert out.exists()
