"""Analysis module — post-processing of experiment results.

Reads combined CSV from run_experiments.py and generates:
  - Comparison tables (strategy × metric, grouped by experiment)
  - Convergence plots (generation vs fitness per experiment)
  - Box plots (metric distribution per strategy)
  - Scaling plots (metric vs problem size)
  - Statistical comparison summary

Usage:
    py analysis.py --input tmp/results/experiments
    py analysis.py --input tmp/results/experiments --plots
    py analysis.py --input tmp/results/experiments --tables-only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════


def load_combined_results(path: Path) -> pd.DataFrame:
    """Load the combined_results.csv from an experiment sweep."""
    df = pd.read_csv(path)
    return df


def load_convergence_logs(experiment_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load convergence JSONs from each experiment subdirectory."""
    logs: Dict[str, pd.DataFrame] = {}
    for conv_file in sorted(experiment_dir.glob("*/convergence.json")):
        exp_name = conv_file.parent.name
        with open(conv_file, encoding="utf-8") as f:
            data = json.load(f)
        if data:
            logs[exp_name] = pd.DataFrame(data)
    return logs


def load_metadata(experiment_dir: Path) -> Dict[str, dict]:
    """Load metadata.json from each experiment subdirectory."""
    metadata: Dict[str, dict] = {}
    for meta_file in sorted(experiment_dir.glob("*/metadata.json")):
        exp_name = meta_file.parent.name
        with open(meta_file, encoding="utf-8") as f:
            metadata[exp_name] = json.load(f)
    return metadata


# ═══════════════════════════════════════════════════════════════════════
# Comparison tables
# ═══════════════════════════════════════════════════════════════════════


def strategy_comparison_table(
    df: pd.DataFrame,
    experiment: Optional[str] = None,
    group: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate metrics by strategy (mean across test instances).

    Returns a DataFrame with one row per strategy and columns:
    sched_rate, avg_wait, cpu_util, mem_util, rejected.
    """
    subset = df.copy()
    if experiment:
        subset = subset[subset["experiment"] == experiment]
    if group:
        subset = subset[subset["group"] == group]

    agg = subset.groupby("strategy").agg(
        sched_rate=("scheduling_success_rate", "mean"),
        avg_wait=("avg_wait_time", "mean"),
        cpu_util=("avg_cpu_utilization", "mean"),
        mem_util=("avg_mem_utilization", "mean"),
        rejected=("rejected_pods", "sum"),
        n_instances=("instance_id", "count"),
    ).round(4)

    return agg.sort_values("sched_rate", ascending=False)


def cross_experiment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compare GP performance across experiments.

    Returns one row per experiment with GP metrics (mean over test instances).
    """
    gp_rows = df[df["strategy"].str.startswith("GP(")]
    if gp_rows.empty:
        return pd.DataFrame()

    agg = gp_rows.groupby(["experiment", "group", "strategy"]).agg(
        sched_rate=("scheduling_success_rate", "mean"),
        avg_wait=("avg_wait_time", "mean"),
        cpu_util=("avg_cpu_utilization", "mean"),
        rejected=("rejected_pods", "sum"),
    ).round(4)

    return agg.reset_index().sort_values(["group", "experiment"])


def format_comparison_text(
    df: pd.DataFrame,
    title: str = "Strategy Comparison",
) -> str:
    """Format a comparison table as aligned text."""
    lines = [
        title,
        "=" * len(title),
        "",
        f"{'Strategy':<22} {'Sched%':>8} {'AvgWait':>9} {'CPU%':>7} "
        f"{'MEM%':>7} {'Rejected':>9}",
        "-" * 70,
    ]
    for strategy, row in df.iterrows():
        lines.append(
            f"{strategy:<22} {row['sched_rate']:>7.1%} "
            f"{row['avg_wait']:>9.3f} "
            f"{row['cpu_util']:>6.1%} "
            f"{row['mem_util']:>6.1%} "
            f"{int(row['rejected']):>9}"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════


def plot_convergence(
    logs: Dict[str, pd.DataFrame],
    output_path: Path,
    *,
    title: str = "GP Convergence",
    filter_experiments: Optional[List[str]] = None,
) -> None:
    """Plot generation-vs-fitness convergence curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_name, log_df in sorted(logs.items()):
        if filter_experiments and exp_name not in filter_experiments:
            continue
        if "gen" in log_df.columns and "max" in log_df.columns:
            ax.plot(log_df["gen"], log_df["max"], label=exp_name, marker=".", markersize=3)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Quality Score (higher is better)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved convergence plot: %s", output_path)


def plot_strategy_boxes(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    *,
    title: Optional[str] = None,
    experiment: Optional[str] = None,
) -> None:
    """Box plot of a metric across strategies."""
    subset = df.copy()
    if experiment:
        subset = subset[subset["experiment"] == experiment]

    strategies = sorted(subset["strategy"].unique())
    data = [subset[subset["strategy"] == s][metric].values for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, tick_labels=strategies, patch_artist=True)

    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric} by Strategy")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved box plot: %s", output_path)


def plot_scaling(
    df: pd.DataFrame,
    metadata: Dict[str, dict],
    output_path: Path,
) -> None:
    """Plot scheduling success rate vs problem scale for GP strategies."""
    scale_exps = df[df["group"] == "scale"]
    if scale_exps.empty:
        return

    gp_rows = scale_exps[scale_exps["strategy"].str.startswith("GP(")]
    if gp_rows.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for exp_name in sorted(gp_rows["experiment"].unique()):
        exp_data = gp_rows[gp_rows["experiment"] == exp_name]
        meta = metadata.get(exp_name, {})
        pods = meta.get("total_pods", exp_name)
        avg_rate = exp_data["scheduling_success_rate"].mean()
        ax.bar(str(pods), avg_rate, alpha=0.7, edgecolor="black")

    ax.set_xlabel("Total Pods")
    ax.set_ylabel("Scheduling Success Rate")
    ax.set_title("GP Performance vs Problem Scale")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved scaling plot: %s", output_path)


def plot_fitness_weights_radar(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Radar plot comparing fitness weight configurations."""
    fw_exps = df[df["group"] == "fitness_weights"]
    if fw_exps.empty:
        return

    gp_rows = fw_exps[fw_exps["strategy"].str.startswith("GP(")]
    if gp_rows.empty:
        return

    metrics = ["scheduling_success_rate", "avg_cpu_utilization", "avg_mem_utilization"]
    labels = ["Sched. Success", "CPU Util.", "MEM Util."]

    fig, ax = plt.subplots(figsize=(8, 6))

    experiments = sorted(gp_rows["experiment"].unique())
    x = np.arange(len(labels))
    width = 0.8 / max(len(experiments), 1)

    for i, exp_name in enumerate(experiments):
        exp_data = gp_rows[gp_rows["experiment"] == exp_name]
        values = [exp_data[m].mean() for m in metrics]
        ax.bar(x + i * width, values, width, label=exp_name, alpha=0.8)

    ax.set_xticks(x + width * (len(experiments) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("GP Performance by Fitness Weight Configuration")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved fitness weights plot: %s", output_path)


def plot_rank_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    metrics: Optional[List[str]] = None,
) -> None:
    """Heatmap of average strategy ranks across metrics."""
    from statistical import average_ranks

    if metrics is None:
        metrics = ["scheduling_success_rate", "avg_wait_time",
                    "avg_cpu_utilization", "avg_mem_utilization"]

    rank_data = {}
    for metric in metrics:
        ascending = metric in ("avg_wait_time",)
        ranks = average_ranks(df, metric, ascending=ascending)
        rank_data[metric.replace("_", "\n")] = ranks

    rank_df = pd.DataFrame(rank_data)
    if rank_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(rank_df) * 0.5 + 1)))
    im = ax.imshow(rank_df.values, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(np.arange(len(rank_df.columns)))
    ax.set_xticklabels(rank_df.columns, fontsize=8)
    ax.set_yticks(np.arange(len(rank_df.index)))
    ax.set_yticklabels(rank_df.index, fontsize=9)

    for i in range(len(rank_df.index)):
        for j in range(len(rank_df.columns)):
            ax.text(j, i, f"{rank_df.iloc[i, j]:.1f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if rank_df.iloc[i, j] > rank_df.values.mean() else "black")

    fig.colorbar(im, ax=ax, label="Average Rank (lower = better)")
    ax.set_title("Strategy Rank Heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved rank heatmap: %s", output_path)


def plot_sensitivity_heatmap(
    df: pd.DataFrame,
    metadata: Dict[str, dict],
    group: str,
    output_path: Path,
) -> None:
    """Heatmap showing GP metric sensitivity to parameter variations."""
    from statistical import sensitivity_table

    metrics = ["scheduling_success_rate", "avg_wait_time", "avg_cpu_utilization"]
    st = sensitivity_table(df, metadata, group, metrics)
    if st.empty:
        return

    mean_cols = [f"mean_{m}" for m in metrics]
    available = [c for c in mean_cols if c in st.columns]
    if not available:
        return

    labels_map = {
        "mean_scheduling_success_rate": "Sched\nRate",
        "mean_avg_wait_time": "Wait\nTime",
        "mean_avg_cpu_utilization": "CPU\nUtil",
    }

    plot_data = st.set_index("experiment")[available]
    display_cols = [labels_map.get(c, c) for c in available]

    fig, ax = plt.subplots(figsize=(8, max(3, len(plot_data) * 0.6 + 1)))
    im = ax.imshow(plot_data.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(display_cols)))
    ax.set_xticklabels(display_cols, fontsize=9)
    ax.set_yticks(np.arange(len(plot_data.index)))
    ax.set_yticklabels(plot_data.index, fontsize=9)

    for i in range(len(plot_data.index)):
        for j in range(len(display_cols)):
            ax.text(j, i, f"{plot_data.iloc[i, j]:.3f}",
                    ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax)
    ax.set_title(f"Sensitivity — {group}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved sensitivity heatmap: %s", output_path)


def plot_feature_importance(
    metadata: Dict[str, dict],
    output_path: Path,
) -> None:
    """Bar chart of GP terminal feature importance across all evolved rules."""
    from statistical import feature_importance_from_metadata

    fi = feature_importance_from_metadata(metadata)
    if fi.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(fi) * 0.4 + 1)))
    y_pos = np.arange(len(fi))
    ax.barh(y_pos, fi["total_count"].values, color="steelblue", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fi["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Total Occurrences Across All Rules")
    ax.set_title("GP Terminal Feature Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance plot: %s", output_path)


def plot_cross_profile(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grouped bar chart comparing GP vs baselines across workload profiles.

    Creates a multi-panel figure with one subplot per metric:
    scheduling_success_rate, avg_wait_time, avg_cpu_utilization.
    X-axis: profiles, grouped bars: strategies.
    """
    profile_exps = df[df["group"] == "profile"]
    if profile_exps.empty:
        return

    metrics = [
        ("scheduling_success_rate", "Scheduling Success Rate"),
        ("avg_wait_time", "Avg Wait Time"),
        ("avg_cpu_utilization", "Avg CPU Utilization"),
    ]

    experiments = sorted(profile_exps["experiment"].unique())
    strategies = sorted(profile_exps["strategy"].unique())
    n_strategies = len(strategies)

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_strategies))
    width = 0.8 / max(n_strategies, 1)

    for ax, (metric, label) in zip(axes, metrics):
        x = np.arange(len(experiments))

        for i, strategy in enumerate(strategies):
            values = []
            for exp in experiments:
                subset = profile_exps[
                    (profile_exps["experiment"] == exp)
                    & (profile_exps["strategy"] == strategy)
                ]
                values.append(subset[metric].mean() if not subset.empty else 0.0)

            ax.bar(x + i * width, values, width, label=strategy,
                   color=colors[i], edgecolor="black", linewidth=0.5)

        # Profile names as x-tick labels
        profile_labels = [e.replace("g_", "") for e in experiments]
        ax.set_xticks(x + width * (n_strategies - 1) / 2)
        ax.set_xticklabels(profile_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(n_strategies, 4),
               fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Cross-Profile Strategy Comparison", fontsize=13, y=1.06)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cross-profile plot: %s", output_path)


# ═══════════════════════════════════════════════════════════════════════
# Statistical summary
# ═══════════════════════════════════════════════════════════════════════


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-strategy statistics across all experiments.

    Returns a DataFrame with mean, std, min, max for key metrics.
    """
    metrics_cols = [
        "scheduling_success_rate", "avg_wait_time",
        "avg_cpu_utilization", "avg_mem_utilization",
    ]

    stats = df.groupby("strategy")[metrics_cols].agg(["mean", "std", "min", "max"])
    stats.columns = ["_".join(col).strip() for col in stats.columns.values]
    return stats.round(4)


# ═══════════════════════════════════════════════════════════════════════
# Full report
# ═══════════════════════════════════════════════════════════════════════


def generate_report(input_dir: Path, output_dir: Path, *, plots: bool = True) -> None:
    """Generate a complete analysis report from experiment results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = input_dir / "combined_results.csv"
    if not combined_csv.exists():
        print(f"Error: {combined_csv} not found. Run experiments first.")
        return

    df = load_combined_results(combined_csv)
    logs = load_convergence_logs(input_dir)
    metadata = load_metadata(input_dir)

    report_lines = [
        "EXPERIMENT ANALYSIS REPORT",
        "=" * 60,
        f"Total rows: {len(df)}",
        f"Experiments: {df['experiment'].nunique()}",
        f"Strategies: {df['strategy'].nunique()}",
        "",
    ]

    # ── Per-experiment tables ────────────────────────────────────
    for exp_name in sorted(df["experiment"].unique()):
        table = strategy_comparison_table(df, experiment=exp_name)
        text = format_comparison_text(table, title=f"Experiment: {exp_name}")
        report_lines.append(text)
        report_lines.append("")
        # Also save as CSV
        table.to_csv(output_dir / f"table_{exp_name}.csv")

    # ── Cross-experiment GP comparison ───────────────────────────
    cross = cross_experiment_table(df)
    if not cross.empty:
        report_lines.append("")
        report_lines.append("CROSS-EXPERIMENT GP COMPARISON")
        report_lines.append("=" * 60)
        report_lines.append(cross.to_string())
        cross.to_csv(output_dir / "cross_experiment_gp.csv", index=False)

    # ── Global statistics ────────────────────────────────────────
    stats = compute_statistics(df)
    report_lines.append("")
    report_lines.append("GLOBAL STRATEGY STATISTICS")
    report_lines.append("=" * 60)
    report_lines.append(stats.to_string())
    stats.to_csv(output_dir / "global_statistics.csv")

    # ── Training metadata ────────────────────────────────────────
    if metadata:
        report_lines.append("")
        report_lines.append("TRAINING SUMMARY")
        report_lines.append("=" * 60)
        report_lines.append(
            f"{'Experiment':<25} {'Engine':<8} {'Fitness':>8} "
            f"{'Time':>8} {'Pods':>5} {'Nodes':>5}"
        )
        report_lines.append("-" * 65)
        for name, meta in sorted(metadata.items()):
            report_lines.append(
                f"{name:<25} {meta.get('engine', '?'):<8} "
                f"{meta.get('best_fitness', 0):>8.4f} "
                f"{meta.get('training_time_s', 0):>7.1f}s "
                f"{meta.get('total_pods', '?'):>5} "
                f"{meta.get('node_count', '?'):>5}"
            )

    # ── Save report ──────────────────────────────────────────────
    report_text = "\n".join(report_lines)
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport saved to: {report_path}")

    # ── Statistical analysis ─────────────────────────────────────
    from statistical import generate_statistical_report

    stat_report = generate_statistical_report(df, metadata)
    stat_path = output_dir / "statistical_report.txt"
    with open(stat_path, "w", encoding="utf-8") as f:
        f.write(stat_report)
    print(f"\nStatistical report saved to: {stat_path}")

    # ── Plots ────────────────────────────────────────────────────
    if plots and logs:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Convergence: all experiments
        plot_convergence(logs, plots_dir / "convergence_all.png")

        # Convergence: by group
        groups = df.groupby("group")["experiment"].unique()
        for group_name, exp_names in groups.items():
            group_logs = {k: v for k, v in logs.items() if k in exp_names}
            if group_logs:
                plot_convergence(
                    group_logs,
                    plots_dir / f"convergence_{group_name}.png",
                    title=f"Convergence — {group_name}",
                )

        # Box plots: key metrics for largest experiment
        largest_exp = df.loc[df["total_pods"].idxmax(), "experiment"] if len(df) > 0 else None
        if largest_exp:
            for metric in ["scheduling_success_rate", "avg_wait_time", "avg_cpu_utilization"]:
                plot_strategy_boxes(
                    df, metric,
                    plots_dir / f"box_{metric}_{largest_exp}.png",
                    experiment=largest_exp,
                )

        # Scaling plot
        if "scale" in df["group"].values:
            plot_scaling(df, metadata, plots_dir / "scaling.png")

        # Fitness weights plot
        if "fitness_weights" in df["group"].values:
            plot_fitness_weights_radar(df, plots_dir / "fitness_weights.png")

        # Rank heatmap
        plot_rank_heatmap(df, plots_dir / "rank_heatmap.png")

        # Sensitivity heatmaps per group
        for grp in ["fitness_weights", "gp_params", "scale", "dynamics"]:
            if grp in df["group"].values:
                plot_sensitivity_heatmap(
                    df, metadata, grp,
                    plots_dir / f"sensitivity_{grp}.png",
                )

        # Feature importance
        if metadata:
            plot_feature_importance(metadata, plots_dir / "feature_importance.png")

        # Cross-profile comparison
        if "profile" in df["group"].values:
            plot_cross_profile(df, plots_dir / "cross_profile.png")

        print(f"Plots saved to: {plots_dir}")


# ═══════════════════════════════════════════════════════════════════════
# Multi-seed report
# ═══════════════════════════════════════════════════════════════════════


def generate_multiseed_report(
    input_dir: Path,
    output_dir: Path,
    *,
    plots: bool = True,
    experiment: Optional[str] = None,
    metric: str = "quality_score",
) -> None:
    """Generate acceptance-criteria report from multi-seed results.

    Reads multiseed_results.csv produced by run_experiments.py --seeds.
    Outputs:
      - acceptance_criteria.txt   (4-criterion verdict per experiment)
      - multiseed_statistics.csv  (median, IQR, Wilcoxon p, Cliff's δ)
      - box_multiseed_<exp>.png   (cross-seed box plots, one per experiment)
    """
    from statistical import acceptance_criteria_report, gp_vs_baselines_multiseed_table

    output_dir.mkdir(parents=True, exist_ok=True)

    ms_csv = input_dir / "multiseed_results.csv"
    if not ms_csv.exists():
        print(
            f"Error: {ms_csv} not found.\n"
            "Run: py run_experiments.py --seeds 42,123,456,789,1337 [other flags]"
        )
        return

    df = pd.read_csv(ms_csv)
    n_seeds = df["run_seed"].nunique() if "run_seed" in df.columns else 1
    experiments = sorted(df["experiment"].unique()) if experiment is None else [experiment]

    print(f"\nMulti-seed analysis: {len(df)} rows, {n_seeds} seeds, {len(experiments)} experiments")

    all_acceptance_lines: list[str] = [
        "MULTI-SEED ACCEPTANCE CRITERIA",
        f"Seeds: {sorted(df['run_seed'].unique().tolist()) if 'run_seed' in df.columns else 'n/a'}",
        f"Metric: {metric}",
        "=" * 80,
        "",
    ]

    all_stats_frames: list[pd.DataFrame] = []

    for exp_name in experiments:
        report = acceptance_criteria_report(df, metric=metric, experiment=exp_name)
        all_acceptance_lines.append(f"--- Experiment: {exp_name} ---")
        all_acceptance_lines.append(report)
        all_acceptance_lines.append("")

        cmp = gp_vs_baselines_multiseed_table(df, metric, experiment=exp_name)
        if not cmp.empty:
            cmp["experiment"] = exp_name
            all_stats_frames.append(cmp)

        if plots:
            _plot_multiseed_boxes(df, metric, exp_name, output_dir)

    # Save acceptance criteria report
    acc_path = output_dir / "acceptance_criteria.txt"
    acc_text = "\n".join(all_acceptance_lines)
    with open(acc_path, "w", encoding="utf-8") as f:
        f.write(acc_text)
    print(acc_text)
    print(f"\nAcceptance criteria report saved to: {acc_path}")

    # Save statistics CSV
    if all_stats_frames:
        stats_df = pd.concat(all_stats_frames, ignore_index=True)
        stats_path = output_dir / "multiseed_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Multi-seed statistics saved to: {stats_path}")

    if plots:
        # Aggregated convergence curves per experiment
        conv_plots = plot_multiseed_convergence(input_dir, output_dir)
        if conv_plots:
            print(f"Convergence plots saved: {len(conv_plots)} files")

        # Terminal frequency chart (requires gp_rule.json — available for new runs)
        term_plot = plot_terminal_frequency(input_dir, output_dir)
        if term_plot:
            print(f"Terminal frequency chart saved: {term_plot}")

        # Verdicts summary
        acc_text = "\n".join(all_acceptance_lines)
        verdict_plot = plot_verdicts_summary(acc_text, output_dir)
        if verdict_plot:
            print(f"Verdicts summary saved: {verdict_plot}")

        # Dissertation-specific plots
        if all_stats_frames:
            stats_combined = pd.concat(all_stats_frames, ignore_index=True)
            diss_heatmap = plot_dissertation_cliffs_heatmap(stats_combined, output_dir)
            if diss_heatmap:
                print(f"Dissertation Cliff's delta heatmap: {diss_heatmap}")
        diss_key = plot_dissertation_key_experiments(df, output_dir, metric=metric)
        if diss_key:
            print(f"Dissertation key experiments plot: {diss_key}")


def _plot_multiseed_boxes(
    df: pd.DataFrame,
    metric: str,
    experiment: str,
    output_dir: Path,
) -> None:
    """Box plot of a metric across strategies for a specific experiment, cross-seeds."""
    subset = df[df["experiment"] == experiment]
    if subset.empty:
        return

    strategies = sorted(subset["strategy"].unique())
    gp_strats = [s for s in strategies if s.startswith("GP(")]
    baseline_strats = [s for s in strategies if not s.startswith("GP(")]
    ordered = gp_strats + sorted(baseline_strats)

    data = [subset[subset["strategy"] == s][metric].dropna().values for s in ordered]

    colors = ["#2196F3" if s.startswith("GP(") else "#90A4AE" for s in ordered]

    fig, ax = plt.subplots(figsize=(max(8, len(ordered) * 0.9), 6))
    bp = ax.boxplot(data, tick_labels=ordered, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{experiment} — {metric} (cross-seed, n={df['run_seed'].nunique() if 'run_seed' in df.columns else '?'} seeds)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)

    safe_exp = experiment.replace("/", "_")
    safe_metric = metric.replace("/", "_")
    out = output_dir / f"box_multiseed_{safe_exp}_{safe_metric}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved multiseed box plot: %s", out)


def plot_multiseed_convergence(input_dir: Path, output_dir: Path) -> list[str]:
    """Aggregated convergence plot per experiment (mean ± std fitness across seeds).

    Reads seed_X/exp_name/convergence.json for every seed dir found in input_dir.
    Returns list of saved plot paths (relative to root).
    """
    import json
    seed_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")],
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
    )
    if not seed_dirs:
        return []

    # Collect per-experiment convergence data: {exp_name: {gen: [fitness_values]}}
    from collections import defaultdict
    conv_data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for seed_dir in seed_dirs:
        for exp_dir in seed_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            conv_file = exp_dir / "convergence.json"
            if not conv_file.exists():
                continue
            with open(conv_file) as f:
                data = json.load(f)
            for entry in data:
                if "max" in entry:
                    conv_data[exp_dir.name][entry["gen"]].append(entry["max"])

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for exp_name, gen_values in sorted(conv_data.items()):
        gens = sorted(gen_values.keys())
        means = np.array([np.mean(gen_values[g]) for g in gens])
        stds  = np.array([np.std(gen_values[g])  for g in gens])
        mins  = np.array([min(gen_values[g])      for g in gens])
        maxs  = np.array([max(gen_values[g])      for g in gens])

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(gens, means, "b-", linewidth=2, label="Mean best fitness")
        ax.fill_between(gens, means - stds, means + stds, alpha=0.2, color="blue", label="±1 std")
        ax.fill_between(gens, mins, maxs, alpha=0.08, color="gray", label="Min/Max range")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Quality Score")
        ax.set_title(f"Convergence: {exp_name} ({len(seed_dirs)} seeds)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=max(0, float(mins.min()) - 0.02))
        fig.tight_layout()

        out = output_dir / f"convergence_{exp_name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        saved.append(str(out))
        logger.info("Saved convergence plot: %s", out)

    return saved


def plot_terminal_frequency(input_dir: Path, output_dir: Path) -> Optional[str]:
    """Horizontal bar chart of terminal usage across all evolved rules.

    Reads gp_rule.json from every seed_X/exp_name/ directory.
    Returns the saved plot path or None if no rules found.
    """
    import json
    from collections import Counter
    from statistical import extract_features_from_expression

    counts: Counter = Counter()
    n_rules = 0

    for seed_dir in input_dir.glob("seed_*"):
        for exp_dir in seed_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            rule_file = exp_dir / "gp_rule.json"
            if not rule_file.exists():
                continue
            with open(rule_file) as f:
                rule_data = json.load(f)
            expr = rule_data.get("best_expression", "")
            if expr:
                counts.update(extract_features_from_expression(expr))
                n_rules += 1

    if not counts or n_rules < 3:
        logger.info("Not enough rule files for terminal frequency chart (found %d)", n_rules)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    top = counts.most_common(15)
    labels = [t[0] for t in top]
    values = [t[1] for t in top]

    # Color by terminal category
    def _color(name: str) -> str:
        if name.startswith("POD_"):     return "#4CAF50"
        if name.startswith("NODE_"):    return "#2196F3"
        if name.startswith("CLUSTER_"): return "#FF9800"
        return "#9C27B0"

    colors = [_color(l) for l in labels]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.45)))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Occurrences across {n_rules} evolved rules")
    ax.set_title("GP Terminal Feature Usage (top-15)")
    ax.grid(True, axis="x", alpha=0.3)

    # Add count labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=7)

    # Legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#4CAF50", label="Pod features"),
        plt.Rectangle((0, 0), 1, 1, color="#2196F3", label="Node features"),
        plt.Rectangle((0, 0), 1, 1, color="#FF9800", label="Cluster features"),
        plt.Rectangle((0, 0), 1, 1, color="#9C27B0", label="Compound features"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right")

    fig.tight_layout()
    out = output_dir / "terminal_frequency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved terminal frequency chart: %s", out)
    return str(out)


def plot_verdicts_summary(acceptance_text: str, output_dir: Path) -> Optional[str]:
    """Horizontal bar chart showing ACCEPTED/REJECTED verdict per experiment.

    Parses the acceptance_criteria.txt report.
    """
    import re
    verdicts: dict[str, tuple[str, int]] = {}  # exp -> (verdict, n_pass)

    current_exp = None
    for line in acceptance_text.splitlines():
        m = re.match(r"--- Experiment: (\S+) ---", line)
        if m:
            current_exp = m.group(1)
        vm = re.match(r"VERDICT: (ACCEPTED|REJECTED)\s+\((\d)/4", line)
        if vm and current_exp:
            verdicts[current_exp] = (vm.group(1), int(vm.group(2)))

    if not verdicts:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    exps = sorted(verdicts.keys())
    scores = [verdicts[e][1] for e in exps]
    accepted = [verdicts[e][0] == "ACCEPTED" for e in exps]
    colors = ["#4CAF50" if a else "#F44336" for a in accepted]

    fig, ax = plt.subplots(figsize=(10, max(4, len(exps) * 0.55)))
    bars = ax.barh(range(len(exps)), scores, color=colors, edgecolor="white")
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels(exps, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Criteria met (out of 4)")
    ax.set_xlim(0, 4.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.axvline(x=3, color="gray", linestyle="--", alpha=0.5, label="Acceptance threshold (3/4)")
    ax.set_title("GP Acceptance Criteria: Verdicts per Experiment")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=8)

    for bar, (exp, (verdict, n)) in zip(bars, verdicts.items()):
        label = f"{verdict} ({n}/4)"
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=7,
                color="#2E7D32" if verdict == "ACCEPTED" else "#C62828")

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#4CAF50", label="ACCEPTED"),
        plt.Rectangle((0, 0), 1, 1, color="#F44336", label="REJECTED"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = output_dir / "verdicts_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved verdicts summary: %s", out)
    return str(out)


def plot_dissertation_cliffs_heatmap(
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> Optional[str]:
    """Horizontal bar chart of Cliff's δ vs LeastAllocated for all experiments.

    Uses the concatenated stats DataFrame from generate_multiseed_report.
    Designed for inclusion in the dissertation (Chapter 5 results).
    """
    if stats_df.empty or "experiment" not in stats_df.columns:
        return None

    la_rows = stats_df[stats_df["baseline"] == "LeastAllocated"].copy()
    if la_rows.empty:
        return None

    # Sort ascending so highest δ is at the top of the horizontal bar chart
    la_rows = la_rows.sort_values("cliffs_d", ascending=True)

    def _bar_color(d: float) -> str:
        a = abs(d)
        if a >= 0.474:
            return "#2E7D32"
        if a >= 0.33:
            return "#66BB6A"
        if a >= 0.147:
            return "#FFA726"
        if d < 0:
            return "#EF5350"
        return "#B0BEC5"

    colors = [_bar_color(d) for d in la_rows["cliffs_d"]]

    fig, ax = plt.subplots(figsize=(9, max(6, len(la_rows) * 0.42)))
    y = range(len(la_rows))
    ax.barh(list(y), la_rows["cliffs_d"].values, color=colors, height=0.65, edgecolor="white", linewidth=0.4)
    ax.axvline(0.147, color="#1565C0", linestyle="--", linewidth=1.2, label="prag |δ|=0.147")
    ax.axvline(0.33, color="#1565C0", linestyle=":", linewidth=0.9, alpha=0.6)
    ax.axvline(0.474, color="#1565C0", linestyle=":", linewidth=0.9, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.7)

    ax.set_yticks(list(y))
    ax.set_yticklabels(la_rows["experiment"].values, fontsize=9)
    ax.set_xlabel("")
    ax.set_title("Cliff's δ: Effect size GP vs LeastAllocated", fontsize=11)

    for i, (_, row) in enumerate(la_rows.iterrows()):
        lbl = f"{row['cliffs_d']:+.3f}"
        x_pos = row["cliffs_d"] + 0.01 if row["cliffs_d"] >= 0 else row["cliffs_d"] - 0.01
        ha = "left" if row["cliffs_d"] >= 0 else "right"
        ax.text(x_pos, i, lbl, va="center", ha=ha, fontsize=7.5)

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#2E7D32", label="large (≥0.474)"),
        plt.Rectangle((0, 0), 1, 1, color="#66BB6A", label="medium (≥0.33)"),
        plt.Rectangle((0, 0), 1, 1, color="#FFA726", label="small (≥0.147)"),
        plt.Rectangle((0, 0), 1, 1, color="#B0BEC5", label="negligible"),
        plt.Rectangle((0, 0), 1, 1, color="#EF5350", label="negativ"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    out = output_dir / "dissertation_cliffs_delta_heatmap.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    logger.info("Saved dissertation Cliff's delta heatmap: %s", out)
    return str(out)


def plot_dissertation_key_experiments(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str = "quality_score",
) -> Optional[str]:
    """2×3 grid of GP vs LeastAllocated box plots for 6 representative experiments.

    Chosen to illustrate the full range: best case (h1_asymmetric), dynamics
    (e2_reschedule), scale (b3_large), cross-profile success (g_cicd),
    structural ceiling (g_webser), and rejected (g_aitrai).
    """
    KEY_EXPS = [
        ("h1_asymmetric", "h1: Asimetric (blind-spot LA)"),
        ("b3_large", "b3: Large scale"),
        ("e2_reschedule", "e2: Node failures (reschedule)"),
        ("g_cicd", "g_cicd: CI/CD"),
        ("g_webser", "g_webser: Web serving"),
        ("g_aitrai", "g_aitrai: AI training (REJECTED)"),
    ]
    available = df["experiment"].unique()
    selected = [(exp, lbl) for exp, lbl in KEY_EXPS if exp in available]
    if len(selected) < 2:
        return None

    n = len(selected)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.8))
    axes = axes.flatten()

    for ax_idx, (exp_name, label) in enumerate(selected):
        ax = axes[ax_idx]
        sub = df[df["experiment"] == exp_name]
        gp_vals = sub[sub["strategy"].str.startswith("GP(")][metric].dropna().values
        la_vals = sub[sub["strategy"] == "LeastAllocated"][metric].dropna().values

        if len(gp_vals) == 0 or len(la_vals) == 0:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            [gp_vals, la_vals],
            tick_labels=["GP", "LeastAlloc"],
            patch_artist=True,
            notch=False,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor("#2196F3")
        bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor("#90A4AE")
        bp["boxes"][1].set_alpha(0.75)

        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_ylabel(metric.replace("_", " "), fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        gp_med = float(np.median(gp_vals))
        la_med = float(np.median(la_vals))
        delta_text = f"Δ={gp_med - la_med:+.4f}"
        ax.text(0.97, 0.04, delta_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#1565C0")

    for ax_idx in range(len(selected), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("GP vs LeastAllocated — experimente reprezentative", fontsize=11, y=1.01)
    fig.tight_layout()
    out = output_dir / "dissertation_key_experiments.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dissertation key experiments plot: %s", out)
    return str(out)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K8s GP Scheduler — Experiment Analysis"
    )
    parser.add_argument(
        "--input", type=str, default="tmp/results/experiments",
        help="Directory with experiment results",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for analysis (default: <input>/analysis)",
    )
    parser.add_argument(
        "--plots", action="store_true", default=True,
        help="Generate plots (default: True)",
    )
    parser.add_argument(
        "--tables-only", action="store_true",
        help="Generate tables only (skip plots)",
    )
    parser.add_argument(
        "--multiseed", action="store_true",
        help="Analyse multiseed_results.csv instead of combined_results.csv; "
             "produces acceptance_criteria.txt and cross-seed box plots.",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Restrict multi-seed analysis to a single experiment name.",
    )
    parser.add_argument(
        "--metric", type=str, default="quality_score",
        help="Primary metric for multi-seed acceptance criteria (default: quality_score).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "analysis"

    if args.multiseed:
        generate_multiseed_report(
            input_dir,
            output_dir,
            plots=not args.tables_only,
            experiment=args.experiment,
            metric=args.metric,
        )
    else:
        do_plots = not args.tables_only
        generate_report(input_dir, output_dir, plots=do_plots)


if __name__ == "__main__":
    main()
