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
        if "gen" in log_df.columns and "min" in log_df.columns:
            ax.plot(log_df["gen"], log_df["min"], label=exp_name, marker=".", markersize=3)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (lower is better)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "analysis"

    do_plots = not args.tables_only
    generate_report(input_dir, output_dir, plots=do_plots)


if __name__ == "__main__":
    main()
