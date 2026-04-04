"""Resource utilization time-series plots (dissertation section 5.7).

Generates:
  - Per-node CPU/MEM utilization over time
  - Cluster-aggregate CPU/MEM over time
  - Pending pod queue depth over time
  - Node failure/recovery event markers
  - Wait-time distribution histogram
  - Free resource time-series
  - Cluster utilization variance over time

Usage::

    from visualization.resource_plots import (
        plot_node_utilization,
        plot_cluster_utilization,
        plot_wait_time_distribution,
        plot_free_resources,
        save_resource_plot,
    )
    fig = plot_cluster_utilization(resource_monitor, title="GP Strategy")
    save_resource_plot(fig, "cluster_util.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from metrics.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


def plot_node_utilization(
    monitor: ResourceMonitor,
    *,
    title: str = "Per-Node Resource Utilization",
    figsize: tuple = (14, 8),
) -> "Figure":
    """Plot CPU and MEM utilization per node over time.

    Creates a 2-row subplot: top = CPU, bottom = MEM.
    Each node is a separate line.  Failure/recovery events are
    shown as vertical markers.
    """
    timestamps = monitor.get_timestamps()
    node_ids = monitor.get_node_ids()

    if not timestamps or not node_ids:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        return fig

    fig, (ax_cpu, ax_mem) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    cmap = _get_colormap(len(node_ids))
    ts = np.array(timestamps)

    for i, nid in enumerate(node_ids):
        color = cmap(i % cmap.N)
        cpu = monitor.get_node_cpu_series(nid)
        mem = monitor.get_node_mem_series(nid)
        ax_cpu.plot(ts, cpu, label=nid, color=color, linewidth=1.2, alpha=0.85)
        ax_mem.plot(ts, mem, label=nid, color=color, linewidth=1.2, alpha=0.85)

    # Mark failure/recovery events
    _add_event_markers(ax_cpu, monitor)
    _add_event_markers(ax_mem, monitor)

    ax_cpu.set_ylabel("CPU Utilization", fontsize=11)
    ax_cpu.set_ylim(-0.05, 1.05)
    ax_cpu.legend(fontsize=8, ncol=min(len(node_ids), 5), loc="upper right")
    ax_cpu.grid(True, alpha=0.3)

    ax_mem.set_ylabel("MEM Utilization", fontsize=11)
    ax_mem.set_xlabel("Simulation Time", fontsize=11)
    ax_mem.set_ylim(-0.05, 1.05)
    ax_mem.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_cluster_utilization(
    monitor: ResourceMonitor,
    *,
    title: str = "Cluster Resource Utilization Over Time",
    figsize: tuple = (14, 6),
) -> "Figure":
    """Plot cluster-aggregate CPU, MEM, and pending queue over time.

    Creates a dual-axis plot: left = utilization (0-1), right = pending count.
    """
    timestamps = monitor.get_timestamps()

    if not timestamps:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        return fig

    ts = np.array(timestamps)
    cpu = np.array(monitor.get_cluster_cpu_series())
    mem = np.array(monitor.get_cluster_mem_series())
    pending = np.array(monitor.get_pending_series())

    fig, ax1 = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Utilization on left axis
    ax1.plot(ts, cpu, label="CPU Utilization", color="#1f77b4", linewidth=1.8)
    ax1.plot(ts, mem, label="MEM Utilization", color="#ff7f0e", linewidth=1.8)
    ax1.fill_between(ts, cpu, alpha=0.15, color="#1f77b4")
    ax1.fill_between(ts, mem, alpha=0.15, color="#ff7f0e")
    ax1.set_ylabel("Utilization", fontsize=11)
    ax1.set_xlabel("Simulation Time", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Pending queue on right axis
    ax2 = ax1.twinx()
    ax2.plot(ts, pending, label="Pending Pods", color="#2ca02c",
             linewidth=1.2, linestyle="--", alpha=0.7)
    ax2.set_ylabel("Pending Pod Count", fontsize=11, color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    # Mark failure/recovery events
    _add_event_markers(ax1, monitor)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


def plot_cluster_comparison(
    monitors: dict[str, ResourceMonitor],
    *,
    title: str = "Cluster Utilization — Strategy Comparison",
    figsize: tuple = (14, 8),
) -> "Figure":
    """Compare cluster CPU utilization across multiple strategies.

    Args:
        monitors: {strategy_name: ResourceMonitor} mapping.
    """
    if not monitors:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        return fig

    fig, (ax_cpu, ax_mem) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    cmap = _get_colormap(len(monitors))

    for i, (name, mon) in enumerate(monitors.items()):
        ts = mon.get_timestamps()
        if not ts:
            continue
        color = cmap(i % cmap.N)
        ax_cpu.plot(ts, mon.get_cluster_cpu_series(),
                    label=name, color=color, linewidth=1.4, alpha=0.85)
        ax_mem.plot(ts, mon.get_cluster_mem_series(),
                    label=name, color=color, linewidth=1.4, alpha=0.85)

    ax_cpu.set_ylabel("CPU Utilization", fontsize=11)
    ax_cpu.set_ylim(-0.05, 1.05)
    ax_cpu.legend(fontsize=9, ncol=min(len(monitors), 4), loc="upper right")
    ax_cpu.grid(True, alpha=0.3)

    ax_mem.set_ylabel("MEM Utilization", fontsize=11)
    ax_mem.set_xlabel("Simulation Time", fontsize=11)
    ax_mem.set_ylim(-0.05, 1.05)
    ax_mem.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def save_resource_plot(fig: "Figure", path, dpi: int = 150) -> None:
    """Save a resource plot figure to disk."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved resource plot: %s", filepath)


def plot_wait_time_distribution(
    wait_times: List[float],
    *,
    title: str = "Pod Wait-Time Distribution",
    figsize: tuple = (10, 5),
    bins: int = 30,
) -> "Figure":
    """Histogram of per-pod wait times with percentile markers."""
    fig, ax = plt.subplots(figsize=figsize)

    if not wait_times:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        return fig

    wt = np.array(wait_times)
    ax.hist(wt, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.8)

    # Percentile markers
    for p, color, ls in [(50, "#2ca02c", "--"), (90, "#ff7f0e", "-."), (99, "#d62728", ":")]:
        val = float(np.percentile(wt, p))
        ax.axvline(val, color=color, linewidth=1.4, linestyle=ls,
                   label=f"P{p} = {val:.2f}")

    ax.set_xlabel("Wait Time")
    ax.set_ylabel("Pod Count")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_free_resources(
    monitor: ResourceMonitor,
    *,
    title: str = "Cluster Free Resources Over Time",
    figsize: tuple = (14, 6),
) -> "Figure":
    """Plot free CPU and free memory over time (cluster-wide absolute values)."""
    timestamps = monitor.get_timestamps()

    if not timestamps:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        return fig

    ts = np.array(timestamps)
    cpu_free = np.array(monitor.get_cluster_cpu_free_series())
    mem_free = np.array(monitor.get_cluster_mem_free_series())

    fig, ax1 = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax1.plot(ts, cpu_free, label="Free CPU (cores)", color="#1f77b4", linewidth=1.6)
    ax1.fill_between(ts, cpu_free, alpha=0.15, color="#1f77b4")
    ax1.set_ylabel("Free CPU (cores)", fontsize=11, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlabel("Simulation Time", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ts, mem_free, label="Free MEM (MiB)", color="#ff7f0e", linewidth=1.6)
    ax2.fill_between(ts, mem_free, alpha=0.12, color="#ff7f0e")
    ax2.set_ylabel("Free Memory (MiB)", fontsize=11, color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    _add_event_markers(ax1, monitor)
    fig.tight_layout()
    return fig


def plot_utilization_variance(
    monitor: ResourceMonitor,
    *,
    title: str = "CPU Utilization Variance Over Time",
    figsize: tuple = (14, 5),
) -> "Figure":
    """Plot the variance of per-node CPU utilization over time (balance indicator)."""
    timestamps = monitor.get_timestamps()

    if not timestamps:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        return fig

    ts = np.array(timestamps)
    var_series = np.array(monitor.get_cpu_util_variance_series())

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ts, var_series, label="CPU Util Variance", color="#9467bd", linewidth=1.4)
    ax.fill_between(ts, var_series, alpha=0.15, color="#9467bd")
    ax.set_ylabel("Variance", fontsize=11)
    ax.set_xlabel("Simulation Time", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Internal helpers ─────────────────────────────────────────────────

def _get_colormap(n_colors: int = 20):
    """Return a matplotlib colormap, compatible across versions."""
    try:
        return plt.colormaps.get_cmap("tab10")
    except AttributeError:
        return plt.cm.get_cmap("tab10", n_colors)


def _add_event_markers(ax, monitor: ResourceMonitor) -> None:
    """Add vertical lines for node failure/recovery events."""
    for t in monitor.get_failure_timestamps():
        ax.axvline(t, color="#d9534f", linewidth=0.8, linestyle=":",
                   alpha=0.6, zorder=0)
    for t in monitor.get_recovery_timestamps():
        ax.axvline(t, color="#5cb85c", linewidth=0.8, linestyle=":",
                   alpha=0.6, zorder=0)
