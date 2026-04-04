"""Gantt chart visualization for Kubernetes pod scheduling.

Produces horizontal bar charts showing pod execution timelines across
cluster nodes.  Each bar represents one pod: its position on the Y-axis
indicates the assigned node, and its horizontal extent spans from
``scheduled_time`` to ``completion_time``.

Usage::

    from visualization.gantt import plot_gantt, plot_gantt_from_engine

    # From a finished SimulationEngine:
    fig = plot_gantt_from_engine(engine, title="GP Strategy")
    fig.savefig("gantt.png", dpi=150, bbox_inches="tight")

    # From raw pod/node data:
    fig = plot_gantt(all_pods, node_ids, title="My Run")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from simulator.engine import SimulationEngine

from models.pod import Pod, PodStatus

logger = logging.getLogger(__name__)


# ── Colour helpers ───────────────────────────────────────────────────

_NAMESPACE_COLORS: Dict[str, str] = {}

_STATUS_HATCH = {
    PodStatus.REJECTED: "///",
    PodStatus.EVICTED: "xxx",
}

_REJECTED_COLOR = "#d9534f"
_DEFAULT_COLORMAP = "tab20"


def _get_colormap(n_colors: int = 20):
    """Return a matplotlib colormap, compatible across versions."""
    try:
        return plt.colormaps.get_cmap(_DEFAULT_COLORMAP)
    except AttributeError:
        return plt.cm.get_cmap(_DEFAULT_COLORMAP, n_colors)


def _namespace_color(namespace: str, cmap) -> tuple:
    """Assign a deterministic colour to each namespace."""
    if namespace not in _NAMESPACE_COLORS:
        idx = len(_NAMESPACE_COLORS) % cmap.N
        _NAMESPACE_COLORS[namespace] = cmap(idx)
    return _NAMESPACE_COLORS[namespace]


# ── Public API ───────────────────────────────────────────────────────

def plot_gantt(
    all_pods: Dict[str, Pod],
    node_ids: List[str],
    *,
    title: str = "Gantt Chart — Pod Scheduling",
    color_by: str = "namespace",
    show_rejected: bool = True,
    show_waiting: bool = True,
    figsize: tuple = (16, 8),
) -> "Figure":
    """Create a Gantt chart from pod data.

    Args:
        all_pods:      ``{pod_id: Pod}`` — complete registry after simulation.
        node_ids:      Ordered list of node IDs (Y-axis labels).
        title:         Chart title.
        color_by:      ``"namespace"`` or ``"priority"`` — bar colour grouping.
        show_rejected: If True, display rejected pods in a separate row.
        show_waiting:  If True, draw lighter bars for the waiting period
                       (arrival → scheduled).
        figsize:       Figure size ``(width, height)`` in inches.

    Returns:
        A ``matplotlib.figure.Figure`` that can be saved or displayed.
    """
    cmap = _get_colormap()

    # Build node-index mapping
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    n_rows = len(node_ids)
    if show_rejected:
        n_rows += 1  # extra row for rejected pods

    fig, ax = plt.subplots(figsize=figsize)

    legend_handles: Dict[str, mpatches.Patch] = {}

    # ── Scheduled / Completed pods ───────────────────────────────
    for pod in all_pods.values():
        if pod.status in (PodStatus.PENDING, PodStatus.REJECTED, PodStatus.EVICTED):
            continue
        if pod.assigned_node_id is None or pod.assigned_node_id not in node_index:
            continue

        y = node_index[pod.assigned_node_id]
        start = pod.scheduled_time if pod.scheduled_time is not None else pod.arrival_time
        end = pod.completion_time if pod.completion_time is not None else start + pod.duration

        if end <= start:
            continue

        # Bar colour
        color = _resolve_color(pod, color_by, cmap)
        label_key = _resolve_label_key(pod, color_by)

        # Waiting period (arrival → scheduled)
        if show_waiting and pod.scheduled_time is not None and pod.scheduled_time > pod.arrival_time:
            wait_start = pod.arrival_time
            wait_dur = pod.scheduled_time - pod.arrival_time
            ax.barh(
                y, wait_dur, left=wait_start, height=0.4,
                color=color, alpha=0.25, edgecolor="gray", linewidth=0.5,
            )

        # Execution bar
        duration = end - start
        bar = ax.barh(
            y, duration, left=start, height=0.7,
            color=color, edgecolor="black", linewidth=0.5, alpha=0.85,
        )

        # Label on bar (only if wide enough)
        if duration > (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02 or True:
            ax.text(
                start + duration / 2, y, pod.pod_id,
                ha="center", va="center", fontsize=6, fontweight="bold",
                color="black",
            )

        # Legend entry
        if label_key not in legend_handles:
            legend_handles[label_key] = mpatches.Patch(
                facecolor=color, edgecolor="black", label=label_key, alpha=0.85,
            )

    # ── Rejected pods ────────────────────────────────────────────
    if show_rejected:
        rejected_y = len(node_ids)
        for pod in all_pods.values():
            if pod.status != PodStatus.REJECTED:
                continue
            rej_time = pod.completion_time if pod.completion_time is not None else pod.arrival_time
            # Draw a thin marker at rejection time
            ax.barh(
                rejected_y, max(0.5, pod.duration * 0.1), left=pod.arrival_time,
                height=0.5, color=_REJECTED_COLOR, alpha=0.6,
                edgecolor="darkred", linewidth=0.5, hatch="///",
            )

        if any(p.status == PodStatus.REJECTED for p in all_pods.values()):
            legend_handles["Rejected"] = mpatches.Patch(
                facecolor=_REJECTED_COLOR, edgecolor="darkred",
                label="Rejected", alpha=0.6, hatch="///",
            )

    # ── Axis formatting ──────────────────────────────────────────
    y_labels = list(node_ids)
    if show_rejected:
        y_labels.append("Rejected")

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Node", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    if legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            loc="upper right", fontsize=8, framealpha=0.9,
        )

    fig.tight_layout()
    return fig


def plot_gantt_from_engine(
    engine: "SimulationEngine",
    *,
    title: str = "Gantt Chart — Pod Scheduling",
    color_by: str = "namespace",
    show_rejected: bool = True,
    show_waiting: bool = True,
    figsize: tuple = (16, 8),
) -> "Figure":
    """Convenience wrapper that extracts data from a finished SimulationEngine."""
    node_ids = sorted(engine.cluster.nodes.keys())
    return plot_gantt(
        all_pods=engine.cluster.all_pods,
        node_ids=node_ids,
        title=title,
        color_by=color_by,
        show_rejected=show_rejected,
        show_waiting=show_waiting,
        figsize=figsize,
    )


def save_gantt(
    fig: "Figure",
    path: str | Path,
    dpi: int = 150,
) -> None:
    """Save a Gantt figure to disk and close it to free memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gantt chart saved to %s", path)


# ── Internal helpers ─────────────────────────────────────────────────

def _resolve_color(pod: Pod, color_by: str, cmap):
    """Pick a bar colour based on the grouping strategy."""
    if color_by == "namespace":
        return _namespace_color(pod.namespace, cmap)
    elif color_by == "priority":
        # Map priority 0–1000 to colormap
        norm = min(pod.priority / 1000.0, 1.0)
        return cmap(norm)
    else:
        return _namespace_color(pod.namespace, cmap)


def _resolve_label_key(pod: Pod, color_by: str) -> str:
    """Legend label for this pod's group."""
    if color_by == "namespace":
        return pod.namespace
    elif color_by == "priority":
        if pod.priority <= 200:
            return "Low priority"
        elif pod.priority <= 600:
            return "Medium priority"
        else:
            return "High priority"
    return pod.namespace
