"""GP tree visualization — render evolved scheduling rules as tree diagrams.

Uses DEAP's built-in graph extraction + matplotlib for rendering,
with no external dependency on graphviz.

Usage::

    from visualization.gp_tree import plot_gp_tree, save_gp_tree
    fig = plot_gp_tree(individual, engine)
    save_gp_tree(fig, "best_rule.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ── Color palette for node types ─────────────────────────────────────
_COLORS = {
    "function":  "#4a90d9",   # blue — internal nodes
    "terminal":  "#5bb370",   # green — leaf features
    "constant":  "#e8a838",   # amber — ephemeral constants
}

_FUNCTION_NAMES = {"add", "sub", "mul", "protected_div", "neg", "min", "max", "if_positive"}


def _node_type(label: str) -> str:
    """Classify a node label as function, terminal, or constant."""
    if label in _FUNCTION_NAMES:
        return "function"
    try:
        float(label)
        return "constant"
    except ValueError:
        return "terminal"


def _layout_tree(nodes, edges):
    """Compute (x, y) positions for each node using a simple tree layout.

    Uses a bottom-up algorithm: leaves get sequential x positions,
    parents are centred above their children.
    """
    # Build adjacency: parent → [children] (ordered)
    children: Dict[int, list] = {}
    all_nodes = set()
    for node_id in nodes:
        children[node_id] = []
        all_nodes.add(node_id)
    for parent, child in edges:
        children[parent].append(child)

    # Find root (node with no incoming edges)
    child_set = {c for _, c in edges}
    roots = [n for n in all_nodes if n not in child_set]
    root = roots[0] if roots else 0

    # Compute depth (y) for each node
    depth: Dict[int, int] = {}
    def _set_depth(n, d):
        depth[n] = d
        for c in children.get(n, []):
            _set_depth(c, d + 1)
    _set_depth(root, 0)

    max_depth = max(depth.values()) if depth else 0

    # Assign x positions bottom-up
    pos: Dict[int, tuple] = {}
    x_counter = [0]

    def _assign_x(n):
        kids = children.get(n, [])
        if not kids:
            # Leaf
            pos[n] = (x_counter[0], max_depth - depth[n])
            x_counter[0] += 1
        else:
            for c in kids:
                _assign_x(c)
            # Centre parent above children
            child_xs = [pos[c][0] for c in kids]
            pos[n] = ((min(child_xs) + max(child_xs)) / 2, max_depth - depth[n])

    _assign_x(root)
    return pos


def plot_gp_tree(
    individual: Any,
    *,
    title: str = "Evolved GP Scheduling Rule",
    figsize: Optional[tuple] = None,
    simplified_expr: Optional[str] = None,
) -> "Figure":
    """Render a DEAP GP individual as a tree diagram.

    Args:
        individual:       A DEAP PrimitiveTree individual.
        title:            Plot title.
        figsize:          Figure size (auto-calculated if None).
        simplified_expr:  Optional simplified expression to show below.

    Returns:
        matplotlib Figure.
    """
    from deap import gp

    # Extract graph structure from DEAP
    nodes_dict, edges, labels = gp.graph(individual)

    if not nodes_dict:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, str(individual), ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return fig

    # Compute layout
    pos = _layout_tree(nodes_dict, edges)

    # Auto-size figure
    if figsize is None:
        n_leaves = sum(1 for n in nodes_dict if not any(p == n for p, _ in edges))
        n_depth = max((p[1] for p in pos.values()), default=0) + 1
        figsize = (max(6, n_leaves * 1.2), max(4, n_depth * 1.5 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges first
    for parent, child in edges:
        px, py = pos[parent]
        cx, cy = pos[child]
        ax.plot([px, cx], [py, cy], color="#888888", linewidth=1.2, zorder=1)

    # Draw nodes
    for node_id in nodes_dict:
        x, y = pos[node_id]
        label = str(labels[node_id])
        ntype = _node_type(label)
        color = _COLORS[ntype]

        # Shorten long labels
        display = label
        if label == "protected_div":
            display = "÷"
        elif label == "if_positive":
            display = "if>0"

        bbox = dict(
            boxstyle="round,pad=0.3",
            facecolor=color,
            edgecolor="white",
            alpha=0.9,
        )
        fontsize = 8 if len(display) > 8 else 9
        ax.text(
            x, y, display,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white",
            bbox=bbox, zorder=2,
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=_COLORS["function"], label="Function"),
        mpatches.Patch(color=_COLORS["terminal"], label="Terminal (feature)"),
        mpatches.Patch(color=_COLORS["constant"], label="Constant"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.8)

    # Title and subtitle
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    if simplified_expr:
        ax.text(
            0.5, -0.02, f"Simplified: {simplified_expr}",
            transform=ax.transAxes, ha="center", fontsize=8,
            style="italic", color="#555555",
        )

    ax.set_axis_off()
    ax.margins(0.15)
    fig.tight_layout()
    return fig


def save_gp_tree(fig: "Figure", path: str | Path, dpi: int = 150) -> Path:
    """Save a GP tree figure to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("GP tree saved to %s", path)
    return path


def plot_pareto_front(
    pareto_individuals: list,
    *,
    objective_names: tuple = ("Wait Time", "Resource Waste", "Rejection Rate"),
    title: str = "Pareto Front — NSGA-II",
    figsize: tuple = (12, 4),
) -> "Figure":
    """Plot the Pareto front from NSGA-II as 2D projections.

    Creates three subplots showing pairwise objective trade-offs:
      - Wait vs Waste
      - Wait vs Rejection
      - Waste vs Rejection
    """
    if not pareto_individuals:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No Pareto front data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    objs = np.array([ind.fitness.values for ind in pareto_individuals])

    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(objs[:, i], objs[:, j], c="#4a90d9", edgecolors="white",
                   s=50, alpha=0.8, zorder=2)
        ax.set_xlabel(objective_names[i], fontsize=9)
        ax.set_ylabel(objective_names[j], fontsize=9)
        ax.grid(True, alpha=0.3)

        # Highlight the "best sum" individual
        best_idx = np.argmin(objs.sum(axis=1))
        ax.scatter(
            objs[best_idx, i], objs[best_idx, j],
            c="#ef4444", s=100, marker="*", zorder=3, label="Best (min sum)",
        )
        ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
