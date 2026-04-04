"""GP Primitives — function set and terminal definitions for Kubernetes scheduling.

This module defines:
  - Protected arithmetic functions (safe division, etc.)
  - The TERMINAL_NAMES list (15 Kubernetes-specific features)
  - A helper to extract terminal values from (Pod, Node, ClusterState)
"""

from __future__ import annotations

from typing import Dict, List

from models.cluster_state import ClusterState
from models.node import Node
from models.pod import Pod


# ═══════════════════════════════════════════════════════════════════════════
# Function Set (8 functions matching the dissertation specification)
# ═══════════════════════════════════════════════════════════════════════════

def protected_div(x: float, y: float) -> float:
    """Protected division: returns 1.0 when the divisor is near zero."""
    if abs(y) < 1e-9:
        return 1.0
    return x / y


def neg(x: float) -> float:
    return -x


def safe_min(x: float, y: float) -> float:
    return min(x, y)


def safe_max(x: float, y: float) -> float:
    return max(x, y)


def if_positive(condition: float, then_val: float, else_val: float) -> float:
    """if condition > 0 return then_val else else_val."""
    return then_val if condition > 0 else else_val


def add(x: float, y: float) -> float:
    return x + y


def sub(x: float, y: float) -> float:
    return x - y


def mul(x: float, y: float) -> float:
    return x * y


# Registry: name → (callable, arity)
FUNCTION_SET: Dict[str, tuple] = {
    "add":           (add, 2),
    "sub":           (sub, 2),
    "mul":           (mul, 2),
    "protected_div": (protected_div, 2),
    "neg":           (neg, 1),
    "min":           (safe_min, 2),
    "max":           (safe_max, 2),
    "if_positive":   (if_positive, 3),
}


# ═══════════════════════════════════════════════════════════════════════════
# Terminal Set (15 terminals — Kubernetes scheduling features)
# ═══════════════════════════════════════════════════════════════════════════

TERMINAL_NAMES: List[str] = [
    # ── Pod features ─────────────────────────────────────────────────
    "POD_CPU_REQ",       # CPU requested by the pod (cores)
    "POD_MEM_REQ",       # Memory requested by the pod (MiB)
    "POD_GPU_REQ",       # GPUs requested by the pod
    "POD_PRIORITY",      # Priority class (0–1000)
    "POD_QOS",           # QoS class (1=BE, 2=Burstable, 3=Guaranteed)
    "POD_WAIT_TIME",     # Time spent in the pending queue
    "POD_DURATION",      # Expected runtime of the pod

    # ── Node features ────────────────────────────────────────────────
    "NODE_CPU_AVAIL",    # CPU available on the candidate node
    "NODE_MEM_AVAIL",    # Memory available on the candidate node
    "NODE_GPU_AVAIL",    # GPUs available on the candidate node
    "NODE_CPU_UTIL",     # CPU utilisation ratio (0–1)
    "NODE_MEM_UTIL",     # Memory utilisation ratio (0–1)
    "NODE_POD_COUNT",    # Number of pods currently on the node
    "NODE_COST",         # Cost per hour of the node
    "NODE_TAINT_COUNT",  # Number of taints on the node (0 = unconstrained)
    "NODE_CPU_MEM_IMBALANCE",  # |cpu_util − mem_util| on the node
    "NODE_CPU_FREE_AFTER",     # CPU free ratio after placing this pod
    "NODE_MEM_FREE_AFTER",     # Memory free ratio after placing this pod

    # ── Cluster features ─────────────────────────────────────────────
    "PENDING_COUNT",     # Total pods in pending queue
    "CLUSTER_CPU_UTIL",  # Cluster-wide CPU utilisation
    "CLUSTER_MEM_UTIL",  # Cluster-wide memory utilisation
    "CLUSTER_GPU_UTIL",  # Cluster-wide GPU utilisation
    "PENDING_PRESSURE",  # pending / (pending + scheduled + 1) — scheduler pressure
    "CLUSTER_UTIL_STD",  # Std-dev of node CPU utilisation (balance signal)

    # ── Compound feature ─────────────────────────────────────────────
    "RESOURCE_FIT",      # How well the pod fits on the node (0–1)

    # ── Dynamics feature ─────────────────────────────────────────────
    "CLUSTER_HEALTHY_RATIO",  # Fraction of healthy (available) nodes (0–1)

    # ── Preemption feature ───────────────────────────────────────────
    "NODE_PREEMPTABLE_COUNT",  # Number of lower-priority pods on node that could be evicted

    # ── Overcommit feature ───────────────────────────────────────────
    "NODE_OVERCOMMIT_RATIO",   # max(cpu_overcommit, mem_overcommit) on this node (>1 = risk)

    # ── Anti-affinity feature ────────────────────────────────────────
    "NODE_AFFINITY_CONFLICT",  # 1.0 if pod has affinity conflict on this node, else 0.0

    # ── Replica-group feature ────────────────────────────────────────
    "REPLICA_GROUP_COLOCATED",  # Pods from same replica group already on this node

    # ── Namespace feature ────────────────────────────────────────────
    "NAMESPACE_PENDING_RATIO",  # Fraction of pending pods from the same namespace
]


def extract_terminal_values(
    pod: Pod,
    node: Node,
    cluster: ClusterState,
    current_time: float,
) -> Dict[str, float]:
    """Build the terminal-value mapping for a (pod, node) pair.

    This is the bridge between the simulation state and the GP tree
    evaluation.  Each terminal maps to a concrete numeric value.

    Args:
        pod:          The pod being scheduled.
        node:         The candidate node.
        cluster:      Current cluster state.
        current_time: Simulation clock (for wait-time calculation).

    Returns:
        Dict terminal_name → float value.
    """
    # Wait time: if the pod is still pending, compute from arrival to now
    wait_time = current_time - pod.arrival_time if pod.scheduled_time is None else pod.wait_time

    # Resource fit: geometric mean of (available / requested) ratios, clamped to [0, 1]
    cpu_fit = (node.cpu_available / pod.cpu_request) if pod.cpu_request > 0 else 1.0
    mem_fit = (node.mem_available / pod.mem_request) if pod.mem_request > 0 else 1.0
    resource_fit = min(1.0, (min(cpu_fit, 1.0) + min(mem_fit, 1.0)) / 2.0)

    # Preemption: count lower-priority pods on this node
    preemptable = sum(1 for p in node.pods.values() if p.priority < pod.priority)

    # Look-ahead: free ratio after placing this pod
    cpu_free_after = max(0.0, node.cpu_available - pod.cpu_request) / node.cpu_capacity if node.cpu_capacity > 0 else 0.0
    mem_free_after = max(0.0, node.mem_available - pod.mem_request) / node.mem_capacity if node.mem_capacity > 0 else 0.0

    # Pending pressure: normalized scheduler pressure
    pending = cluster.pending_count
    scheduled = cluster.scheduled_pod_count
    pending_pressure = pending / (pending + scheduled + 1)

    return {
        "POD_CPU_REQ":      pod.cpu_request,
        "POD_MEM_REQ":      pod.mem_request,
        "POD_GPU_REQ":      pod.gpu_request,
        "POD_PRIORITY":     float(pod.priority),
        "POD_QOS":          float(pod.qos_class.value),
        "POD_WAIT_TIME":    wait_time,
        "POD_DURATION":     pod.duration,

        "NODE_CPU_AVAIL":   node.cpu_available,
        "NODE_MEM_AVAIL":   node.mem_available,
        "NODE_GPU_AVAIL":   node.gpu_available,
        "NODE_CPU_UTIL":    node.cpu_utilization,
        "NODE_MEM_UTIL":    node.mem_utilization,
        "NODE_POD_COUNT":   float(node.pod_count),
        "NODE_COST":        node.cost_per_hour,
        "NODE_TAINT_COUNT": float(len(node.taints)),
        "NODE_CPU_MEM_IMBALANCE": abs(node.cpu_utilization - node.mem_utilization),
        "NODE_CPU_FREE_AFTER":    cpu_free_after,
        "NODE_MEM_FREE_AFTER":    mem_free_after,

        "PENDING_COUNT":    float(pending),
        "CLUSTER_CPU_UTIL": cluster.cluster_cpu_utilization,
        "CLUSTER_MEM_UTIL": cluster.cluster_mem_utilization,
        "CLUSTER_GPU_UTIL": cluster.cluster_gpu_utilization,
        "PENDING_PRESSURE": pending_pressure,
        "CLUSTER_UTIL_STD": cluster.cluster_cpu_util_std,

        "RESOURCE_FIT":     resource_fit,

        "CLUSTER_HEALTHY_RATIO": cluster.healthy_node_ratio,

        "NODE_PREEMPTABLE_COUNT": float(preemptable),

        "NODE_OVERCOMMIT_RATIO": max(node.cpu_overcommit_ratio, node.mem_overcommit_ratio),

        "NODE_AFFINITY_CONFLICT": 1.0 if node.has_affinity_conflict(pod) else 0.0,

        "REPLICA_GROUP_COLOCATED": float(cluster.replica_group_count_on_node(
            pod.replica_group, node.node_id,
        )),

        "NAMESPACE_PENDING_RATIO": cluster.namespace_pending_fraction(pod.namespace),
    }
