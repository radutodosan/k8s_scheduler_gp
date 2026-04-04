"""ClusterState — aggregate view of the Kubernetes cluster at a point in time."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models.node import Node
from models.pod import Pod, PodStatus


@dataclass
class SchedulingResult:
    """Outcome of a single scheduling decision.

    Attributes:
        pod:       The pod that was (or was not) scheduled.
        node_id:   The node it was assigned to (None if rejected).
        success:   True if the pod was placed on a node.
        reason:    Human-readable reason for rejection (empty on success).
    """

    pod: Pod
    node_id: Optional[str]
    success: bool
    reason: str = ""


@dataclass
class ClusterState:
    """Holds the complete mutable state of the simulated cluster.

    Responsibilities:
      - maintain the set of nodes and their resource accounting
      - maintain the pending queue of pods waiting to be scheduled
      - expose cluster-level aggregate metrics (for GP terminals)
      - execute placement / release while keeping accounting consistent

    The ClusterState does NOT decide *where* to place a pod — that is the
    scheduling strategy's job.  It only exposes information and executes
    the final binding.
    """

    nodes: Dict[str, Node] = field(default_factory=dict)
    pending_pods: List[Pod] = field(default_factory=list)

    # ── Bookkeeping for metrics ──────────────────────────────────────
    all_pods: Dict[str, Pod] = field(default_factory=dict)

    # ── Node management ──────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    # ── Pod lifecycle ────────────────────────────────────────────────

    def enqueue_pod(self, pod: Pod) -> None:
        """Add a newly arrived pod to the pending queue."""
        pod.status = PodStatus.PENDING
        self.pending_pods.append(pod)
        self.all_pods[pod.pod_id] = pod

    def bind_pod(self, pod: Pod, node_id: str, current_time: float) -> SchedulingResult:
        """Bind *pod* to *node_id*, updating resource accounting.

        Returns a SchedulingResult reflecting the outcome.
        """
        node = self.nodes[node_id]
        if not node.can_fit(pod):
            return SchedulingResult(
                pod=pod,
                node_id=None,
                success=False,
                reason=(
                    f"Insufficient resources on {node_id}: "
                    f"need cpu={pod.cpu_request} mem={pod.mem_request}, "
                    f"avail cpu={node.cpu_available} mem={node.mem_available}"
                ),
            )

        node.allocate(pod)
        pod.schedule_on(node_id, current_time)
        # Remove from pending queue
        self.pending_pods = [p for p in self.pending_pods if p.pod_id != pod.pod_id]
        return SchedulingResult(pod=pod, node_id=node_id, success=True)

    def release_pod(self, pod: Pod, current_time: float) -> None:
        """Release resources when *pod* completes or is evicted."""
        if pod.assigned_node_id and pod.assigned_node_id in self.nodes:
            self.nodes[pod.assigned_node_id].release(pod)
        pod.complete(current_time)

    def reject_pod(self, pod: Pod, current_time: float, reason: str = "") -> SchedulingResult:
        """Mark a pod as rejected and remove it from the pending queue."""
        pod.reject(current_time)
        self.pending_pods = [p for p in self.pending_pods if p.pod_id != pod.pod_id]
        return SchedulingResult(pod=pod, node_id=None, success=False, reason=reason)

    # ── Feasibility helpers ──────────────────────────────────────────

    def feasible_nodes(self, pod: Pod) -> List[Node]:
        """Return all available nodes that have sufficient resources for *pod*."""
        return [n for n in self.nodes.values() if n.is_available and n.can_fit(pod)]

    def evict_pods_from_node(self, node_id: str) -> List[Pod]:
        """Release all pods from a node (for failure handling).

        Returns pods sorted by eviction priority (BEST_EFFORT first,
        GUARANTEED last), matching the Kubernetes eviction order.
        Does NOT modify pod status — caller handles re-enqueue.
        """
        node = self.nodes[node_id]
        pods_to_evict = sorted(node.pods.values(), key=lambda p: p.qos_class.value)
        for pod in pods_to_evict:
            node.release(pod)
        return pods_to_evict

    # ── Cluster-level aggregates (used as GP terminals) ──────────────

    @property
    def total_cpu_capacity(self) -> float:
        return sum(n.cpu_capacity for n in self.nodes.values())

    @property
    def total_cpu_allocated(self) -> float:
        return sum(n.cpu_allocated for n in self.nodes.values())

    @property
    def total_mem_capacity(self) -> float:
        return sum(n.mem_capacity for n in self.nodes.values())

    @property
    def total_mem_allocated(self) -> float:
        return sum(n.mem_allocated for n in self.nodes.values())

    @property
    def total_gpu_capacity(self) -> float:
        return sum(n.gpu_capacity for n in self.nodes.values())

    @property
    def total_gpu_allocated(self) -> float:
        return sum(n.gpu_allocated for n in self.nodes.values())

    @property
    def cluster_cpu_utilization(self) -> float:
        cap = self.total_cpu_capacity
        return self.total_cpu_allocated / cap if cap > 0 else 0.0

    @property
    def cluster_mem_utilization(self) -> float:
        cap = self.total_mem_capacity
        return self.total_mem_allocated / cap if cap > 0 else 0.0

    @property
    def cluster_gpu_utilization(self) -> float:
        cap = self.total_gpu_capacity
        return self.total_gpu_allocated / cap if cap > 0 else 0.0

    @property
    def pending_count(self) -> int:
        return len(self.pending_pods)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def scheduled_pod_count(self) -> int:
        return sum(1 for p in self.all_pods.values() if p.status == PodStatus.SCHEDULED)

    @property
    def completed_pod_count(self) -> int:
        return sum(1 for p in self.all_pods.values() if p.status == PodStatus.COMPLETED)

    @property
    def rejected_pod_count(self) -> int:
        return sum(1 for p in self.all_pods.values() if p.status == PodStatus.REJECTED)

    @property
    def available_node_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.is_available)

    @property
    def healthy_node_ratio(self) -> float:
        """Fraction of nodes that are currently available (0.0–1.0)."""
        if not self.nodes:
            return 1.0
        return self.available_node_count / len(self.nodes)

    @property
    def cluster_cpu_util_variance(self) -> float:
        """Variance of CPU utilization across available nodes."""
        nodes = [n for n in self.nodes.values() if n.is_available]
        if len(nodes) < 2:
            return 0.0
        mean = sum(n.cpu_utilization for n in nodes) / len(nodes)
        return sum((n.cpu_utilization - mean) ** 2 for n in nodes) / len(nodes)

    @property
    def cluster_cpu_util_std(self) -> float:
        """Standard deviation of CPU utilization across available nodes."""
        return math.sqrt(self.cluster_cpu_util_variance)

    def namespace_pending_fraction(self, namespace: str) -> float:
        """Fraction of pending pods that belong to *namespace*."""
        total = len(self.pending_pods)
        if total == 0:
            return 0.0
        same = sum(1 for p in self.pending_pods if p.namespace == namespace)
        return same / total

    def replica_group_count_on_node(self, replica_group: str, node_id: str) -> int:
        """Count pods from *replica_group* running on *node_id*."""
        if not replica_group or node_id not in self.nodes:
            return 0
        return sum(
            1 for p in self.nodes[node_id].pods.values()
            if p.replica_group == replica_group
        )
