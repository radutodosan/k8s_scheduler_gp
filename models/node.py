"""Node model — represents a Kubernetes worker node with resource accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet

from models.pod import Pod


@dataclass
class Node:
    """A Kubernetes worker node with capacity tracking.

    Resource accounting follows the Kubernetes model:
      allocatable = capacity - system reserved
    We simplify by treating capacity as the allocatable amount directly.

    Attributes:
        node_id:       Unique identifier.
        cpu_capacity:  Total allocatable CPU cores.
        mem_capacity:  Total allocatable memory (MiB).
        cpu_allocated: CPU currently allocated to pods.
        mem_allocated: Memory currently allocated to pods.
        cost_per_hour: Hourly cost (for cost-aware scheduling).
        pods:          Pods currently running on this node, keyed by pod_id.
    """

    node_id: str
    cpu_capacity: float
    mem_capacity: float
    gpu_capacity: float = 0.0
    cost_per_hour: float = 1.0
    taints: FrozenSet[str] = field(default_factory=frozenset)
    labels: Dict[str, str] = field(default_factory=dict)

    # Mutable state
    cpu_allocated: float = 0.0
    mem_allocated: float = 0.0
    gpu_allocated: float = 0.0
    cpu_limit_total: float = 0.0   # sum of effective limits (for overcommit)
    mem_limit_total: float = 0.0
    pods: Dict[str, Pod] = field(default_factory=dict)
    is_available: bool = True

    # ── Derived properties ───────────────────────────────────────────

    @property
    def cpu_available(self) -> float:
        return self.cpu_capacity - self.cpu_allocated

    @property
    def mem_available(self) -> float:
        return self.mem_capacity - self.mem_allocated

    @property
    def gpu_available(self) -> float:
        return self.gpu_capacity - self.gpu_allocated

    @property
    def cpu_utilization(self) -> float:
        """CPU utilization ratio (0.0 – 1.0)."""
        if self.cpu_capacity == 0:
            return 0.0
        return self.cpu_allocated / self.cpu_capacity

    @property
    def mem_utilization(self) -> float:
        """Memory utilization ratio (0.0 – 1.0)."""
        if self.mem_capacity == 0:
            return 0.0
        return self.mem_allocated / self.mem_capacity

    @property
    def gpu_utilization(self) -> float:
        """GPU utilization ratio (0.0 – 1.0)."""
        if self.gpu_capacity == 0:
            return 0.0
        return self.gpu_allocated / self.gpu_capacity

    @property
    def pod_count(self) -> int:
        return len(self.pods)

    @property
    def cpu_overcommit_ratio(self) -> float:
        """Ratio of total CPU limits to capacity (>1 means overcommitted)."""
        if self.cpu_capacity == 0:
            return 0.0
        return self.cpu_limit_total / self.cpu_capacity

    @property
    def mem_overcommit_ratio(self) -> float:
        """Ratio of total memory limits to capacity."""
        if self.mem_capacity == 0:
            return 0.0
        return self.mem_limit_total / self.mem_capacity

    # ── Feasibility check ────────────────────────────────────────────

    def can_fit(self, pod: Pod) -> bool:
        """Return True if the node has enough resources for *pod*.

        Checks:
          1. Resource capacity (CPU + memory)
          2. Taints & tolerations — pod must tolerate all node taints
          3. Node selector — pod's label requirements must match node labels
        """
        if not pod.fits_on(self.cpu_available, self.mem_available, self.gpu_available):
            return False
        if self.taints and not self.taints.issubset(pod.tolerations):
            return False
        if pod.node_selector:
            for key, val in pod.node_selector.items():
                if self.labels.get(key) != val:
                    return False
        # Anti-affinity: reject if a pod with the same key is already here
        if pod.anti_affinity_key:
            for existing in self.pods.values():
                if existing.anti_affinity_key == pod.anti_affinity_key:
                    return False
        return True

    # ── Allocation / Release ─────────────────────────────────────────

    def allocate(self, pod: Pod) -> None:
        """Reserve resources for *pod* on this node.

        Raises ValueError if the pod does not fit.
        """
        if not self.can_fit(pod):
            raise ValueError(
                f"Pod {pod.pod_id} does not fit on node {self.node_id} "
                f"(need cpu={pod.cpu_request}, mem={pod.mem_request}; "
                f"avail cpu={self.cpu_available}, mem={self.mem_available})"
            )
        self.cpu_allocated += pod.cpu_request
        self.mem_allocated += pod.mem_request
        self.gpu_allocated += pod.gpu_request
        self.cpu_limit_total += pod.effective_cpu_limit
        self.mem_limit_total += pod.effective_mem_limit
        self.pods[pod.pod_id] = pod

    def release(self, pod: Pod) -> None:
        """Free resources held by *pod* on this node."""
        self.cpu_allocated = max(0.0, self.cpu_allocated - pod.cpu_request)
        self.mem_allocated = max(0.0, self.mem_allocated - pod.mem_request)
        self.gpu_allocated = max(0.0, self.gpu_allocated - pod.gpu_request)
        self.cpu_limit_total = max(0.0, self.cpu_limit_total - pod.effective_cpu_limit)
        self.mem_limit_total = max(0.0, self.mem_limit_total - pod.effective_mem_limit)
        self.pods.pop(pod.pod_id, None)

    def has_affinity_conflict(self, pod: Pod) -> bool:
        """True if a pod with the same anti_affinity_key is already on this node."""
        if not pod.anti_affinity_key:
            return False
        return any(p.anti_affinity_key == pod.anti_affinity_key for p in self.pods.values())

    # ── Availability ─────────────────────────────────────────────────

    def mark_failed(self) -> None:
        """Mark this node as unavailable (simulating a node failure)."""
        self.is_available = False

    def mark_recovered(self) -> None:
        """Mark this node as available again after recovery."""
        self.is_available = True

    def tolerates(self, pod: Pod) -> bool:
        """Check whether *pod* tolerates all of this node's taints."""
        return self.taints.issubset(pod.tolerations)

    def matches_selector(self, pod: Pod) -> bool:
        """Check whether node labels satisfy *pod*'s node_selector."""
        return all(self.labels.get(k) == v for k, v in pod.node_selector.items())
