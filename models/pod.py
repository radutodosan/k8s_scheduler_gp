"""Pod model — represents a Kubernetes Pod with resource requests and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, Optional


class QoSClass(IntEnum):
    """Kubernetes Quality-of-Service classes (ascending priority for eviction).

    BestEffort pods are evicted first, Guaranteed pods last.
    """

    BEST_EFFORT = 1
    BURSTABLE = 2
    GUARANTEED = 3


class PodStatus(IntEnum):
    """Lifecycle status of a Pod within the simulator."""

    PENDING = 0       # In scheduling queue, waiting for a node
    SCHEDULED = 1     # Assigned to a node, running
    COMPLETED = 2     # Finished successfully
    REJECTED = 3      # Could not be scheduled (no feasible node)
    EVICTED = 4       # Removed from node (resource pressure — future)


@dataclass
class Pod:
    """A Kubernetes Pod with resource requests and scheduling metadata.

    Attributes:
        pod_id:       Unique identifier.
        cpu_request:  CPU cores requested (e.g. 0.5 = 500m).
        mem_request:  Memory requested in MiB.
        priority:     Scheduling priority (0–1000, higher = more important).
        qos_class:    QoS tier (determines eviction order).
        arrival_time: Simulation time when the pod enters the pending queue.
        duration:     Expected runtime in simulation time-units (0 = unknown).
        namespace:    Logical grouping (for fairness analysis).
        status:       Current lifecycle status.
        scheduled_time: Time when pod was assigned to a node (None if not yet).
        completion_time: Time when pod finished (None if not yet).
        assigned_node_id: ID of the node the pod is running on (None if pending).
    """

    pod_id: str
    cpu_request: float
    mem_request: float
    priority: int = 0
    qos_class: QoSClass = QoSClass.BEST_EFFORT
    arrival_time: float = 0.0
    duration: float = 0.0
    namespace: str = "default"
    tolerations: FrozenSet[str] = field(default_factory=frozenset)
    node_selector: Dict[str, str] = field(default_factory=dict)
    cpu_limit: float = 0.0    # 0 = same as request (no overcommit)
    mem_limit: float = 0.0    # 0 = same as request (no overcommit)
    gpu_request: float = 0.0     # GPUs requested (e.g. 1.0 = 1 GPU)
    anti_affinity_key: str = ""  # empty = no anti-affinity constraint
    workload_type: str = ""       # profile tag: web_serving, ai_training, etc.
    replica_group: str = ""       # deployment/replicaset grouping tag

    # Mutable state — set during simulation
    status: PodStatus = PodStatus.PENDING
    scheduled_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_node_id: Optional[str] = None
    _time_executed: float = field(default=0.0, repr=False)

    # ── Derived properties ───────────────────────────────────────────

    @property
    def effective_cpu_limit(self) -> float:
        """Actual CPU limit (falls back to request when unset)."""
        return self.cpu_limit if self.cpu_limit > 0 else self.cpu_request

    @property
    def effective_mem_limit(self) -> float:
        """Actual memory limit (falls back to request when unset)."""
        return self.mem_limit if self.mem_limit > 0 else self.mem_request

    @property
    def wait_time(self) -> float:
        """Time spent in the pending queue (0 if not yet scheduled)."""
        if self.scheduled_time is None:
            return 0.0
        return self.scheduled_time - self.arrival_time

    def fits_on(self, cpu_available: float, mem_available: float,
                gpu_available: float = float('inf')) -> bool:
        """Check whether the pod's requests fit within the given resources."""
        return (self.cpu_request <= cpu_available
                and self.mem_request <= mem_available
                and self.gpu_request <= gpu_available)

    @property
    def remaining_duration(self) -> float:
        """Duration remaining after accounting for previously elapsed execution time."""
        return max(0.0, self.duration - self._time_executed)

    # ── Mutation helpers (called by simulator) ───────────────────────

    def schedule_on(self, node_id: str, current_time: float) -> None:
        """Mark this pod as scheduled on *node_id* at *current_time*."""
        self.status = PodStatus.SCHEDULED
        self.assigned_node_id = node_id
        self.scheduled_time = current_time

    def complete(self, current_time: float) -> None:
        """Mark this pod as successfully completed."""
        self.status = PodStatus.COMPLETED
        self.completion_time = current_time

    def reject(self, current_time: float) -> None:
        """Mark this pod as rejected (no feasible node found)."""
        self.status = PodStatus.REJECTED
        self.completion_time = current_time

    def evict(self, current_time: float) -> None:
        """Reset this pod for rescheduling after eviction from a failed node.

        Accumulates the time already executed so that the remaining
        duration is shortened accordingly (avoids infinite reschedule loops).
        """
        if self.scheduled_time is not None:
            self._time_executed += current_time - self.scheduled_time
        self.status = PodStatus.PENDING
        self.assigned_node_id = None
        self.scheduled_time = None

    def add_restart_overhead(self, overhead: float) -> None:
        """Increase the pod's total duration by *overhead* time units.

        Models the extra cost of restarting a pod on a new node after
        eviction (container image pull, init, health-check warm-up).
        Only meaningful when the pod will be rescheduled.
        """
        self.duration += overhead
