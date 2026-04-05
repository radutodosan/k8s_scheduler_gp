"""MetricsCollector — gathers per-run simulation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from models.cluster_state import ClusterState, SchedulingResult
from models.pod import Pod, PodStatus


@dataclass
class SchedulingMetrics:
    """Aggregated metrics for a single simulation run.

    Matches the evaluation criteria defined in the dissertation:
      - scheduling_success_rate
      - avg_wait_time  /  wait_time percentiles
      - avg_cpu_utilization / avg_mem_utilization
      - rejection_count (and reasons + timeline)
      - fairness (per-namespace / per-priority breakdown)
      - preemption / eviction counts
      - throughput
    """

    total_pods: int = 0
    scheduled_pods: int = 0
    completed_pods: int = 0
    rejected_pods: int = 0
    evicted_pods: int = 0
    preemption_count: int = 0
    node_failure_count: int = 0

    # Throughput tracking
    simulation_duration: float = 0.0

    # Cost tracking: Σ(node.cost_per_hour × uptime_hours) for all nodes
    total_cost: float = 0.0

    total_wait_time: float = 0.0
    per_pod_wait_times: List[float] = field(default_factory=list)
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    rejection_timeline: List[Tuple[float, str]] = field(default_factory=list)

    # Scheduling attempt counts per pod (pod_id → attempts)
    scheduling_attempts: Dict[str, int] = field(default_factory=dict)

    # Resource utilisation samples (snapshot each scheduling cycle)
    cpu_util_samples: List[float] = field(default_factory=list)
    mem_util_samples: List[float] = field(default_factory=list)
    gpu_util_samples: List[float] = field(default_factory=list)

    # Per-namespace counts for fairness analysis
    pods_per_namespace: Dict[str, int] = field(default_factory=dict)
    scheduled_per_namespace: Dict[str, int] = field(default_factory=dict)

    # Per-priority counts
    pods_per_priority: Dict[int, int] = field(default_factory=dict)
    scheduled_per_priority: Dict[int, int] = field(default_factory=dict)

    # ── Derived metrics ──────────────────────────────────────────────

    @property
    def scheduling_success_rate(self) -> float:
        if self.total_pods == 0:
            return 0.0
        return self.scheduled_pods / self.total_pods

    @property
    def avg_wait_time(self) -> float:
        if self.scheduled_pods == 0:
            return 0.0
        return self.total_wait_time / self.scheduled_pods

    def wait_time_percentile(self, p: float) -> float:
        """Return the *p*-th percentile (0–100) of per-pod wait times."""
        if not self.per_pod_wait_times:
            return 0.0
        sorted_wt = sorted(self.per_pod_wait_times)
        k = (p / 100.0) * (len(sorted_wt) - 1)
        lo = int(math.floor(k))
        hi = min(lo + 1, len(sorted_wt) - 1)
        frac = k - lo
        return sorted_wt[lo] + frac * (sorted_wt[hi] - sorted_wt[lo])

    @property
    def wait_time_p50(self) -> float:
        return self.wait_time_percentile(50)

    @property
    def wait_time_p90(self) -> float:
        return self.wait_time_percentile(90)

    @property
    def wait_time_p95(self) -> float:
        return self.wait_time_percentile(95)

    @property
    def wait_time_p99(self) -> float:
        return self.wait_time_percentile(99)

    @property
    def avg_cpu_utilization(self) -> float:
        if not self.cpu_util_samples:
            return 0.0
        return sum(self.cpu_util_samples) / len(self.cpu_util_samples)

    @property
    def avg_mem_utilization(self) -> float:
        if not self.mem_util_samples:
            return 0.0
        return sum(self.mem_util_samples) / len(self.mem_util_samples)

    @property
    def avg_gpu_utilization(self) -> float:
        if not self.gpu_util_samples:
            return 0.0
        return sum(self.gpu_util_samples) / len(self.gpu_util_samples)

    @property
    def throughput(self) -> float:
        """Completed pods per unit time (pods/s)."""
        if self.simulation_duration <= 0:
            return 0.0
        return self.completed_pods / self.simulation_duration

    @property
    def avg_scheduling_attempts(self) -> float:
        """Mean scheduling attempts across all pods that were attempted."""
        if not self.scheduling_attempts:
            return 0.0
        return sum(self.scheduling_attempts.values()) / len(self.scheduling_attempts)

    @property
    def cost_per_pod(self) -> float:
        """Average infrastructure cost per completed pod."""
        if self.completed_pods == 0:
            return 0.0
        return self.total_cost / self.completed_pods

    def to_dict(self) -> Dict[str, object]:
        """Flat dictionary suitable for CSV / JSON export."""
        return {
            "total_pods": self.total_pods,
            "scheduled_pods": self.scheduled_pods,
            "completed_pods": self.completed_pods,
            "rejected_pods": self.rejected_pods,
            "scheduling_success_rate": round(self.scheduling_success_rate, 4),
            "avg_wait_time": round(self.avg_wait_time, 4),
            "wait_time_p50": round(self.wait_time_p50, 4),
            "wait_time_p90": round(self.wait_time_p90, 4),
            "wait_time_p95": round(self.wait_time_p95, 4),
            "wait_time_p99": round(self.wait_time_p99, 4),
            "avg_cpu_utilization": round(self.avg_cpu_utilization, 4),
            "avg_mem_utilization": round(self.avg_mem_utilization, 4),
            "avg_gpu_utilization": round(self.avg_gpu_utilization, 4),
            "throughput": round(self.throughput, 4),
            "evicted_pods": self.evicted_pods,
            "preemption_count": self.preemption_count,
            "node_failure_count": self.node_failure_count,
            "avg_scheduling_attempts": round(self.avg_scheduling_attempts, 2),
            "total_cost": round(self.total_cost, 4),
            "cost_per_pod": round(self.cost_per_pod, 4),
            "rejection_reasons": dict(self.rejection_reasons),
        }


class MetricsCollector:
    """Records events during a simulation run and produces SchedulingMetrics."""

    def __init__(self) -> None:
        self._metrics = SchedulingMetrics()
        self._current_time: float = 0.0

    def set_time(self, t: float) -> None:
        """Update the collector's clock (called by the engine each event)."""
        self._current_time = t

    def record_pod_arrival(self, pod: Pod) -> None:
        self._metrics.total_pods += 1
        ns = pod.namespace
        self._metrics.pods_per_namespace[ns] = self._metrics.pods_per_namespace.get(ns, 0) + 1
        pri = pod.priority
        self._metrics.pods_per_priority[pri] = self._metrics.pods_per_priority.get(pri, 0) + 1

    def record_scheduling_attempt(self, pod_id: str) -> None:
        """Increment the scheduling attempt count for a pod."""
        self._metrics.scheduling_attempts[pod_id] = (
            self._metrics.scheduling_attempts.get(pod_id, 0) + 1
        )

    def record_scheduling_result(self, result: SchedulingResult) -> None:
        if result.success:
            self._metrics.scheduled_pods += 1
            wt = result.pod.wait_time
            self._metrics.total_wait_time += wt
            self._metrics.per_pod_wait_times.append(wt)
            ns = result.pod.namespace
            self._metrics.scheduled_per_namespace[ns] = (
                self._metrics.scheduled_per_namespace.get(ns, 0) + 1
            )
            pri = result.pod.priority
            self._metrics.scheduled_per_priority[pri] = (
                self._metrics.scheduled_per_priority.get(pri, 0) + 1
            )
        else:
            self._metrics.rejected_pods += 1
            reason = result.reason or "unknown"
            self._metrics.rejection_reasons[reason] = (
                self._metrics.rejection_reasons.get(reason, 0) + 1
            )
            self._metrics.rejection_timeline.append((self._current_time, reason))

    def record_pod_completion(self, pod: Pod) -> None:
        self._metrics.completed_pods += 1

    def record_pod_rejection(self, pod: Pod, reason: str = "killed_by_failure") -> None:
        """Record a pod killed / permanently rejected outside normal scheduling."""
        self._metrics.rejected_pods += 1
        self._metrics.rejection_reasons[reason] = (
            self._metrics.rejection_reasons.get(reason, 0) + 1
        )
        self._metrics.rejection_timeline.append((self._current_time, reason))

    def sample_utilization(self, cluster: ClusterState) -> None:
        """Take a snapshot of cluster resource utilisation."""
        self._metrics.cpu_util_samples.append(cluster.cluster_cpu_utilization)
        self._metrics.mem_util_samples.append(cluster.cluster_mem_utilization)
        self._metrics.gpu_util_samples.append(cluster.cluster_gpu_utilization)

    def record_pod_eviction(self, pod: Pod) -> None:
        """Record that a pod was evicted from a failed node."""
        self._metrics.evicted_pods += 1

    def record_preemption(self, victim: Pod, preemptor: Pod) -> None:
        """Record that *victim* was preempted by *preemptor*."""
        self._metrics.preemption_count += 1

    def record_node_failure(self) -> None:
        """Record a node failure event."""
        self._metrics.node_failure_count += 1

    def get_metrics(self) -> SchedulingMetrics:
        return self._metrics

    def reset(self) -> None:
        self._metrics = SchedulingMetrics()
        self._current_time = 0.0
