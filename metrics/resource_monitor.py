"""ResourceMonitor — records per-node resource snapshots over simulation time.

Captures timestamped CPU/MEM utilization for every node at each
scheduling cycle, enabling time-series visualisation of cluster
resource dynamics (dissertation section 5.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.cluster_state import ClusterState


@dataclass
class ResourceSnapshot:
    """A single point-in-time observation of cluster resources."""

    timestamp: float
    # Per-node utilisation: {node_id: value}
    node_cpu_util: Dict[str, float] = field(default_factory=dict)
    node_mem_util: Dict[str, float] = field(default_factory=dict)
    node_cpu_free: Dict[str, float] = field(default_factory=dict)
    node_mem_free: Dict[str, float] = field(default_factory=dict)
    node_pod_count: Dict[str, int] = field(default_factory=dict)
    node_available: Dict[str, bool] = field(default_factory=dict)
    # Cluster aggregates
    cluster_cpu_util: float = 0.0
    cluster_mem_util: float = 0.0
    cluster_cpu_free: float = 0.0
    cluster_mem_free: float = 0.0
    cpu_util_variance: float = 0.0
    pending_count: int = 0
    completed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "node_cpu_util": {k: round(v, 4) for k, v in self.node_cpu_util.items()},
            "node_mem_util": {k: round(v, 4) for k, v in self.node_mem_util.items()},
            "node_cpu_free": {k: round(v, 4) for k, v in self.node_cpu_free.items()},
            "node_mem_free": {k: round(v, 4) for k, v in self.node_mem_free.items()},
            "node_pod_count": dict(self.node_pod_count),
            "node_available": dict(self.node_available),
            "cluster_cpu_util": round(self.cluster_cpu_util, 4),
            "cluster_mem_util": round(self.cluster_mem_util, 4),
            "cluster_cpu_free": round(self.cluster_cpu_free, 4),
            "cluster_mem_free": round(self.cluster_mem_free, 4),
            "cpu_util_variance": round(self.cpu_util_variance, 6),
            "pending_count": self.pending_count,
            "completed_count": self.completed_count,
        }


class ResourceMonitor:
    """Collects time-series resource data during a simulation run.

    Usage::

        monitor = ResourceMonitor()
        # During simulation (called by engine):
        monitor.record_snapshot(current_time, cluster)
        # After simulation:
        timeline = monitor.get_timeline()
        monitor.export_json(path)
    """

    def __init__(self) -> None:
        self._snapshots: List[ResourceSnapshot] = []

    @property
    def snapshots(self) -> List[ResourceSnapshot]:
        return self._snapshots

    def record_snapshot(self, timestamp: float, cluster: ClusterState) -> None:
        """Capture current resource state from the cluster."""
        snap = ResourceSnapshot(timestamp=timestamp)

        for node_id, node in cluster.nodes.items():
            snap.node_cpu_util[node_id] = node.cpu_utilization
            snap.node_mem_util[node_id] = node.mem_utilization
            snap.node_cpu_free[node_id] = node.cpu_available
            snap.node_mem_free[node_id] = node.mem_available
            snap.node_pod_count[node_id] = node.pod_count
            snap.node_available[node_id] = node.is_available

        snap.cluster_cpu_util = cluster.cluster_cpu_utilization
        snap.cluster_mem_util = cluster.cluster_mem_utilization
        snap.cluster_cpu_free = cluster.total_cpu_capacity - cluster.total_cpu_allocated
        snap.cluster_mem_free = cluster.total_mem_capacity - cluster.total_mem_allocated
        snap.cpu_util_variance = cluster.cluster_cpu_util_variance
        snap.pending_count = cluster.pending_count
        snap.completed_count = cluster.completed_pod_count

        self._snapshots.append(snap)

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Return all snapshots as a list of dicts (JSON-serialisable)."""
        return [s.to_dict() for s in self._snapshots]

    def get_node_ids(self) -> List[str]:
        """Return sorted list of node IDs observed across all snapshots."""
        ids: set[str] = set()
        for s in self._snapshots:
            ids.update(s.node_cpu_util.keys())
        return sorted(ids)

    def get_timestamps(self) -> List[float]:
        """Return ordered list of snapshot timestamps."""
        return [s.timestamp for s in self._snapshots]

    def get_node_cpu_series(self, node_id: str) -> List[float]:
        """Return CPU utilization time-series for a specific node."""
        return [s.node_cpu_util.get(node_id, 0.0) for s in self._snapshots]

    def get_node_mem_series(self, node_id: str) -> List[float]:
        """Return MEM utilization time-series for a specific node."""
        return [s.node_mem_util.get(node_id, 0.0) for s in self._snapshots]

    def get_cluster_cpu_series(self) -> List[float]:
        """Return cluster-wide CPU utilization time-series."""
        return [s.cluster_cpu_util for s in self._snapshots]

    def get_cluster_mem_series(self) -> List[float]:
        """Return cluster-wide MEM utilization time-series."""
        return [s.cluster_mem_util for s in self._snapshots]

    def get_pending_series(self) -> List[int]:
        """Return pending pod count time-series."""
        return [s.pending_count for s in self._snapshots]

    def get_cluster_cpu_free_series(self) -> List[float]:
        """Return cluster-wide free CPU (absolute cores) time-series."""
        return [s.cluster_cpu_free for s in self._snapshots]

    def get_cluster_mem_free_series(self) -> List[float]:
        """Return cluster-wide free memory (MiB) time-series."""
        return [s.cluster_mem_free for s in self._snapshots]

    def get_cpu_util_variance_series(self) -> List[float]:
        """Return CPU utilization variance time-series."""
        return [s.cpu_util_variance for s in self._snapshots]

    def get_completed_series(self) -> List[int]:
        """Return completed pod count time-series."""
        return [s.completed_count for s in self._snapshots]

    def get_failure_timestamps(self) -> List[float]:
        """Return timestamps where at least one node became unavailable."""
        failures: List[float] = []
        prev_available: Dict[str, bool] = {}
        for s in self._snapshots:
            for nid, avail in s.node_available.items():
                if not avail and prev_available.get(nid, True):
                    failures.append(s.timestamp)
                    break
            prev_available = dict(s.node_available)
        return failures

    def get_recovery_timestamps(self) -> List[float]:
        """Return timestamps where a node recovered from failure."""
        recoveries: List[float] = []
        prev_available: Dict[str, bool] = {}
        for s in self._snapshots:
            for nid, avail in s.node_available.items():
                if avail and not prev_available.get(nid, True):
                    recoveries.append(s.timestamp)
                    break
            prev_available = dict(s.node_available)
        return recoveries

    def throughput(self) -> float:
        """Compute throughput as completed_pods / simulation_duration (pods/s).

        Returns 0.0 if no snapshots or duration is zero.
        """
        if len(self._snapshots) < 2:
            return 0.0
        duration = self._snapshots[-1].timestamp - self._snapshots[0].timestamp
        if duration <= 0:
            return 0.0
        completed = self._snapshots[-1].completed_count
        return completed / duration

    def reset(self) -> None:
        """Clear all recorded snapshots."""
        self._snapshots.clear()

    def export_json(self, path) -> None:
        """Export the full timeline to a JSON file."""
        import json
        from pathlib import Path
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.get_timeline(), fh, indent=2)
