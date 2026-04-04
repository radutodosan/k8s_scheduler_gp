"""RoundRobinStrategy — cycles through nodes in fixed order."""

from __future__ import annotations

from typing import Optional

from models.cluster_state import ClusterState
from models.pod import Pod
from scheduling.strategy import ISchedulingStrategy


class RoundRobinStrategy(ISchedulingStrategy):
    """Assigns pods to nodes in a cyclic order, skipping infeasible ones.

    Maintains an internal index that advances after each successful
    placement, distributing pods evenly across nodes.
    """

    def __init__(self) -> None:
        self._index: int = 0
        self._node_ids: list[str] = []

    @property
    def name(self) -> str:
        return "RoundRobin"

    def on_episode_start(self, cluster: ClusterState) -> None:
        self._node_ids = sorted(cluster.nodes.keys())
        self._index = 0

    def select_node(self, pod: Pod, cluster: ClusterState) -> Optional[str]:
        if not self._node_ids:
            self._node_ids = sorted(cluster.nodes.keys())
        n = len(self._node_ids)
        if n == 0:
            return None

        # Try each node starting from current index
        for _ in range(n):
            node_id = self._node_ids[self._index % n]
            self._index = (self._index + 1) % n
            node = cluster.nodes[node_id]
            if node.can_fit(pod):
                return node_id

        return None
