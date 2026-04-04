"""FirstFitStrategy — picks the first node that has enough resources."""

from __future__ import annotations

from typing import Optional

from models.cluster_state import ClusterState
from models.pod import Pod
from scheduling.strategy import ISchedulingStrategy


class FirstFitStrategy(ISchedulingStrategy):
    """Iterates over nodes in a deterministic order and returns the first
    one that can accommodate the pod.

    Fast and simple — O(N) worst-case where N is node count.
    """

    @property
    def name(self) -> str:
        return "FirstFit"

    def select_node(self, pod: Pod, cluster: ClusterState) -> Optional[str]:
        for node_id in sorted(cluster.nodes.keys()):
            if cluster.nodes[node_id].can_fit(pod):
                return node_id
        return None
