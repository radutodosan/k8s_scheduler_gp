"""MostAllocatedStrategy — bin-packing: fill up nodes before using new ones."""

from __future__ import annotations

from typing import Optional

from models.cluster_state import ClusterState
from models.pod import Pod
from scheduling.strategy import ISchedulingStrategy


class MostAllocatedStrategy(ISchedulingStrategy):
    """Selects the feasible node with the *least* available resources.

    Score per node:
        ((cpu_utilization) + (mem_utilization)) / 2

    Opposite of LeastAllocated — packs pods tightly to minimise the
    number of active nodes (useful for cost reduction / autoscaling).
    """

    @property
    def name(self) -> str:
        return "MostAllocated"

    def select_node(self, pod: Pod, cluster: ClusterState) -> Optional[str]:
        feasible = cluster.feasible_nodes(pod)
        if not feasible:
            return None

        best_id: Optional[str] = None
        best_score = -1.0

        for node in feasible:
            if node.gpu_capacity > 0:
                score = (node.cpu_utilization + node.mem_utilization + node.gpu_utilization) / 3.0
            else:
                score = (node.cpu_utilization + node.mem_utilization) / 2.0

            if score > best_score:
                best_score = score
                best_id = node.node_id

        return best_id
