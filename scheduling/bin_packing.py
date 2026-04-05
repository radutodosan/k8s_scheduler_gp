"""BinPackingStrategy — Best Fit Decreasing scheduling heuristic.

Selects the node where the pod fits with the *least* remaining capacity
after placement.  This packs pods tightly onto fewer nodes, maximising
utilisation per node and leaving other nodes free (useful for
auto-scaling or power-saving scenarios).

This corresponds to the Best Fit Decreasing (BFD) variant of the
classic bin packing problem, widely studied in combinatorial
optimization.
"""

from __future__ import annotations

from typing import Optional

from models.cluster_state import ClusterState
from models.pod import Pod
from scheduling.strategy import ISchedulingStrategy


class BinPackingStrategy(ISchedulingStrategy):
    """Pack pods as tightly as possible onto existing nodes.

    Score per feasible node:
        score = (cpu_util_after + mem_util_after) / 2

    The node with the *highest* score (most loaded after placement)
    is selected.  If GPU is requested, the score becomes:
        score = (cpu_util_after + mem_util_after + gpu_util_after) / 3

    This is the inverse of LeastAllocated — it prefers the fullest
    node that still has room.
    """

    @property
    def name(self) -> str:
        return "BinPacking"

    def select_node(self, pod: Pod, cluster: ClusterState) -> Optional[str]:
        feasible = cluster.feasible_nodes(pod)
        if not feasible:
            return None

        best_id: Optional[str] = None
        best_score = -1.0

        for node in feasible:
            cpu_after = (
                (node.cpu_allocated + pod.cpu_request) / node.cpu_capacity
                if node.cpu_capacity > 0
                else 0.0
            )
            mem_after = (
                (node.mem_allocated + pod.mem_request) / node.mem_capacity
                if node.mem_capacity > 0
                else 0.0
            )

            if node.gpu_capacity > 0 and pod.gpu_request > 0:
                gpu_after = (
                    (node.gpu_allocated + pod.gpu_request) / node.gpu_capacity
                )
                score = (cpu_after + mem_after + gpu_after) / 3.0
            else:
                score = (cpu_after + mem_after) / 2.0

            if score > best_score:
                best_score = score
                best_id = node.node_id

        return best_id
