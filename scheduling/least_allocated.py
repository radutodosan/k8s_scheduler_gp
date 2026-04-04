"""LeastAllocatedStrategy — Kubernetes-style scoring favoring the emptiest node."""

from __future__ import annotations

from typing import Optional

from models.cluster_state import ClusterState
from models.pod import Pod
from scheduling.strategy import ISchedulingStrategy


class LeastAllocatedStrategy(ISchedulingStrategy):
    """Selects the feasible node with the most available resources.

    Score per node:
        ((cpu_available / cpu_capacity) + (mem_available / mem_capacity)) / 2

    This is the default scoring plugin in the Kubernetes scheduler
    (``NodeResourcesLeastAllocated``).  It spreads the load, maximising
    headroom for future pods at the cost of lower packing efficiency.
    """

    @property
    def name(self) -> str:
        return "LeastAllocated"

    def select_node(self, pod: Pod, cluster: ClusterState) -> Optional[str]:
        feasible = cluster.feasible_nodes(pod)
        if not feasible:
            return None

        best_id: Optional[str] = None
        best_score = -1.0

        for node in feasible:
            cpu_frac = node.cpu_available / node.cpu_capacity if node.cpu_capacity > 0 else 0.0
            mem_frac = node.mem_available / node.mem_capacity if node.mem_capacity > 0 else 0.0
            gpu_frac = node.gpu_available / node.gpu_capacity if node.gpu_capacity > 0 else 0.0

            if node.gpu_capacity > 0:
                score = (cpu_frac + mem_frac + gpu_frac) / 3.0
            else:
                score = (cpu_frac + mem_frac) / 2.0

            if score > best_score:
                best_score = score
                best_id = node.node_id

        return best_id
