"""BalancedAllocationStrategy — minimise the gap between CPU and memory utilisation."""

from __future__ import annotations

from typing import Optional

from models.cluster_state import ClusterState
from models.pod import Pod
from scheduling.strategy import ISchedulingStrategy


class BalancedAllocationStrategy(ISchedulingStrategy):
    """Picks the node where CPU and memory utilisation are most balanced.

    Score per node (lower imbalance → higher score):
        1 - |cpu_utilization_after - mem_utilization_after|

    This reflects the Kubernetes ``NodeResourcesBalancedAllocation``
    plugin.  By balancing the two resource dimensions, it avoids
    situations where one resource is exhausted while the other is
    under-used.

    *After-placement* utilisation is estimated by adding the pod's
    request to the node's current allocation before computing the
    score.  This predicts the actual balance if the pod were placed
    there.
    """

    @property
    def name(self) -> str:
        return "BalancedAllocation"

    def select_node(self, pod: Pod, cluster: ClusterState) -> Optional[str]:
        feasible = cluster.feasible_nodes(pod)
        if not feasible:
            return None

        best_id: Optional[str] = None
        best_score = -1.0

        for node in feasible:
            # Predicted utilisation after placing this pod
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
            gpu_after = (
                (node.gpu_allocated + pod.gpu_request) / node.gpu_capacity
                if node.gpu_capacity > 0
                else 0.0
            )
            imbalance = abs(cpu_after - mem_after)
            if node.gpu_capacity > 0:
                imbalance = max(imbalance, abs(cpu_after - gpu_after), abs(mem_after - gpu_after))
            score = 1.0 - imbalance

            if score > best_score:
                best_score = score
                best_id = node.node_id

        return best_id
