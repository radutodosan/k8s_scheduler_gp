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

    **Fit-tightness score** per feasible node:

        fit_cpu = pod.cpu_request / node.cpu_available
        fit_mem = pod.mem_request / node.mem_available
        score   = (fit_cpu + fit_mem) / 2

    The node where the pod uses the *highest fraction of what is still
    available* is selected.  This matches the Best Fit Decreasing (BFD)
    semantics: place each item in the bin where it fits the tightest,
    preserving large gaps on other nodes for future large items.

    **Why this differs from MostAllocated:**
    MostAllocated scores by current utilisation (allocated / capacity).
    The ranking it produces is identical to utilisation_after for
    equal-capacity nodes, but diverges on heterogeneous clusters.
    Fit-tightness naturally routes CPU-hungry pods to CPU-heavy nodes
    and memory-hungry pods to memory-heavy nodes, reducing fragmentation.
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
            # Fraction of *available* CPU / MEM the pod would consume
            fit_cpu = (
                pod.cpu_request / node.cpu_available
                if node.cpu_available > 0
                else 1.0
            )
            fit_mem = (
                pod.mem_request / node.mem_available
                if node.mem_available > 0
                else 1.0
            )

            if node.gpu_capacity > 0 and pod.gpu_request > 0 and node.gpu_available > 0:
                fit_gpu = pod.gpu_request / node.gpu_available
                score = (fit_cpu + fit_mem + fit_gpu) / 3.0
            else:
                score = (fit_cpu + fit_mem) / 2.0

            if score > best_score:
                best_score = score
                best_id = node.node_id

        return best_id
