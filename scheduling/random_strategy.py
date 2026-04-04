"""RandomStrategy — picks a random feasible node."""

from __future__ import annotations

import random
from typing import Optional

from models.cluster_state import ClusterState
from models.pod import Pod
from scheduling.strategy import ISchedulingStrategy


class RandomStrategy(ISchedulingStrategy):
    """Selects a uniformly random node among those with sufficient resources."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "Random"

    def select_node(self, pod: Pod, cluster: ClusterState) -> Optional[str]:
        feasible = cluster.feasible_nodes(pod)
        if not feasible:
            return None
        return self._rng.choice(feasible).node_id
