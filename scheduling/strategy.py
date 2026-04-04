"""ISchedulingStrategy — abstract interface for all scheduling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from models.cluster_state import ClusterState
from models.pod import Pod


class ISchedulingStrategy(ABC):
    """Contract that every scheduling strategy must implement.

    The simulator calls `select_node` for each pending pod that needs
    placement.  The strategy returns the best node id or None if no
    feasible node exists.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name (used in logs and reports)."""
        ...

    @abstractmethod
    def select_node(self, pod: Pod, cluster: ClusterState) -> Optional[str]:
        """Choose a node for *pod* given the current *cluster* state.

        Returns:
            node_id of the selected node, or None if the pod cannot be placed.
        """
        ...

    def on_episode_start(self, cluster: ClusterState) -> None:
        """Optional hook called at the start of a simulation episode.

        Strategies may use this to reset internal state.
        """

    def on_episode_end(self, cluster: ClusterState) -> None:
        """Optional hook called at the end of a simulation episode."""
