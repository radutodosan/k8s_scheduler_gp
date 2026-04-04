"""GPSchedulingStrategy — scheduling strategy that uses a GP-evolved scoring rule.

The strategy evaluates a GP expression tree on every feasible (pod, node) pair
and selects the node with the highest score.
"""

from __future__ import annotations

from typing import Any, Optional

from gp.interface import IGeneticEngine
from gp.primitives import extract_terminal_values
from models.cluster_state import ClusterState
from models.pod import Pod
from scheduling.strategy import ISchedulingStrategy


class GPSchedulingStrategy(ISchedulingStrategy):
    """Uses a GP individual (expression tree) to score (pod, node) pairs.

    At construction time, receives a trained GP engine and the best
    individual.  At each scheduling decision, it:
      1. Filters feasible nodes (via ClusterState)
      2. Extracts terminal values for each (pod, node) pair
      3. Evaluates the GP tree → score
      4. Returns the node with the maximum score
    """

    def __init__(
        self,
        gp_engine: IGeneticEngine,
        individual: Any,
        current_time_fn: Optional[callable] = None,
    ) -> None:
        """
        Args:
            gp_engine:       The GP engine used for evaluation.
            individual:      A trained GP individual (expression tree).
            current_time_fn: Callable returning current simulation time.
                             If None, wait_time defaults to 0.
        """
        self._engine = gp_engine
        self._individual = individual
        self._current_time_fn = current_time_fn
        self._current_time: float = 0.0

    @property
    def name(self) -> str:
        return f"GP({self._engine.name})"

    def set_current_time(self, t: float) -> None:
        """Update the simulation clock (called by the SimulationEngine)."""
        self._current_time = t

    def select_node(self, pod: Pod, cluster: ClusterState) -> Optional[str]:
        feasible = cluster.feasible_nodes(pod)
        if not feasible:
            return None

        current_time = (
            self._current_time_fn() if self._current_time_fn else self._current_time
        )

        best_node_id: Optional[str] = None
        best_score = float("-inf")

        for node in feasible:
            terminal_values = extract_terminal_values(pod, node, cluster, current_time)
            score = self._engine.evaluate_individual(self._individual, terminal_values)

            if score > best_score:
                best_score = score
                best_node_id = node.node_id

        return best_node_id

    @property
    def individual(self) -> Any:
        """Access the underlying GP individual (for logging / export)."""
        return self._individual

    @property
    def expression(self) -> str:
        """Human-readable form of the GP rule."""
        return self._engine.get_expression_string(self._individual)
