"""IGeneticEngine — abstract interface for GP engines.

This interface decouples the GP implementation from the rest of the system,
allowing Phase 2 to add a second engine (e.g. gplearn) without touching
the simulator or experiment runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class GPResult:
    """Output of a GP training run.

    Attributes:
        best_individual:     The best individual (engine-specific type).
        best_fitness:        Fitness value of the best individual.
        best_expression:     Human-readable string of the best individual.
        generations:         Number of generations completed.
        log:                 Per-generation statistics (list of dicts).
        hall_of_fame:        Top-k individuals (engine-specific type).
        pareto_front:        Pareto-optimal individuals (NSGA-II only).
    """

    best_individual: Any
    best_fitness: float
    best_expression: str
    generations: int
    log: List[Dict[str, Any]]
    hall_of_fame: List[Any]
    pareto_front: List[Any] = None


class IGeneticEngine(ABC):
    """Contract for pluggable GP engines (DEAP, gplearn, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier (e.g. 'deap', 'gplearn')."""
        ...

    @abstractmethod
    def setup(
        self,
        terminal_names: List[str],
        function_set: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Configure primitives, terminals, operators and GP parameters.

        Args:
            terminal_names: Ordered list of terminal (feature) names that the
                            scoring function will receive at evaluation time.
            function_set:   Optional list of function names to use
                            (implementation maps names to callables).
            **kwargs:       Engine-specific parameters (population_size,
                            n_generations, crossover_prob, etc.).
        """
        ...

    @abstractmethod
    def train(
        self,
        fitness_function: Callable[..., float],
        seed: int = 42,
    ) -> GPResult:
        """Run the evolutionary process.

        Args:
            fitness_function: A callable that accepts an individual (tree /
                              expression) and returns a scalar fitness.
                              Lower is better (minimisation by convention).
            seed:             Random seed for reproducibility.

        Returns:
            GPResult with the best individual and training statistics.
        """
        ...

    @abstractmethod
    def evaluate_individual(
        self,
        individual: Any,
        terminal_values: Dict[str, float],
    ) -> float:
        """Evaluate a single individual on a set of terminal values.

        This is the core scoring step used during simulation:
        Score(pod, node) = evaluate_individual(rule, features).

        Args:
            individual:      The GP tree / expression.
            terminal_values: Mapping terminal_name → float value.

        Returns:
            Scalar score.  Higher means the (pod, node) pair is preferred.
        """
        ...

    @abstractmethod
    def get_expression_string(self, individual: Any) -> str:
        """Return a human-readable representation of *individual*."""
        ...
