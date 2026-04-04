"""WorkloadGenerator — generates synthetic pod workloads for simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from config.schema import WorkloadConfig
from models.pod import Pod


class IWorkloadGenerator(ABC):
    """Contract for workload generators.

    Implementations produce a deterministic list of Pod objects
    (with arrival_time, duration, resource requests, etc.) for a given
    seed and configuration.
    """

    @abstractmethod
    def generate(self, config: WorkloadConfig, seed: int) -> List[Pod]:
        """Generate a list of pods according to *config*.

        The list is ordered by arrival_time.  Each pod has a unique pod_id.

        Args:
            config: Workload parameters (total pods, distributions, etc.).
            seed:   Random seed for reproducibility.

        Returns:
            Ordered list of Pod objects.
        """
        ...
