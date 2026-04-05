"""TraceReplayGenerator — replay real cluster traces as simulation workloads.

Parses CSV trace files (e.g. from Alibaba or Google cluster traces)
and converts them into Pod objects suitable for the simulator.

Expected CSV columns (case-insensitive, order-independent):
  - ``cpu_request``  (cores)
  - ``mem_request``  (MiB)
  - ``arrival_time`` (seconds from trace start)
  - ``duration``     (seconds)

Optional columns:
  - ``gpu_request``  (GPUs, default 0)
  - ``priority``     (integer 0–1000, default 100)
  - ``qos_class``    (best_effort / burstable / guaranteed, default burstable)
  - ``namespace``    (string, default "trace")
  - ``pod_id``       (string, auto-generated if missing)

Usage::

    from workload.trace_replay import TraceReplayGenerator

    gen = TraceReplayGenerator("traces/alibaba_batch.csv")
    pods = gen.generate(config, seed=42)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional

from config.schema import WorkloadConfig
from models.pod import Pod, QoSClass
from workload.generator import IWorkloadGenerator

logger = logging.getLogger(__name__)

_QOS_MAP = {
    "best_effort": QoSClass.BEST_EFFORT,
    "burstable": QoSClass.BURSTABLE,
    "guaranteed": QoSClass.GUARANTEED,
}


class TraceReplayGenerator(IWorkloadGenerator):
    """Replays a CSV trace file as a simulation workload.

    The ``seed`` parameter controls how many pods are sampled when
    ``config.total_pods`` is smaller than the trace file.  If the trace
    has fewer rows than ``config.total_pods``, all rows are used.
    """

    def __init__(self, trace_path: str | Path) -> None:
        self._trace_path = Path(trace_path)
        if not self._trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {self._trace_path}")

    def generate(self, config: WorkloadConfig, seed: int = 0) -> List[Pod]:
        """Parse the trace CSV and return pods sorted by arrival_time.

        Args:
            config: Used for ``total_pods`` limit and default ranges.
            seed:   Seed for deterministic sampling when the trace has
                    more rows than ``config.total_pods``.
        """
        import random as _random

        raw_pods = self._parse_csv()

        # Sample if the trace is larger than the requested pod count
        if len(raw_pods) > config.total_pods:
            rng = _random.Random(seed)
            raw_pods = rng.sample(raw_pods, config.total_pods)

        # Sort by arrival time
        raw_pods.sort(key=lambda p: p.arrival_time)

        # Assign sequential pod_ids if they have placeholder ids
        for i, pod in enumerate(raw_pods):
            if pod.pod_id.startswith("trace-"):
                pod.pod_id = f"trace-{i:05d}"

        logger.info(
            "TraceReplayGenerator: loaded %d pods from %s",
            len(raw_pods), self._trace_path.name,
        )
        return raw_pods

    def _parse_csv(self) -> List[Pod]:
        """Read the CSV file and convert each row to a Pod."""
        pods: List[Pod] = []

        with open(self._trace_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"Empty or malformed CSV: {self._trace_path}")

            # Normalise header names to lowercase
            normalised = {h.strip().lower(): h for h in reader.fieldnames}
            self._validate_columns(normalised)

            for row_idx, row in enumerate(reader):
                # Normalise row keys
                norm_row = {k.strip().lower(): v.strip() for k, v in row.items()}

                cpu = float(norm_row["cpu_request"])
                mem = float(norm_row["mem_request"])
                arrival = float(norm_row["arrival_time"])
                duration = float(norm_row["duration"])

                # Clamp invalid values
                cpu = max(0.01, cpu)
                mem = max(1.0, mem)
                duration = max(0.1, duration)
                arrival = max(0.0, arrival)

                gpu = float(norm_row.get("gpu_request", "0"))
                priority = int(float(norm_row.get("priority", "100")))
                qos_str = norm_row.get("qos_class", "burstable").lower()
                qos = _QOS_MAP.get(qos_str, QoSClass.BURSTABLE)
                namespace = norm_row.get("namespace", "trace")
                pod_id = norm_row.get("pod_id", f"trace-{row_idx:05d}")

                pods.append(Pod(
                    pod_id=pod_id,
                    cpu_request=cpu,
                    mem_request=mem,
                    gpu_request=gpu,
                    priority=priority,
                    qos_class=qos,
                    arrival_time=arrival,
                    duration=duration,
                    namespace=namespace,
                ))

        return pods

    @staticmethod
    def _validate_columns(normalised: dict) -> None:
        """Check that required columns are present."""
        required = {"cpu_request", "mem_request", "arrival_time", "duration"}
        missing = required - set(normalised.keys())
        if missing:
            raise ValueError(
                f"Trace CSV missing required columns: {sorted(missing)}. "
                f"Found: {sorted(normalised.keys())}"
            )
