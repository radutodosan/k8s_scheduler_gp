"""MetricsReporter — aggregates results across runs and exports them."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from metrics.collector import SchedulingMetrics


class MetricsReporter:
    """Collects SchedulingMetrics from multiple runs and exports reports."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def add_run(
        self,
        strategy_name: str,
        instance_id: str,
        seed: int,
        metrics: SchedulingMetrics,
    ) -> None:
        """Register the results of one simulation run."""
        row = metrics.to_dict()
        row["strategy"] = strategy_name
        row["instance_id"] = instance_id
        row["seed"] = seed
        self._records.append(row)

    def export_csv(self, path: str | Path) -> None:
        """Write all recorded results to a CSV file."""
        if not self._records:
            return
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "strategy",
            "instance_id",
            "seed",
            "total_pods",
            "scheduled_pods",
            "completed_pods",
            "rejected_pods",
            "scheduling_success_rate",
            "avg_wait_time",
            "wait_time_p50",
            "wait_time_p90",
            "wait_time_p95",
            "wait_time_p99",
            "avg_cpu_utilization",
            "avg_mem_utilization",
            "throughput",
            "evicted_pods",
            "preemption_count",
            "node_failure_count",
            "avg_scheduling_attempts",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._records)

    def export_json(self, path: str | Path) -> None:
        """Write all recorded results to a JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self._records, fh, indent=2, default=str)

    def summary_table(self) -> str:
        """Return a plain-text formatted summary table."""
        if not self._records:
            return "(no data)"

        header = (
            f"{'Strategy':<20} {'Instance':<15} {'Sched%':>8} "
            f"{'AvgWait':>8} {'P90Wait':>8} {'CPU%':>6} {'MEM%':>6} "
            f"{'Rej':>5} {'Preempt':>7}"
        )
        lines = [header, "-" * len(header)]
        for r in self._records:
            lines.append(
                f"{r['strategy']:<20} {r['instance_id']:<15} "
                f"{r['scheduling_success_rate']:>8.2%} "
                f"{r['avg_wait_time']:>8.2f} "
                f"{r.get('wait_time_p90', 0.0):>8.2f} "
                f"{r['avg_cpu_utilization']:>5.1%} "
                f"{r['avg_mem_utilization']:>5.1%} "
                f"{r['rejected_pods']:>5} "
                f"{r.get('preemption_count', 0):>7}"
            )
        return "\n".join(lines)
