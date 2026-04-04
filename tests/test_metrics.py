"""Tests for metrics — SchedulingMetrics, MetricsCollector, MetricsReporter."""

import json
import csv
from pathlib import Path

import pytest

from metrics.collector import MetricsCollector, SchedulingMetrics
from metrics.reporter import MetricsReporter
from models.cluster_state import ClusterState, SchedulingResult
from models.node import Node
from models.pod import Pod, PodStatus


# ══════════════════════════════════════════════════════════════════════
# SchedulingMetrics
# ══════════════════════════════════════════════════════════════════════


class TestSchedulingMetrics:
    def test_defaults_are_zero(self):
        m = SchedulingMetrics()
        assert m.scheduling_success_rate == 0.0
        assert m.avg_wait_time == 0.0
        assert m.avg_cpu_utilization == 0.0
        assert m.avg_mem_utilization == 0.0

    def test_success_rate(self):
        m = SchedulingMetrics(total_pods=10, scheduled_pods=8)
        assert m.scheduling_success_rate == pytest.approx(0.8)

    def test_avg_wait_time(self):
        m = SchedulingMetrics(scheduled_pods=4, total_wait_time=12.0)
        assert m.avg_wait_time == pytest.approx(3.0)

    def test_avg_utilization(self):
        m = SchedulingMetrics(cpu_util_samples=[0.2, 0.4, 0.6])
        assert m.avg_cpu_utilization == pytest.approx(0.4)

    def test_to_dict(self):
        m = SchedulingMetrics(total_pods=5, scheduled_pods=3)
        d = m.to_dict()
        assert d["total_pods"] == 5
        assert "scheduling_success_rate" in d


# ══════════════════════════════════════════════════════════════════════
# MetricsCollector
# ══════════════════════════════════════════════════════════════════════


class TestMetricsCollector:
    def test_record_pod_arrival(self):
        collector = MetricsCollector()
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0, namespace="prod", priority=100)
        collector.record_pod_arrival(pod)

        m = collector.get_metrics()
        assert m.total_pods == 1
        assert m.pods_per_namespace["prod"] == 1
        assert m.pods_per_priority[100] == 1

    def test_record_scheduling_success(self):
        collector = MetricsCollector()
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0, priority=100)
        pod.schedule_on("node-0", 5.0)  # wait = 5.0 - 0.0 = 5.0
        result = SchedulingResult(pod=pod, node_id="node-0", success=True)

        collector.record_scheduling_result(result)
        m = collector.get_metrics()
        assert m.scheduled_pods == 1
        assert m.total_wait_time == pytest.approx(5.0)

    def test_record_scheduling_failure(self):
        collector = MetricsCollector()
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0)
        result = SchedulingResult(pod=pod, node_id=None, success=False, reason="No capacity")

        collector.record_scheduling_result(result)
        m = collector.get_metrics()
        assert m.rejected_pods == 1
        assert "No capacity" in m.rejection_reasons

    def test_sample_utilization(self, cluster_3_nodes, small_pod):
        collector = MetricsCollector()
        # Empty cluster — 0 util
        collector.sample_utilization(cluster_3_nodes)
        # After allocating a pod
        cluster_3_nodes.enqueue_pod(small_pod)
        cluster_3_nodes.bind_pod(small_pod, "node-000", 0.0)
        collector.sample_utilization(cluster_3_nodes)

        m = collector.get_metrics()
        assert len(m.cpu_util_samples) == 2
        assert m.cpu_util_samples[0] == 0.0
        assert m.cpu_util_samples[1] > 0.0

    def test_reset(self):
        collector = MetricsCollector()
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0)
        collector.record_pod_arrival(pod)
        collector.reset()
        m = collector.get_metrics()
        assert m.total_pods == 0


# ══════════════════════════════════════════════════════════════════════
# MetricsReporter
# ══════════════════════════════════════════════════════════════════════


class TestMetricsReporter:
    def test_export_csv(self, tmp_path):
        reporter = MetricsReporter()
        m = SchedulingMetrics(total_pods=10, scheduled_pods=8, rejected_pods=2)
        reporter.add_run("GP(deap)", "test-0", 42, m)

        csv_path = tmp_path / "results.csv"
        reporter.export_csv(csv_path)

        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["strategy"] == "GP(deap)"
        assert rows[0]["total_pods"] == "10"

    def test_export_json(self, tmp_path):
        reporter = MetricsReporter()
        m = SchedulingMetrics(total_pods=5, scheduled_pods=5)
        reporter.add_run("baseline", "test-0", 1, m)

        json_path = tmp_path / "results.json"
        reporter.export_json(json_path)

        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert len(data) == 1

    def test_summary_table_not_empty(self):
        reporter = MetricsReporter()
        m = SchedulingMetrics(total_pods=5, scheduled_pods=3)
        reporter.add_run("test", "i0", 1, m)
        table = reporter.summary_table()
        assert len(table) > 0
        assert "test" in table

    def test_summary_table_empty(self):
        reporter = MetricsReporter()
        table = reporter.summary_table()
        assert "no data" in table.lower()

    def test_export_csv_empty(self, tmp_path):
        """Exporting with no data should not crash."""
        reporter = MetricsReporter()
        csv_path = tmp_path / "empty.csv"
        reporter.export_csv(csv_path)
        # File may or may not be created when empty — just no crash


# ══════════════════════════════════════════════════════════════════════
# Wait-time percentiles
# ══════════════════════════════════════════════════════════════════════


class TestWaitTimePercentiles:
    def test_empty_returns_zero(self):
        m = SchedulingMetrics()
        assert m.wait_time_p50 == 0.0
        assert m.wait_time_p90 == 0.0

    def test_single_value(self):
        m = SchedulingMetrics(per_pod_wait_times=[5.0])
        assert m.wait_time_p50 == pytest.approx(5.0)
        assert m.wait_time_p99 == pytest.approx(5.0)

    def test_known_percentiles(self):
        # 100 values: 0, 1, 2, ..., 99
        m = SchedulingMetrics(per_pod_wait_times=list(range(100)))
        assert m.wait_time_p50 == pytest.approx(49.5)
        assert m.wait_time_p90 == pytest.approx(89.1)

    def test_to_dict_includes_percentiles(self):
        m = SchedulingMetrics(per_pod_wait_times=[1.0, 2.0, 3.0])
        d = m.to_dict()
        assert "wait_time_p50" in d
        assert "wait_time_p90" in d
        assert "wait_time_p95" in d
        assert "wait_time_p99" in d


# ══════════════════════════════════════════════════════════════════════
# Scheduling attempts & preemptions
# ══════════════════════════════════════════════════════════════════════


class TestSchedulingAttempts:
    def test_record_attempt(self):
        collector = MetricsCollector()
        collector.record_scheduling_attempt("p1")
        collector.record_scheduling_attempt("p1")
        collector.record_scheduling_attempt("p2")
        m = collector.get_metrics()
        assert m.scheduling_attempts["p1"] == 2
        assert m.scheduling_attempts["p2"] == 1
        assert m.avg_scheduling_attempts == pytest.approx(1.5)

    def test_record_preemption(self):
        collector = MetricsCollector()
        victim = Pod(pod_id="v", cpu_request=0.5, mem_request=256.0)
        preemptor = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0)
        collector.record_preemption(victim, preemptor)
        m = collector.get_metrics()
        assert m.preemption_count == 1

    def test_rejection_timeline(self):
        collector = MetricsCollector()
        collector.set_time(10.0)
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0)
        result = SchedulingResult(pod=pod, node_id=None, success=False, reason="No nodes")
        collector.record_scheduling_result(result)
        m = collector.get_metrics()
        assert len(m.rejection_timeline) == 1
        assert m.rejection_timeline[0] == (10.0, "No nodes")

    def test_collector_per_pod_wait_times(self):
        collector = MetricsCollector()
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0, arrival_time=0.0)
        pod.schedule_on("n0", 3.0)
        result = SchedulingResult(pod=pod, node_id="n0", success=True)
        collector.record_scheduling_result(result)
        m = collector.get_metrics()
        assert m.per_pod_wait_times == [pytest.approx(3.0)]


# ══════════════════════════════════════════════════════════════════════
# Reporter CSV with new fields
# ══════════════════════════════════════════════════════════════════════


class TestReporterNewFields:
    def test_csv_has_percentile_columns(self, tmp_path):
        reporter = MetricsReporter()
        m = SchedulingMetrics(
            total_pods=3, scheduled_pods=3,
            per_pod_wait_times=[1.0, 2.0, 3.0],
        )
        reporter.add_run("test", "i0", 1, m)
        csv_path = tmp_path / "out.csv"
        reporter.export_csv(csv_path)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "wait_time_p90" in row
        assert "preemption_count" in row
        assert "avg_scheduling_attempts" in row
