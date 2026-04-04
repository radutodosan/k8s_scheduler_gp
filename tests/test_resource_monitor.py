"""Tests for ResourceMonitor — time-series resource data capture."""

import json
from pathlib import Path

import pytest

from metrics.resource_monitor import ResourceMonitor, ResourceSnapshot
from models.cluster_state import ClusterState
from models.node import Node
from models.pod import Pod, QoSClass
from simulator.engine import SimulationEngine


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_cluster(n_nodes: int = 3, cpu: float = 4.0, mem: float = 8192.0) -> ClusterState:
    cs = ClusterState()
    for i in range(n_nodes):
        cs.add_node(Node(
            node_id=f"node-{i:03d}",
            cpu_capacity=cpu,
            mem_capacity=mem,
        ))
    return cs


def _make_pod(pod_id: str = "p-0", cpu: float = 1.0, mem: float = 1024.0) -> Pod:
    return Pod(
        pod_id=pod_id,
        cpu_request=cpu,
        mem_request=mem,
        priority=100,
        qos_class=QoSClass.BURSTABLE,
        arrival_time=0.0,
        duration=10.0,
    )


# ═══════════════════════════════════════════════════════════════════════
# ResourceSnapshot
# ═══════════════════════════════════════════════════════════════════════


class TestResourceSnapshot:
    def test_to_dict(self):
        snap = ResourceSnapshot(timestamp=1.0)
        snap.node_cpu_util["node-000"] = 0.5
        snap.cluster_cpu_util = 0.5
        d = snap.to_dict()
        assert d["timestamp"] == 1.0
        assert d["node_cpu_util"]["node-000"] == 0.5
        assert d["cluster_cpu_util"] == 0.5

    def test_defaults(self):
        snap = ResourceSnapshot(timestamp=0.0)
        assert snap.pending_count == 0
        assert snap.completed_count == 0
        assert snap.node_cpu_util == {}


# ═══════════════════════════════════════════════════════════════════════
# ResourceMonitor — recording
# ═══════════════════════════════════════════════════════════════════════


class TestRecordSnapshot:
    def test_records_snapshot(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2)
        mon.record_snapshot(0.0, cs)
        assert len(mon.snapshots) == 1

    def test_captures_all_nodes(self):
        mon = ResourceMonitor()
        cs = _make_cluster(3)
        mon.record_snapshot(0.0, cs)
        snap = mon.snapshots[0]
        assert len(snap.node_cpu_util) == 3
        assert len(snap.node_mem_util) == 3

    def test_captures_utilization_with_pods(self):
        cs = _make_cluster(2)
        pod = _make_pod()
        cs.enqueue_pod(pod)
        cs.bind_pod(pod, "node-000", 0.0)

        mon = ResourceMonitor()
        mon.record_snapshot(1.0, cs)
        snap = mon.snapshots[0]
        assert snap.node_cpu_util["node-000"] > 0
        assert snap.node_cpu_util["node-001"] == 0.0

    def test_captures_pending_count(self):
        cs = _make_cluster(1)
        pod = _make_pod()
        cs.enqueue_pod(pod)

        mon = ResourceMonitor()
        mon.record_snapshot(0.0, cs)
        assert mon.snapshots[0].pending_count == 1

    def test_captures_node_availability(self):
        cs = _make_cluster(2)
        cs.nodes["node-000"].mark_failed()
        mon = ResourceMonitor()
        mon.record_snapshot(0.0, cs)
        assert mon.snapshots[0].node_available["node-000"] is False
        assert mon.snapshots[0].node_available["node-001"] is True

    def test_multiple_snapshots(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2)
        for t in [0.0, 1.0, 2.0]:
            mon.record_snapshot(t, cs)
        assert len(mon.snapshots) == 3

    def test_captures_cluster_aggregates(self):
        cs = _make_cluster(2)
        mon = ResourceMonitor()
        mon.record_snapshot(0.0, cs)
        assert mon.snapshots[0].cluster_cpu_util == 0.0


# ═══════════════════════════════════════════════════════════════════════
# ResourceMonitor — query methods
# ═══════════════════════════════════════════════════════════════════════


class TestMonitorQueries:
    @pytest.fixture
    def loaded_monitor(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2, cpu=4.0, mem=8192.0)
        mon.record_snapshot(0.0, cs)

        pod = _make_pod("p-0", cpu=2.0, mem=4096.0)
        cs.enqueue_pod(pod)
        cs.bind_pod(pod, "node-000", 0.5)
        mon.record_snapshot(1.0, cs)

        cs.release_pod(pod, 2.0)
        mon.record_snapshot(2.0, cs)
        return mon

    def test_get_node_ids(self, loaded_monitor):
        ids = loaded_monitor.get_node_ids()
        assert ids == ["node-000", "node-001"]

    def test_get_timestamps(self, loaded_monitor):
        ts = loaded_monitor.get_timestamps()
        assert ts == [0.0, 1.0, 2.0]

    def test_get_node_cpu_series(self, loaded_monitor):
        cpu = loaded_monitor.get_node_cpu_series("node-000")
        assert len(cpu) == 3
        assert cpu[0] == 0.0  # no pods at t=0
        assert cpu[1] == 0.5  # 2.0/4.0 at t=1
        assert cpu[2] == 0.0  # released at t=2

    def test_get_node_mem_series(self, loaded_monitor):
        mem = loaded_monitor.get_node_mem_series("node-000")
        assert len(mem) == 3
        assert mem[1] == 0.5  # 4096/8192

    def test_get_cluster_cpu_series(self, loaded_monitor):
        cpu = loaded_monitor.get_cluster_cpu_series()
        assert len(cpu) == 3
        assert cpu[0] == 0.0
        assert cpu[1] == 0.25  # 2.0/(4.0*2) = 0.25

    def test_get_cluster_mem_series(self, loaded_monitor):
        mem = loaded_monitor.get_cluster_mem_series()
        assert len(mem) == 3
        assert mem[1] == 0.25  # 4096/(8192*2)

    def test_get_pending_series(self, loaded_monitor):
        pending = loaded_monitor.get_pending_series()
        assert pending[0] == 0
        assert pending[1] == 0  # was scheduled
        assert pending[2] == 0

    def test_get_timeline_returns_dicts(self, loaded_monitor):
        timeline = loaded_monitor.get_timeline()
        assert len(timeline) == 3
        assert all(isinstance(d, dict) for d in timeline)
        assert all("timestamp" in d for d in timeline)


# ═══════════════════════════════════════════════════════════════════════
# Failure/recovery event detection
# ═══════════════════════════════════════════════════════════════════════


class TestFailureRecoveryDetection:
    def test_no_failures(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2)
        mon.record_snapshot(0.0, cs)
        mon.record_snapshot(1.0, cs)
        assert mon.get_failure_timestamps() == []

    def test_detects_failure(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2)
        mon.record_snapshot(0.0, cs)
        cs.nodes["node-000"].mark_failed()
        mon.record_snapshot(1.0, cs)
        assert 1.0 in mon.get_failure_timestamps()

    def test_detects_recovery(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2)
        cs.nodes["node-000"].mark_failed()
        mon.record_snapshot(0.0, cs)
        cs.nodes["node-000"].mark_recovered()
        mon.record_snapshot(1.0, cs)
        assert 1.0 in mon.get_recovery_timestamps()

    def test_no_false_recovery_at_start(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2)
        mon.record_snapshot(0.0, cs)
        assert mon.get_recovery_timestamps() == []


# ═══════════════════════════════════════════════════════════════════════
# Throughput
# ═══════════════════════════════════════════════════════════════════════


class TestThroughput:
    def test_empty_monitor(self):
        mon = ResourceMonitor()
        assert mon.throughput() == 0.0

    def test_single_snapshot(self):
        mon = ResourceMonitor()
        cs = _make_cluster(1)
        mon.record_snapshot(0.0, cs)
        assert mon.throughput() == 0.0

    def test_positive_throughput(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2)
        pod1 = _make_pod("p-0", cpu=1.0)
        pod2 = _make_pod("p-1", cpu=1.0)
        cs.enqueue_pod(pod1)
        cs.enqueue_pod(pod2)
        mon.record_snapshot(0.0, cs)

        cs.bind_pod(pod1, "node-000", 0.5)
        cs.bind_pod(pod2, "node-001", 0.5)
        cs.release_pod(pod1, 5.0)
        cs.release_pod(pod2, 8.0)
        mon.record_snapshot(10.0, cs)

        tp = mon.throughput()
        assert tp == pytest.approx(2 / 10.0)  # 2 completed pods / 10s


# ═══════════════════════════════════════════════════════════════════════
# Export & reset
# ═══════════════════════════════════════════════════════════════════════


class TestExportReset:
    def test_export_json(self, tmp_path):
        mon = ResourceMonitor()
        cs = _make_cluster(2)
        mon.record_snapshot(0.0, cs)
        mon.record_snapshot(1.0, cs)

        path = tmp_path / "timeline.json"
        mon.export_json(path)
        assert path.exists()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["timestamp"] == 0.0

    def test_export_creates_dirs(self, tmp_path):
        mon = ResourceMonitor()
        cs = _make_cluster(1)
        mon.record_snapshot(0.0, cs)

        path = tmp_path / "sub" / "dir" / "tl.json"
        mon.export_json(path)
        assert path.exists()

    def test_reset(self):
        mon = ResourceMonitor()
        cs = _make_cluster(1)
        mon.record_snapshot(0.0, cs)
        assert len(mon.snapshots) == 1
        mon.reset()
        assert len(mon.snapshots) == 0


# ═══════════════════════════════════════════════════════════════════════
# Free resource & variance series (Phase 4 additions)
# ═══════════════════════════════════════════════════════════════════════


class TestFreeResourceSeries:
    def test_cluster_cpu_free_series_empty(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2)
        mon.record_snapshot(0.0, cs)
        free = mon.get_cluster_cpu_free_series()
        # All capacity is free
        assert free[0] == pytest.approx(8.0)  # 2 * 4.0

    def test_cluster_mem_free_series_after_allocation(self):
        mon = ResourceMonitor()
        cs = _make_cluster(1, cpu=4.0, mem=1000.0)
        pod = _make_pod("p", cpu=1.0, mem=400.0)
        cs.enqueue_pod(pod)
        cs.bind_pod(pod, "node-000", 0.0)
        mon.record_snapshot(1.0, cs)
        free = mon.get_cluster_mem_free_series()
        assert free[0] == pytest.approx(600.0)

    def test_cpu_util_variance_series(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2, cpu=4.0, mem=8192.0)
        # Load one node with 50% CPU, other at 0%
        pod = _make_pod("heavy", cpu=2.0, mem=100.0)
        cs.enqueue_pod(pod)
        cs.bind_pod(pod, "node-000", 0.0)
        mon.record_snapshot(0.0, cs)
        var = mon.get_cpu_util_variance_series()
        # mean = 0.25, variance = ((0.5-0.25)^2 + (0-0.25)^2)/2 = 0.0625
        assert var[0] == pytest.approx(0.0625)

    def test_completed_series(self):
        mon = ResourceMonitor()
        cs = _make_cluster(1)
        pod = _make_pod("p", cpu=1.0, mem=1024.0)
        cs.enqueue_pod(pod)
        cs.bind_pod(pod, "node-000", 0.0)
        mon.record_snapshot(0.0, cs)
        cs.release_pod(pod, 5.0)
        mon.record_snapshot(5.0, cs)
        completed = mon.get_completed_series()
        assert completed == [0, 1]

    def test_snapshot_to_dict_has_new_fields(self):
        mon = ResourceMonitor()
        cs = _make_cluster(1, cpu=4.0, mem=1000.0)
        mon.record_snapshot(0.0, cs)
        d = mon.snapshots[0].to_dict()
        assert "cluster_cpu_free" in d
        assert "cluster_mem_free" in d
        assert "cpu_util_variance" in d
        assert "node_cpu_free" in d
        assert "node_mem_free" in d


# ═══════════════════════════════════════════════════════════════════════
# SimulationEngine integration
# ═══════════════════════════════════════════════════════════════════════


class TestEngineIntegration:
    def test_engine_has_resource_monitor(self):
        from scheduling.round_robin import RoundRobinStrategy
        from config.schema import ClusterConfig, NodeConfig
        engine = SimulationEngine(
            strategy=RoundRobinStrategy(),
            cluster_config=ClusterConfig(
                node_templates=[NodeConfig(count=2, cpu_capacity=4.0, mem_capacity=8192.0)]
            ),
        )
        assert engine.resource_monitor is not None
        assert isinstance(engine.resource_monitor, ResourceMonitor)

    def test_engine_records_snapshots(self):
        from scheduling.round_robin import RoundRobinStrategy
        from config.schema import ClusterConfig, NodeConfig
        from simulator.engine import SimulationEngine

        strategy = RoundRobinStrategy()
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=ClusterConfig(
                node_templates=[NodeConfig(count=2, cpu_capacity=4.0, mem_capacity=8192.0)]
            ),
        )
        engine.build_cluster()
        pods = [
            _make_pod(f"p-{i}", cpu=0.5, mem=512.0)
            for i in range(5)
        ]
        for i, p in enumerate(pods):
            p.arrival_time = float(i)
        engine.load_workload(pods)
        engine.run()

        mon = engine.resource_monitor
        assert len(mon.snapshots) > 0
        assert mon.get_node_ids() == ["node-000", "node-001"]

    def test_throughput_calculated_after_run(self):
        from scheduling.round_robin import RoundRobinStrategy
        from config.schema import ClusterConfig, NodeConfig
        from simulator.engine import SimulationEngine

        engine = SimulationEngine(
            strategy=RoundRobinStrategy(),
            cluster_config=ClusterConfig(
                node_templates=[NodeConfig(count=2, cpu_capacity=4.0, mem_capacity=8192.0)]
            ),
        )
        engine.build_cluster()
        pods = [
            _make_pod(f"p-{i}", cpu=0.5, mem=512.0)
            for i in range(5)
        ]
        for i, p in enumerate(pods):
            p.arrival_time = float(i)
        engine.load_workload(pods)
        engine.run()

        m = engine.collector.get_metrics()
        assert m.simulation_duration > 0
        assert m.throughput > 0

    def test_throughput_in_to_dict(self):
        from scheduling.round_robin import RoundRobinStrategy
        from config.schema import ClusterConfig, NodeConfig
        from simulator.engine import SimulationEngine

        engine = SimulationEngine(
            strategy=RoundRobinStrategy(),
            cluster_config=ClusterConfig(
                node_templates=[NodeConfig(count=2, cpu_capacity=4.0, mem_capacity=8192.0)]
            ),
        )
        engine.build_cluster()
        pods = [_make_pod(f"p-{i}", cpu=0.5, mem=512.0) for i in range(3)]
        for i, p in enumerate(pods):
            p.arrival_time = float(i)
        engine.load_workload(pods)
        engine.run()

        d = engine.collector.get_metrics().to_dict()
        assert "throughput" in d
        assert d["throughput"] > 0
