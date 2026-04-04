"""Tests for resource utilization plots (visualization/resource_plots.py)."""

import matplotlib
matplotlib.use("Agg")

import pytest

from metrics.resource_monitor import ResourceMonitor
from models.cluster_state import ClusterState
from models.node import Node
from models.pod import Pod, QoSClass
from visualization.resource_plots import (
    plot_node_utilization,
    plot_cluster_utilization,
    plot_cluster_comparison,
    save_resource_plot,
)


def _make_cluster(n: int = 2) -> ClusterState:
    cs = ClusterState()
    for i in range(n):
        cs.add_node(Node(f"node-{i:03d}", cpu_capacity=4.0, mem_capacity=8192.0))
    return cs


def _make_pod(pid: str, cpu: float = 1.0) -> Pod:
    return Pod(pid, cpu_request=cpu, mem_request=1024.0,
               priority=100, qos_class=QoSClass.BURSTABLE,
               arrival_time=0.0, duration=10.0)


def _loaded_monitor() -> ResourceMonitor:
    mon = ResourceMonitor()
    cs = _make_cluster(2)
    mon.record_snapshot(0.0, cs)
    pod = _make_pod("p-0", cpu=2.0)
    cs.enqueue_pod(pod)
    cs.bind_pod(pod, "node-000", 0.5)
    mon.record_snapshot(1.0, cs)
    cs.release_pod(pod, 2.0)
    mon.record_snapshot(2.0, cs)
    return mon


# ═══════════════════════════════════════════════════════════════════════


class TestPlotNodeUtilization:
    def test_returns_figure(self):
        mon = _loaded_monitor()
        fig = plot_node_utilization(mon)
        assert fig is not None

    def test_empty_monitor(self):
        mon = ResourceMonitor()
        fig = plot_node_utilization(mon)
        assert fig is not None  # "(no data)" placeholder

    def test_save(self, tmp_path):
        mon = _loaded_monitor()
        fig = plot_node_utilization(mon)
        out = tmp_path / "node_util.png"
        save_resource_plot(fig, out)
        assert out.exists()


class TestPlotClusterUtilization:
    def test_returns_figure(self):
        mon = _loaded_monitor()
        fig = plot_cluster_utilization(mon)
        assert fig is not None

    def test_empty_monitor(self):
        mon = ResourceMonitor()
        fig = plot_cluster_utilization(mon)
        assert fig is not None

    def test_title_kwarg(self):
        mon = _loaded_monitor()
        fig = plot_cluster_utilization(mon, title="Custom Title")
        assert fig is not None

    def test_save(self, tmp_path):
        mon = _loaded_monitor()
        fig = plot_cluster_utilization(mon)
        out = tmp_path / "cluster_util.png"
        save_resource_plot(fig, out)
        assert out.exists()


class TestPlotClusterComparison:
    def test_returns_figure(self):
        monitors = {"Strategy A": _loaded_monitor(), "Strategy B": _loaded_monitor()}
        fig = plot_cluster_comparison(monitors)
        assert fig is not None

    def test_empty_monitors(self):
        fig = plot_cluster_comparison({})
        assert fig is not None

    def test_single_strategy(self):
        fig = plot_cluster_comparison({"Only": _loaded_monitor()})
        assert fig is not None

    def test_save(self, tmp_path):
        monitors = {"A": _loaded_monitor(), "B": _loaded_monitor()}
        fig = plot_cluster_comparison(monitors)
        out = tmp_path / "comparison.png"
        save_resource_plot(fig, out)
        assert out.exists()


class TestSaveResourcePlot:
    def test_creates_parent_dirs(self, tmp_path):
        mon = _loaded_monitor()
        fig = plot_cluster_utilization(mon)
        out = tmp_path / "sub" / "dir" / "plot.png"
        save_resource_plot(fig, out)
        assert out.exists()


class TestFailureMarkers:
    def test_failure_markers_in_node_plot(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2)
        mon.record_snapshot(0.0, cs)
        cs.nodes["node-000"].mark_failed()
        mon.record_snapshot(1.0, cs)
        cs.nodes["node-000"].mark_recovered()
        mon.record_snapshot(2.0, cs)
        fig = plot_node_utilization(mon, title="With Failures")
        assert fig is not None

    def test_failure_markers_in_cluster_plot(self):
        mon = ResourceMonitor()
        cs = _make_cluster(2)
        mon.record_snapshot(0.0, cs)
        cs.nodes["node-000"].mark_failed()
        mon.record_snapshot(1.0, cs)
        fig = plot_cluster_utilization(mon)
        assert fig is not None
