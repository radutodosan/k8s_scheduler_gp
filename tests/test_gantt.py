"""Tests for visualization.gantt — Gantt chart generation."""

import pytest

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing

from matplotlib.figure import Figure

from config.schema import ClusterConfig, NodeConfig
from models.pod import Pod, PodStatus, QoSClass
from models.cluster_state import ClusterState
from models.node import Node
from scheduling.first_fit import FirstFitStrategy
from simulator.engine import SimulationEngine
from visualization.gantt import plot_gantt, plot_gantt_from_engine, save_gantt


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def scheduled_pods():
    """Pods that have gone through a simulation lifecycle."""
    pods = {}

    p1 = Pod(pod_id="pod-0", cpu_request=0.5, mem_request=256.0,
             arrival_time=0.0, duration=5.0, namespace="default")
    p1.schedule_on("node-000", 1.0)
    p1.complete(6.0)
    pods["pod-0"] = p1

    p2 = Pod(pod_id="pod-1", cpu_request=1.0, mem_request=512.0,
             arrival_time=2.0, duration=8.0, namespace="production")
    p2.schedule_on("node-001", 3.0)
    p2.complete(11.0)
    pods["pod-1"] = p2

    p3 = Pod(pod_id="pod-2", cpu_request=0.3, mem_request=128.0,
             arrival_time=1.0, duration=3.0, namespace="default")
    p3.schedule_on("node-000", 1.5)
    p3.complete(4.5)
    pods["pod-2"] = p3

    # Rejected pod
    p4 = Pod(pod_id="pod-3", cpu_request=50.0, mem_request=99999.0,
             arrival_time=0.0, duration=10.0, namespace="batch")
    p4.reject(2.0)
    pods["pod-3"] = p4

    return pods


@pytest.fixture
def node_ids():
    return ["node-000", "node-001", "node-002"]


# ── Tests ────────────────────────────────────────────────────────────

class TestPlotGantt:
    def test_returns_figure(self, scheduled_pods, node_ids):
        fig = plot_gantt(scheduled_pods, node_ids)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_color_by_namespace(self, scheduled_pods, node_ids):
        fig = plot_gantt(scheduled_pods, node_ids, color_by="namespace")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_color_by_priority(self, scheduled_pods, node_ids):
        fig = plot_gantt(scheduled_pods, node_ids, color_by="priority")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_no_rejected_row(self, scheduled_pods, node_ids):
        fig = plot_gantt(scheduled_pods, node_ids, show_rejected=False)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "Rejected" not in labels
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_rejected_row_present(self, scheduled_pods, node_ids):
        fig = plot_gantt(scheduled_pods, node_ids, show_rejected=True)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "Rejected" in labels
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_empty_pods(self, node_ids):
        fig = plot_gantt({}, node_ids)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_custom_title(self, scheduled_pods, node_ids):
        fig = plot_gantt(scheduled_pods, node_ids, title="My Custom Title")
        ax = fig.axes[0]
        assert ax.get_title() == "My Custom Title"
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotGanttFromEngine:
    def test_from_simulation_engine(self):
        """Full integration: run a simulation and produce a Gantt chart."""
        pods = [
            Pod(pod_id=f"p-{i}", cpu_request=0.5, mem_request=256.0,
                arrival_time=float(i), duration=5.0, namespace="default")
            for i in range(5)
        ]
        strategy = FirstFitStrategy()
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=ClusterConfig(
                node_templates=[NodeConfig(count=2, cpu_capacity=4.0, mem_capacity=8192.0)]
            ),
        )
        engine.build_cluster()
        engine.load_workload(pods)
        engine.run()

        fig = plot_gantt_from_engine(engine, title="Test Gantt")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestSaveGantt:
    def test_saves_png(self, tmp_path, scheduled_pods, node_ids):
        fig = plot_gantt(scheduled_pods, node_ids)
        out_path = tmp_path / "test_gantt.png"
        save_gantt(fig, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path, scheduled_pods, node_ids):
        fig = plot_gantt(scheduled_pods, node_ids)
        out_path = tmp_path / "subdir" / "nested" / "gantt.png"
        save_gantt(fig, out_path)
        assert out_path.exists()
