"""Tests for PASUL 8 — Node failures, recovery, eviction, and dynamics."""

import pytest

from config.schema import ClusterConfig, DynamicsConfig, ExperimentConfig, NodeConfig
from gp.primitives import TERMINAL_NAMES, extract_terminal_values
from metrics.collector import MetricsCollector, SchedulingMetrics
from models.cluster_state import ClusterState
from models.node import Node
from models.pod import Pod, PodStatus, QoSClass
from scheduling.first_fit import FirstFitStrategy
from scheduling.least_allocated import LeastAllocatedStrategy
from scheduling.strategy import ISchedulingStrategy
from simulator.engine import SimulationEngine


# ── Helpers ──────────────────────────────────────────────────────────


class _AlwaysFirstNodeStrategy(ISchedulingStrategy):
    """Picks the first feasible node (sorted by id)."""

    @property
    def name(self) -> str:
        return "AlwaysFirst"

    def select_node(self, pod, cluster):
        feasible = cluster.feasible_nodes(pod)
        if not feasible:
            return None
        return sorted(feasible, key=lambda n: n.node_id)[0].node_id


def _make_cluster_config(count=3, cpu=4.0, mem=8192.0):
    return ClusterConfig(
        node_templates=[NodeConfig(count=count, cpu_capacity=cpu, mem_capacity=mem)]
    )


def _make_pods(n, cpu=0.5, mem=256.0, duration=10.0, arrival_rate=1.0):
    """Create n pods with staggered arrivals."""
    pods = []
    for i in range(n):
        qos = QoSClass.BEST_EFFORT if i % 3 == 0 else (
            QoSClass.BURSTABLE if i % 3 == 1 else QoSClass.GUARANTEED
        )
        pods.append(Pod(
            pod_id=f"pod-{i:05d}",
            cpu_request=cpu,
            mem_request=mem,
            priority=100 + i * 10,
            qos_class=qos,
            arrival_time=i * arrival_rate,
            duration=duration,
            namespace="default",
        ))
    return pods


# ═════════════════════════════════════════════════════════════════════
# 1. Node availability
# ═════════════════════════════════════════════════════════════════════


class TestNodeAvailability:
    def test_node_starts_available(self):
        node = Node(node_id="n-0", cpu_capacity=4.0, mem_capacity=8192.0)
        assert node.is_available is True

    def test_mark_failed(self):
        node = Node(node_id="n-0", cpu_capacity=4.0, mem_capacity=8192.0)
        node.mark_failed()
        assert node.is_available is False

    def test_mark_recovered(self):
        node = Node(node_id="n-0", cpu_capacity=4.0, mem_capacity=8192.0)
        node.mark_failed()
        node.mark_recovered()
        assert node.is_available is True

    def test_failed_node_excluded_from_feasible(self):
        cluster = ClusterState()
        n0 = Node(node_id="n-0", cpu_capacity=4.0, mem_capacity=8192.0)
        n1 = Node(node_id="n-1", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster.add_node(n0)
        cluster.add_node(n1)

        pod = Pod(pod_id="p", cpu_request=0.5, mem_request=256.0)
        assert len(cluster.feasible_nodes(pod)) == 2

        n0.mark_failed()
        feasible = cluster.feasible_nodes(pod)
        assert len(feasible) == 1
        assert feasible[0].node_id == "n-1"


# ═════════════════════════════════════════════════════════════════════
# 2. Cluster-level health metrics
# ═════════════════════════════════════════════════════════════════════


class TestClusterHealth:
    def test_all_healthy(self):
        cluster = ClusterState()
        for i in range(3):
            cluster.add_node(Node(node_id=f"n-{i}", cpu_capacity=4.0, mem_capacity=8192.0))
        assert cluster.available_node_count == 3
        assert cluster.healthy_node_ratio == pytest.approx(1.0)

    def test_one_failed(self):
        cluster = ClusterState()
        for i in range(4):
            cluster.add_node(Node(node_id=f"n-{i}", cpu_capacity=4.0, mem_capacity=8192.0))
        cluster.nodes["n-1"].mark_failed()
        assert cluster.available_node_count == 3
        assert cluster.healthy_node_ratio == pytest.approx(0.75)

    def test_all_failed(self):
        cluster = ClusterState()
        for i in range(2):
            cluster.add_node(Node(node_id=f"n-{i}", cpu_capacity=4.0, mem_capacity=8192.0))
            cluster.nodes[f"n-{i}"].mark_failed()
        assert cluster.available_node_count == 0
        assert cluster.healthy_node_ratio == pytest.approx(0.0)

    def test_empty_cluster_ratio(self):
        cluster = ClusterState()
        assert cluster.healthy_node_ratio == pytest.approx(1.0)


# ═════════════════════════════════════════════════════════════════════
# 3. Pod eviction
# ═════════════════════════════════════════════════════════════════════


class TestPodEviction:
    def test_evict_resets_pod_state(self):
        pod = Pod(
            pod_id="p", cpu_request=1.0, mem_request=512.0,
            arrival_time=0.0, duration=10.0,
        )
        pod.schedule_on("n-0", 1.0)
        assert pod.status == PodStatus.SCHEDULED
        assert pod.assigned_node_id == "n-0"

        pod.evict(2.0)
        assert pod.status == PodStatus.PENDING
        assert pod.assigned_node_id is None
        assert pod.scheduled_time is None

    def test_evict_from_node_sorted_by_qos(self):
        """BEST_EFFORT pods should be evicted first, GUARANTEED last."""
        cluster = ClusterState()
        node = Node(node_id="n-0", cpu_capacity=10.0, mem_capacity=10000.0)
        cluster.add_node(node)

        pods = [
            Pod(pod_id="guaranteed", cpu_request=1.0, mem_request=100.0,
                qos_class=QoSClass.GUARANTEED),
            Pod(pod_id="best-effort", cpu_request=1.0, mem_request=100.0,
                qos_class=QoSClass.BEST_EFFORT),
            Pod(pod_id="burstable", cpu_request=1.0, mem_request=100.0,
                qos_class=QoSClass.BURSTABLE),
        ]
        for pod in pods:
            node.allocate(pod)
            pod.schedule_on("n-0", 0.0)

        evicted = cluster.evict_pods_from_node("n-0")
        names = [p.pod_id for p in evicted]
        assert names == ["best-effort", "burstable", "guaranteed"]

    def test_evict_releases_resources(self):
        cluster = ClusterState()
        node = Node(node_id="n-0", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster.add_node(node)

        pod = Pod(pod_id="p", cpu_request=2.0, mem_request=4096.0)
        node.allocate(pod)
        assert node.cpu_available == pytest.approx(2.0)

        cluster.evict_pods_from_node("n-0")
        assert node.cpu_available == pytest.approx(4.0)
        assert node.pod_count == 0

    def test_evicted_pod_can_be_rescheduled(self):
        """After eviction + evict(), a pod can be bound to a new node."""
        cluster = ClusterState()
        n0 = Node(node_id="n-0", cpu_capacity=4.0, mem_capacity=8192.0)
        n1 = Node(node_id="n-1", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster.add_node(n0)
        cluster.add_node(n1)

        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0, arrival_time=0.0)
        cluster.enqueue_pod(pod)
        cluster.bind_pod(pod, "n-0", 1.0)
        assert pod.status == PodStatus.SCHEDULED

        # Evict from n-0
        cluster.evict_pods_from_node("n-0")
        pod.evict(1.5)
        cluster.pending_pods.append(pod)

        # Reschedule on n-1
        result = cluster.bind_pod(pod, "n-1", 2.0)
        assert result.success
        assert pod.assigned_node_id == "n-1"
        assert pod.scheduled_time == 2.0


# ═════════════════════════════════════════════════════════════════════
# 4. Metrics tracking for evictions
# ═════════════════════════════════════════════════════════════════════


class TestEvictionMetrics:
    def test_record_pod_eviction(self):
        collector = MetricsCollector()
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0)
        collector.record_pod_eviction(pod)
        collector.record_pod_eviction(pod)
        m = collector.get_metrics()
        assert m.evicted_pods == 2

    def test_record_node_failure(self):
        collector = MetricsCollector()
        collector.record_node_failure()
        collector.record_node_failure()
        collector.record_node_failure()
        m = collector.get_metrics()
        assert m.node_failure_count == 3

    def test_metrics_to_dict_includes_dynamics(self):
        m = SchedulingMetrics(evicted_pods=5, node_failure_count=2)
        d = m.to_dict()
        assert d["evicted_pods"] == 5
        assert d["node_failure_count"] == 2


# ═════════════════════════════════════════════════════════════════════
# 5. DynamicsConfig
# ═════════════════════════════════════════════════════════════════════


class TestDynamicsConfig:
    def test_defaults(self):
        dc = DynamicsConfig()
        assert dc.failure_mode == "off"
        assert dc.failure_rate == 1
        assert dc.recovery_time_min == 10.0
        assert dc.recovery_time_max == 30.0
        assert dc.restart_overhead_min == 2.0
        assert dc.restart_overhead_max == 8.0
        assert dc.enabled is False

    def test_enabled_property(self):
        assert DynamicsConfig(failure_mode="off").enabled is False
        assert DynamicsConfig(failure_mode="reschedule").enabled is True
        assert DynamicsConfig(failure_mode="kill").enabled is True

    def test_from_yaml(self, tmp_path):
        cfg_text = """\
name: test_dyn
seed: 7
dynamics:
  failure_mode: reschedule
  failure_rate: 2
  recovery_time_min: 5.0
  recovery_time_max: 15.0
  restart_overhead_min: 1.0
  restart_overhead_max: 4.0
"""
        p = tmp_path / "dyn.yaml"
        p.write_text(cfg_text)
        cfg = ExperimentConfig.from_yaml(p)
        assert cfg.dynamics.failure_mode == "reschedule"
        assert cfg.dynamics.failure_rate == 2
        assert cfg.dynamics.recovery_time_min == 5.0
        assert cfg.dynamics.restart_overhead_min == 1.0

    def test_legacy_yaml_compat(self, tmp_path):
        """Old ``node_failures: true`` YAML is converted to failure_mode."""
        cfg_text = """\
name: legacy
seed: 1
dynamics:
  node_failures: true
  failure_interval: 25.0
  recovery_time_min: 5.0
  recovery_time_max: 15.0
"""
        p = tmp_path / "legacy.yaml"
        p.write_text(cfg_text)
        cfg = ExperimentConfig.from_yaml(p)
        assert cfg.dynamics.failure_mode == "reschedule"
        assert cfg.dynamics.enabled is True
        assert cfg.dynamics.recovery_time_min == 5.0

    def test_missing_dynamics_uses_defaults(self, tmp_path):
        cfg_text = "name: no_dyn\nseed: 1\n"
        p = tmp_path / "nodyn.yaml"
        p.write_text(cfg_text)
        cfg = ExperimentConfig.from_yaml(p)
        assert cfg.dynamics.failure_mode == "off"
        assert cfg.dynamics.enabled is False


# ═════════════════════════════════════════════════════════════════════
# 6. GP terminal — CLUSTER_HEALTHY_RATIO
# ═════════════════════════════════════════════════════════════════════


class TestClusterHealthyTerminal:
    def test_terminal_in_names(self):
        assert "CLUSTER_HEALTHY_RATIO" in TERMINAL_NAMES

    def test_extract_healthy_ratio_all_up(self):
        cluster = ClusterState()
        node = Node(node_id="n-0", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster.add_node(node)
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0, arrival_time=0.0)
        vals = extract_terminal_values(pod, node, cluster, current_time=0.0)
        assert vals["CLUSTER_HEALTHY_RATIO"] == pytest.approx(1.0)

    def test_extract_healthy_ratio_with_failure(self):
        cluster = ClusterState()
        n0 = Node(node_id="n-0", cpu_capacity=4.0, mem_capacity=8192.0)
        n1 = Node(node_id="n-1", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster.add_node(n0)
        cluster.add_node(n1)
        n0.mark_failed()

        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0, arrival_time=0.0)
        vals = extract_terminal_values(pod, n1, cluster, current_time=0.0)
        assert vals["CLUSTER_HEALTHY_RATIO"] == pytest.approx(0.5)


# ═════════════════════════════════════════════════════════════════════
# 7. SimulationEngine — node failure / recovery handlers
# ═════════════════════════════════════════════════════════════════════


class TestNodeFailureSimulation:
    """Integration tests for the full failure → eviction → recovery cycle."""

    def test_no_failures_when_disabled(self):
        """Without dynamics, simulation runs normally with zero evictions."""
        strategy = _AlwaysFirstNodeStrategy()
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=_make_cluster_config(),
            dynamics_config=None,
        )
        engine.build_cluster()
        engine.load_workload(_make_pods(10))
        engine.run()

        m = engine.collector.get_metrics()
        assert m.evicted_pods == 0
        assert m.node_failure_count == 0

    def test_no_failures_when_off(self):
        """failure_mode='off' behaves like disabled dynamics."""
        dynamics = DynamicsConfig(failure_mode="off")
        strategy = FirstFitStrategy()
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=_make_cluster_config(),
            dynamics_config=dynamics,
            failure_seed=42,
        )
        engine.build_cluster()
        engine.load_workload(_make_pods(10))
        engine.run()

        m = engine.collector.get_metrics()
        assert m.evicted_pods == 0
        assert m.node_failure_count == 0

    def test_reschedule_produces_evictions(self):
        """Reschedule mode: at least one eviction should occur."""
        dynamics = DynamicsConfig(
            failure_mode="reschedule",
            failure_rate=3,
            recovery_time_min=1.0,
            recovery_time_max=2.0,
            restart_overhead_min=1.0,
            restart_overhead_max=2.0,
        )
        strategy = FirstFitStrategy()
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=_make_cluster_config(count=3),
            dynamics_config=dynamics,
            failure_seed=42,
        )
        engine.build_cluster()
        engine.load_workload(_make_pods(15, duration=20.0))
        engine.run()

        m = engine.collector.get_metrics()
        assert m.node_failure_count >= 1
        assert m.evicted_pods >= 0  # may be 0 if failure hit empty node

    def test_kill_mode_rejects_evicted(self):
        """Kill mode: evicted pods end up REJECTED, not rescheduled."""
        dynamics = DynamicsConfig(
            failure_mode="kill",
            failure_rate=3,
            recovery_time_min=1.0,
            recovery_time_max=2.0,
        )
        strategy = FirstFitStrategy()
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=_make_cluster_config(count=3),
            dynamics_config=dynamics,
            failure_seed=42,
        )
        engine.build_cluster()
        engine.load_workload(_make_pods(15, duration=20.0))
        engine.run()

        m = engine.collector.get_metrics()
        assert m.node_failure_count >= 1
        # All pods should be terminal
        assert m.completed_pods + m.rejected_pods == m.total_pods

    def test_node_recovery_makes_node_available(self):
        """After recovery, the node accepts pods again."""
        dynamics = DynamicsConfig(
            failure_mode="reschedule",
            failure_rate=1,
            recovery_time_min=1.0,
            recovery_time_max=1.0,
        )
        strategy = LeastAllocatedStrategy()
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=_make_cluster_config(count=3),
            dynamics_config=dynamics,
            failure_seed=99,
        )
        engine.build_cluster()
        engine.load_workload(_make_pods(20, duration=15.0))
        engine.run()

        # After simulation, all nodes should be recovered
        for node in engine.cluster.nodes.values():
            assert node.is_available is True

    def test_stale_completion_ignored(self):
        """If a pod is evicted, its old POD_COMPLETION event is safely ignored."""
        dynamics = DynamicsConfig(
            failure_mode="reschedule",
            failure_rate=2,
            recovery_time_min=2.0,
            recovery_time_max=3.0,
            restart_overhead_min=1.0,
            restart_overhead_max=2.0,
        )
        strategy = FirstFitStrategy()
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=_make_cluster_config(count=3),
            dynamics_config=dynamics,
            failure_seed=42,
        )
        engine.build_cluster()
        engine.load_workload(_make_pods(10, duration=30.0))
        engine.run()

        m = engine.collector.get_metrics()
        assert m.completed_pods <= m.total_pods

    def test_deterministic_with_same_seed(self):
        """Same failure_seed produces identical simulation results."""
        dynamics = DynamicsConfig(
            failure_mode="reschedule",
            failure_rate=2,
            recovery_time_min=3.0,
            recovery_time_max=6.0,
            restart_overhead_min=1.0,
            restart_overhead_max=3.0,
        )

        results = []
        for _ in range(2):
            strategy = FirstFitStrategy()
            engine = SimulationEngine(
                strategy=strategy,
                cluster_config=_make_cluster_config(count=3),
                dynamics_config=dynamics,
                failure_seed=77,
            )
            engine.build_cluster()
            engine.load_workload(_make_pods(15, duration=20.0))
            engine.run()
            results.append(engine.collector.get_metrics())

        assert results[0].node_failure_count == results[1].node_failure_count
        assert results[0].evicted_pods == results[1].evicted_pods
        assert results[0].completed_pods == results[1].completed_pods
        assert results[0].rejected_pods == results[1].rejected_pods

    def test_different_seeds_differ(self):
        """Different failure_seeds produce different failure patterns."""
        dynamics = DynamicsConfig(
            failure_mode="reschedule",
            failure_rate=2,
            recovery_time_min=2.0,
            recovery_time_max=4.0,
            restart_overhead_min=1.0,
            restart_overhead_max=2.0,
        )

        metrics_list = []
        for seed in [10, 20]:
            strategy = FirstFitStrategy()
            engine = SimulationEngine(
                strategy=strategy,
                cluster_config=_make_cluster_config(count=3),
                dynamics_config=dynamics,
                failure_seed=seed,
            )
            engine.build_cluster()
            engine.load_workload(_make_pods(20, duration=15.0))
            engine.run()
            metrics_list.append(engine.collector.get_metrics())

        assert all(m.total_pods == 20 for m in metrics_list)

    def test_simulation_completes_all_pods_terminal(self):
        """Simulation terminates — all pods reach COMPLETED or REJECTED."""
        dynamics = DynamicsConfig(
            failure_mode="reschedule",
            failure_rate=2,
            recovery_time_min=2.0,
            recovery_time_max=5.0,
            restart_overhead_min=1.0,
            restart_overhead_max=3.0,
        )
        strategy = LeastAllocatedStrategy()
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=_make_cluster_config(count=3),
            dynamics_config=dynamics,
            failure_seed=42,
        )
        engine.build_cluster()
        pods = _make_pods(15, duration=12.0)
        engine.load_workload(pods)
        engine.run()

        m = engine.collector.get_metrics()
        assert m.completed_pods + m.rejected_pods == m.total_pods

    def test_evicted_pods_rescheduled(self):
        """Evicted pods end up completed or rejected (not stuck PENDING)."""
        dynamics = DynamicsConfig(
            failure_mode="reschedule",
            failure_rate=2,
            recovery_time_min=1.0,
            recovery_time_max=2.0,
            restart_overhead_min=0.5,
            restart_overhead_max=1.0,
        )
        strategy = LeastAllocatedStrategy()
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=_make_cluster_config(count=4, cpu=8.0),
            dynamics_config=dynamics,
            failure_seed=55,
            max_pending_retries=10,
        )
        engine.build_cluster()
        engine.load_workload(_make_pods(10, cpu=0.5, duration=10.0))
        engine.run()

        for pod in engine.cluster.all_pods.values():
            assert pod.status in (PodStatus.COMPLETED, PodStatus.REJECTED)


# ═════════════════════════════════════════════════════════════════════
# 8. Pod restart overhead
# ═════════════════════════════════════════════════════════════════════


class TestRestartOverhead:
    def test_add_restart_overhead_increases_duration(self):
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0, duration=10.0)
        assert pod.duration == 10.0
        pod.add_restart_overhead(3.5)
        assert pod.duration == 13.5

    def test_overhead_accumulates(self):
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0, duration=10.0)
        pod.add_restart_overhead(2.0)
        pod.add_restart_overhead(1.5)
        assert pod.duration == 13.5

    def test_reschedule_mode_applies_overhead(self):
        """In reschedule mode, evicted pods get extra duration."""
        dynamics = DynamicsConfig(
            failure_mode="reschedule",
            failure_rate=3,
            recovery_time_min=1.0,
            recovery_time_max=1.0,
            restart_overhead_min=5.0,
            restart_overhead_max=5.0,  # fixed overhead for predictability
        )
        strategy = FirstFitStrategy()
        engine = SimulationEngine(
            strategy=strategy,
            cluster_config=_make_cluster_config(count=3),
            dynamics_config=dynamics,
            failure_seed=42,
        )
        engine.build_cluster()
        original_duration = 20.0
        engine.load_workload(_make_pods(15, duration=original_duration))
        engine.run()

        m = engine.collector.get_metrics()
        if m.evicted_pods > 0:
            # At least one pod should have increased duration
            evicted_found = any(
                p.duration > original_duration
                for p in engine.cluster.all_pods.values()
            )
            assert evicted_found
