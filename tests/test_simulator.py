"""Tests for simulator — Event ordering, EventQueue, SimulationEngine."""

import pytest

from config.schema import ClusterConfig, NodeConfig
from models.pod import Pod, PodStatus, QoSClass
from scheduling.strategy import ISchedulingStrategy
from models.cluster_state import ClusterState
from simulator.event import (
    Event,
    EventType,
    PRIORITY_POD_ARRIVAL,
    PRIORITY_POD_COMPLETION,
    PRIORITY_SCHEDULE_CYCLE,
)
from simulator.event_queue import EventQueue
from simulator.engine import SimulationEngine

from typing import Optional


# ── Event ordering ───────────────────────────────────────────────────


class TestEventOrdering:
    def test_earlier_timestamp_first(self):
        e1 = Event(timestamp=1.0, priority=0, event_type=EventType.POD_ARRIVAL)
        e2 = Event(timestamp=2.0, priority=0, event_type=EventType.POD_ARRIVAL)
        assert e1 < e2

    def test_same_timestamp_lower_priority_first(self):
        e_cycle = Event(timestamp=1.0, priority=PRIORITY_SCHEDULE_CYCLE, event_type=EventType.SCHEDULE_CYCLE)
        e_arrival = Event(timestamp=1.0, priority=PRIORITY_POD_ARRIVAL, event_type=EventType.POD_ARRIVAL)
        assert e_cycle < e_arrival  # SCHEDULE_CYCLE prio=0 < POD_ARRIVAL prio=1

    def test_payload_excluded_from_comparison(self):
        e1 = Event(timestamp=1.0, priority=0, event_type=EventType.POD_ARRIVAL, payload="a")
        e2 = Event(timestamp=1.0, priority=0, event_type=EventType.POD_ARRIVAL, payload="z")
        assert e1 == e2  # payload doesn't affect ordering


# ── EventQueue ───────────────────────────────────────────────────────


class TestEventQueue:
    def test_push_pop_order(self):
        q = EventQueue()
        q.push(Event(timestamp=3.0, priority=0, event_type=EventType.POD_ARRIVAL))
        q.push(Event(timestamp=1.0, priority=0, event_type=EventType.POD_ARRIVAL))
        q.push(Event(timestamp=2.0, priority=0, event_type=EventType.POD_ARRIVAL))

        assert q.pop().timestamp == 1.0
        assert q.pop().timestamp == 2.0
        assert q.pop().timestamp == 3.0

    def test_empty_pop_raises(self):
        q = EventQueue()
        with pytest.raises(IndexError):
            q.pop()

    def test_empty_peek_raises(self):
        q = EventQueue()
        with pytest.raises(IndexError):
            q.peek()

    def test_is_empty(self):
        q = EventQueue()
        assert q.is_empty is True
        q.push(Event(timestamp=0.0, priority=0, event_type=EventType.POD_ARRIVAL))
        assert q.is_empty is False

    def test_len(self):
        q = EventQueue()
        assert len(q) == 0
        q.push(Event(timestamp=0.0, priority=0, event_type=EventType.POD_ARRIVAL))
        assert len(q) == 1


# ── SimulationEngine ─────────────────────────────────────────────────


class _AlwaysFirstNodeStrategy(ISchedulingStrategy):
    """Test strategy: always picks the first feasible node."""

    @property
    def name(self) -> str:
        return "always_first"

    def select_node(self, pod: Pod, cluster: ClusterState) -> Optional[str]:
        feasible = cluster.feasible_nodes(pod)
        return feasible[0].node_id if feasible else None


class _AlwaysRejectStrategy(ISchedulingStrategy):
    """Test strategy: always returns None (no placement)."""

    @property
    def name(self) -> str:
        return "always_reject"

    def select_node(self, pod: Pod, cluster: ClusterState) -> Optional[str]:
        return None


def _make_cluster_config(count=2, cpu=4.0, mem=8192.0):
    return ClusterConfig(
        node_templates=[NodeConfig(count=count, cpu_capacity=cpu, mem_capacity=mem)]
    )


def _make_pods(n, cpu=0.5, mem=256.0, duration=5.0, arrival_gap=1.0):
    return [
        Pod(
            pod_id=f"pod-{i:03d}",
            cpu_request=cpu,
            mem_request=mem,
            priority=100,
            arrival_time=i * arrival_gap,
            duration=duration,
            namespace="default",
        )
        for i in range(n)
    ]


class TestSimulationEngineBasic:
    def test_all_pods_scheduled(self):
        """All small pods should be scheduled on adequate nodes."""
        pods = _make_pods(4, cpu=0.5, mem=256.0, duration=5.0)
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=_make_cluster_config(count=2, cpu=4.0),
        )
        engine.build_cluster()
        engine.load_workload(pods)
        engine.run()

        m = engine.collector.get_metrics()
        assert m.scheduled_pods == 4
        assert m.rejected_pods == 0

    def test_pods_rejected_no_capacity(self):
        """Oversized pods should be rejected after max retries."""
        pods = _make_pods(2, cpu=100.0, mem=256.0, duration=5.0)
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=_make_cluster_config(count=1, cpu=4.0),
            max_pending_retries=2,
        )
        engine.build_cluster()
        engine.load_workload(pods)
        engine.run()

        m = engine.collector.get_metrics()
        assert m.rejected_pods == 2
        assert m.scheduled_pods == 0

    def test_strategy_always_reject(self):
        """If strategy returns None, pods get rejected after max_pending_retries."""
        pods = _make_pods(3, cpu=0.5, mem=256.0, duration=5.0)
        engine = SimulationEngine(
            strategy=_AlwaysRejectStrategy(),
            cluster_config=_make_cluster_config(count=2, cpu=4.0),
            max_pending_retries=2,
        )
        engine.build_cluster()
        engine.load_workload(pods)
        engine.run()

        m = engine.collector.get_metrics()
        assert m.rejected_pods == 3

    def test_completion_frees_resources(self):
        """Pod completion should free resources for subsequent pods."""
        # 1 node, 1 CPU — only 1 pod at a time
        # pod-0 arrives at t=0 (dur=2), pod-1 arrives at t=1 (dur=2)
        # pod-0 finishes at t=2, freeing room for pod-1
        pods = _make_pods(2, cpu=1.0, mem=256.0, duration=2.0, arrival_gap=1.0)
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=_make_cluster_config(count=1, cpu=1.0, mem=8192.0),
            max_pending_retries=5,
            schedule_interval=0.5,
        )
        engine.build_cluster()
        engine.load_workload(pods)
        engine.run()

        m = engine.collector.get_metrics()
        assert m.scheduled_pods == 2
        assert m.completed_pods == 2
        assert m.rejected_pods == 0

    def test_empty_workload(self):
        """Engine should handle empty pod list gracefully."""
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=_make_cluster_config(),
        )
        engine.build_cluster()
        engine.load_workload([])
        engine.run()

        m = engine.collector.get_metrics()
        assert m.total_pods == 0


class TestPreemption:
    """Priority preemption: high-priority pods evict low-priority ones."""

    def test_preemption_evicts_low_priority(self):
        """A high-priority pod preempts a low-priority pod to get resources."""
        # 1 node, 2 CPU — low-priority pod takes 2 CPU, high-priority arrives later
        low_pod = Pod(
            pod_id="low", cpu_request=2.0, mem_request=256.0,
            priority=10, arrival_time=0.0, duration=100.0,
        )
        high_pod = Pod(
            pod_id="high", cpu_request=2.0, mem_request=256.0,
            priority=900, arrival_time=2.0, duration=5.0,
        )
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=_make_cluster_config(count=1, cpu=2.0, mem=8192.0),
            max_pending_retries=2,
            schedule_interval=0.5,
        )
        engine.build_cluster()
        engine.load_workload([low_pod, high_pod])
        engine.run()

        m = engine.collector.get_metrics()
        # High-priority pod should have been scheduled via preemption
        assert high_pod.status in (PodStatus.SCHEDULED, PodStatus.COMPLETED)
        assert m.scheduled_pods >= 2  # Both got scheduled at some point

    def test_no_preemption_if_equal_priority(self):
        """Preemption only applies when the incoming pod has higher priority."""
        pods = [
            Pod(pod_id=f"p{i}", cpu_request=2.0, mem_request=256.0,
                priority=100, arrival_time=i * 1.0, duration=100.0)
            for i in range(2)
        ]
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=_make_cluster_config(count=1, cpu=2.0, mem=8192.0),
            max_pending_retries=2,
        )
        engine.build_cluster()
        engine.load_workload(pods)
        engine.run()

        m = engine.collector.get_metrics()
        # Second pod can't preempt (same priority) → rejected after retries
        assert m.rejected_pods >= 1


class TestTaintsInSimulation:
    """End-to-end simulation with tainted nodes."""

    def test_pod_rejected_due_to_taint(self):
        """Pod without toleration gets rejected on tainted-only cluster."""
        config = ClusterConfig(
            node_templates=[NodeConfig(
                count=1, cpu_capacity=8.0, mem_capacity=8192.0,
                taints=["gpu"],
            )]
        )
        pod = Pod(pod_id="no-tol", cpu_request=1.0, mem_request=256.0,
                  priority=100, arrival_time=0.0, duration=5.0)
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=config,
            max_pending_retries=2,
        )
        engine.build_cluster()
        engine.load_workload([pod])
        engine.run()

        m = engine.collector.get_metrics()
        assert m.rejected_pods == 1
        assert m.scheduled_pods == 0

    def test_pod_with_toleration_scheduled(self):
        """Pod with correct toleration gets scheduled on tainted node."""
        config = ClusterConfig(
            node_templates=[NodeConfig(
                count=1, cpu_capacity=8.0, mem_capacity=8192.0,
                taints=["gpu"],
            )]
        )
        pod = Pod(pod_id="gpu-pod", cpu_request=1.0, mem_request=256.0,
                  priority=100, arrival_time=0.0, duration=5.0,
                  tolerations=frozenset(["gpu"]))
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=config,
            max_pending_retries=2,
        )
        engine.build_cluster()
        engine.load_workload([pod])
        engine.run()

        m = engine.collector.get_metrics()
        assert m.scheduled_pods == 1
        assert m.rejected_pods == 0


class TestLabelsInSimulation:
    """End-to-end simulation with node labels and selectors."""

    def test_pod_with_selector_placed_on_matching_node(self):
        """Pod with node_selector is placed only on the matching node."""
        config = ClusterConfig(
            node_templates=[
                NodeConfig(count=1, cpu_capacity=8.0, mem_capacity=8192.0,
                           labels={"disktype": "ssd"}),
                NodeConfig(count=1, cpu_capacity=8.0, mem_capacity=8192.0,
                           labels={"disktype": "hdd"}),
            ]
        )
        pod = Pod(pod_id="ssd-pod", cpu_request=1.0, mem_request=256.0,
                  priority=100, arrival_time=0.0, duration=5.0,
                  node_selector={"disktype": "ssd"})
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=config,
            max_pending_retries=2,
        )
        engine.build_cluster()
        engine.load_workload([pod])
        engine.run()

        m = engine.collector.get_metrics()
        assert m.scheduled_pods == 1
        assert pod.assigned_node_id == "node-000"  # First node has ssd


class TestOOMKill:
    """Overcommit OOM-kill: pods with limits > capacity get killed."""

    def test_oom_kills_overcommitted_pods(self):
        """Two pods with high limits overcommit the node → OOM kills lowest."""
        from models.pod import QoSClass
        config = _make_cluster_config(count=1, cpu=4.0, mem=4096.0)
        # Two pods: each requests 2 CPU but limits 3 CPU → total limits 6 > 4
        p1 = Pod(pod_id="p1", cpu_request=2.0, mem_request=1024.0,
                 cpu_limit=3.0, mem_limit=1024.0,
                 priority=100, qos_class=QoSClass.BEST_EFFORT,
                 arrival_time=0.0, duration=50.0)
        p2 = Pod(pod_id="p2", cpu_request=2.0, mem_request=1024.0,
                 cpu_limit=3.0, mem_limit=1024.0,
                 priority=200, qos_class=QoSClass.BURSTABLE,
                 arrival_time=0.5, duration=50.0)
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=config,
            max_pending_retries=2,
        )
        engine.build_cluster()
        engine.load_workload([p1, p2])
        engine.run()

        m = engine.collector.get_metrics()
        # At least one pod should be OOM-killed (rejected)
        assert m.rejected_pods >= 1

    def test_guaranteed_pods_survive_oom(self):
        """GUARANTEED pods are not killed by OOM, only BE/Burstable are."""
        from models.pod import QoSClass
        config = _make_cluster_config(count=1, cpu=4.0, mem=4096.0)
        # Guaranteed pod + BestEffort pod, both overcommitting
        p_guaranteed = Pod(pod_id="pg", cpu_request=2.0, mem_request=2048.0,
                           cpu_limit=3.0, mem_limit=3072.0,
                           priority=100, qos_class=QoSClass.GUARANTEED,
                           arrival_time=0.0, duration=50.0)
        p_be = Pod(pod_id="pbe", cpu_request=2.0, mem_request=2048.0,
                   cpu_limit=3.0, mem_limit=3072.0,
                   priority=50, qos_class=QoSClass.BEST_EFFORT,
                   arrival_time=0.5, duration=50.0)
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=config,
            max_pending_retries=2,
        )
        engine.build_cluster()
        engine.load_workload([p_guaranteed, p_be])
        engine.run()

        # Guaranteed pod should survive (not rejected by OOM)
        assert p_guaranteed.status in (PodStatus.SCHEDULED, PodStatus.COMPLETED)


class TestAntiAffinityInSimulation:
    """End-to-end simulation with anti-affinity constraints."""

    def test_anti_affinity_spreads_pods(self):
        """Pods with same anti_affinity_key can't land on the same node."""
        config = _make_cluster_config(count=2, cpu=8.0, mem=8192.0)
        p1 = Pod(pod_id="web-0", cpu_request=1.0, mem_request=256.0,
                 priority=100, arrival_time=0.0, duration=50.0,
                 anti_affinity_key="app-web")
        p2 = Pod(pod_id="web-1", cpu_request=1.0, mem_request=256.0,
                 priority=100, arrival_time=0.5, duration=50.0,
                 anti_affinity_key="app-web")
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=config,
            max_pending_retries=2,
        )
        engine.build_cluster()
        engine.load_workload([p1, p2])
        engine.run()

        m = engine.collector.get_metrics()
        assert m.scheduled_pods == 2
        # They must be on different nodes
        assert p1.assigned_node_id != p2.assigned_node_id

    def test_anti_affinity_rejected_when_no_alternative(self):
        """With only 1 node, second pod with same key gets rejected."""
        config = _make_cluster_config(count=1, cpu=8.0, mem=8192.0)
        p1 = Pod(pod_id="api-0", cpu_request=1.0, mem_request=256.0,
                 priority=100, arrival_time=0.0, duration=100.0,
                 anti_affinity_key="app-api")
        p2 = Pod(pod_id="api-1", cpu_request=1.0, mem_request=256.0,
                 priority=100, arrival_time=0.5, duration=5.0,
                 anti_affinity_key="app-api")
        engine = SimulationEngine(
            strategy=_AlwaysFirstNodeStrategy(),
            cluster_config=config,
            max_pending_retries=2,
        )
        engine.build_cluster()
        engine.load_workload([p1, p2])
        engine.run()

        m = engine.collector.get_metrics()
        assert m.rejected_pods >= 1
