"""Tests for models.cluster_state — pod lifecycle, binding, aggregates."""

import pytest

from models.cluster_state import ClusterState, SchedulingResult
from models.node import Node
from models.pod import Pod, PodStatus, QoSClass


class TestClusterStateSetup:
    def test_add_node(self, cluster_3_nodes):
        assert cluster_3_nodes.node_count == 3
        assert "node-000" in cluster_3_nodes.nodes

    def test_empty_cluster(self):
        cluster = ClusterState()
        assert cluster.node_count == 0
        assert cluster.pending_count == 0


class TestClusterPodLifecycle:
    def test_enqueue_pod(self, cluster_3_nodes, small_pod):
        cluster_3_nodes.enqueue_pod(small_pod)
        assert cluster_3_nodes.pending_count == 1
        assert small_pod.status == PodStatus.PENDING
        assert small_pod.pod_id in cluster_3_nodes.all_pods

    def test_bind_pod_success(self, cluster_3_nodes, small_pod):
        cluster_3_nodes.enqueue_pod(small_pod)
        result = cluster_3_nodes.bind_pod(small_pod, "node-000", 1.0)

        assert result.success is True
        assert result.node_id == "node-000"
        assert small_pod.status == PodStatus.SCHEDULED
        assert cluster_3_nodes.pending_count == 0

    def test_bind_pod_insufficient_resources(self, cluster_3_nodes, oversized_pod):
        cluster_3_nodes.enqueue_pod(oversized_pod)
        result = cluster_3_nodes.bind_pod(oversized_pod, "node-000", 1.0)

        assert result.success is False
        assert "Insufficient" in result.reason

    def test_release_pod(self, cluster_3_nodes, small_pod):
        cluster_3_nodes.enqueue_pod(small_pod)
        cluster_3_nodes.bind_pod(small_pod, "node-000", 1.0)
        cluster_3_nodes.release_pod(small_pod, 11.0)

        assert small_pod.status == PodStatus.COMPLETED
        node = cluster_3_nodes.get_node("node-000")
        assert node.cpu_allocated == pytest.approx(0.0)

    def test_reject_pod(self, cluster_3_nodes, small_pod):
        cluster_3_nodes.enqueue_pod(small_pod)
        result = cluster_3_nodes.reject_pod(small_pod, 5.0, reason="No capacity")

        assert result.success is False
        assert small_pod.status == PodStatus.REJECTED
        assert cluster_3_nodes.pending_count == 0


class TestClusterFeasibleNodes:
    def test_feasible_nodes_all_fit(self, cluster_3_nodes, small_pod):
        nodes = cluster_3_nodes.feasible_nodes(small_pod)
        assert len(nodes) == 3

    def test_feasible_nodes_none_fit(self, cluster_3_nodes, oversized_pod):
        nodes = cluster_3_nodes.feasible_nodes(oversized_pod)
        assert len(nodes) == 0

    def test_feasible_reduces_after_allocation(self, cluster_3_nodes):
        """Filling nodes reduces feasible set."""
        big = Pod(pod_id="big", cpu_request=3.5, mem_request=100.0)
        cluster_3_nodes.enqueue_pod(big)
        cluster_3_nodes.bind_pod(big, "node-000", 0.0)

        another_big = Pod(pod_id="big2", cpu_request=3.5, mem_request=100.0)
        feasible = cluster_3_nodes.feasible_nodes(another_big)
        # node-000 only has 0.5 CPU left; nodes 1 and 2 should still fit
        assert len(feasible) == 2


class TestClusterAggregates:
    def test_utilization_empty(self, cluster_3_nodes):
        assert cluster_3_nodes.cluster_cpu_utilization == 0.0
        assert cluster_3_nodes.cluster_mem_utilization == 0.0

    def test_utilization_after_bind(self, cluster_3_nodes, small_pod):
        cluster_3_nodes.enqueue_pod(small_pod)
        cluster_3_nodes.bind_pod(small_pod, "node-001", 0.0)
        # total capacity: 3 * 4 = 12 CPU
        expected_cpu = 0.5 / 12.0
        assert cluster_3_nodes.cluster_cpu_utilization == pytest.approx(expected_cpu)

    def test_pod_counts(self, cluster_3_nodes):
        p1 = Pod(pod_id="p1", cpu_request=0.1, mem_request=100.0, duration=5.0)
        p2 = Pod(pod_id="p2", cpu_request=0.1, mem_request=100.0)

        cluster_3_nodes.enqueue_pod(p1)
        cluster_3_nodes.enqueue_pod(p2)
        cluster_3_nodes.bind_pod(p1, "node-000", 0.0)
        cluster_3_nodes.reject_pod(p2, 1.0, "test")

        assert cluster_3_nodes.scheduled_pod_count == 1
        assert cluster_3_nodes.rejected_pod_count == 1

    def test_zero_capacity_cluster(self):
        """Empty cluster gives 0.0 utilization (no division by zero)."""
        cluster = ClusterState()
        assert cluster.cluster_cpu_utilization == 0.0
        assert cluster.cluster_mem_utilization == 0.0
