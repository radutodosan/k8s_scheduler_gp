"""Tests for models.node — resource accounting, allocation, release."""

import pytest

from models.node import Node
from models.pod import Pod, QoSClass


class TestNodeCreation:
    def test_defaults(self, standard_node):
        assert standard_node.cpu_allocated == 0.0
        assert standard_node.mem_allocated == 0.0
        assert standard_node.pod_count == 0

    def test_available_resources(self, standard_node):
        assert standard_node.cpu_available == 8.0
        assert standard_node.mem_available == 16384.0


class TestNodeUtilization:
    def test_empty_node(self, standard_node):
        assert standard_node.cpu_utilization == 0.0
        assert standard_node.mem_utilization == 0.0

    def test_zero_capacity_guard(self):
        """Division by zero returns 0.0."""
        node = Node(node_id="zero", cpu_capacity=0.0, mem_capacity=0.0)
        assert node.cpu_utilization == 0.0
        assert node.mem_utilization == 0.0

    def test_after_allocation(self, standard_node, small_pod):
        standard_node.allocate(small_pod)
        assert standard_node.cpu_utilization == pytest.approx(0.5 / 8.0)
        assert standard_node.mem_utilization == pytest.approx(512.0 / 16384.0)


class TestNodeCanFit:
    def test_fits(self, standard_node, small_pod):
        assert standard_node.can_fit(small_pod) is True

    def test_oversized_pod(self, standard_node, oversized_pod):
        assert standard_node.can_fit(oversized_pod) is False

    def test_no_room_after_allocation(self, tiny_node, small_pod):
        """After filling the node, a new pod shouldn't fit."""
        # tiny_node: 1 CPU, 1024 MiB; small_pod: 0.5 CPU, 512 MiB
        tiny_node.allocate(small_pod)
        second_pod = Pod(
            pod_id="pod-2", cpu_request=0.6, mem_request=600.0,
        )
        assert tiny_node.can_fit(second_pod) is False


class TestNodeAllocate:
    def test_allocate_updates_accounting(self, standard_node, small_pod):
        standard_node.allocate(small_pod)
        assert standard_node.cpu_allocated == pytest.approx(0.5)
        assert standard_node.mem_allocated == pytest.approx(512.0)
        assert standard_node.pod_count == 1
        assert small_pod.pod_id in standard_node.pods

    def test_allocate_raises_if_no_room(self, tiny_node, oversized_pod):
        with pytest.raises(ValueError, match="does not fit"):
            tiny_node.allocate(oversized_pod)


class TestNodeRelease:
    def test_release_frees_resources(self, standard_node, small_pod):
        standard_node.allocate(small_pod)
        standard_node.release(small_pod)
        assert standard_node.cpu_allocated == pytest.approx(0.0)
        assert standard_node.mem_allocated == pytest.approx(0.0)
        assert standard_node.pod_count == 0

    def test_release_clamps_to_zero(self, standard_node):
        """Releasing a pod that wasn't allocated should not go negative."""
        ghost_pod = Pod(pod_id="ghost", cpu_request=1.0, mem_request=1024.0)
        standard_node.release(ghost_pod)
        assert standard_node.cpu_allocated == 0.0
        assert standard_node.mem_allocated == 0.0


class TestNodeTaints:
    def test_no_taints_all_pods_fit(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0)
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0)
        assert node.can_fit(pod) is True

    def test_tainted_node_rejects_pod_without_toleration(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0,
                     taints=frozenset(["gpu"]))
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0)
        assert node.can_fit(pod) is False

    def test_tainted_node_accepts_pod_with_toleration(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0,
                     taints=frozenset(["gpu"]))
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0,
                  tolerations=frozenset(["gpu"]))
        assert node.can_fit(pod) is True

    def test_multiple_taints_require_all_tolerations(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0,
                     taints=frozenset(["gpu", "spot"]))
        pod_partial = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0,
                          tolerations=frozenset(["gpu"]))
        pod_full = Pod(pod_id="p2", cpu_request=1.0, mem_request=512.0,
                       tolerations=frozenset(["gpu", "spot"]))
        assert node.can_fit(pod_partial) is False
        assert node.can_fit(pod_full) is True

    def test_tolerates_method(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0,
                     taints=frozenset(["gpu"]))
        pod_yes = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0,
                      tolerations=frozenset(["gpu"]))
        pod_no = Pod(pod_id="p2", cpu_request=1.0, mem_request=512.0)
        assert node.tolerates(pod_yes) is True
        assert node.tolerates(pod_no) is False


class TestNodeLabels:
    def test_no_selector_fits_any_node(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0,
                     labels={"disktype": "ssd"})
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0)
        assert node.can_fit(pod) is True

    def test_selector_matches_label(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0,
                     labels={"disktype": "ssd", "zone": "zone-a"})
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0,
                  node_selector={"disktype": "ssd"})
        assert node.can_fit(pod) is True

    def test_selector_mismatch_rejects(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0,
                     labels={"disktype": "hdd"})
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0,
                  node_selector={"disktype": "ssd"})
        assert node.can_fit(pod) is False

    def test_selector_missing_label_rejects(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0)
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0,
                  node_selector={"zone": "zone-a"})
        assert node.can_fit(pod) is False

    def test_matches_selector_method(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0,
                     labels={"disktype": "ssd"})
        pod_yes = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0,
                      node_selector={"disktype": "ssd"})
        pod_no = Pod(pod_id="p2", cpu_request=1.0, mem_request=512.0,
                     node_selector={"disktype": "hdd"})
        assert node.matches_selector(pod_yes) is True
        assert node.matches_selector(pod_no) is False


class TestNodeOvercommit:
    def test_overcommit_ratio_zero_when_empty(self):
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=4096.0)
        assert node.cpu_overcommit_ratio == 0.0
        assert node.mem_overcommit_ratio == 0.0

    def test_overcommit_ratio_tracks_limits(self):
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=4096.0)
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=1024.0,
                  cpu_limit=3.0, mem_limit=3072.0)
        node.allocate(pod)
        assert node.cpu_overcommit_ratio == pytest.approx(3.0 / 4.0)
        assert node.mem_overcommit_ratio == pytest.approx(3072.0 / 4096.0)

    def test_overcommit_exceeds_one(self):
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=4096.0)
        p1 = Pod(pod_id="p1", cpu_request=2.0, mem_request=2048.0,
                 cpu_limit=3.0, mem_limit=3072.0)
        p2 = Pod(pod_id="p2", cpu_request=2.0, mem_request=2048.0,
                 cpu_limit=3.0, mem_limit=3072.0)
        node.allocate(p1)
        node.allocate(p2)
        assert node.cpu_overcommit_ratio == pytest.approx(6.0 / 4.0)
        assert node.mem_overcommit_ratio == pytest.approx(6144.0 / 4096.0)

    def test_release_updates_limit_totals(self):
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=4096.0)
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=1024.0,
                  cpu_limit=3.0, mem_limit=3072.0)
        node.allocate(pod)
        node.release(pod)
        assert node.cpu_limit_total == pytest.approx(0.0)
        assert node.mem_limit_total == pytest.approx(0.0)

    def test_no_limit_defaults_to_request(self):
        """When cpu_limit=0.0, effective limit = request."""
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=4096.0)
        pod = Pod(pod_id="p", cpu_request=1.5, mem_request=512.0)  # no limits set
        node.allocate(pod)
        assert node.cpu_limit_total == pytest.approx(1.5)
        assert node.mem_limit_total == pytest.approx(512.0)


class TestNodeAntiAffinity:
    def test_no_affinity_key_always_fits(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0)
        p1 = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0)
        p2 = Pod(pod_id="p2", cpu_request=1.0, mem_request=512.0)
        node.allocate(p1)
        assert node.can_fit(p2) is True

    def test_different_keys_fit(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0)
        p1 = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0,
                 anti_affinity_key="app-web")
        p2 = Pod(pod_id="p2", cpu_request=1.0, mem_request=512.0,
                 anti_affinity_key="app-api")
        node.allocate(p1)
        assert node.can_fit(p2) is True

    def test_same_key_rejected(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0)
        p1 = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0,
                 anti_affinity_key="app-web")
        p2 = Pod(pod_id="p2", cpu_request=1.0, mem_request=512.0,
                 anti_affinity_key="app-web")
        node.allocate(p1)
        assert node.can_fit(p2) is False

    def test_has_affinity_conflict_method(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0)
        p1 = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0,
                 anti_affinity_key="app-web")
        node.allocate(p1)
        p2_conflict = Pod(pod_id="p2", cpu_request=1.0, mem_request=512.0,
                          anti_affinity_key="app-web")
        p2_ok = Pod(pod_id="p3", cpu_request=1.0, mem_request=512.0,
                    anti_affinity_key="app-api")
        assert node.has_affinity_conflict(p2_conflict) is True
        assert node.has_affinity_conflict(p2_ok) is False

    def test_release_clears_conflict(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0)
        p1 = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0,
                 anti_affinity_key="app-web")
        p2 = Pod(pod_id="p2", cpu_request=1.0, mem_request=512.0,
                 anti_affinity_key="app-web")
        node.allocate(p1)
        assert node.can_fit(p2) is False
        node.release(p1)
        assert node.can_fit(p2) is True


class TestNodeGPU:
    def test_gpu_defaults_zero(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0)
        assert node.gpu_capacity == 0.0
        assert node.gpu_allocated == 0.0
        assert node.gpu_available == 0.0
        assert node.gpu_utilization == 0.0

    def test_gpu_allocation(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0, gpu_capacity=4.0)
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0, gpu_request=2.0)
        assert node.can_fit(pod) is True
        node.allocate(pod)
        assert node.gpu_allocated == 2.0
        assert node.gpu_available == 2.0
        assert node.gpu_utilization == pytest.approx(0.5)

    def test_gpu_release(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0, gpu_capacity=4.0)
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0, gpu_request=2.0)
        node.allocate(pod)
        node.release(pod)
        assert node.gpu_allocated == 0.0
        assert node.gpu_available == 4.0

    def test_gpu_pod_rejected_if_no_gpu(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0, gpu_capacity=0.0)
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0, gpu_request=1.0)
        assert node.can_fit(pod) is False

    def test_gpu_pod_rejected_if_insufficient(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=8192.0, gpu_capacity=2.0)
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0, gpu_request=3.0)
        assert node.can_fit(pod) is False

    def test_zero_gpu_pod_fits_anywhere(self):
        """Pod with no GPU request fits on any node (with or without GPUs)."""
        node_no_gpu = Node(node_id="n1", cpu_capacity=8.0, mem_capacity=8192.0)
        node_with_gpu = Node(node_id="n2", cpu_capacity=8.0, mem_capacity=8192.0, gpu_capacity=4.0)
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=512.0, gpu_request=0.0)
        assert node_no_gpu.can_fit(pod) is True
        assert node_with_gpu.can_fit(pod) is True
