"""Tests for models.pod — Pod lifecycle, status transitions, QoS ordering."""

import pytest

from models.pod import Pod, PodStatus, QoSClass


class TestQoSClass:
    def test_ordering(self):
        """QoS ints allow direct comparison (BE < BURSTABLE < GUARANTEED)."""
        assert QoSClass.BEST_EFFORT < QoSClass.BURSTABLE < QoSClass.GUARANTEED

    def test_values(self):
        assert QoSClass.BEST_EFFORT == 1
        assert QoSClass.BURSTABLE == 2
        assert QoSClass.GUARANTEED == 3


class TestPodStatus:
    def test_ordering(self):
        assert PodStatus.PENDING < PodStatus.SCHEDULED < PodStatus.COMPLETED


class TestPodCreation:
    def test_defaults(self):
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=1024.0)
        assert pod.priority == 0
        assert pod.qos_class == QoSClass.BEST_EFFORT
        assert pod.status == PodStatus.PENDING
        assert pod.namespace == "default"
        assert pod.duration == 0.0
        assert pod.assigned_node_id is None
        assert pod.scheduled_time is None
        assert pod.completion_time is None

    def test_custom_attributes(self, large_pod):
        assert large_pod.cpu_request == 6.0
        assert large_pod.priority == 500
        assert large_pod.qos_class == QoSClass.GUARANTEED
        assert large_pod.namespace == "production"


class TestPodStatusTransitions:
    def test_schedule_on(self, small_pod):
        small_pod.schedule_on("node-000", 5.0)
        assert small_pod.status == PodStatus.SCHEDULED
        assert small_pod.assigned_node_id == "node-000"
        assert small_pod.scheduled_time == 5.0

    def test_complete(self, small_pod):
        small_pod.schedule_on("node-000", 1.0)
        small_pod.complete(11.0)
        assert small_pod.status == PodStatus.COMPLETED
        assert small_pod.completion_time == 11.0

    def test_reject(self, small_pod):
        small_pod.reject(3.0)
        assert small_pod.status == PodStatus.REJECTED
        assert small_pod.completion_time == 3.0


class TestPodWaitTime:
    def test_wait_time_after_scheduling(self, small_pod):
        """wait_time = scheduled_time - arrival_time."""
        small_pod.schedule_on("node-000", 5.0)
        assert small_pod.wait_time == pytest.approx(5.0)

    def test_wait_time_not_scheduled(self, small_pod):
        """Before scheduling, wait_time is 0.0."""
        assert small_pod.wait_time == 0.0


class TestPodFitsOn:
    def test_fits(self, small_pod):
        assert small_pod.fits_on(cpu_available=4.0, mem_available=8192.0) is True

    def test_not_enough_cpu(self, small_pod):
        assert small_pod.fits_on(cpu_available=0.1, mem_available=8192.0) is False

    def test_not_enough_mem(self, small_pod):
        assert small_pod.fits_on(cpu_available=4.0, mem_available=100.0) is False

    def test_exact_fit(self):
        pod = Pod(pod_id="exact", cpu_request=2.0, mem_request=1024.0)
        assert pod.fits_on(cpu_available=2.0, mem_available=1024.0) is True
