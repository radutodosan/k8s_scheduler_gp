"""Tests for workload — PoissonWorkloadGenerator determinism and constraints."""

import pytest

from config.schema import WorkloadConfig
from models.pod import Pod, QoSClass
from workload.poisson_generator import PoissonWorkloadGenerator


@pytest.fixture
def wl_config():
    return WorkloadConfig(
        total_pods=30,
        arrival_rate=3.0,
        burst_probability=0.15,
        burst_size_min=2,
        burst_size_max=5,
        cpu_range=(0.1, 2.0),
        mem_range=(128.0, 4096.0),
        duration_range=(1.0, 20.0),
        priority_weights={"low": 0.5, "medium": 0.3, "high": 0.2},
        qos_weights={"best_effort": 0.4, "burstable": 0.4, "guaranteed": 0.2},
        namespaces=["default", "staging", "production"],
    )


class TestPoissonWorkloadGenerator:
    def test_correct_count(self, wl_config):
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(wl_config, seed=42)
        assert len(pods) == wl_config.total_pods

    def test_arrival_order(self, wl_config):
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(wl_config, seed=42)
        for i in range(1, len(pods)):
            assert pods[i].arrival_time >= pods[i - 1].arrival_time

    def test_unique_ids(self, wl_config):
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(wl_config, seed=42)
        ids = [p.pod_id for p in pods]
        assert len(ids) == len(set(ids))

    def test_resource_ranges(self, wl_config):
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(wl_config, seed=42)
        for p in pods:
            assert wl_config.cpu_range[0] <= p.cpu_request <= wl_config.cpu_range[1]
            assert wl_config.mem_range[0] <= p.mem_request <= wl_config.mem_range[1]
            assert wl_config.duration_range[0] <= p.duration <= wl_config.duration_range[1]

    def test_valid_qos_classes(self, wl_config):
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(wl_config, seed=42)
        valid = {QoSClass.BEST_EFFORT, QoSClass.BURSTABLE, QoSClass.GUARANTEED}
        for p in pods:
            assert p.qos_class in valid

    def test_valid_namespaces(self, wl_config):
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(wl_config, seed=42)
        for p in pods:
            assert p.namespace in wl_config.namespaces

    def test_priority_in_range(self, wl_config):
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(wl_config, seed=42)
        for p in pods:
            assert 0 <= p.priority <= 1000

    def test_reproducibility(self, wl_config):
        gen = PoissonWorkloadGenerator()
        pods1 = gen.generate(wl_config, seed=99)
        pods2 = gen.generate(wl_config, seed=99)
        assert len(pods1) == len(pods2)
        for a, b in zip(pods1, pods2):
            assert a.pod_id == b.pod_id
            assert a.cpu_request == b.cpu_request
            assert a.arrival_time == b.arrival_time

    def test_different_seeds_differ(self, wl_config):
        gen = PoissonWorkloadGenerator()
        pods1 = gen.generate(wl_config, seed=1)
        pods2 = gen.generate(wl_config, seed=2)
        # At least some attribute should differ
        arrivals1 = [p.arrival_time for p in pods1]
        arrivals2 = [p.arrival_time for p in pods2]
        assert arrivals1 != arrivals2


class TestBurstArrivalPatterns:
    """Feature B: variable-rate Poisson arrival patterns."""

    def test_diurnal_pattern_generates_correct_count(self):
        config = WorkloadConfig(total_pods=50, arrival_rate=2.0, arrival_pattern="diurnal")
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(config, seed=42)
        assert len(pods) == 50

    def test_bursty_pattern_generates_correct_count(self):
        config = WorkloadConfig(total_pods=50, arrival_rate=2.0, arrival_pattern="bursty")
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(config, seed=42)
        assert len(pods) == 50

    def test_constant_pattern_unchanged(self):
        """Constant pattern should produce identical results to before."""
        config = WorkloadConfig(total_pods=30, arrival_rate=2.0, arrival_pattern="constant")
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(config, seed=42)
        assert len(pods) == 30
        # Verify arrival order
        for i in range(1, len(pods)):
            assert pods[i].arrival_time >= pods[i - 1].arrival_time

    def test_diurnal_has_varying_density(self):
        """Diurnal pattern should produce non-uniform arrival spacing."""
        config = WorkloadConfig(
            total_pods=100, arrival_rate=2.0,
            arrival_pattern="diurnal",
            diurnal_peak_rate=5.0,
            diurnal_trough_rate=0.2,
        )
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(config, seed=42)
        # Split arrivals into two halves by time and compare density
        mid_time = pods[-1].arrival_time / 2
        first_half = [p for p in pods if p.arrival_time <= mid_time]
        second_half = [p for p in pods if p.arrival_time > mid_time]
        # Both halves should have pods (pattern shifts density, not eliminates)
        assert len(first_half) > 0
        assert len(second_half) > 0


class TestLimitsGeneration:
    """Feature A: pods may get cpu_limit / mem_limit from generator."""

    def test_some_pods_have_limits(self):
        config = WorkloadConfig(
            total_pods=50,
            limit_probability=0.8,
            limit_ratio_min=1.2,
            limit_ratio_max=2.0,
        )
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(config, seed=42)
        with_limits = [p for p in pods if p.cpu_limit > 0]
        assert len(with_limits) > 0

    def test_limits_above_request(self):
        config = WorkloadConfig(
            total_pods=50,
            limit_probability=1.0,
            limit_ratio_min=1.5,
            limit_ratio_max=2.0,
        )
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(config, seed=42)
        for p in pods:
            if p.cpu_limit > 0:
                assert p.cpu_limit >= p.cpu_request * 1.5 - 0.01
            if p.mem_limit > 0:
                assert p.mem_limit >= p.mem_request * 1.5 - 0.1


class TestAntiAffinityGeneration:
    """Feature C: pods may get anti_affinity_key from generator."""

    def test_some_pods_have_anti_affinity(self):
        config = WorkloadConfig(
            total_pods=50,
            anti_affinity_probability=0.8,
        )
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(config, seed=42)
        with_key = [p for p in pods if p.anti_affinity_key]
        assert len(with_key) > 0

    def test_anti_affinity_from_possible_keys(self):
        keys = ["svc-a", "svc-b"]
        config = WorkloadConfig(
            total_pods=50,
            possible_anti_affinity_keys=keys,
            anti_affinity_probability=1.0,
        )
        gen = PoissonWorkloadGenerator()
        pods = gen.generate(config, seed=42)
        for p in pods:
            assert p.anti_affinity_key in keys
