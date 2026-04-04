"""Tests for workload profiles — profile definitions, apply, mixed mode, generation."""

import random

import pytest

from config.schema import WorkloadConfig
from models.pod import QoSClass
from workload.poisson_generator import PoissonWorkloadGenerator
from workload.profiles import (
    DEFAULT_PROFILE_MIX,
    PROFILE_NAMES,
    PROFILES,
    apply_profile,
    get_pod_overrides,
    pick_profile_for_pod,
)


# ── Profile registry tests ──────────────────────────────────────────

class TestProfileDefinitions:
    def test_all_profiles_present(self):
        expected = {"web_serving", "ai_training", "ci_cd", "batch_processing", "microservices"}
        assert expected == set(PROFILES.keys())

    def test_profile_names_includes_mixed(self):
        assert "mixed" in PROFILE_NAMES

    def test_each_profile_has_cpu_mem_duration(self):
        for name, p in PROFILES.items():
            assert "cpu_range" in p, f"{name} missing cpu_range"
            assert "mem_range" in p, f"{name} missing mem_range"
            assert "duration_range" in p, f"{name} missing duration_range"
            assert len(p["cpu_range"]) == 2
            assert p["cpu_range"][0] < p["cpu_range"][1]
            assert p["mem_range"][0] < p["mem_range"][1]

    def test_each_profile_has_priority_and_qos_weights(self):
        for name, p in PROFILES.items():
            pw = p.get("priority_weights", {})
            qw = p.get("qos_weights", {})
            assert set(pw.keys()) <= {"low", "medium", "high"}, name
            assert set(qw.keys()) <= {"best_effort", "burstable", "guaranteed"}, name

    def test_default_mix_sums_to_one(self):
        assert abs(sum(DEFAULT_PROFILE_MIX.values()) - 1.0) < 1e-9

    def test_default_mix_keys_are_valid_profiles(self):
        for key in DEFAULT_PROFILE_MIX:
            assert key in PROFILES


# ── apply_profile tests ─────────────────────────────────────────────

class TestApplyProfile:
    def test_apply_web_serving_overrides_cpu(self):
        base = WorkloadConfig(total_pods=50)
        cfg = apply_profile(base, "web_serving")
        assert cfg.cpu_range == PROFILES["web_serving"]["cpu_range"]
        assert cfg.total_pods == 50  # preserved

    def test_apply_ai_training_overrides_mem(self):
        base = WorkloadConfig(total_pods=50)
        cfg = apply_profile(base, "ai_training")
        assert cfg.mem_range == PROFILES["ai_training"]["mem_range"]

    def test_apply_keeps_unoverridden_fields(self):
        base = WorkloadConfig(total_pods=77, arrival_rate=5.0)
        cfg = apply_profile(base, "batch_processing")
        assert cfg.total_pods == 77
        assert cfg.arrival_rate == 5.0

    def test_apply_mixed_sets_diurnal(self):
        base = WorkloadConfig(total_pods=50)
        cfg = apply_profile(base, "mixed")
        assert cfg.arrival_pattern == "diurnal"

    def test_apply_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            apply_profile(WorkloadConfig(), "nonexistent")

    def test_apply_ci_cd_sets_bursty(self):
        cfg = apply_profile(WorkloadConfig(), "ci_cd")
        assert cfg.arrival_pattern == "bursty"


# ── pick_profile_for_pod tests ──────────────────────────────────────

class TestPickProfile:
    def test_returns_valid_profile(self):
        rng = random.Random(42)
        for _ in range(100):
            name = pick_profile_for_pod(DEFAULT_PROFILE_MIX, rng)
            assert name in PROFILES

    def test_distribution_roughly_correct(self):
        rng = random.Random(123)
        counts = {}
        n = 10000
        for _ in range(n):
            name = pick_profile_for_pod(DEFAULT_PROFILE_MIX, rng)
            counts[name] = counts.get(name, 0) + 1
        # Web serving should be ~30%, check it's in 25-35%
        web_pct = counts.get("web_serving", 0) / n
        assert 0.25 < web_pct < 0.35
        # AI training should be ~10%, check 5-15%
        ai_pct = counts.get("ai_training", 0) / n
        assert 0.05 < ai_pct < 0.15


# ── get_pod_overrides tests ─────────────────────────────────────────

class TestGetPodOverrides:
    def test_returns_expected_keys(self):
        rng = random.Random(42)
        ov = get_pod_overrides("web_serving", rng)
        assert "cpu_range" in ov
        assert "mem_range" in ov
        assert "duration_range" in ov

    def test_unknown_profile_returns_empty(self):
        rng = random.Random(42)
        assert get_pod_overrides("nonexistent", rng) == {}


# ── Generation with profiles ────────────────────────────────────────

class TestProfileGeneration:
    @pytest.fixture
    def gen(self):
        return PoissonWorkloadGenerator()

    def test_web_serving_profile_cpu_range(self, gen):
        cfg = WorkloadConfig(total_pods=50, profile="web_serving")
        pods = gen.generate(cfg, seed=42)
        assert len(pods) == 50
        for p in pods:
            assert 0.1 <= p.cpu_request <= 1.0
            assert p.workload_type == "web_serving"

    def test_ai_training_profile_heavy_resources(self, gen):
        cfg = WorkloadConfig(total_pods=30, profile="ai_training")
        pods = gen.generate(cfg, seed=99)
        assert len(pods) == 30
        for p in pods:
            assert p.cpu_request >= 2.0
            assert p.mem_request >= 4096.0
            assert p.workload_type == "ai_training"

    def test_ci_cd_profile_short_duration(self, gen):
        cfg = WorkloadConfig(total_pods=40, profile="ci_cd")
        pods = gen.generate(cfg, seed=7)
        for p in pods:
            assert p.duration <= 15.0
            assert p.workload_type == "ci_cd"

    def test_batch_processing_namespaces(self, gen):
        cfg = WorkloadConfig(total_pods=40, profile="batch_processing")
        pods = gen.generate(cfg, seed=55)
        batch_ns = set(PROFILES["batch_processing"]["namespaces"])
        for p in pods:
            assert p.namespace in batch_ns

    def test_microservices_anti_affinity(self, gen):
        cfg = WorkloadConfig(total_pods=100, profile="microservices")
        pods = gen.generate(cfg, seed=33)
        # At least some pods should have anti-affinity keys
        aa_pods = [p for p in pods if p.anti_affinity_key]
        assert len(aa_pods) > 10  # ~60% probability

    def test_no_profile_is_generic(self, gen):
        cfg = WorkloadConfig(total_pods=20, profile="")
        pods = gen.generate(cfg, seed=42)
        assert len(pods) == 20
        for p in pods:
            assert p.workload_type == ""

    def test_backward_compat_no_profile(self, gen):
        """Config without profile should produce identical results to before."""
        cfg = WorkloadConfig(total_pods=20)
        pods = gen.generate(cfg, seed=42)
        assert len(pods) == 20
        # workload_type should be empty
        assert all(p.workload_type == "" for p in pods)


class TestMixedProfileGeneration:
    @pytest.fixture
    def gen(self):
        return PoissonWorkloadGenerator()

    def test_mixed_generates_multiple_types(self, gen):
        cfg = WorkloadConfig(total_pods=200, profile="mixed")
        pods = gen.generate(cfg, seed=42)
        types = set(p.workload_type for p in pods)
        # Should have at least 3 different workload types
        assert len(types) >= 3
        # All types should be valid profile names
        for t in types:
            assert t in PROFILES

    def test_mixed_different_cpu_ranges(self, gen):
        cfg = WorkloadConfig(total_pods=200, profile="mixed")
        pods = gen.generate(cfg, seed=42)
        web_pods = [p for p in pods if p.workload_type == "web_serving"]
        ai_pods = [p for p in pods if p.workload_type == "ai_training"]
        if web_pods and ai_pods:
            avg_web_cpu = sum(p.cpu_request for p in web_pods) / len(web_pods)
            avg_ai_cpu = sum(p.cpu_request for p in ai_pods) / len(ai_pods)
            # AI training pods should have higher average CPU
            assert avg_ai_cpu > avg_web_cpu

    def test_mixed_respects_custom_mix(self, gen):
        cfg = WorkloadConfig(
            total_pods=500,
            profile="mixed",
            profile_mix={
                "web_serving": 0.0,
                "ai_training": 1.0,
                "ci_cd": 0.0,
                "batch_processing": 0.0,
                "microservices": 0.0,
            },
        )
        pods = gen.generate(cfg, seed=42)
        # All pods should be ai_training
        assert all(p.workload_type == "ai_training" for p in pods)

    def test_mixed_pod_count_correct(self, gen):
        cfg = WorkloadConfig(total_pods=100, profile="mixed")
        pods = gen.generate(cfg, seed=42)
        assert len(pods) == 100

    def test_mixed_deterministic(self, gen):
        cfg = WorkloadConfig(total_pods=50, profile="mixed")
        pods1 = gen.generate(cfg, seed=42)
        pods2 = gen.generate(cfg, seed=42)
        for a, b in zip(pods1, pods2):
            assert a.pod_id == b.pod_id
            assert a.cpu_request == b.cpu_request
            assert a.workload_type == b.workload_type


# ── Serialisation round-trip ─────────────────────────────────────────

class TestProfileSerialisation:
    def test_workload_type_round_trip(self):
        from generate_dataset import _pod_from_dict, _pod_to_dict
        from models.pod import Pod

        pod = Pod(
            pod_id="test-pod",
            cpu_request=1.0,
            mem_request=1024.0,
            workload_type="ai_training",
        )
        d = _pod_to_dict(pod)
        assert d["workload_type"] == "ai_training"
        restored = _pod_from_dict(d)
        assert restored.workload_type == "ai_training"

    def test_empty_workload_type_backward_compat(self):
        from generate_dataset import _pod_from_dict

        # Old format without workload_type key
        d = {
            "pod_id": "old-pod",
            "cpu_request": 0.5,
            "mem_request": 512.0,
            "priority": 100,
            "qos_class": "BEST_EFFORT",
            "arrival_time": 0.0,
            "duration": 10.0,
            "namespace": "default",
        }
        pod = _pod_from_dict(d)
        assert pod.workload_type == ""
