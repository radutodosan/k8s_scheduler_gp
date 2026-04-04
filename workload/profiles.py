"""Workload profiles — realistic K8s workload archetypes.

Each profile overrides a subset of WorkloadConfig fields to produce
pods that resemble a specific real-world application class (web
servers, ML training jobs, CI pipelines, etc.).

Usage::

    from workload.profiles import PROFILES, apply_profile

    # Apply a single profile
    cfg = apply_profile(WorkloadConfig(total_pods=100), "web_serving")

    # Get profile for per-pod override in mixed mode
    overrides = PROFILES["ai_training"]
"""

from __future__ import annotations

import random as _random
from dataclasses import asdict
from typing import Any, Dict, List

from config.schema import WorkloadConfig

# ── Profile definitions ──────────────────────────────────────────────
# Each profile is a dict of WorkloadConfig field overrides.
# Fields not listed keep the base config's values.

PROFILES: Dict[str, Dict[str, Any]] = {
    "web_serving": {
        "description": "Stateless HTTP services — light CPU, moderate MEM, long-running, high priority, spread replicas",
        "cpu_range": [0.1, 1.0],
        "mem_range": [128.0, 1024.0],
        "duration_range": [20.0, 60.0],
        "priority_weights": {"low": 0.1, "medium": 0.3, "high": 0.6},
        "qos_weights": {"best_effort": 0.1, "burstable": 0.4, "guaranteed": 0.5},
        "namespaces": ["web", "production", "frontend"],
        "arrival_pattern": "diurnal",
        "anti_affinity_probability": 0.7,
        "possible_anti_affinity_keys": ["web-frontend", "web-api", "web-gateway"],
        "limit_probability": 0.6,
        "limit_ratio_min": 1.0,
        "limit_ratio_max": 1.5,
        "replica_group_probability": 0.5,
        "replica_group_size_range": [2, 5],
    },
    "ai_training": {
        "description": "GPU-hungry ML training jobs — heavy CPU/MEM, long duration, low priority, bursty arrival",
        "cpu_range": [2.0, 6.0],
        "mem_range": [4096.0, 16384.0],
        "gpu_range": [1.0, 4.0],
        "gpu_probability": 0.8,
        "duration_range": [40.0, 120.0],
        "priority_weights": {"low": 0.6, "medium": 0.3, "high": 0.1},
        "qos_weights": {"best_effort": 0.5, "burstable": 0.35, "guaranteed": 0.15},
        "namespaces": ["ml-training", "data-science", "research"],
        "arrival_pattern": "bursty",
        "anti_affinity_probability": 0.1,
        "possible_anti_affinity_keys": ["training-job"],
        "limit_probability": 0.8,
        "limit_ratio_min": 1.2,
        "limit_ratio_max": 2.5,
        "possible_taints": ["gpu", "dedicated"],
        "taint_toleration_probability": 0.6,
        "replica_group_probability": 0.3,
        "replica_group_size_range": [2, 4],
    },
    "ci_cd": {
        "description": "CI/CD pipeline jobs — moderate CPU, short-lived, bursty (commit pushes), expendable",
        "cpu_range": [0.5, 2.0],
        "mem_range": [512.0, 4096.0],
        "duration_range": [2.0, 15.0],
        "priority_weights": {"low": 0.3, "medium": 0.5, "high": 0.2},
        "qos_weights": {"best_effort": 0.6, "burstable": 0.3, "guaranteed": 0.1},
        "namespaces": ["ci", "testing", "build"],
        "arrival_pattern": "bursty",
        "bursty_spike_multiplier": 8.0,
        "bursty_spike_probability": 0.2,
        "anti_affinity_probability": 0.05,
        "possible_anti_affinity_keys": ["ci-runner"],
        "limit_probability": 0.4,
        "limit_ratio_min": 1.0,
        "limit_ratio_max": 3.0,
        "replica_group_probability": 0.4,
        "replica_group_size_range": [3, 6],
    },
    "batch_processing": {
        "description": "Data processing / ETL jobs — heavy CPU/MEM, medium duration, low priority, constant rate",
        "cpu_range": [1.0, 4.0],
        "mem_range": [2048.0, 8192.0],
        "duration_range": [25.0, 80.0],
        "priority_weights": {"low": 0.7, "medium": 0.2, "high": 0.1},
        "qos_weights": {"best_effort": 0.3, "burstable": 0.5, "guaranteed": 0.2},
        "namespaces": ["batch", "etl", "data-pipeline"],
        "arrival_pattern": "constant",
        "anti_affinity_probability": 0.15,
        "possible_anti_affinity_keys": ["batch-worker"],
        "limit_probability": 0.5,
        "limit_ratio_min": 1.0,
        "limit_ratio_max": 2.0,
        "replica_group_probability": 0.2,
        "replica_group_size_range": [2, 4],
    },
    "microservices": {
        "description": "Distributed microservices — varied resource needs, medium duration, diurnal, spread across nodes",
        "cpu_range": [0.1, 1.5],
        "mem_range": [256.0, 2048.0],
        "duration_range": [15.0, 50.0],
        "priority_weights": {"low": 0.2, "medium": 0.5, "high": 0.3},
        "qos_weights": {"best_effort": 0.2, "burstable": 0.4, "guaranteed": 0.4},
        "namespaces": ["backend", "auth", "payments", "notifications"],
        "arrival_pattern": "diurnal",
        "anti_affinity_probability": 0.6,
        "possible_anti_affinity_keys": ["svc-auth", "svc-payments", "svc-orders", "svc-notify"],
        "limit_probability": 0.7,
        "limit_ratio_min": 1.0,
        "limit_ratio_max": 1.8,
        "replica_group_probability": 0.6,
        "replica_group_size_range": [2, 5],
    },
}

# Default weights when generating a mixed workload
DEFAULT_PROFILE_MIX: Dict[str, float] = {
    "web_serving": 0.30,
    "ai_training": 0.10,
    "ci_cd": 0.15,
    "batch_processing": 0.20,
    "microservices": 0.25,
}

PROFILE_NAMES = list(PROFILES.keys()) + ["mixed"]

# Fields in a profile dict that are NOT WorkloadConfig attributes
_PROFILE_META_KEYS = {"description", "replica_group_probability", "replica_group_size_range"}

# Allowed override keys (WorkloadConfig field names)
_ALLOWED_OVERRIDE_KEYS = {
    "cpu_range", "mem_range", "gpu_range", "gpu_probability",
    "duration_range", "priority_weights",
    "qos_weights", "namespaces", "arrival_pattern", "anti_affinity_probability",
    "possible_anti_affinity_keys", "limit_probability", "limit_ratio_min",
    "limit_ratio_max", "bursty_spike_multiplier", "bursty_spike_probability",
    "possible_taints", "taint_toleration_probability",
}


def validate_profiles() -> None:
    """Validate all profile definitions at import-time safe check.

    Raises ``ValueError`` if any profile contains unknown keys or
    invalid override values.
    """
    wc_fields = {f.name for f in WorkloadConfig.__dataclass_fields__.values()}
    for name, overrides in PROFILES.items():
        for key in overrides:
            if key in _PROFILE_META_KEYS:
                continue
            if key not in _ALLOWED_OVERRIDE_KEYS:
                raise ValueError(
                    f"Profile {name!r} has unknown override key {key!r}. "
                    f"Allowed: {sorted(_ALLOWED_OVERRIDE_KEYS)}"
                )
            if key not in wc_fields:
                raise ValueError(
                    f"Profile {name!r}: override key {key!r} is not a WorkloadConfig field"
                )
        # Validate range pairs
        for range_key in ("cpu_range", "mem_range", "duration_range", "gpu_range"):
            if range_key in overrides:
                r = overrides[range_key]
                if len(r) != 2 or r[0] > r[1]:
                    raise ValueError(f"Profile {name!r}: {range_key} must be [min, max] with min <= max")
        # Validate replica group config
        if "replica_group_size_range" in overrides:
            rg = overrides["replica_group_size_range"]
            if len(rg) != 2 or rg[0] > rg[1] or rg[0] < 1:
                raise ValueError(f"Profile {name!r}: replica_group_size_range must be [min, max] with min >= 1")


# Run validation at import time to catch typos early
validate_profiles()


def apply_profile(base: WorkloadConfig, profile_name: str) -> WorkloadConfig:
    """Return a *new* WorkloadConfig with profile overrides applied.

    For ``"mixed"`` profile, only ``arrival_pattern`` is set to
    ``"diurnal"``; per-pod overrides are handled by the generator.
    """
    if profile_name == "mixed":
        # Mixed mode: keep base config but switch to diurnal arrivals
        d = asdict(base)
        d["arrival_pattern"] = "diurnal"
        return WorkloadConfig(**d)

    if profile_name not in PROFILES:
        raise ValueError(
            f"Unknown profile {profile_name!r}. "
            f"Available: {', '.join(PROFILE_NAMES)}"
        )

    overrides = PROFILES[profile_name]
    d = asdict(base)
    for key, val in overrides.items():
        if key in _PROFILE_META_KEYS:
            continue
        d[key] = val
    return WorkloadConfig(**d)


def pick_profile_for_pod(
    profile_mix: Dict[str, float],
    rng: _random.Random,
) -> str:
    """Weighted random selection of a profile name for mixed mode."""
    labels = list(profile_mix.keys())
    weights = list(profile_mix.values())
    total = sum(weights)
    if total <= 0:
        return rng.choice(labels)
    r = rng.random() * total
    cumulative = 0.0
    for label, w in zip(labels, weights):
        cumulative += w
        if r <= cumulative:
            return label
    return labels[-1]


def get_pod_overrides(profile_name: str, rng: _random.Random) -> Dict[str, Any]:
    """Return per-pod field overrides for a given profile.

    Used in mixed mode: the generator picks a profile per pod and
    then applies these overrides to the individual pod's parameters.
    Returns cpu_range, mem_range, duration_range, priority_weights,
    qos_weights, namespace, anti_affinity_probability, and
    possible_anti_affinity_keys from the profile definition.
    """
    if profile_name not in PROFILES:
        return {}
    p = PROFILES[profile_name]
    return {
        "cpu_range": p.get("cpu_range"),
        "mem_range": p.get("mem_range"),
        "duration_range": p.get("duration_range"),
        "priority_weights": p.get("priority_weights"),
        "qos_weights": p.get("qos_weights"),
        "namespaces": p.get("namespaces"),
        "anti_affinity_probability": p.get("anti_affinity_probability"),
        "possible_anti_affinity_keys": p.get("possible_anti_affinity_keys"),
        "limit_probability": p.get("limit_probability"),
        "limit_ratio_min": p.get("limit_ratio_min"),
        "limit_ratio_max": p.get("limit_ratio_max"),
        "gpu_range": p.get("gpu_range"),
        "gpu_probability": p.get("gpu_probability"),
    }
