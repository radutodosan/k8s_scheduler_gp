"""Concrete workload generator — Poisson arrivals with burst support.

Produces a deterministic, ordered list of Pods from a WorkloadConfig
and a random seed.  Supports workload *profiles* (web_serving,
ai_training, ci_cd, batch_processing, microservices, mixed) that
override resource ranges and scheduling parameters per-pod.
"""

from __future__ import annotations

import random as _random
from typing import Any, Dict, List, Optional

import numpy as np

from config.schema import WorkloadConfig
from models.pod import Pod, QoSClass
from workload.generator import IWorkloadGenerator
from workload.profiles import (
    PROFILES,
    apply_profile,
    get_pod_overrides,
    pick_profile_for_pod,
)


# ── Mapping helpers ──────────────────────────────────────────────────────

_PRIORITY_MAP = {
    "low": (0, 200),
    "medium": (201, 600),
    "high": (601, 1000),
}

_QOS_MAP = {
    "best_effort": QoSClass.BEST_EFFORT,
    "burstable": QoSClass.BURSTABLE,
    "guaranteed": QoSClass.GUARANTEED,
}

# ── Replica group defaults (when a profile doesn't specify them) ─────
_DEFAULT_REPLICA_GROUP_PROB = 0.15
_DEFAULT_REPLICA_GROUP_SIZE_RANGE = [2, 4]


class PoissonWorkloadGenerator(IWorkloadGenerator):
    """Generates pods with Poisson-distributed inter-arrival times.

    Supports occasional *bursts*: with probability ``burst_probability``
    per time-step, a cluster of pods arrives simultaneously.
    """

    def generate(self, config: WorkloadConfig, seed: int) -> List[Pod]:
        # Apply profile overrides to the base config (no-op when profile="")
        effective_config = config
        is_mixed = config.profile == "mixed"
        if config.profile and not is_mixed:
            effective_config = apply_profile(config, config.profile)

        rng = np.random.RandomState(seed)
        py_rng = _random.Random(seed)

        pods: List[Pod] = []
        pod_counter = 0
        current_time = 0.0
        group_counter = 0

        # Pre-compute weighted-choice helpers (used for non-mixed and as fallback)
        priority_labels, priority_weights = _unpack_weights(effective_config.priority_weights)
        qos_labels, qos_weights = _unpack_weights(effective_config.qos_weights)

        while pod_counter < effective_config.total_pods:
            # Determine how many pods arrive at this instant
            if py_rng.random() < effective_config.burst_probability:
                count = py_rng.randint(effective_config.burst_size_min, effective_config.burst_size_max)
            else:
                count = 1

            # Don't exceed the total
            count = min(count, effective_config.total_pods - pod_counter)

            i = 0
            while i < count and pod_counter < effective_config.total_pods:
                # In mixed mode, pick a profile per pod
                pod_overrides: Optional[Dict[str, Any]] = None
                workload_type = config.profile if config.profile and not is_mixed else ""
                chosen_profile = ""
                if is_mixed:
                    chosen_profile = pick_profile_for_pod(config.profile_mix, py_rng)
                    pod_overrides = get_pod_overrides(chosen_profile, py_rng)
                    workload_type = chosen_profile
                elif config.profile:
                    chosen_profile = config.profile

                # Replica group decision
                rg_prob, rg_range = _replica_group_params(chosen_profile)
                replica_group_tag = ""
                replica_count = 1
                if py_rng.random() < rg_prob:
                    replica_count = py_rng.randint(rg_range[0], rg_range[1])
                    replica_count = min(replica_count, effective_config.total_pods - pod_counter, count - i)
                    if replica_count > 1:
                        group_counter += 1
                        replica_group_tag = f"rg-{group_counter:04d}"

                # Generate the first (template) pod
                template_pod = self._make_pod(
                    pod_id=f"pod-{pod_counter:05d}",
                    arrival_time=round(current_time, 4),
                    config=effective_config,
                    rng=rng,
                    py_rng=py_rng,
                    priority_labels=priority_labels,
                    priority_weights=priority_weights,
                    qos_labels=qos_labels,
                    qos_weights=qos_weights,
                    workload_type=workload_type,
                    pod_overrides=pod_overrides,
                    replica_group=replica_group_tag,
                )
                pods.append(template_pod)
                pod_counter += 1
                i += 1

                # Generate replicas with same resources/priority/namespace
                for r in range(1, replica_count):
                    if pod_counter >= effective_config.total_pods:
                        break
                    replica = Pod(
                        pod_id=f"pod-{pod_counter:05d}",
                        cpu_request=template_pod.cpu_request,
                        mem_request=template_pod.mem_request,
                        gpu_request=template_pod.gpu_request,
                        priority=template_pod.priority,
                        qos_class=template_pod.qos_class,
                        arrival_time=template_pod.arrival_time,
                        duration=template_pod.duration,
                        namespace=template_pod.namespace,
                        tolerations=template_pod.tolerations,
                        node_selector=template_pod.node_selector,
                        cpu_limit=template_pod.cpu_limit,
                        mem_limit=template_pod.mem_limit,
                        anti_affinity_key=template_pod.anti_affinity_key,
                        workload_type=template_pod.workload_type,
                        replica_group=replica_group_tag,
                    )
                    pods.append(replica)
                    pod_counter += 1
                    i += 1

            # Advance time by a Poisson inter-arrival interval
            if pod_counter < effective_config.total_pods:
                effective_rate = _effective_arrival_rate(
                    effective_config, current_time,
                )
                interval = rng.exponential(1.0 / effective_rate)
                current_time += interval

        return pods

    # ── Private ──────────────────────────────────────────────────────

    @staticmethod
    def _make_pod(
        pod_id: str,
        arrival_time: float,
        config: WorkloadConfig,
        rng: np.random.RandomState,
        py_rng: _random.Random,
        priority_labels: List[str],
        priority_weights: List[float],
        qos_labels: List[str],
        qos_weights: List[float],
        workload_type: str = "",
        pod_overrides: Optional[Dict[str, Any]] = None,
        replica_group: str = "",
    ) -> Pod:
        # Per-pod overrides (mixed mode) take precedence over config
        ov = pod_overrides or {}

        cpu_r = ov.get("cpu_range") or config.cpu_range
        mem_r = ov.get("mem_range") or config.mem_range
        dur_r = ov.get("duration_range") or config.duration_range

        cpu = round(rng.uniform(cpu_r[0], cpu_r[1]), 2)
        mem = round(rng.uniform(mem_r[0], mem_r[1]), 1)
        duration = round(rng.uniform(dur_r[0], dur_r[1]), 2)

        # GPU — optional resource
        gpu_r = ov.get("gpu_range") or config.gpu_range
        gpu_prob = ov.get("gpu_probability") if ov.get("gpu_probability") is not None else config.gpu_probability
        if gpu_prob > 0 and gpu_r[1] > 0 and rng.random() < gpu_prob:
            gpu = float(max(1, round(rng.uniform(gpu_r[0], gpu_r[1]))))
        else:
            gpu = 0.0

        # Priority — use override weights if provided
        ov_pri = ov.get("priority_weights")
        if ov_pri:
            p_labels, p_weights = _unpack_weights(ov_pri)
        else:
            p_labels, p_weights = priority_labels, priority_weights
        pri_label = _weighted_choice(p_labels, p_weights, py_rng)
        lo, hi = _PRIORITY_MAP[pri_label]
        priority = py_rng.randint(lo, hi)

        # QoS — use override weights if provided
        ov_qos = ov.get("qos_weights")
        if ov_qos:
            q_labels, q_weights = _unpack_weights(ov_qos)
        else:
            q_labels, q_weights = qos_labels, qos_weights
        qos_label = _weighted_choice(q_labels, q_weights, py_rng)
        qos_class = _QOS_MAP[qos_label]

        # Namespace
        ns_choices = ov.get("namespaces") or config.namespaces
        namespace = py_rng.choice(ns_choices)

        # Anti-affinity — use per-pod override probability/keys
        aa_prob = ov.get("anti_affinity_probability") if ov.get("anti_affinity_probability") is not None else config.anti_affinity_probability
        aa_keys = ov.get("possible_anti_affinity_keys") or config.possible_anti_affinity_keys

        # Limits ratio — use per-pod override
        lim_prob = ov.get("limit_probability") if ov.get("limit_probability") is not None else config.limit_probability
        lim_min = ov.get("limit_ratio_min") if ov.get("limit_ratio_min") is not None else config.limit_ratio_min
        lim_max = ov.get("limit_ratio_max") if ov.get("limit_ratio_max") is not None else config.limit_ratio_max

        return Pod(
            pod_id=pod_id,
            cpu_request=cpu,
            mem_request=mem,
            gpu_request=gpu,
            priority=priority,
            qos_class=qos_class,
            arrival_time=arrival_time,
            duration=duration,
            namespace=namespace,
            tolerations=_make_tolerations(config.possible_taints, config.taint_toleration_probability, py_rng),
            node_selector=_make_node_selector(config.possible_labels, config.node_selector_probability, py_rng),
            cpu_limit=_make_limit(cpu, lim_min, lim_max, lim_prob, rng),
            mem_limit=_make_limit(mem, lim_min, lim_max, lim_prob, rng),
            anti_affinity_key=_make_anti_affinity_key(aa_keys, aa_prob, py_rng),
            workload_type=workload_type,
            replica_group=replica_group,
        )


# ── Module-level helpers ─────────────────────────────────────────────────

def _unpack_weights(d: dict) -> tuple:
    """Return (labels, normalised_weights) from a {label: weight} dict."""
    labels = list(d.keys())
    weights = list(d.values())
    total = sum(weights)
    if total <= 0:
        weights = [1.0 / len(labels)] * len(labels)
    else:
        weights = [w / total for w in weights]
    return labels, weights


def _weighted_choice(labels: List[str], weights: List[float], rng: _random.Random) -> str:
    """Weighted random choice using cumulative distribution."""
    r = rng.random()
    cumulative = 0.0
    for label, w in zip(labels, weights):
        cumulative += w
        if r <= cumulative:
            return label
    return labels[-1]


def _make_tolerations(
    possible_taints: List[str],
    probability: float,
    rng: _random.Random,
) -> frozenset:
    """Randomly assign toleration for each possible taint."""
    if not possible_taints:
        return frozenset()
    return frozenset(t for t in possible_taints if rng.random() < probability)


def _make_node_selector(
    possible_labels: dict,
    probability: float,
    rng: _random.Random,
) -> dict:
    """Randomly assign a node_selector from possible label values."""
    if not possible_labels or rng.random() >= probability:
        return {}
    # Pick one random label key and a random value for it
    key = rng.choice(list(possible_labels.keys()))
    val = rng.choice(possible_labels[key])
    return {key: val}


def _make_limit(
    request: float,
    ratio_min: float,
    ratio_max: float,
    probability: float,
    rng: np.random.RandomState,
) -> float:
    """Optionally set a limit higher than the request (overcommit)."""
    if rng.random() >= probability:
        return 0.0  # 0 = same as request
    ratio = rng.uniform(ratio_min, ratio_max)
    return round(request * ratio, 2)


def _make_anti_affinity_key(
    possible_keys: List[str],
    probability: float,
    rng: _random.Random,
) -> str:
    """Randomly assign an anti-affinity key to a pod."""
    if not possible_keys or rng.random() >= probability:
        return ""
    return rng.choice(possible_keys)


def _replica_group_params(profile_name: str) -> tuple:
    """Return (probability, size_range) for replica groups.

    Uses profile-specific values when defined, else defaults.
    """
    if profile_name and profile_name in PROFILES:
        p = PROFILES[profile_name]
        prob = p.get("replica_group_probability", _DEFAULT_REPLICA_GROUP_PROB)
        size_range = p.get("replica_group_size_range", _DEFAULT_REPLICA_GROUP_SIZE_RANGE)
        return prob, size_range
    return _DEFAULT_REPLICA_GROUP_PROB, _DEFAULT_REPLICA_GROUP_SIZE_RANGE


def _effective_arrival_rate(config: WorkloadConfig, current_time: float) -> float:
    """Compute the effective Poisson λ based on arrival_pattern mode.

    - ``constant``: λ = arrival_rate (unchanged)
    - ``diurnal``:  λ varies sinusoidally — peak at t≡12 (mod 24),
                    trough at t≡0 (mod 24), scaled between trough_rate
                    and peak_rate multipliers.
    - ``bursty``:   λ = arrival_rate normally, with random spikes at
                    ``bursty_spike_multiplier × arrival_rate``.
    """
    base = config.arrival_rate
    mode = config.arrival_pattern

    if mode == "diurnal":
        import math
        # Map time to a 24-hour cycle: peak at hour 12, trough at hour 0
        hour = current_time % 24.0
        # Sinusoidal: 0 at hour 0, 1 at hour 12
        phase = (1.0 - math.cos(2.0 * math.pi * hour / 24.0)) / 2.0
        lo = config.diurnal_trough_rate
        hi = config.diurnal_peak_rate
        multiplier = lo + (hi - lo) * phase
        return max(0.01, base * multiplier)

    if mode == "bursty":
        # Use a deterministic hash of quantised time to decide spike
        import hashlib
        bucket = int(current_time * 10)  # 0.1s buckets
        h = int(hashlib.md5(str(bucket).encode()).hexdigest()[:8], 16)
        if (h / 0xFFFFFFFF) < config.bursty_spike_probability:
            return base * config.bursty_spike_multiplier
        return base

    # "constant" or unrecognised → original rate
    return base
