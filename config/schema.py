"""Experiment configuration schema and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigValidationError(ValueError):
    """Raised when experiment configuration contains invalid values."""


def _check_range(name: str, values: List[float], *, min_bound: float = 0.0) -> None:
    """Assert *values* is a 2-element list with values[0] <= values[1]."""
    if len(values) != 2:
        raise ConfigValidationError(f"{name} must have exactly 2 elements, got {len(values)}")
    if values[0] < min_bound:
        raise ConfigValidationError(f"{name}[0] must be >= {min_bound}, got {values[0]}")
    if values[0] > values[1]:
        raise ConfigValidationError(f"{name}[0] ({values[0]}) must be <= {name}[1] ({values[1]})")


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigValidationError(f"{name} must be in [0, 1], got {value}")


def _check_positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    if allow_zero:
        if value < 0:
            raise ConfigValidationError(f"{name} must be >= 0, got {value}")
    else:
        if value <= 0:
            raise ConfigValidationError(f"{name} must be > 0, got {value}")


def _check_weights(name: str, weights: Dict[str, float]) -> None:
    """Assert all values are non-negative and sum to ~1.0."""
    for k, v in weights.items():
        if v < 0:
            raise ConfigValidationError(f"{name}[{k!r}] must be >= 0, got {v}")
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        raise ConfigValidationError(f"{name} values must sum to ~1.0, got {total:.4f}")


# ─── Cluster configuration ───────────────────────────────────────────────

@dataclass
class NodeConfig:
    """Template for creating simulation nodes."""

    count: int = 5
    cpu_capacity: float = 8.0
    mem_capacity: float = 16384.0   # MiB
    gpu_capacity: float = 0.0       # GPUs (0 = no GPU)
    cost_per_hour: float = 1.0
    taints: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        _check_positive("NodeConfig.count", self.count)
        _check_positive("NodeConfig.cpu_capacity", self.cpu_capacity)
        _check_positive("NodeConfig.mem_capacity", self.mem_capacity)
        _check_positive("NodeConfig.gpu_capacity", self.gpu_capacity, allow_zero=True)
        _check_positive("NodeConfig.cost_per_hour", self.cost_per_hour, allow_zero=True)


@dataclass
class ClusterConfig:
    """Cluster-level experiment settings."""

    node_templates: List[NodeConfig] = field(default_factory=lambda: [NodeConfig()])

    def validate(self) -> None:
        if not self.node_templates:
            raise ConfigValidationError("ClusterConfig.node_templates must not be empty")
        for i, nt in enumerate(self.node_templates):
            try:
                nt.validate()
            except ConfigValidationError as e:
                raise ConfigValidationError(f"node_templates[{i}]: {e}") from e


# ─── Workload configuration ─────────────────────────────────────────────

@dataclass
class WorkloadConfig:
    """Controls synthetic workload generation."""

    total_pods: int = 100
    arrival_rate: float = 1.0           # avg pods per time-unit (Poisson λ)
    burst_probability: float = 0.1      # chance of a burst each time-step
    burst_size_min: int = 3
    burst_size_max: int = 8
    cpu_range: List[float] = field(default_factory=lambda: [0.1, 2.0])
    mem_range: List[float] = field(default_factory=lambda: [128.0, 4096.0])
    gpu_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    gpu_probability: float = 0.0  # probability a pod requests GPU
    duration_range: List[float] = field(default_factory=lambda: [5.0, 50.0])
    priority_weights: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.5, "medium": 0.3, "high": 0.2}
    )
    qos_weights: Dict[str, float] = field(
        default_factory=lambda: {"best_effort": 0.4, "burstable": 0.4, "guaranteed": 0.2}
    )
    namespaces: List[str] = field(default_factory=lambda: ["default", "production", "batch"])
    # Taints that may appear on nodes; pods may randomly tolerate some of them
    possible_taints: List[str] = field(default_factory=lambda: ["gpu", "spot", "dedicated"])
    taint_toleration_probability: float = 0.3  # probability a pod tolerates a given taint
    # Label keys and possible values for node_selector assignment
    possible_labels: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "disktype": ["ssd", "hdd"],
            "zone": ["zone-a", "zone-b"],
        }
    )
    node_selector_probability: float = 0.2  # probability a pod has a node_selector
    # Limits multiplier: cpu_limit = cpu_request * U(limit_ratio_min, limit_ratio_max)
    limit_ratio_min: float = 1.0   # 1.0 = no overcommit
    limit_ratio_max: float = 2.0   # 2.0 = up to 2× request
    limit_probability: float = 0.5  # probability a pod gets limits > requests
    # Anti-affinity: pods with the same key avoid the same node
    possible_anti_affinity_keys: List[str] = field(
        default_factory=lambda: ["app-web", "app-api", "app-worker"]
    )
    anti_affinity_probability: float = 0.3  # probability a pod has an anti-affinity key
    # Arrival pattern mode: "constant" (uniform Poisson), "diurnal" (time-of-day),
    # "bursty" (random spikes)
    arrival_pattern: str = "constant"
    diurnal_peak_rate: float = 3.0   # λ multiplier at peak hour
    diurnal_trough_rate: float = 0.3  # λ multiplier at lowest hour
    bursty_spike_multiplier: float = 5.0  # λ multiplier during a spike
    bursty_spike_probability: float = 0.15  # probability of entering a spike interval
    # Workload profile: "" (generic), "web_serving", "ai_training", "ci_cd",
    # "batch_processing", "microservices", "mixed" (weighted combination)
    profile: str = ""  # empty = generic (no profile), preserves backward compat
    profile_mix: Dict[str, float] = field(
        default_factory=lambda: {
            "web_serving": 0.30,
            "ai_training": 0.10,
            "ci_cd": 0.15,
            "batch_processing": 0.20,
            "microservices": 0.25,
        }
    )

    # ── Valid choices ────────────────────────────────────────────────
    VALID_ARRIVAL_PATTERNS = ("constant", "diurnal", "bursty")
    VALID_PROFILES = ("", "web_serving", "ai_training", "ci_cd",
                      "batch_processing", "microservices", "mixed")

    def validate(self) -> None:
        _check_positive("WorkloadConfig.total_pods", self.total_pods)
        _check_positive("WorkloadConfig.arrival_rate", self.arrival_rate)
        _check_probability("WorkloadConfig.burst_probability", self.burst_probability)
        _check_positive("WorkloadConfig.burst_size_min", self.burst_size_min)
        _check_positive("WorkloadConfig.burst_size_max", self.burst_size_max)
        if self.burst_size_min > self.burst_size_max:
            raise ConfigValidationError(
                f"burst_size_min ({self.burst_size_min}) must be <= burst_size_max ({self.burst_size_max})"
            )
        _check_range("WorkloadConfig.cpu_range", self.cpu_range, min_bound=0.0)
        _check_range("WorkloadConfig.mem_range", self.mem_range, min_bound=0.0)
        _check_range("WorkloadConfig.gpu_range", self.gpu_range, min_bound=0.0)
        _check_probability("WorkloadConfig.gpu_probability", self.gpu_probability)
        _check_range("WorkloadConfig.duration_range", self.duration_range, min_bound=0.0)
        _check_weights("WorkloadConfig.priority_weights", self.priority_weights)
        _check_weights("WorkloadConfig.qos_weights", self.qos_weights)
        if not self.namespaces:
            raise ConfigValidationError("WorkloadConfig.namespaces must not be empty")
        _check_probability("WorkloadConfig.taint_toleration_probability", self.taint_toleration_probability)
        _check_probability("WorkloadConfig.node_selector_probability", self.node_selector_probability)
        _check_positive("WorkloadConfig.limit_ratio_min", self.limit_ratio_min)
        _check_positive("WorkloadConfig.limit_ratio_max", self.limit_ratio_max)
        if self.limit_ratio_min > self.limit_ratio_max:
            raise ConfigValidationError(
                f"limit_ratio_min ({self.limit_ratio_min}) must be <= limit_ratio_max ({self.limit_ratio_max})"
            )
        _check_probability("WorkloadConfig.limit_probability", self.limit_probability)
        _check_probability("WorkloadConfig.anti_affinity_probability", self.anti_affinity_probability)

        if self.arrival_pattern not in self.VALID_ARRIVAL_PATTERNS:
            raise ConfigValidationError(
                f"arrival_pattern must be one of {self.VALID_ARRIVAL_PATTERNS}, got {self.arrival_pattern!r}"
            )
        _check_positive("WorkloadConfig.diurnal_peak_rate", self.diurnal_peak_rate)
        _check_positive("WorkloadConfig.diurnal_trough_rate", self.diurnal_trough_rate)
        _check_positive("WorkloadConfig.bursty_spike_multiplier", self.bursty_spike_multiplier)
        _check_probability("WorkloadConfig.bursty_spike_probability", self.bursty_spike_probability)

        if self.profile not in self.VALID_PROFILES:
            raise ConfigValidationError(
                f"profile must be one of {self.VALID_PROFILES}, got {self.profile!r}"
            )
        if self.profile == "mixed":
            if not self.profile_mix:
                raise ConfigValidationError("profile_mix must not be empty when profile='mixed'")
            for k in self.profile_mix:
                if k not in self.VALID_PROFILES or k in ("", "mixed"):
                    raise ConfigValidationError(f"profile_mix key {k!r} is not a valid base profile")
            total = sum(self.profile_mix.values())
            if abs(total - 1.0) > 0.05:
                raise ConfigValidationError(f"profile_mix values must sum to ~1.0, got {total:.4f}")


# ─── GP configuration ───────────────────────────────────────────────────

@dataclass
class GPConfig:
    """Parameters for the GP engine."""

    engine: str = "deap"                 # 'deap' or 'gplearn' (Phase 2)
    population_size: int = 150
    n_generations: int = 50
    tournament_size: int = 3
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    max_tree_depth: int = 10
    elitism_ratio: float = 0.05
    parsimony_coefficient: float = 0.001  # bloat control
    multi_objective: bool = False         # NSGA-II (3 objectives: wait, waste, reject)

    VALID_ENGINES = ("deap", "gplearn")

    def validate(self) -> None:
        if self.engine not in self.VALID_ENGINES:
            raise ConfigValidationError(
                f"GPConfig.engine must be one of {self.VALID_ENGINES}, got {self.engine!r}"
            )
        _check_positive("GPConfig.population_size", self.population_size)
        _check_positive("GPConfig.n_generations", self.n_generations)
        _check_positive("GPConfig.tournament_size", self.tournament_size)
        if self.tournament_size > self.population_size:
            raise ConfigValidationError(
                f"tournament_size ({self.tournament_size}) must be <= population_size ({self.population_size})"
            )
        _check_probability("GPConfig.crossover_prob", self.crossover_prob)
        _check_probability("GPConfig.mutation_prob", self.mutation_prob)
        _check_positive("GPConfig.max_tree_depth", self.max_tree_depth)
        _check_probability("GPConfig.elitism_ratio", self.elitism_ratio)
        _check_positive("GPConfig.parsimony_coefficient", self.parsimony_coefficient, allow_zero=True)
        if self.multi_objective and self.engine != "deap":
            raise ConfigValidationError(
                "multi_objective=True is only supported with engine='deap'"
            )


# ─── Dynamics configuration ───────────────────────────────────────────────

@dataclass
class DynamicsConfig:
    """Configuration for dynamic simulation events (node failures).

    failure_mode:
        - ``"off"``        — no failures (stable cluster)
        - ``"reschedule"`` — evicted pods are re-queued with a restart
          overhead penalty and continue execution on another node
        - ``"kill"``       — evicted pods are permanently rejected

    failure_rate:
        Controls the *number* of failure events as a fraction of
        cluster nodes: 1 → 10 %, 2 → 20 %, 3 → 30 % (max).  The
        concrete number of failures =
        ``max(1, round(num_nodes * failure_rate * 0.1))``.

    recovery_time_min / recovery_time_max:
        Uniform random range for how long a failed node stays down.

    restart_overhead_min / restart_overhead_max:
        Extra time added to each rescheduled pod's remaining duration
        (only relevant for ``"reschedule"`` mode).  Models container
        image pull, init containers, health-check warm-up, etc.
    """

    failure_mode: str = "off"                # "off" | "reschedule" | "kill"
    failure_rate: int = 1                    # 1=10%, 2=20%, 3=30%
    recovery_time_min: float = 10.0
    recovery_time_max: float = 30.0
    restart_overhead_min: float = 2.0
    restart_overhead_max: float = 8.0

    # ── legacy alias ────────────────────────────────────────────
    # ``node_failures: bool`` is still accepted in YAML for backward
    # compat — _from_dict converts it to the new fields.

    VALID_FAILURE_MODES = ("off", "reschedule", "kill")

    @property
    def enabled(self) -> bool:
        """True when any failure mode is active."""
        return self.failure_mode in ("reschedule", "kill")

    def validate(self) -> None:
        if self.failure_mode not in self.VALID_FAILURE_MODES:
            raise ConfigValidationError(
                f"failure_mode must be one of {self.VALID_FAILURE_MODES}, got {self.failure_mode!r}"
            )
        if not 1 <= self.failure_rate <= 3:
            raise ConfigValidationError(
                f"failure_rate must be 1, 2, or 3, got {self.failure_rate}"
            )
        _check_positive("DynamicsConfig.recovery_time_min", self.recovery_time_min)
        _check_positive("DynamicsConfig.recovery_time_max", self.recovery_time_max)
        if self.recovery_time_min > self.recovery_time_max:
            raise ConfigValidationError(
                f"recovery_time_min ({self.recovery_time_min}) must be <= recovery_time_max ({self.recovery_time_max})"
            )
        _check_positive("DynamicsConfig.restart_overhead_min", self.restart_overhead_min, allow_zero=True)
        _check_positive("DynamicsConfig.restart_overhead_max", self.restart_overhead_max, allow_zero=True)
        if self.restart_overhead_min > self.restart_overhead_max:
            raise ConfigValidationError(
                f"restart_overhead_min ({self.restart_overhead_min}) must be <= restart_overhead_max ({self.restart_overhead_max})"
            )


# ─── Fitness weights ─────────────────────────────────────────────────────

@dataclass
class FitnessWeights:
    """Weights α, β, γ for the combined fitness function."""

    alpha_wait_time: float = 0.4
    beta_resource_waste: float = 0.3
    gamma_failed_pods: float = 0.3

    def validate(self) -> None:
        _check_positive("FitnessWeights.alpha_wait_time", self.alpha_wait_time, allow_zero=True)
        _check_positive("FitnessWeights.beta_resource_waste", self.beta_resource_waste, allow_zero=True)
        _check_positive("FitnessWeights.gamma_failed_pods", self.gamma_failed_pods, allow_zero=True)
        total = self.alpha_wait_time + self.beta_resource_waste + self.gamma_failed_pods
        if abs(total - 1.0) > 0.01:
            raise ConfigValidationError(
                f"Fitness weights must sum to ~1.0, got {total:.4f}"
            )


# ─── Experiment-level settings ───────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Top-level configuration for a full experiment.

    An experiment consists of one or more simulation runs with a given
    cluster, workload, GP setup, and evaluation criteria.
    """

    name: str = "default_experiment"
    seed: int = 42
    num_training_instances: int = 5
    num_test_instances: int = 5
    dynamic_instances: bool = False      # regenerate training instances each generation
    output_dir: str = "tmp/results"
    output_format: str = "csv"           # 'csv' or 'json'

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    fitness: FitnessWeights = field(default_factory=FitnessWeights)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)

    # ── Serialisation helpers ────────────────────────────────────────

    @staticmethod
    def from_yaml(path: str | Path) -> ExperimentConfig:
        """Load configuration from a YAML file.

        Missing keys are filled with defaults.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
        return ExperimentConfig._from_dict(raw)

    @staticmethod
    def _from_dict(d: Dict[str, Any]) -> ExperimentConfig:
        cluster_raw = d.get("cluster", {})
        node_templates = [
            NodeConfig(**nt) for nt in cluster_raw.get("node_templates", [{}])
        ]
        cluster = ClusterConfig(node_templates=node_templates)

        workload = WorkloadConfig(**d.get("workload", {}))
        gp = GPConfig(**d.get("gp", {}))
        fitness = FitnessWeights(**d.get("fitness", {}))

        # Dynamics — support legacy ``node_failures: bool`` key
        dyn_raw = dict(d.get("dynamics", {}))
        if "node_failures" in dyn_raw:
            legacy = dyn_raw.pop("node_failures")
            if "failure_mode" not in dyn_raw:
                dyn_raw["failure_mode"] = "reschedule" if legacy else "off"
        if "failure_interval" in dyn_raw:
            dyn_raw.pop("failure_interval")  # no longer used
        dynamics = DynamicsConfig(**dyn_raw)

        cfg = ExperimentConfig(
            name=d.get("name", "default_experiment"),
            seed=d.get("seed", 42),
            num_training_instances=d.get("num_training_instances", 5),
            num_test_instances=d.get("num_test_instances", 5),
            dynamic_instances=d.get("dynamic_instances", False),
            output_dir=d.get("output_dir", "tmp/results"),
            output_format=d.get("output_format", "csv"),
            cluster=cluster,
            workload=workload,
            gp=gp,
            fitness=fitness,
            dynamics=dynamics,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate the entire configuration tree."""
        _check_positive("ExperimentConfig.num_training_instances", self.num_training_instances)
        _check_positive("ExperimentConfig.num_test_instances", self.num_test_instances)
        if self.output_format not in ("csv", "json"):
            raise ConfigValidationError(
                f"output_format must be 'csv' or 'json', got {self.output_format!r}"
            )
        self.cluster.validate()
        self.workload.validate()
        self.gp.validate()
        self.fitness.validate()
        self.dynamics.validate()
