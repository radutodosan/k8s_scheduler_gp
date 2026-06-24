"""Tests for config.schema — YAML loading, defaults, nested dataclasses."""

import pytest
from pathlib import Path

from config.schema import (
    ClusterConfig,
    ConfigValidationError,
    DynamicsConfig,
    ExperimentConfig,
    FitnessWeights,
    GPConfig,
    NodeConfig,
    WorkloadConfig,
)
from gp.primitives import TERMINAL_NAMES


class TestExperimentConfigDefaults:
    def test_default_values(self):
        cfg = ExperimentConfig()
        assert cfg.seed == 42
        assert cfg.num_training_instances == 5
        assert cfg.num_validation_instances == 0
        assert cfg.output_format == "csv"
        assert cfg.dynamic_instances is False
        assert isinstance(cfg.cluster, ClusterConfig)
        assert isinstance(cfg.workload, WorkloadConfig)
        assert isinstance(cfg.gp, GPConfig)
        assert isinstance(cfg.fitness, FitnessWeights)


class TestYamlLoading:
    def test_load_smoke_config(self):
        path = Path(__file__).resolve().parent.parent / "config" / "smoke_test_config.yaml"
        if not path.exists():
            pytest.skip("smoke_test_config.yaml not found")
        cfg = ExperimentConfig.from_yaml(str(path))
        assert cfg.name == "smoke_test"
        assert cfg.seed == 42
        assert cfg.workload.total_pods == 20
        assert cfg.gp.population_size == 15
        assert len(cfg.cluster.node_templates) == 1
        assert cfg.cluster.node_templates[0].count == 3

    def test_load_default_config(self):
        path = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"
        if not path.exists():
            pytest.skip("default_config.yaml not found")
        cfg = ExperimentConfig.from_yaml(str(path))
        assert cfg.workload.total_pods > 0
        assert cfg.gp.n_generations > 0

    def test_from_yaml_with_tmp(self, tmp_path):
        """Write a minimal YAML and load it."""
        yaml_content = """\
name: "test_exp"
seed: 123
workload:
  total_pods: 5
gp:
  population_size: 10
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        cfg = ExperimentConfig.from_yaml(str(yaml_file))
        assert cfg.name == "test_exp"
        assert cfg.seed == 123
        assert cfg.workload.total_pods == 5
        assert cfg.gp.population_size == 10

    def test_dynamic_instances_from_yaml(self, tmp_path):
        """dynamic_instances should be parsed from YAML."""
        yaml_content = """\
name: "dyn_test"
dynamic_instances: true
"""
        yaml_file = tmp_path / "dyn.yaml"
        yaml_file.write_text(yaml_content)

        cfg = ExperimentConfig.from_yaml(str(yaml_file))
        assert cfg.dynamic_instances is True


class TestFitnessWeights:
    def test_defaults_sum_to_one(self):
        w = FitnessWeights()
        total = (
            w.alpha_wait_time
            + w.beta_resource_waste
            + w.gamma_failed_pods
            + w.delta_evicted_pods
            + w.epsilon_preemptions
            + getattr(w, "eta_churn", 0.0)
            + w.zeta_scheduling_attempts
        )
        assert total == pytest.approx(1.0)


class TestNodeConfig:
    def test_defaults(self):
        nc = NodeConfig()
        assert nc.count == 5
        assert nc.cpu_capacity == 8.0


# ── Validation tests ─────────────────────────────────────────────────


class TestWorkloadConfigValidation:
    def test_defaults_pass(self):
        WorkloadConfig().validate()

    def test_invalid_cpu_range_order(self):
        cfg = WorkloadConfig(cpu_range=[5.0, 1.0])
        with pytest.raises(ConfigValidationError, match="cpu_range"):
            cfg.validate()

    def test_invalid_cpu_range_negative(self):
        cfg = WorkloadConfig(cpu_range=[-1.0, 2.0])
        with pytest.raises(ConfigValidationError, match="cpu_range"):
            cfg.validate()

    def test_invalid_arrival_pattern(self):
        cfg = WorkloadConfig(arrival_pattern="exponential")
        with pytest.raises(ConfigValidationError, match="arrival_pattern"):
            cfg.validate()

    def test_invalid_burst_probability(self):
        cfg = WorkloadConfig(burst_probability=1.5)
        with pytest.raises(ConfigValidationError, match="burst_probability"):
            cfg.validate()

    def test_invalid_burst_size_order(self):
        cfg = WorkloadConfig(burst_size_min=10, burst_size_max=2)
        with pytest.raises(ConfigValidationError, match="burst_size_min"):
            cfg.validate()

    def test_invalid_priority_weights(self):
        cfg = WorkloadConfig(priority_weights={"low": 0.1, "high": 0.1})
        with pytest.raises(ConfigValidationError, match="priority_weights"):
            cfg.validate()

    def test_invalid_profile(self):
        cfg = WorkloadConfig(profile="unknown_profile")
        with pytest.raises(ConfigValidationError, match="profile"):
            cfg.validate()

    def test_mixed_profile_invalid_keys(self):
        cfg = WorkloadConfig(profile="mixed", profile_mix={"nonexistent": 1.0})
        with pytest.raises(ConfigValidationError, match="profile_mix"):
            cfg.validate()

    def test_mixed_profile_empty(self):
        cfg = WorkloadConfig(profile="mixed", profile_mix={})
        with pytest.raises(ConfigValidationError, match="profile_mix"):
            cfg.validate()

    def test_invalid_limit_ratio_order(self):
        cfg = WorkloadConfig(limit_ratio_min=3.0, limit_ratio_max=1.0)
        with pytest.raises(ConfigValidationError, match="limit_ratio_min"):
            cfg.validate()

    def test_invalid_anti_affinity_probability(self):
        cfg = WorkloadConfig(anti_affinity_probability=-0.1)
        with pytest.raises(ConfigValidationError, match="anti_affinity_probability"):
            cfg.validate()


class TestGPConfigValidation:
    def test_defaults_pass(self):
        GPConfig().validate()

    def test_invalid_engine(self):
        cfg = GPConfig(engine="pytorch")
        with pytest.raises(ConfigValidationError, match="engine"):
            cfg.validate()

    def test_invalid_crossover_prob(self):
        cfg = GPConfig(crossover_prob=1.5)
        with pytest.raises(ConfigValidationError, match="crossover_prob"):
            cfg.validate()

    def test_tournament_exceeds_population(self):
        cfg = GPConfig(population_size=10, tournament_size=20)
        with pytest.raises(ConfigValidationError, match="tournament_size"):
            cfg.validate()

    def test_negative_population(self):
        cfg = GPConfig(population_size=-5)
        with pytest.raises(ConfigValidationError, match="population_size"):
            cfg.validate()

    def test_invalid_fitness_aggregation(self):
        cfg = GPConfig(fitness_aggregation="median")
        with pytest.raises(ConfigValidationError, match="fitness_aggregation"):
            cfg.validate()

    def test_invalid_n_workers(self):
        cfg = GPConfig(n_workers=0)
        with pytest.raises(ConfigValidationError, match="n_workers"):
            cfg.validate()

    def test_selected_terminals_defaults_to_all(self):
        cfg = GPConfig()
        assert cfg.selected_terminals() == list(TERMINAL_NAMES)

    def test_selected_terminals_preserves_canonical_order(self):
        cfg = GPConfig(
            terminal_mandatory=["NODE_CPU_AVAIL", "POD_CPU_REQ"],
            terminal_optional_enabled=["RESOURCE_FIT", "POD_MEM_REQ"],
        )
        cfg.validate()
        expected = [
            name
            for name in TERMINAL_NAMES
            if name in {"NODE_CPU_AVAIL", "POD_CPU_REQ", "RESOURCE_FIT", "POD_MEM_REQ"}
        ]
        assert cfg.selected_terminals() == expected

    def test_invalid_mandatory_terminal_name(self):
        cfg = GPConfig(terminal_mandatory=["NOT_A_REAL_TERMINAL"])
        with pytest.raises(ConfigValidationError, match="Unknown mandatory terminal"):
            cfg.validate()

    def test_invalid_optional_terminal_name(self):
        cfg = GPConfig(terminal_optional_enabled=["NOT_A_REAL_TERMINAL"])
        with pytest.raises(ConfigValidationError, match="Unknown optional terminal"):
            cfg.validate()


class TestFitnessWeightsValidation:
    def test_defaults_pass(self):
        FitnessWeights().validate()

    def test_weights_sum_not_one(self):
        w = FitnessWeights(alpha_wait_time=0.5, beta_resource_waste=0.5, gamma_failed_pods=0.5)
        with pytest.raises(ConfigValidationError, match="sum"):
            w.validate()

    def test_negative_weight(self):
        w = FitnessWeights(alpha_wait_time=-0.1)
        with pytest.raises(ConfigValidationError, match="alpha_wait_time"):
            w.validate()


class TestDynamicsConfigValidation:
    def test_defaults_pass(self):
        DynamicsConfig().validate()

    def test_invalid_failure_mode(self):
        cfg = DynamicsConfig(failure_mode="crash")
        with pytest.raises(ConfigValidationError, match="failure_mode"):
            cfg.validate()

    def test_invalid_failure_rate(self):
        cfg = DynamicsConfig(failure_rate=5)
        with pytest.raises(ConfigValidationError, match="failure_rate"):
            cfg.validate()

    def test_recovery_time_order(self):
        cfg = DynamicsConfig(recovery_time_min=50.0, recovery_time_max=10.0)
        with pytest.raises(ConfigValidationError, match="recovery_time_min"):
            cfg.validate()


class TestExperimentConfigValidation:
    def test_defaults_pass(self):
        ExperimentConfig().validate()

    def test_from_yaml_validates(self, tmp_path):
        """Invalid YAML should raise ConfigValidationError on load."""
        yaml_content = """\
name: "bad"
workload:
  arrival_pattern: "invalid_mode"
"""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ConfigValidationError, match="arrival_pattern"):
            ExperimentConfig.from_yaml(str(yaml_file))

    def test_invalid_output_format(self):
        cfg = ExperimentConfig(output_format="xml")
        with pytest.raises(ConfigValidationError, match="output_format"):
            cfg.validate()

    def test_invalid_validation_instances(self):
        cfg = ExperimentConfig(num_validation_instances=-1)
        with pytest.raises(ConfigValidationError, match="num_validation_instances"):
            cfg.validate()

    def test_full_yaml_configs_pass(self):
        """Both default and smoke YAML configs pass validation."""
        config_dir = Path(__file__).resolve().parent.parent / "config"
        for name in ("default_config.yaml", "smoke_test_config.yaml"):
            path = config_dir / name
            if path.exists():
                cfg = ExperimentConfig.from_yaml(str(path))
                cfg.validate()  # should not raise
