"""Integration test — full pipeline: config → generate → train → evaluate → export."""

import csv
from pathlib import Path

import pytest

from config.schema import (
    ClusterConfig,
    ExperimentConfig,
    FitnessWeights,
    GPConfig,
    NodeConfig,
    WorkloadConfig,
)
from gp.deap_engine import DeapGeneticEngine
from gp.fitness import FitnessEvaluator
from gp.primitives import TERMINAL_NAMES
from metrics.reporter import MetricsReporter
from workload.poisson_generator import PoissonWorkloadGenerator


class TestEndToEndPipeline:
    """Validates the complete experiment pipeline mirrors main.py logic."""

    @pytest.fixture
    def mini_config(self):
        """Minimal-but-complete ExperimentConfig for fast integration tests."""
        return ExperimentConfig(
            name="integration_test",
            seed=777,
            num_training_instances=1,
            num_test_instances=1,
            output_dir="tmp/test_output",
            output_format="csv",
            cluster=ClusterConfig(
                node_templates=[NodeConfig(count=2, cpu_capacity=4.0, mem_capacity=8192.0)]
            ),
            workload=WorkloadConfig(
                total_pods=10,
                arrival_rate=2.0,
                burst_probability=0.0,
                burst_size_min=1,
                burst_size_max=1,
                cpu_range=(0.1, 1.0),
                mem_range=(128.0, 1024.0),
                duration_range=(2.0, 8.0),
                priority_weights={"low": 0.5, "medium": 0.3, "high": 0.2},
                qos_weights={"best_effort": 0.5, "burstable": 0.3, "guaranteed": 0.2},
                namespaces=["default"],
            ),
            gp=GPConfig(
                engine="deap",
                population_size=8,
                n_generations=2,
                tournament_size=3,
                crossover_prob=0.7,
                mutation_prob=0.2,
                max_tree_depth=4,
                elitism_ratio=0.1,
                parsimony_coefficient=0.0,
            ),
            fitness=FitnessWeights(
                alpha_wait_time=0.4,
                beta_resource_waste=0.3,
                gamma_failed_pods=0.3,
            ),
        )

    def test_full_pipeline(self, mini_config, tmp_path):
        cfg = mini_config

        # 1. Generate workload instances
        gen = PoissonWorkloadGenerator()
        training = [gen.generate(cfg.workload, seed=cfg.seed + i) for i in range(cfg.num_training_instances)]
        test = [gen.generate(cfg.workload, seed=cfg.seed + cfg.num_training_instances + i) for i in range(cfg.num_test_instances)]

        assert len(training) == 1
        assert len(test) == 1
        assert len(training[0]) == cfg.workload.total_pods

        # 2. Setup GP
        engine = DeapGeneticEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=cfg.gp.population_size,
            n_generations=cfg.gp.n_generations,
            tournament_size=cfg.gp.tournament_size,
            crossover_prob=cfg.gp.crossover_prob,
            mutation_prob=cfg.gp.mutation_prob,
            max_tree_depth=cfg.gp.max_tree_depth,
            elitism_ratio=cfg.gp.elitism_ratio,
            parsimony_coefficient=cfg.gp.parsimony_coefficient,
        )

        # 3. Build fitness evaluator
        evaluator = FitnessEvaluator(
            gp_engine=engine,
            training_instances=training,
            cluster_config=cfg.cluster,
            fitness_weights=cfg.fitness,
        )

        # 4. Train
        result = engine.train(fitness_function=evaluator, seed=cfg.seed)
        assert result.best_individual is not None
        assert result.best_fitness < float("inf")

        # 5. Evaluate on test set
        test_metrics = evaluator.evaluate_on_instances(result.best_individual, test)
        assert len(test_metrics) == 1
        m = test_metrics[0]
        assert m.total_pods == cfg.workload.total_pods
        # OOM kills can cause a pod to be both scheduled and later rejected
        assert m.scheduled_pods + m.rejected_pods >= m.total_pods

        # 6. Export CSV
        reporter = MetricsReporter()
        reporter.add_run(f"GP({engine.name})", "test-0", cfg.seed, m)
        csv_path = tmp_path / "integration_results.csv"
        reporter.export_csv(csv_path)

        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert int(rows[0]["total_pods"]) == cfg.workload.total_pods

    def test_reproducibility(self, mini_config):
        """Same config + seed produces same results."""
        cfg = mini_config

        def run_pipeline():
            gen = PoissonWorkloadGenerator()
            training = [gen.generate(cfg.workload, seed=cfg.seed)]
            engine = DeapGeneticEngine()
            engine.setup(
                terminal_names=list(TERMINAL_NAMES),
                population_size=cfg.gp.population_size,
                n_generations=cfg.gp.n_generations,
                max_tree_depth=cfg.gp.max_tree_depth,
            )
            evaluator = FitnessEvaluator(
                gp_engine=engine,
                training_instances=training,
                cluster_config=cfg.cluster,
                fitness_weights=cfg.fitness,
            )
            result = engine.train(fitness_function=evaluator, seed=cfg.seed)
            return result.best_fitness, result.best_expression

        f1, e1 = run_pipeline()
        f2, e2 = run_pipeline()
        assert f1 == f2
        assert e1 == e2
