"""Tests for gp — primitives, DeapGeneticEngine, FitnessEvaluator."""

import math

import pytest

from gp.primitives import (
    FUNCTION_SET,
    TERMINAL_NAMES,
    extract_terminal_values,
    protected_div,
    if_positive,
    neg,
    safe_min,
    safe_max,
)
from gp.deap_engine import DeapGeneticEngine
from gp.fitness import FitnessEvaluator
from models.cluster_state import ClusterState
from models.node import Node
from models.pod import Pod, QoSClass
from config.schema import ClusterConfig, FitnessWeights, NodeConfig, WorkloadConfig


# ══════════════════════════════════════════════════════════════════════
# Primitives
# ══════════════════════════════════════════════════════════════════════


class TestProtectedDiv:
    def test_normal_division(self):
        assert protected_div(10.0, 2.0) == pytest.approx(5.0)

    def test_zero_denominator(self):
        assert protected_div(5.0, 0.0) == 1.0

    def test_near_zero_denominator(self):
        assert protected_div(5.0, 1e-10) == 1.0

    def test_negative_denominator(self):
        assert protected_div(6.0, -3.0) == pytest.approx(-2.0)


class TestIfPositive:
    def test_positive_condition(self):
        assert if_positive(1.0, 10.0, 20.0) == 10.0

    def test_zero_condition(self):
        """Zero is NOT positive — returns else branch."""
        assert if_positive(0.0, 10.0, 20.0) == 20.0

    def test_negative_condition(self):
        assert if_positive(-1.0, 10.0, 20.0) == 20.0


class TestOtherPrimitives:
    def test_neg(self):
        assert neg(3.0) == -3.0

    def test_safe_min(self):
        assert safe_min(2.0, 5.0) == 2.0

    def test_safe_max(self):
        assert safe_max(2.0, 5.0) == 5.0


class TestFunctionSet:
    def test_all_functions_registered(self):
        expected = {"add", "sub", "mul", "protected_div", "neg", "min", "max", "if_positive"}
        assert set(FUNCTION_SET.keys()) == expected

    def test_arities(self):
        assert FUNCTION_SET["neg"][1] == 1
        assert FUNCTION_SET["if_positive"][1] == 3
        for name in ("add", "sub", "mul", "protected_div", "min", "max"):
            assert FUNCTION_SET[name][1] == 2


class TestTerminalNames:
    def test_count(self):
        assert len(TERMINAL_NAMES) == 31

    def test_contains_key_features(self):
        assert "POD_CPU_REQ" in TERMINAL_NAMES
        assert "NODE_CPU_AVAIL" in TERMINAL_NAMES
        assert "CLUSTER_CPU_UTIL" in TERMINAL_NAMES
        assert "RESOURCE_FIT" in TERMINAL_NAMES
        assert "NODE_TAINT_COUNT" in TERMINAL_NAMES
        assert "NODE_PREEMPTABLE_COUNT" in TERMINAL_NAMES
        assert "NODE_OVERCOMMIT_RATIO" in TERMINAL_NAMES
        assert "NODE_AFFINITY_CONFLICT" in TERMINAL_NAMES

    def test_contains_new_terminals(self):
        new = [
            "POD_DURATION", "NODE_CPU_MEM_IMBALANCE",
            "NODE_CPU_FREE_AFTER", "NODE_MEM_FREE_AFTER",
            "PENDING_PRESSURE", "CLUSTER_UTIL_STD",
            "REPLICA_GROUP_COLOCATED", "NAMESPACE_PENDING_RATIO",
        ]
        for name in new:
            assert name in TERMINAL_NAMES


class TestExtractTerminalValues:
    def test_returns_all_terminals(self):
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0, arrival_time=0.0)
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster = ClusterState()
        cluster.add_node(node)

        vals = extract_terminal_values(pod, node, cluster, current_time=2.0)
        assert set(vals.keys()) == set(TERMINAL_NAMES)

    def test_wait_time_calculation(self):
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0, arrival_time=1.0)
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster = ClusterState()
        cluster.add_node(node)

        vals = extract_terminal_values(pod, node, cluster, current_time=5.0)
        assert vals["POD_WAIT_TIME"] == pytest.approx(4.0)

    def test_resource_fit_clamped(self):
        """RESOURCE_FIT should be in [0, 1] even for oversized pods."""
        pod = Pod(pod_id="p", cpu_request=100.0, mem_request=100000.0, arrival_time=0.0)
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster = ClusterState()
        cluster.add_node(node)

        vals = extract_terminal_values(pod, node, cluster, current_time=0.0)
        assert 0.0 <= vals["RESOURCE_FIT"] <= 1.0

    def test_pod_duration_terminal(self):
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0, duration=10.5)
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster = ClusterState()
        cluster.add_node(node)
        vals = extract_terminal_values(pod, node, cluster, current_time=0.0)
        assert vals["POD_DURATION"] == pytest.approx(10.5)

    def test_cpu_mem_imbalance(self):
        """Imbalance should equal |cpu_util - mem_util|."""
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=8192.0)
        # Allocate 2 cpu (50%) and 0 mem (0%) => imbalance = 0.5
        filler = Pod(pod_id="f", cpu_request=2.0, mem_request=0.0)
        node.allocate(filler)
        pod = Pod(pod_id="p", cpu_request=0.1, mem_request=100.0)
        cluster = ClusterState()
        cluster.add_node(node)
        vals = extract_terminal_values(pod, node, cluster, current_time=0.0)
        assert vals["NODE_CPU_MEM_IMBALANCE"] == pytest.approx(0.5)

    def test_free_after_terminals(self):
        """Free-after should reflect the surplus after placing the pod."""
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=1000.0)
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=200.0)
        cluster = ClusterState()
        cluster.add_node(node)
        vals = extract_terminal_values(pod, node, cluster, current_time=0.0)
        # (4-1)/4 = 0.75, (1000-200)/1000 = 0.8
        assert vals["NODE_CPU_FREE_AFTER"] == pytest.approx(0.75)
        assert vals["NODE_MEM_FREE_AFTER"] == pytest.approx(0.8)

    def test_pending_pressure(self):
        pod = Pod(pod_id="p1", cpu_request=0.5, mem_request=256.0)
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster = ClusterState()
        cluster.add_node(node)
        # Enqueue 3 pods
        for i in range(3):
            cluster.enqueue_pod(Pod(pod_id=f"q{i}", cpu_request=0.1, mem_request=64.0))
        vals = extract_terminal_values(pod, node, cluster, current_time=0.0)
        # pending=3, scheduled=0 => 3/(3+0+1) = 0.75
        assert vals["PENDING_PRESSURE"] == pytest.approx(0.75)

    def test_cluster_util_std(self):
        """With one node, std should be 0."""
        pod = Pod(pod_id="p", cpu_request=0.5, mem_request=256.0)
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster = ClusterState()
        cluster.add_node(node)
        vals = extract_terminal_values(pod, node, cluster, current_time=0.0)
        assert vals["CLUSTER_UTIL_STD"] == pytest.approx(0.0)

    def test_replica_group_colocated(self):
        node = Node(node_id="n", cpu_capacity=8.0, mem_capacity=16384.0)
        # Place two pods from group "deploy-A" on the node
        for i in range(2):
            p = Pod(pod_id=f"r{i}", cpu_request=0.5, mem_request=256.0, replica_group="deploy-A")
            node.allocate(p)
        cluster = ClusterState()
        cluster.add_node(node)
        pod = Pod(pod_id="new", cpu_request=0.5, mem_request=256.0, replica_group="deploy-A")
        vals = extract_terminal_values(pod, node, cluster, current_time=0.0)
        assert vals["REPLICA_GROUP_COLOCATED"] == pytest.approx(2.0)

    def test_replica_group_empty(self):
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster = ClusterState()
        cluster.add_node(node)
        pod = Pod(pod_id="p", cpu_request=0.5, mem_request=256.0)  # no replica_group
        vals = extract_terminal_values(pod, node, cluster, current_time=0.0)
        assert vals["REPLICA_GROUP_COLOCATED"] == pytest.approx(0.0)

    def test_namespace_pending_ratio(self):
        node = Node(node_id="n", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster = ClusterState()
        cluster.add_node(node)
        cluster.enqueue_pod(Pod(pod_id="a1", cpu_request=0.1, mem_request=64.0, namespace="ns-a"))
        cluster.enqueue_pod(Pod(pod_id="a2", cpu_request=0.1, mem_request=64.0, namespace="ns-a"))
        cluster.enqueue_pod(Pod(pod_id="b1", cpu_request=0.1, mem_request=64.0, namespace="ns-b"))
        pod = Pod(pod_id="target", cpu_request=0.5, mem_request=256.0, namespace="ns-a")
        vals = extract_terminal_values(pod, node, cluster, current_time=0.0)
        # 2 of 3 pending pods are in ns-a
        assert vals["NAMESPACE_PENDING_RATIO"] == pytest.approx(2.0 / 3.0)


# ══════════════════════════════════════════════════════════════════════
# DeapGeneticEngine
# ══════════════════════════════════════════════════════════════════════


class TestDeapEngineSetup:
    def test_train_before_setup_raises(self):
        engine = DeapGeneticEngine()
        with pytest.raises(RuntimeError, match="setup"):
            engine.train(fitness_function=lambda ind: 1.0, seed=1)

    def test_setup_twice_no_crash(self):
        """Repeated setup() should not raise (creator guard)."""
        engine = DeapGeneticEngine()
        engine.setup(terminal_names=list(TERMINAL_NAMES), population_size=5, n_generations=1)
        engine.setup(terminal_names=list(TERMINAL_NAMES), population_size=5, n_generations=1)

    def test_name(self):
        engine = DeapGeneticEngine()
        assert engine.name == "deap"


class TestDeapEngineTraining:
    def test_minimal_training(self):
        engine = DeapGeneticEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=8,
            n_generations=2,
            max_tree_depth=4,
        )
        result = engine.train(fitness_function=lambda ind: 1.0, seed=42)

        assert result.best_individual is not None
        assert result.best_fitness is not None
        assert result.generations == 2
        assert isinstance(result.best_expression, str)
        assert len(result.log) > 0

    def test_reproducibility(self):
        """Same seed should produce same best fitness."""
        def run(seed):
            engine = DeapGeneticEngine()
            engine.setup(
                terminal_names=list(TERMINAL_NAMES),
                population_size=10,
                n_generations=3,
                max_tree_depth=5,
            )
            return engine.train(fitness_function=lambda ind: 1.0, seed=seed)

        r1 = run(123)
        r2 = run(123)
        assert r1.best_fitness == r2.best_fitness
        assert r1.best_expression == r2.best_expression


class TestDeapEngineEvaluation:
    @pytest.fixture
    def engine_and_individual(self):
        engine = DeapGeneticEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=8,
            n_generations=2,
            max_tree_depth=4,
        )
        result = engine.train(fitness_function=lambda ind: 1.0, seed=42)
        return engine, result.best_individual

    def test_evaluate_returns_finite(self, engine_and_individual):
        engine, ind = engine_and_individual
        vals = {name: 1.0 for name in TERMINAL_NAMES}
        score = engine.evaluate_individual(ind, vals)
        assert math.isfinite(score)

    def test_get_expression_string(self, engine_and_individual):
        engine, ind = engine_and_individual
        expr = engine.get_expression_string(ind)
        assert isinstance(expr, str)
        assert len(expr) > 0


class TestDeapNSGA2:
    """Tests for NSGA-II multi-objective training."""

    def test_nsga2_training(self):
        engine = DeapGeneticEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=10,
            n_generations=3,
            max_tree_depth=4,
            multi_objective=True,
        )

        def fake_mo_fitness(ind):
            return (1.0, 0.5, 0.2)

        result = engine.train(fitness_function=fake_mo_fitness, seed=42)
        assert result.best_individual is not None
        assert result.best_fitness is not None
        assert result.generations == 3
        assert result.pareto_front is not None
        assert len(result.pareto_front) > 0

    def test_nsga2_pareto_front_dominance(self):
        """Pareto front should contain non-dominated individuals."""
        engine = DeapGeneticEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=12,
            n_generations=3,
            max_tree_depth=4,
            multi_objective=True,
        )

        call_count = [0]

        def varying_fitness(ind):
            call_count[0] += 1
            # Vary objectives so there's a real Pareto front
            v = hash(str(ind)) % 100 / 100.0
            return (v, 1.0 - v, 0.3)

        result = engine.train(fitness_function=varying_fitness, seed=42)
        # All Pareto front members should have 3-tuple fitness
        for ind in result.pareto_front:
            assert len(ind.fitness.values) == 3

    def test_nsga2_with_fitness_evaluator(self):
        """Full integration: NSGA-II with FitnessEvaluator.evaluate_objectives."""
        engine = DeapGeneticEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=8,
            n_generations=2,
            max_tree_depth=4,
            multi_objective=True,
        )

        pods = [
            Pod(
                pod_id=f"p-{i}", cpu_request=0.5, mem_request=256.0,
                arrival_time=float(i), duration=5.0,
            )
            for i in range(5)
        ]
        cluster_cfg = ClusterConfig(
            node_templates=[NodeConfig(count=2, cpu_capacity=4.0, mem_capacity=8192.0)]
        )
        weights = FitnessWeights()

        evaluator = FitnessEvaluator(
            gp_engine=engine,
            training_instances=[pods],
            cluster_config=cluster_cfg,
            fitness_weights=weights,
        )

        result = engine.train(
            fitness_function=evaluator.evaluate_objectives, seed=42,
        )
        assert result.best_individual is not None
        assert math.isfinite(result.best_fitness)
        assert result.pareto_front is not None


# ══════════════════════════════════════════════════════════════════════
# FitnessEvaluator
# ══════════════════════════════════════════════════════════════════════


class TestFitnessEvaluator:
    @pytest.fixture
    def evaluator_setup(self):
        """Prepare a GP engine + training pods + FitnessEvaluator."""
        engine = DeapGeneticEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=8,
            n_generations=2,
            max_tree_depth=4,
        )

        # Simple training pods
        pods = [
            Pod(
                pod_id=f"tp-{i}", cpu_request=0.5, mem_request=256.0,
                arrival_time=float(i), duration=5.0,
            )
            for i in range(5)
        ]

        cluster_cfg = ClusterConfig(
            node_templates=[NodeConfig(count=2, cpu_capacity=4.0, mem_capacity=8192.0)]
        )
        weights = FitnessWeights(alpha_wait_time=0.4, beta_resource_waste=0.3, gamma_failed_pods=0.3)

        evaluator = FitnessEvaluator(
            gp_engine=engine,
            training_instances=[pods],
            cluster_config=cluster_cfg,
            fitness_weights=weights,
        )
        return engine, evaluator

    def test_call_returns_finite(self, evaluator_setup):
        engine, evaluator = evaluator_setup
        result = engine.train(fitness_function=evaluator, seed=42)
        assert math.isfinite(result.best_fitness)

    def test_evaluate_on_test_instances(self, evaluator_setup):
        engine, evaluator = evaluator_setup
        result = engine.train(fitness_function=evaluator, seed=42)

        test_pods = [
            Pod(
                pod_id=f"test-{i}", cpu_request=0.3, mem_request=200.0,
                arrival_time=float(i), duration=3.0,
            )
            for i in range(3)
        ]
        metrics_list = evaluator.evaluate_on_instances(result.best_individual, [test_pods])
        assert len(metrics_list) == 1
        m = metrics_list[0]
        assert m.total_pods == 3
        assert m.scheduled_pods + m.rejected_pods == m.total_pods


# ══════════════════════════════════════════════════════════════════════
# Dynamic Training Instances
# ══════════════════════════════════════════════════════════════════════


class TestDynamicInstances:
    @pytest.fixture
    def dynamic_evaluator_setup(self):
        """Prepare a GP engine + FitnessEvaluator with dynamic instances."""
        from workload.poisson_generator import PoissonWorkloadGenerator

        engine = DeapGeneticEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=8,
            n_generations=3,
            max_tree_depth=4,
        )

        wl_config = WorkloadConfig(total_pods=10, arrival_rate=2.0)
        generator = PoissonWorkloadGenerator()

        # Generate initial static instances (used at gen 0)
        initial_instances = [
            generator.generate(wl_config, seed=42 + i) for i in range(2)
        ]

        cluster_cfg = ClusterConfig(
            node_templates=[NodeConfig(count=2, cpu_capacity=4.0, mem_capacity=8192.0)]
        )
        weights = FitnessWeights()

        evaluator = FitnessEvaluator(
            gp_engine=engine,
            training_instances=initial_instances,
            cluster_config=cluster_cfg,
            fitness_weights=weights,
            dynamic_instances=True,
            workload_generator=generator,
            workload_config=wl_config,
            base_seed=42,
            num_instances=2,
        )
        return engine, evaluator

    def test_rotate_changes_instances(self, dynamic_evaluator_setup):
        """Rotating to a different generation should produce different pods."""
        _, evaluator = dynamic_evaluator_setup
        old_arrivals = [p.arrival_time for inst in evaluator._training_instances for p in inst]
        evaluator.rotate_instances(5)
        new_arrivals = [p.arrival_time for inst in evaluator._training_instances for p in inst]
        assert old_arrivals != new_arrivals

    def test_rotate_deterministic(self, dynamic_evaluator_setup):
        """Same generation number should always produce the same instances."""
        _, evaluator = dynamic_evaluator_setup
        evaluator.rotate_instances(3)
        arrivals_a = [p.arrival_time for inst in evaluator._training_instances for p in inst]
        evaluator.rotate_instances(3)
        arrivals_b = [p.arrival_time for inst in evaluator._training_instances for p in inst]
        assert arrivals_a == arrivals_b

    def test_static_rotate_is_noop(self):
        """When dynamic_instances=False, rotate_instances does nothing."""
        engine = DeapGeneticEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=8,
            n_generations=2,
            max_tree_depth=4,
        )
        pods = [
            Pod(pod_id="s-0", cpu_request=0.5, mem_request=256.0,
                arrival_time=0.0, duration=5.0)
        ]
        evaluator = FitnessEvaluator(
            gp_engine=engine,
            training_instances=[pods],
            cluster_config=ClusterConfig(),
            fitness_weights=FitnessWeights(),
        )
        evaluator.rotate_instances(99)
        # Instances unchanged
        assert evaluator._training_instances == [pods]

    def test_dynamic_requires_generator(self):
        """dynamic_instances=True without generator should raise ValueError."""
        engine = DeapGeneticEngine()
        engine.setup(terminal_names=list(TERMINAL_NAMES))
        with pytest.raises(ValueError, match="workload_generator"):
            FitnessEvaluator(
                gp_engine=engine,
                training_instances=[[]],
                cluster_config=ClusterConfig(),
                fitness_weights=FitnessWeights(),
                dynamic_instances=True,
            )

    def test_dynamic_training_runs(self, dynamic_evaluator_setup):
        """Full GP training with dynamic instances should complete."""
        engine, evaluator = dynamic_evaluator_setup
        result = engine.train(fitness_function=evaluator, seed=42)
        assert math.isfinite(result.best_fitness)
        assert result.generations == 3
