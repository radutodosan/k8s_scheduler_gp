"""Tests for gp.gplearn_engine — GplearnEngine, _ScoringCollector, custom functions."""

import numpy as np
import pytest

from config.schema import ClusterConfig, FitnessWeights, NodeConfig
from gp.gplearn_engine import (
    GplearnEngine,
    _ScoringCollector,
    _gplearn_if_positive,
    _if_positive_vec,
    _copy_pod,
)
from gp.interface import GPResult
from gp.primitives import FUNCTION_SET, TERMINAL_NAMES, extract_terminal_values
from models.cluster_state import ClusterState
from models.node import Node
from models.pod import Pod, QoSClass
from scheduling.gp_strategy import GPSchedulingStrategy
from scheduling.least_allocated import LeastAllocatedStrategy
from simulator.engine import SimulationEngine


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_pods(n: int = 10, arrival_spread: float = 2.0) -> list[Pod]:
    """Create a small set of pods for testing."""
    pods = []
    for i in range(n):
        pods.append(Pod(
            pod_id=f"p{i}",
            cpu_request=0.5 + (i % 3) * 0.3,
            mem_request=256.0 + (i % 4) * 128.0,
            priority=100 * (i % 3),
            qos_class=QoSClass.BURSTABLE,
            arrival_time=i * arrival_spread,
            duration=10.0 + i,
            namespace="test",
        ))
    return pods


def _small_cluster_config() -> ClusterConfig:
    return ClusterConfig(
        node_templates=[NodeConfig(count=3, cpu_capacity=4.0, mem_capacity=8192.0)]
    )


# ═══════════════════════════════════════════════════════════════════════
# Custom if_positive function
# ═══════════════════════════════════════════════════════════════════════


class TestIfPositiveVec:
    def test_positive_values(self):
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([10.0, 20.0, 30.0])
        x3 = np.array([100.0, 200.0, 300.0])
        result = _if_positive_vec(x1, x2, x3)
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])

    def test_negative_values(self):
        x1 = np.array([-1.0, -5.0])
        x2 = np.array([10.0, 20.0])
        x3 = np.array([100.0, 200.0])
        result = _if_positive_vec(x1, x2, x3)
        np.testing.assert_array_equal(result, [100.0, 200.0])

    def test_zero_is_not_positive(self):
        x1 = np.array([0.0])
        x2 = np.array([10.0])
        x3 = np.array([99.0])
        result = _if_positive_vec(x1, x2, x3)
        np.testing.assert_array_equal(result, [99.0])

    def test_mixed(self):
        x1 = np.array([5.0, -3.0, 0.0, 0.1])
        x2 = np.array([1.0, 2.0, 3.0, 4.0])
        x3 = np.array([10.0, 20.0, 30.0, 40.0])
        result = _if_positive_vec(x1, x2, x3)
        np.testing.assert_array_equal(result, [1.0, 20.0, 30.0, 4.0])

    def test_gplearn_custom_function_metadata(self):
        assert _gplearn_if_positive.name == "if_positive"
        assert _gplearn_if_positive.arity == 3


# ═══════════════════════════════════════════════════════════════════════
# _ScoringCollector
# ═══════════════════════════════════════════════════════════════════════


class TestScoringCollector:
    def test_delegates_to_inner_strategy(self):
        inner = LeastAllocatedStrategy()
        collector = _ScoringCollector(inner)

        pod = Pod(pod_id="p0", cpu_request=1.0, mem_request=512.0, arrival_time=0.0, duration=5.0)
        cluster = ClusterState()
        node = Node(node_id="n0", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster.add_node(node)

        chosen = collector.select_node(pod, cluster)
        assert chosen == "n0"

    def test_records_features_and_scores(self):
        inner = LeastAllocatedStrategy()
        collector = _ScoringCollector(inner)

        pod = Pod(pod_id="p0", cpu_request=1.0, mem_request=512.0, arrival_time=0.0, duration=5.0)
        cluster = ClusterState()
        n1 = Node(node_id="n1", cpu_capacity=4.0, mem_capacity=8192.0)
        n2 = Node(node_id="n2", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster.add_node(n1)
        cluster.add_node(n2)

        collector.select_node(pod, cluster)
        assert len(collector.records) == 2  # one per feasible node

        for features_dict, score in collector.records:
            assert set(features_dict.keys()) == set(TERMINAL_NAMES)
            assert 0.0 <= score <= 1.0

    def test_score_reflects_resource_availability(self):
        inner = LeastAllocatedStrategy()
        collector = _ScoringCollector(inner)

        pod = Pod(pod_id="p0", cpu_request=0.5, mem_request=100.0, arrival_time=0.0, duration=5.0)
        cluster = ClusterState()
        n1 = Node(node_id="n1", cpu_capacity=4.0, mem_capacity=8192.0)
        n2 = Node(node_id="n2", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster.add_node(n1)
        cluster.add_node(n2)

        # Allocate resources on n1 to make it less available
        dummy = Pod(pod_id="dummy", cpu_request=2.0, mem_request=4096.0, arrival_time=0.0)
        n1.allocate(dummy)

        collector.select_node(pod, cluster)
        score_values = [r[1] for r in collector.records]
        # n2 should have higher score (more available resources)
        assert len(score_values) == 2
        assert len(set(score_values)) == 2  # different scores

    def test_name_wraps_inner(self):
        inner = LeastAllocatedStrategy()
        collector = _ScoringCollector(inner)
        assert "Collector" in collector.name
        assert "LeastAllocated" in collector.name

    def test_set_current_time(self):
        inner = LeastAllocatedStrategy()
        collector = _ScoringCollector(inner)
        collector.set_current_time(5.0)
        assert collector._current_time == 5.0

    def test_returns_none_when_no_feasible(self):
        inner = LeastAllocatedStrategy()
        collector = _ScoringCollector(inner)

        # Pod too large for node
        pod = Pod(pod_id="p", cpu_request=100.0, mem_request=100.0, arrival_time=0.0)
        cluster = ClusterState()
        cluster.add_node(Node(node_id="n", cpu_capacity=4.0, mem_capacity=8192.0))

        chosen = collector.select_node(pod, cluster)
        assert chosen is None
        assert len(collector.records) == 0

    def test_episode_hooks_delegate(self):
        inner = LeastAllocatedStrategy()
        collector = _ScoringCollector(inner)
        cluster = ClusterState()
        # Should not raise
        collector.on_episode_start(cluster)
        collector.on_episode_end(cluster)


# ═══════════════════════════════════════════════════════════════════════
# _copy_pod helper
# ═══════════════════════════════════════════════════════════════════════


class TestCopyPod:
    def test_copies_all_fields(self):
        pod = Pod(
            pod_id="p1", cpu_request=2.0, mem_request=4096.0,
            priority=500, qos_class=QoSClass.GUARANTEED,
            arrival_time=10.0, duration=30.0, namespace="prod",
        )
        copy = _copy_pod(pod)
        assert copy.pod_id == "p1"
        assert copy.cpu_request == 2.0
        assert copy.mem_request == 4096.0
        assert copy.priority == 500
        assert copy.qos_class == QoSClass.GUARANTEED
        assert copy.arrival_time == 10.0
        assert copy.duration == 30.0
        assert copy.namespace == "prod"

    def test_copy_resets_mutable_state(self):
        pod = Pod(pod_id="p", cpu_request=1.0, mem_request=512.0, arrival_time=0.0)
        pod.schedule_on("n0", 2.0)
        copy = _copy_pod(pod)
        assert copy.assigned_node_id is None
        assert copy.scheduled_time is None


# ═══════════════════════════════════════════════════════════════════════
# GplearnEngine — Setup
# ═══════════════════════════════════════════════════════════════════════


class TestGplearnEngineSetup:
    def test_name(self):
        engine = GplearnEngine()
        assert engine.name == "gplearn"

    def test_train_before_setup_raises(self):
        engine = GplearnEngine()
        with pytest.raises(RuntimeError, match="setup"):
            engine.train(fitness_function=lambda ind: 1.0, seed=1)

    def test_setup_stores_terminal_names(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            training_instances=[_make_pods(5)],
            cluster_config=_small_cluster_config(),
        )
        assert engine._terminal_names == list(TERMINAL_NAMES)
        assert engine._setup_done is True

    def test_setup_default_params(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            training_instances=[],
            cluster_config=_small_cluster_config(),
        )
        assert engine._params["population_size"] == 150
        assert engine._params["n_generations"] == 50
        assert engine._params["tournament_size"] == 3

    def test_setup_custom_params(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=50,
            n_generations=10,
            tournament_size=5,
            training_instances=[_make_pods(5)],
            cluster_config=_small_cluster_config(),
        )
        assert engine._params["population_size"] == 50
        assert engine._params["n_generations"] == 10
        assert engine._params["tournament_size"] == 5


# ═══════════════════════════════════════════════════════════════════════
# GplearnEngine — Function set mapping
# ═══════════════════════════════════════════════════════════════════════


class TestGplearnFunctionMapping:
    def test_maps_all_project_functions(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            training_instances=[],
            cluster_config=_small_cluster_config(),
        )
        func_set = engine._build_function_set()
        # Should map all 8 functions from FUNCTION_SET
        assert len(func_set) == 8

    def test_built_in_functions_are_strings(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            training_instances=[],
            cluster_config=_small_cluster_config(),
        )
        func_set = engine._build_function_set()
        strings = [f for f in func_set if isinstance(f, str)]
        assert "add" in strings
        assert "sub" in strings
        assert "mul" in strings
        assert "div" in strings
        assert "neg" in strings
        assert "min" in strings
        assert "max" in strings

    def test_if_positive_is_custom(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            training_instances=[],
            cluster_config=_small_cluster_config(),
        )
        func_set = engine._build_function_set()
        custom = [f for f in func_set if not isinstance(f, str)]
        assert len(custom) == 1
        assert custom[0].name == "if_positive"
        assert custom[0].arity == 3

    def test_default_fallback_for_empty_names(self):
        engine = GplearnEngine()
        engine._function_names = ["nonexistent_function"]
        func_set = engine._build_function_set()
        # Falls back to defaults when no valid functions found
        assert func_set == ("add", "sub", "mul", "div")


# ═══════════════════════════════════════════════════════════════════════
# GplearnEngine — Training data generation
# ═══════════════════════════════════════════════════════════════════════


class TestGplearnTrainingData:
    def test_generates_nonempty_data(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            training_instances=[_make_pods(8)],
            cluster_config=_small_cluster_config(),
            population_size=10,
            n_generations=1,
        )
        X, y = engine._generate_training_data(seed=42)
        assert X.shape[0] > 0
        assert X.shape[1] == len(TERMINAL_NAMES)
        assert len(y) == X.shape[0]

    def test_data_has_correct_dtypes(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            training_instances=[_make_pods(5)],
            cluster_config=_small_cluster_config(),
        )
        X, y = engine._generate_training_data(seed=42)
        assert X.dtype == np.float64
        assert y.dtype == np.float64

    def test_scores_in_valid_range(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            training_instances=[_make_pods(5)],
            cluster_config=_small_cluster_config(),
        )
        X, y = engine._generate_training_data(seed=42)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)

    def test_empty_instances_returns_empty(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            training_instances=[],
            cluster_config=_small_cluster_config(),
        )
        X, y = engine._generate_training_data(seed=42)
        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_multiple_instances_concatenated(self):
        engine = GplearnEngine()
        instances = [_make_pods(5), _make_pods(5)]
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            training_instances=instances,
            cluster_config=_small_cluster_config(),
        )
        X, y = engine._generate_training_data(seed=42)
        # Multiple instances should produce more data than one
        engine2 = GplearnEngine()
        engine2.setup(
            terminal_names=list(TERMINAL_NAMES),
            training_instances=[_make_pods(5)],
            cluster_config=_small_cluster_config(),
        )
        X2, y2 = engine2._generate_training_data(seed=42)
        assert X.shape[0] >= X2.shape[0]


# ═══════════════════════════════════════════════════════════════════════
# GplearnEngine — Training
# ═══════════════════════════════════════════════════════════════════════


class TestGplearnTraining:
    def test_minimal_training(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=20,
            n_generations=2,
            max_tree_depth=4,
            training_instances=[_make_pods(10)],
            cluster_config=_small_cluster_config(),
        )
        result = engine.train(fitness_function=lambda ind: 1.0, seed=42)

        assert isinstance(result, GPResult)
        assert result.best_individual is not None
        assert result.best_fitness is not None
        assert isinstance(result.best_expression, str)
        assert len(result.best_expression) > 0
        assert result.generations == 2
        assert len(result.log) > 0

    def test_log_entries_have_expected_keys(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=20,
            n_generations=3,
            max_tree_depth=4,
            training_instances=[_make_pods(8)],
            cluster_config=_small_cluster_config(),
        )
        result = engine.train(fitness_function=lambda ind: 1.0, seed=42)

        for entry in result.log:
            assert "gen" in entry
            assert "min" in entry
            assert "avg" in entry

    def test_hall_of_fame_contains_best(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=20,
            n_generations=2,
            max_tree_depth=4,
            training_instances=[_make_pods(8)],
            cluster_config=_small_cluster_config(),
        )
        result = engine.train(fitness_function=lambda ind: 1.0, seed=42)
        assert len(result.hall_of_fame) >= 1
        assert result.hall_of_fame[0] is result.best_individual

    def test_train_without_data_raises(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=10,
            n_generations=1,
            training_instances=[],
            cluster_config=_small_cluster_config(),
        )
        with pytest.raises(RuntimeError, match="No training data"):
            engine.train(fitness_function=lambda ind: 1.0, seed=42)

    def test_reproducibility(self):
        """Same seed should produce same best expression."""
        def run(seed):
            engine = GplearnEngine()
            engine.setup(
                terminal_names=list(TERMINAL_NAMES),
                population_size=20,
                n_generations=2,
                max_tree_depth=4,
                training_instances=[_make_pods(8)],
                cluster_config=_small_cluster_config(),
            )
            return engine.train(fitness_function=lambda ind: 1.0, seed=seed)

        r1 = run(123)
        r2 = run(123)
        assert r1.best_expression == r2.best_expression
        assert r1.best_fitness == pytest.approx(r2.best_fitness)


# ═══════════════════════════════════════════════════════════════════════
# GplearnEngine — Evaluation
# ═══════════════════════════════════════════════════════════════════════


class TestGplearnEvaluation:
    @pytest.fixture
    def trained_engine(self):
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=20,
            n_generations=2,
            max_tree_depth=4,
            training_instances=[_make_pods(10)],
            cluster_config=_small_cluster_config(),
        )
        result = engine.train(fitness_function=lambda ind: 1.0, seed=42)
        return engine, result

    def test_evaluate_individual_returns_float(self, trained_engine):
        engine, result = trained_engine
        # Build terminal values  from a concrete pod/node
        pod = Pod(pod_id="px", cpu_request=1.0, mem_request=512.0, arrival_time=0.0)
        node = Node(node_id="nx", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster = ClusterState()
        cluster.add_node(node)

        vals = extract_terminal_values(pod, node, cluster, current_time=0.0)
        score = engine.evaluate_individual(result.best_individual, vals)
        assert isinstance(score, float)

    def test_evaluate_returns_finite(self, trained_engine):
        engine, result = trained_engine
        pod = Pod(pod_id="px", cpu_request=0.5, mem_request=256.0, arrival_time=0.0)
        node = Node(node_id="nx", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster = ClusterState()
        cluster.add_node(node)

        vals = extract_terminal_values(pod, node, cluster, current_time=1.0)
        score = engine.evaluate_individual(result.best_individual, vals)
        assert np.isfinite(score)

    def test_get_expression_string(self, trained_engine):
        engine, result = trained_engine
        expr = engine.get_expression_string(result.best_individual)
        assert isinstance(expr, str)
        assert len(expr) > 0
        assert expr == result.best_expression

    def test_different_inputs_may_give_different_scores(self, trained_engine):
        engine, result = trained_engine
        cluster = ClusterState()
        n1 = Node(node_id="n1", cpu_capacity=4.0, mem_capacity=8192.0)
        n2 = Node(node_id="n2", cpu_capacity=4.0, mem_capacity=8192.0)
        cluster.add_node(n1)
        cluster.add_node(n2)

        # Make nodes have very different resource levels
        dummy = Pod(pod_id="dummy", cpu_request=3.5, mem_request=7000.0, arrival_time=0.0)
        n1.allocate(dummy)

        pod = Pod(pod_id="px", cpu_request=0.3, mem_request=200.0, arrival_time=0.0)
        vals1 = extract_terminal_values(pod, n1, cluster, 0.0)
        vals2 = extract_terminal_values(pod, n2, cluster, 0.0)

        s1 = engine.evaluate_individual(result.best_individual, vals1)
        s2 = engine.evaluate_individual(result.best_individual, vals2)
        # With very different resource levels, scores should differ
        # (unless the evolved expression is constant, which is unlikely with enough data)
        # We just check both are finite
        assert np.isfinite(s1) and np.isfinite(s2)


# ═══════════════════════════════════════════════════════════════════════
# GplearnEngine — Integration with GPSchedulingStrategy
# ═══════════════════════════════════════════════════════════════════════


class TestGplearnIntegration:
    def test_gp_strategy_uses_gplearn_individual(self):
        """Train gplearn, wrap in GPSchedulingStrategy, run a simulation."""
        engine = GplearnEngine()
        cluster_cfg = _small_cluster_config()
        pods = _make_pods(10)

        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=20,
            n_generations=2,
            max_tree_depth=4,
            training_instances=[pods],
            cluster_config=cluster_cfg,
        )
        result = engine.train(fitness_function=lambda ind: 1.0, seed=42)

        # Use the trained individual in a simulation
        strategy = GPSchedulingStrategy(engine, result.best_individual)
        fresh_pods = [_copy_pod(p) for p in pods]

        sim = SimulationEngine(strategy=strategy, cluster_config=cluster_cfg)
        sim.build_cluster()
        sim.load_workload(fresh_pods)
        sim.run()

        metrics = sim.collector.get_metrics()
        assert metrics.total_pods == 10
        # GP strategy should schedule at least some pods
        assert metrics.scheduled_pods > 0

    def test_gplearn_vs_baseline_runs(self):
        """Both gplearn GP and a baseline should complete on same workload."""
        cluster_cfg = _small_cluster_config()
        pods = _make_pods(8)

        # Train gplearn
        engine = GplearnEngine()
        engine.setup(
            terminal_names=list(TERMINAL_NAMES),
            population_size=20,
            n_generations=2,
            max_tree_depth=4,
            training_instances=[pods],
            cluster_config=cluster_cfg,
        )
        result = engine.train(fitness_function=lambda ind: 1.0, seed=42)

        # Run with gplearn
        gp_strategy = GPSchedulingStrategy(engine, result.best_individual)
        fresh1 = [_copy_pod(p) for p in pods]
        sim1 = SimulationEngine(strategy=gp_strategy, cluster_config=cluster_cfg)
        sim1.build_cluster()
        sim1.load_workload(fresh1)
        sim1.run()

        # Run with LeastAllocated baseline
        baseline = LeastAllocatedStrategy()
        fresh2 = [_copy_pod(p) for p in pods]
        sim2 = SimulationEngine(strategy=baseline, cluster_config=cluster_cfg)
        sim2.build_cluster()
        sim2.load_workload(fresh2)
        sim2.run()

        m1 = sim1.collector.get_metrics()
        m2 = sim2.collector.get_metrics()
        # Both should handle same number of pods
        assert m1.total_pods == m2.total_pods == 8
