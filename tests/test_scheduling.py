"""Tests for scheduling — GPSchedulingStrategy + ISchedulingStrategy contract."""

import pytest

from models.cluster_state import ClusterState
from models.node import Node
from models.pod import Pod, QoSClass
from gp.deap_engine import DeapGeneticEngine
from gp.primitives import TERMINAL_NAMES
from scheduling.gp_strategy import GPSchedulingStrategy


@pytest.fixture
def trained_gp():
    """Train a minimal GP individual for testing scheduling decisions."""
    engine = DeapGeneticEngine()
    engine.setup(
        terminal_names=list(TERMINAL_NAMES),
        population_size=10,
        n_generations=2,
        tournament_size=3,
        crossover_prob=0.7,
        mutation_prob=0.2,
        max_tree_depth=4,
        elitism_ratio=0.1,
        parsimony_coefficient=0.0,
    )
    # Trivial fitness: constant (just evolve something)
    result = engine.train(fitness_function=lambda ind: 1.0, seed=42)
    return engine, result.best_individual


class TestGPSchedulingStrategy:
    def test_selects_feasible_node(self, trained_gp, cluster_3_nodes, small_pod):
        engine, individual = trained_gp
        strategy = GPSchedulingStrategy(engine, individual)
        strategy.set_current_time(0.0)

        cluster_3_nodes.enqueue_pod(small_pod)
        node_id = strategy.select_node(small_pod, cluster_3_nodes)

        assert node_id is not None
        assert node_id in cluster_3_nodes.nodes

    def test_returns_none_when_no_feasible(self, trained_gp, cluster_3_nodes, oversized_pod):
        engine, individual = trained_gp
        strategy = GPSchedulingStrategy(engine, individual)
        strategy.set_current_time(0.0)

        cluster_3_nodes.enqueue_pod(oversized_pod)
        node_id = strategy.select_node(oversized_pod, cluster_3_nodes)

        assert node_id is None

    def test_name_includes_engine(self, trained_gp):
        engine, individual = trained_gp
        strategy = GPSchedulingStrategy(engine, individual)
        assert "deap" in strategy.name.lower()

    def test_expression_is_string(self, trained_gp):
        engine, individual = trained_gp
        strategy = GPSchedulingStrategy(engine, individual)
        assert isinstance(strategy.expression, str)
        assert len(strategy.expression) > 0

    def test_hooks_no_op(self, trained_gp, cluster_3_nodes):
        """on_episode_start/end should not raise."""
        engine, individual = trained_gp
        strategy = GPSchedulingStrategy(engine, individual)
        strategy.on_episode_start(cluster_3_nodes)
        strategy.on_episode_end(cluster_3_nodes)
