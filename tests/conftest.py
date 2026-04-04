"""Shared fixtures for the k8s_scheduler_gp test suite."""

import pytest

from config.schema import (
    ClusterConfig,
    ExperimentConfig,
    FitnessWeights,
    GPConfig,
    NodeConfig,
    WorkloadConfig,
)
from models.cluster_state import ClusterState
from models.node import Node
from models.pod import Pod, QoSClass


# ── Pod fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def small_pod():
    """A pod that fits on any standard node."""
    return Pod(
        pod_id="pod-small",
        cpu_request=0.5,
        mem_request=512.0,
        priority=100,
        qos_class=QoSClass.BURSTABLE,
        arrival_time=0.0,
        duration=10.0,
        namespace="default",
    )


@pytest.fixture
def large_pod():
    """A pod that consumes significant resources."""
    return Pod(
        pod_id="pod-large",
        cpu_request=6.0,
        mem_request=12000.0,
        priority=500,
        qos_class=QoSClass.GUARANTEED,
        arrival_time=1.0,
        duration=20.0,
        namespace="production",
    )


@pytest.fixture
def oversized_pod():
    """A pod that exceeds any single node's capacity."""
    return Pod(
        pod_id="pod-oversized",
        cpu_request=100.0,
        mem_request=999999.0,
        priority=900,
        qos_class=QoSClass.GUARANTEED,
        arrival_time=0.0,
        duration=5.0,
    )


# ── Node fixtures ────────────────────────────────────────────────────


@pytest.fixture
def standard_node():
    """A node with moderate capacity."""
    return Node(
        node_id="node-000",
        cpu_capacity=8.0,
        mem_capacity=16384.0,
        cost_per_hour=1.0,
    )


@pytest.fixture
def tiny_node():
    """A node with very limited resources."""
    return Node(
        node_id="node-tiny",
        cpu_capacity=1.0,
        mem_capacity=1024.0,
        cost_per_hour=0.5,
    )


# ── Cluster fixtures ─────────────────────────────────────────────────


@pytest.fixture
def cluster_3_nodes():
    """ClusterState with 3 standard nodes (4 CPU, 8 GiB each)."""
    cluster = ClusterState()
    for i in range(3):
        cluster.add_node(
            Node(
                node_id=f"node-{i:03d}",
                cpu_capacity=4.0,
                mem_capacity=8192.0,
                cost_per_hour=1.0,
            )
        )
    return cluster


# ── Config fixtures ──────────────────────────────────────────────────


@pytest.fixture
def smoke_cluster_config():
    """ClusterConfig matching smoke_test_config.yaml."""
    return ClusterConfig(
        node_templates=[
            NodeConfig(count=3, cpu_capacity=4.0, mem_capacity=8192.0, cost_per_hour=1.0)
        ]
    )


@pytest.fixture
def smoke_workload_config():
    """WorkloadConfig matching smoke_test_config.yaml."""
    return WorkloadConfig(
        total_pods=20,
        arrival_rate=2.0,
        burst_probability=0.1,
        burst_size_min=2,
        burst_size_max=4,
        cpu_range=(0.1, 1.5),
        mem_range=(128.0, 2048.0),
        duration_range=(3.0, 15.0),
        priority_weights={"low": 0.5, "medium": 0.3, "high": 0.2},
        qos_weights={"best_effort": 0.4, "burstable": 0.4, "guaranteed": 0.2},
        namespaces=["default", "production"],
    )


@pytest.fixture
def smoke_gp_config():
    """GPConfig for fast testing."""
    return GPConfig(
        engine="deap",
        population_size=10,
        n_generations=3,
        tournament_size=3,
        crossover_prob=0.8,
        mutation_prob=0.2,
        max_tree_depth=5,
        elitism_ratio=0.1,
        parsimony_coefficient=0.001,
    )


@pytest.fixture
def smoke_fitness_weights():
    return FitnessWeights(
        alpha_wait_time=0.4,
        beta_resource_waste=0.3,
        gamma_failed_pods=0.3,
    )
