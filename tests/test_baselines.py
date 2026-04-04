"""Tests for baseline scheduling strategies."""

import pytest

from models.cluster_state import ClusterState
from models.node import Node
from models.pod import Pod, QoSClass
from scheduling.random_strategy import RandomStrategy
from scheduling.round_robin import RoundRobinStrategy
from scheduling.first_fit import FirstFitStrategy
from scheduling.least_allocated import LeastAllocatedStrategy
from scheduling.most_allocated import MostAllocatedStrategy
from scheduling.balanced_allocation import BalancedAllocationStrategy


# ── Shared helpers ───────────────────────────────────────────────────

@pytest.fixture
def cluster_3_uniform():
    """3 identical nodes, each 4 CPU / 8 GiB."""
    cluster = ClusterState()
    for i in range(3):
        cluster.add_node(
            Node(node_id=f"node-{i:03d}", cpu_capacity=4.0, mem_capacity=8192.0)
        )
    return cluster


@pytest.fixture
def cluster_uneven():
    """Nodes with different loads for scoring-based strategies."""
    cluster = ClusterState()
    # node-000: empty
    cluster.add_node(Node(node_id="node-000", cpu_capacity=4.0, mem_capacity=8192.0))
    # node-001: half loaded
    n1 = Node(node_id="node-001", cpu_capacity=4.0, mem_capacity=8192.0)
    n1.cpu_allocated = 2.0
    n1.mem_allocated = 4096.0
    cluster.add_node(n1)
    # node-002: nearly full
    n2 = Node(node_id="node-002", cpu_capacity=4.0, mem_capacity=8192.0)
    n2.cpu_allocated = 3.5
    n2.mem_allocated = 7000.0
    cluster.add_node(n2)
    return cluster


@pytest.fixture
def small_pod():
    return Pod(
        pod_id="pod-small", cpu_request=0.5, mem_request=512.0,
        priority=100, qos_class=QoSClass.BURSTABLE,
        arrival_time=0.0, duration=10.0,
    )


@pytest.fixture
def oversized_pod():
    return Pod(
        pod_id="pod-huge", cpu_request=100.0, mem_request=999999.0,
        priority=100, arrival_time=0.0, duration=5.0,
    )


# ══════════════════════════════════════════════════════════════════════
# RandomStrategy
# ══════════════════════════════════════════════════════════════════════

class TestRandomStrategy:
    def test_selects_feasible_node(self, cluster_3_uniform, small_pod):
        strategy = RandomStrategy(seed=42)
        node_id = strategy.select_node(small_pod, cluster_3_uniform)
        assert node_id is not None
        assert node_id in cluster_3_uniform.nodes

    def test_returns_none_no_feasible(self, cluster_3_uniform, oversized_pod):
        strategy = RandomStrategy(seed=42)
        assert strategy.select_node(oversized_pod, cluster_3_uniform) is None

    def test_deterministic_with_seed(self, cluster_3_uniform, small_pod):
        results_a = [RandomStrategy(seed=7).select_node(small_pod, cluster_3_uniform) for _ in range(5)]
        results_b = [RandomStrategy(seed=7).select_node(small_pod, cluster_3_uniform) for _ in range(5)]
        assert results_a == results_b

    def test_name(self):
        assert RandomStrategy().name == "Random"


# ══════════════════════════════════════════════════════════════════════
# RoundRobinStrategy
# ══════════════════════════════════════════════════════════════════════

class TestRoundRobinStrategy:
    def test_cycles_through_nodes(self, cluster_3_uniform, small_pod):
        strategy = RoundRobinStrategy()
        strategy.on_episode_start(cluster_3_uniform)

        seen = []
        for _ in range(6):  # 2 full cycles
            node_id = strategy.select_node(small_pod, cluster_3_uniform)
            assert node_id is not None
            seen.append(node_id)

        # Should cycle: 0,1,2,0,1,2
        assert seen == ["node-000", "node-001", "node-002"] * 2

    def test_skips_infeasible(self, cluster_3_uniform):
        strategy = RoundRobinStrategy()
        strategy.on_episode_start(cluster_3_uniform)

        # Fill node-000 completely
        cluster_3_uniform.nodes["node-000"].cpu_allocated = 4.0
        cluster_3_uniform.nodes["node-000"].mem_allocated = 8192.0

        pod = Pod(pod_id="p", cpu_request=0.5, mem_request=256.0,
                  arrival_time=0.0, duration=5.0)
        node_id = strategy.select_node(pod, cluster_3_uniform)
        assert node_id in ("node-001", "node-002")

    def test_returns_none_all_full(self, cluster_3_uniform, oversized_pod):
        strategy = RoundRobinStrategy()
        strategy.on_episode_start(cluster_3_uniform)
        assert strategy.select_node(oversized_pod, cluster_3_uniform) is None

    def test_name(self):
        assert RoundRobinStrategy().name == "RoundRobin"


# ══════════════════════════════════════════════════════════════════════
# FirstFitStrategy
# ══════════════════════════════════════════════════════════════════════

class TestFirstFitStrategy:
    def test_picks_first_available(self, cluster_3_uniform, small_pod):
        strategy = FirstFitStrategy()
        node_id = strategy.select_node(small_pod, cluster_3_uniform)
        assert node_id == "node-000"

    def test_skips_full_node(self, cluster_3_uniform, small_pod):
        cluster_3_uniform.nodes["node-000"].cpu_allocated = 4.0
        strategy = FirstFitStrategy()
        node_id = strategy.select_node(small_pod, cluster_3_uniform)
        assert node_id == "node-001"

    def test_returns_none_all_full(self, cluster_3_uniform, oversized_pod):
        strategy = FirstFitStrategy()
        assert strategy.select_node(oversized_pod, cluster_3_uniform) is None

    def test_name(self):
        assert FirstFitStrategy().name == "FirstFit"


# ══════════════════════════════════════════════════════════════════════
# LeastAllocatedStrategy
# ══════════════════════════════════════════════════════════════════════

class TestLeastAllocatedStrategy:
    def test_selects_emptiest_node(self, cluster_uneven, small_pod):
        strategy = LeastAllocatedStrategy()
        node_id = strategy.select_node(small_pod, cluster_uneven)
        assert node_id == "node-000"  # empty node has most free resources

    def test_returns_none_no_feasible(self, cluster_3_uniform, oversized_pod):
        strategy = LeastAllocatedStrategy()
        assert strategy.select_node(oversized_pod, cluster_3_uniform) is None

    def test_chooses_among_feasible_only(self, cluster_uneven):
        """Even if node-000 is emptiest, if it can't fit, choose the next."""
        # Make node-000 unable to fit by filling CPU
        cluster_uneven.nodes["node-000"].cpu_allocated = 4.0
        pod = Pod(pod_id="p", cpu_request=0.5, mem_request=256.0,
                  arrival_time=0.0, duration=5.0)
        strategy = LeastAllocatedStrategy()
        node_id = strategy.select_node(pod, cluster_uneven)
        assert node_id == "node-001"  # half-loaded, more free than node-002

    def test_name(self):
        assert LeastAllocatedStrategy().name == "LeastAllocated"


# ══════════════════════════════════════════════════════════════════════
# MostAllocatedStrategy
# ══════════════════════════════════════════════════════════════════════

class TestMostAllocatedStrategy:
    def test_selects_fullest_node(self, cluster_uneven, small_pod):
        strategy = MostAllocatedStrategy()
        node_id = strategy.select_node(small_pod, cluster_uneven)
        assert node_id == "node-002"  # nearest to full

    def test_returns_none_no_feasible(self, cluster_3_uniform, oversized_pod):
        strategy = MostAllocatedStrategy()
        assert strategy.select_node(oversized_pod, cluster_3_uniform) is None

    def test_packs_tight(self, cluster_3_uniform):
        """All nodes empty → any is fine; after loading one, it should prefer it."""
        strategy = MostAllocatedStrategy()
        pod = Pod(pod_id="p1", cpu_request=1.0, mem_request=1024.0,
                  arrival_time=0.0, duration=10.0)
        # Load node-001
        cluster_3_uniform.nodes["node-001"].cpu_allocated = 2.0
        cluster_3_uniform.nodes["node-001"].mem_allocated = 4096.0

        node_id = strategy.select_node(pod, cluster_3_uniform)
        assert node_id == "node-001"

    def test_name(self):
        assert MostAllocatedStrategy().name == "MostAllocated"


# ══════════════════════════════════════════════════════════════════════
# BalancedAllocationStrategy
# ══════════════════════════════════════════════════════════════════════

class TestBalancedAllocationStrategy:
    def test_prefers_balanced_node(self):
        """Node with balanced CPU/mem utilisation should be preferred."""
        cluster = ClusterState()
        # node-A: CPU 50%, MEM 50% → balanced
        na = Node(node_id="node-A", cpu_capacity=4.0, mem_capacity=8192.0)
        na.cpu_allocated = 2.0
        na.mem_allocated = 4096.0
        cluster.add_node(na)
        # node-B: CPU 80%, MEM 10% → imbalanced
        nb = Node(node_id="node-B", cpu_capacity=4.0, mem_capacity=8192.0)
        nb.cpu_allocated = 3.0   # will be 3.5/4 = 87.5% after pod
        nb.mem_allocated = 500.0  # will be ~1000/8192 = 12% after pod
        cluster.add_node(nb)

        pod = Pod(pod_id="p", cpu_request=0.5, mem_request=512.0,
                  arrival_time=0.0, duration=5.0)

        strategy = BalancedAllocationStrategy()
        node_id = strategy.select_node(pod, cluster)
        assert node_id == "node-A"

    def test_returns_none_no_feasible(self, cluster_3_uniform, oversized_pod):
        strategy = BalancedAllocationStrategy()
        assert strategy.select_node(oversized_pod, cluster_3_uniform) is None

    def test_name(self):
        assert BalancedAllocationStrategy().name == "BalancedAllocation"


# ══════════════════════════════════════════════════════════════════════
# ISchedulingStrategy contract (common for all baselines)
# ══════════════════════════════════════════════════════════════════════

ALL_STRATEGIES = [
    RandomStrategy(seed=42),
    RoundRobinStrategy(),
    FirstFitStrategy(),
    LeastAllocatedStrategy(),
    MostAllocatedStrategy(),
    BalancedAllocationStrategy(),
]


class TestStrategyContract:
    @pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
    def test_name_is_nonempty_string(self, strategy):
        assert isinstance(strategy.name, str) and len(strategy.name) > 0

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
    def test_returns_none_on_empty_cluster(self, strategy):
        cluster = ClusterState()
        pod = Pod(pod_id="p", cpu_request=0.5, mem_request=256.0,
                  arrival_time=0.0, duration=5.0)
        assert strategy.select_node(pod, cluster) is None

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
    def test_hooks_do_not_raise(self, strategy, cluster_3_uniform):
        strategy.on_episode_start(cluster_3_uniform)
        strategy.on_episode_end(cluster_3_uniform)
