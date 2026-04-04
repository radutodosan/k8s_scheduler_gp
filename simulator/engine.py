"""SimulationEngine — discrete-event simulation loop for Kubernetes scheduling.

Processes events chronologically:
  POD_ARRIVAL    → enqueue pod in ClusterState
  SCHEDULE_CYCLE → attempt to place all pending pods via ISchedulingStrategy
  POD_COMPLETION → release node resources, update metrics
  NODE_FAILURE   → evict pods, mark node unavailable, schedule recovery
  NODE_RECOVERY  → mark node available, trigger scheduling cycle
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional

from config.schema import ClusterConfig, DynamicsConfig, NodeConfig
from metrics.collector import MetricsCollector
from metrics.resource_monitor import ResourceMonitor
from models.cluster_state import ClusterState
from models.node import Node
from models.pod import Pod, PodStatus
from scheduling.strategy import ISchedulingStrategy
from simulator.event import (
    Event,
    EventType,
    PRIORITY_NODE_FAILURE,
    PRIORITY_NODE_RECOVERY,
    PRIORITY_POD_ARRIVAL,
    PRIORITY_POD_COMPLETION,
    PRIORITY_SCHEDULE_CYCLE,
)
from simulator.event_queue import EventQueue

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Drives a single simulation run (one workload instance).

    Lifecycle:
      1. ``build_cluster()``   — create nodes from config
      2. ``load_workload()``   — inject pod arrival events
      3. ``run()``             — process events until the queue is empty
      4. access ``collector.get_metrics()`` for results

    The engine does NOT own the scheduling strategy — it receives one
    via dependency injection so the same engine can be reused with
    different strategies.
    """

    def __init__(
        self,
        strategy: ISchedulingStrategy,
        cluster_config: ClusterConfig,
        schedule_interval: float = 1.0,
        max_pending_retries: int = 3,
        dynamics_config: Optional[DynamicsConfig] = None,
        failure_seed: int = 0,
    ) -> None:
        """
        Args:
            strategy:           Scheduling strategy to use.
            cluster_config:     Node templates for cluster creation.
            schedule_interval:  Time between SCHEDULE_CYCLE events.
            max_pending_retries: How many consecutive cycles a pod can stay
                                 pending before being rejected.
            dynamics_config:    Configuration for node failures (None = disabled).
            failure_seed:       RNG seed for failure event generation.
        """
        self._strategy = strategy
        self._cluster_config = cluster_config
        self._schedule_interval = schedule_interval
        self._max_pending_retries = max_pending_retries

        self._cluster = ClusterState()
        self._queue = EventQueue()
        self._collector = MetricsCollector()
        self._resource_monitor = ResourceMonitor()
        self._current_time: float = 0.0

        # Track how many cycles each pending pod has waited
        self._pending_cycles: dict[str, int] = {}

        # Dynamics (node failures)
        self._dynamics = dynamics_config
        self._failure_rng = random.Random(failure_seed)
        self._total_pods_loaded: int = 0

    # ── Public API ───────────────────────────────────────────────────

    @property
    def cluster(self) -> ClusterState:
        return self._cluster

    @property
    def collector(self) -> MetricsCollector:
        return self._collector

    @property
    def resource_monitor(self) -> ResourceMonitor:
        return self._resource_monitor

    @property
    def current_time(self) -> float:
        return self._current_time

    def build_cluster(self) -> None:
        """Create nodes from the cluster configuration."""
        node_idx = 0
        for template in self._cluster_config.node_templates:
            for _ in range(template.count):
                node = Node(
                    node_id=f"node-{node_idx:03d}",
                    cpu_capacity=template.cpu_capacity,
                    mem_capacity=template.mem_capacity,
                    gpu_capacity=template.gpu_capacity,
                    cost_per_hour=template.cost_per_hour,
                    taints=frozenset(template.taints),
                    labels=dict(template.labels),
                )
                self._cluster.add_node(node)
                node_idx += 1
        logger.info("Cluster built: %d nodes", self._cluster.node_count)

    def load_workload(self, pods: List[Pod]) -> None:
        """Inject pod arrival events into the event queue."""
        for pod in pods:
            self._queue.push(
                Event(
                    timestamp=pod.arrival_time,
                    priority=PRIORITY_POD_ARRIVAL,
                    event_type=EventType.POD_ARRIVAL,
                    payload=pod,
                )
            )
        # Seed the first schedule cycle
        if pods:
            first_arrival = pods[0].arrival_time
            self._queue.push(
                Event(
                    timestamp=first_arrival,
                    priority=PRIORITY_SCHEDULE_CYCLE,
                    event_type=EventType.SCHEDULE_CYCLE,
                )
            )
        self._total_pods_loaded = len(pods)

        # Schedule node failures if dynamics are enabled
        if self._dynamics and self._dynamics.enabled and pods:
            self._schedule_all_failures(pods)

        logger.info("Loaded %d pods into event queue", len(pods))

    def run(self) -> None:
        """Execute the simulation until all events are processed."""
        self._strategy.on_episode_start(self._cluster)

        while not self._queue.is_empty:
            event = self._queue.pop()
            self._current_time = event.timestamp

            # Keep collector clock in sync
            self._collector.set_time(self._current_time)

            # Update strategy clock if it supports it
            if hasattr(self._strategy, "set_current_time"):
                self._strategy.set_current_time(self._current_time)

            if event.event_type == EventType.POD_ARRIVAL:
                self._handle_pod_arrival(event.payload)

            elif event.event_type == EventType.POD_COMPLETION:
                self._handle_pod_completion(event.payload)

            elif event.event_type == EventType.SCHEDULE_CYCLE:
                self._handle_schedule_cycle()

            elif event.event_type == EventType.NODE_FAILURE:
                self._handle_node_failure()

            elif event.event_type == EventType.NODE_RECOVERY:
                self._handle_node_recovery(event.payload)

        self._strategy.on_episode_end(self._cluster)

        # Record simulation duration for throughput calculation
        self._collector._metrics.simulation_duration = self._current_time

        logger.info(
            "Simulation finished at t=%.2f  scheduled=%d rejected=%d completed=%d",
            self._current_time,
            self._cluster.scheduled_pod_count,
            self._cluster.rejected_pod_count,
            self._cluster.completed_pod_count,
        )

    # ── Event handlers ───────────────────────────────────────────────

    def _handle_pod_arrival(self, pod: Pod) -> None:
        """A new pod enters the pending queue."""
        self._cluster.enqueue_pod(pod)
        self._collector.record_pod_arrival(pod)
        self._pending_cycles[pod.pod_id] = 0
        logger.debug("t=%.2f  POD_ARRIVAL %s", self._current_time, pod.pod_id)

    def _handle_pod_completion(self, pod: Pod) -> None:
        """A running pod finishes — release resources."""
        # Guard: ignore stale completions from evicted/rescheduled pods
        if pod.status != PodStatus.SCHEDULED:
            logger.debug(
                "t=%.2f  Ignoring stale completion for %s (status=%s)",
                self._current_time, pod.pod_id, pod.status.name,
            )
            return
        # Guard: check expected completion time (catches rescheduled pods)
        if pod.scheduled_time is not None and pod.remaining_duration > 0:
            expected = pod.scheduled_time + pod.remaining_duration
            if abs(self._current_time - expected) > 1e-6:
                logger.debug(
                    "t=%.2f  Ignoring stale completion for %s (expected at %.2f)",
                    self._current_time, pod.pod_id, expected,
                )
                return
        self._cluster.release_pod(pod, self._current_time)
        self._collector.record_pod_completion(pod)
        logger.debug("t=%.2f  POD_COMPLETION %s", self._current_time, pod.pod_id)

    def _handle_schedule_cycle(self) -> None:
        """Attempt to schedule all pending pods."""
        logger.debug(
            "t=%.2f  SCHEDULE_CYCLE  pending=%d",
            self._current_time,
            self._cluster.pending_count,
        )

        # Snapshot utilisation before scheduling
        self._collector.sample_utilization(self._cluster)
        self._resource_monitor.record_snapshot(self._current_time, self._cluster)

        # OOM kill check: if total limits on a node exceed capacity,
        # evict lowest-priority BestEffort/Burstable pods until safe.
        self._check_overcommit_oom()

        # Sort pending pods: higher priority first, then earlier arrival
        pending = sorted(
            self._cluster.pending_pods,
            key=lambda p: (-p.priority, p.arrival_time),
        )

        still_pending: List[Pod] = []

        for pod in pending:
            self._collector.record_scheduling_attempt(pod.pod_id)
            node_id = self._strategy.select_node(pod, self._cluster)

            if node_id is not None:
                result = self._cluster.bind_pod(pod, node_id, self._current_time)
                self._collector.record_scheduling_result(result)

                if result.success and pod.duration > 0:
                    rd = pod.remaining_duration
                    if rd > 0:
                        # Schedule the pod's completion event
                        self._queue.push(
                            Event(
                                timestamp=self._current_time + rd,
                                priority=PRIORITY_POD_COMPLETION,
                                event_type=EventType.POD_COMPLETION,
                                payload=pod,
                            )
                        )
                    else:
                        # Pod already served its full duration through
                        # partial execution stints — complete immediately.
                        self._cluster.release_pod(pod, self._current_time)
                        self._collector.record_pod_completion(pod)
                self._pending_cycles.pop(pod.pod_id, None)

            else:
                # No feasible node — try priority preemption
                preempted_node = self._try_preemption(pod)
                if preempted_node is not None:
                    result = self._cluster.bind_pod(pod, preempted_node, self._current_time)
                    self._collector.record_scheduling_result(result)
                    if result.success and pod.duration > 0:
                        rd = pod.remaining_duration
                        if rd > 0:
                            self._queue.push(
                                Event(
                                    timestamp=self._current_time + rd,
                                    priority=PRIORITY_POD_COMPLETION,
                                    event_type=EventType.POD_COMPLETION,
                                    payload=pod,
                                )
                            )
                        else:
                            self._cluster.release_pod(pod, self._current_time)
                            self._collector.record_pod_completion(pod)
                    self._pending_cycles.pop(pod.pod_id, None)
                else:
                    # No preemption possible — check retry limit
                    self._pending_cycles[pod.pod_id] = (
                        self._pending_cycles.get(pod.pod_id, 0) + 1
                    )
                    if self._pending_cycles[pod.pod_id] >= self._max_pending_retries:
                        result = self._cluster.reject_pod(
                            pod,
                            self._current_time,
                            reason="No feasible node after max retries",
                        )
                        self._collector.record_scheduling_result(result)
                        self._pending_cycles.pop(pod.pod_id, None)
                    else:
                        still_pending.append(pod)

        # OOM check after new scheduling decisions (catches fresh overcommit)
        self._check_overcommit_oom()

        # Schedule next cycle if there's still work to do
        not_all_arrived = len(self._cluster.all_pods) < self._total_pods_loaded
        has_pending = len(still_pending) > 0 or self._cluster.pending_count > 0

        if not_all_arrived or has_pending:
            self._queue.push(
                Event(
                    timestamp=self._current_time + self._schedule_interval,
                    priority=PRIORITY_SCHEDULE_CYCLE,
                    event_type=EventType.SCHEDULE_CYCLE,
                )
            )

    # ── Priority preemption ─────────────────────────────────────────

    def _try_preemption(self, pod: Pod) -> Optional[str]:
        """Attempt to free resources for *pod* by evicting lower-priority pods.

        Mirrors Kubernetes priority-based preemption: a high-priority pod
        can evict one or more lower-priority pods from a node to obtain
        the resources it needs.

        Returns the node_id where preemption succeeded, or None.
        """
        best_node_id: Optional[str] = None
        best_evict_count = float("inf")

        for node in self._cluster.nodes.values():
            if not node.is_available:
                continue
            # Must pass taint/label checks (resource-independent constraints)
            if node.taints and not node.taints.issubset(pod.tolerations):
                continue
            if pod.node_selector and not node.matches_selector(pod):
                continue

            # Find lower-priority pods that could be evicted
            evictable = sorted(
                [p for p in node.pods.values() if p.priority < pod.priority],
                key=lambda p: (p.qos_class.value, p.priority),
            )
            if not evictable:
                continue

            # Greedily compute how many evictions are needed
            freed_cpu = node.cpu_available
            freed_mem = node.mem_available
            to_evict: list[Pod] = []
            for victim in evictable:
                if freed_cpu >= pod.cpu_request and freed_mem >= pod.mem_request:
                    break
                freed_cpu += victim.cpu_request
                freed_mem += victim.mem_request
                to_evict.append(victim)

            if freed_cpu >= pod.cpu_request and freed_mem >= pod.mem_request:
                if len(to_evict) < best_evict_count:
                    best_evict_count = len(to_evict)
                    best_node_id = node.node_id

        if best_node_id is None:
            return None

        # Execute the preemption on the chosen node
        node = self._cluster.nodes[best_node_id]
        evictable = sorted(
            [p for p in node.pods.values() if p.priority < pod.priority],
            key=lambda p: (p.qos_class.value, p.priority),
        )
        for victim in evictable:
            if node.cpu_available >= pod.cpu_request and node.mem_available >= pod.mem_request:
                break
            node.release(victim)
            victim.evict(self._current_time)
            self._collector.record_pod_eviction(victim)
            self._collector.record_preemption(victim, pod)
            # Re-queue the evicted pod
            self._cluster.pending_pods.append(victim)
            self._pending_cycles[victim.pod_id] = 0
            logger.debug(
                "t=%.2f  PREEMPT %s (pri=%d) evicted by %s (pri=%d) from %s",
                self._current_time, victim.pod_id, victim.priority,
                pod.pod_id, pod.priority, best_node_id,
            )

        return best_node_id

    # ── Overcommit OOM kill ──────────────────────────────────────────

    def _check_overcommit_oom(self) -> None:
        """Kill pods on overcommitted nodes (total limits > capacity).

        Mimics Kubernetes OOM-killer: when the sum of effective limits on
        a node exceeds its physical capacity, the scheduler evicts pods
        starting with lowest QoS / lowest priority until within capacity.
        Only BEST_EFFORT and BURSTABLE pods are eligible for OOM kill;
        GUARANTEED pods are protected.
        """
        from models.pod import QoSClass

        for node in self._cluster.nodes.values():
            if not node.is_available:
                continue
            if (node.cpu_limit_total <= node.cpu_capacity
                    and node.mem_limit_total <= node.mem_capacity):
                continue

            # Node is overcommitted — find OOM-eligible pods
            eligible = sorted(
                [p for p in node.pods.values()
                 if p.qos_class != QoSClass.GUARANTEED],
                key=lambda p: (p.qos_class.value, p.priority),
            )

            for victim in eligible:
                if (node.cpu_limit_total <= node.cpu_capacity
                        and node.mem_limit_total <= node.mem_capacity):
                    break
                node.release(victim)
                victim.evict(self._current_time)
                victim.reject(self._current_time)  # OOM = permanent death
                self._collector.record_pod_eviction(victim)
                self._collector.record_pod_rejection(victim)
                self._pending_cycles.pop(victim.pod_id, None)
                logger.debug(
                    "t=%.2f  OOM_KILL %s (qos=%s, pri=%d) on %s",
                    self._current_time, victim.pod_id,
                    victim.qos_class.name, victim.priority, node.node_id,
                )

    # ── Node failure / recovery ──────────────────────────────────────

    def _has_active_workload(self) -> bool:
        """Check whether there are pods still being processed."""
        if self._total_pods_loaded == 0:
            return False
        terminal = sum(
            1 for p in self._cluster.all_pods.values()
            if p.status in (PodStatus.COMPLETED, PodStatus.REJECTED)
        )
        return terminal < self._total_pods_loaded

    def _schedule_all_failures(self, pods: List[Pod]) -> None:
        """Pre-schedule a fixed number of node failure events.

        The number of failures = max(1, round(num_nodes * rate * 0.1)).
        Failure times are spread uniformly across [10 %, 90 %] of the
        estimated simulation duration.
        """
        if not self._dynamics or not self._dynamics.enabled:
            return

        num_nodes = self._cluster.node_count
        if num_nodes == 0:
            return

        n_failures = max(1, round(num_nodes * self._dynamics.failure_rate * 0.1))

        # Estimate simulation duration from workload
        if pods:
            est_duration = max(p.arrival_time + p.duration for p in pods)
        else:
            est_duration = 100.0

        lo = est_duration * 0.10
        hi = est_duration * 0.90
        if hi <= lo:
            hi = lo + 1.0

        for _ in range(n_failures):
            t = self._failure_rng.uniform(lo, hi)
            self._queue.push(
                Event(
                    timestamp=t,
                    priority=PRIORITY_NODE_FAILURE,
                    event_type=EventType.NODE_FAILURE,
                )
            )

    def _handle_node_failure(self) -> None:
        """A random available node fails — behaviour depends on ``failure_mode``.

        ``"reschedule"``: evicted pods are re-queued with restart overhead.
        ``"kill"``:       evicted pods are permanently rejected.
        """
        available = [n for n in self._cluster.nodes.values() if n.is_available]
        if not available:
            return

        node = self._failure_rng.choice(available)
        node.mark_failed()
        self._collector.record_node_failure()

        # Evict all pods from the failed node (sorted by QoS: BE first)
        evicted = self._cluster.evict_pods_from_node(node.node_id)

        mode = self._dynamics.failure_mode if self._dynamics else "reschedule"

        for pod in evicted:
            pod.evict(self._current_time)
            self._collector.record_pod_eviction(pod)

            if mode == "kill":
                # Pod is permanently lost — reject it
                pod.reject(self._current_time)
                self._collector.record_pod_rejection(pod)
            elif mode == "reschedule":
                # Add restart overhead before re-queuing
                overhead = self._failure_rng.uniform(
                    self._dynamics.restart_overhead_min,
                    self._dynamics.restart_overhead_max,
                )
                pod.add_restart_overhead(overhead)
                self._cluster.pending_pods.append(pod)
                self._pending_cycles[pod.pod_id] = 0

        logger.info(
            "t=%.2f  NODE_FAILURE %s (%d pods evicted, mode=%s)",
            self._current_time, node.node_id, len(evicted), mode,
        )

        # Schedule recovery
        recovery_time = self._failure_rng.uniform(
            self._dynamics.recovery_time_min,
            self._dynamics.recovery_time_max,
        )
        self._queue.push(
            Event(
                timestamp=self._current_time + recovery_time,
                priority=PRIORITY_NODE_RECOVERY,
                event_type=EventType.NODE_RECOVERY,
                payload=node.node_id,
            )
        )

        # Trigger immediate schedule cycle for rescheduled pods
        if evicted and mode == "reschedule":
            self._queue.push(
                Event(
                    timestamp=self._current_time,
                    priority=PRIORITY_SCHEDULE_CYCLE,
                    event_type=EventType.SCHEDULE_CYCLE,
                )
            )

    def _handle_node_recovery(self, node_id: str) -> None:
        """A failed node comes back online."""
        node = self._cluster.nodes[node_id]
        node.mark_recovered()
        logger.info("t=%.2f  NODE_RECOVERY %s", self._current_time, node_id)

        # Trigger schedule cycle to place pending pods on recovered node
        self._queue.push(
            Event(
                timestamp=self._current_time,
                priority=PRIORITY_SCHEDULE_CYCLE,
                event_type=EventType.SCHEDULE_CYCLE,
            )
        )
