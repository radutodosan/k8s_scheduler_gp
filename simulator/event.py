"""Discrete-event types and Event dataclass for the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    """Types of events processed by the simulation engine.

    POD_ARRIVAL:     A new pod enters the pending queue.
    POD_COMPLETION:  A running pod finishes execution.
    SCHEDULE_CYCLE:  Trigger for the scheduler to process the pending queue.
    NODE_FAILURE:    A node becomes unavailable (reserved — not implemented in Phase 1).
    NODE_RECOVERY:   A failed node comes back online (reserved — Phase 2).
    """

    POD_ARRIVAL = auto()
    POD_COMPLETION = auto()
    SCHEDULE_CYCLE = auto()
    NODE_FAILURE = auto()      # A node becomes unavailable
    NODE_RECOVERY = auto()     # A failed node comes back online


@dataclass(order=True)
class Event:
    """A single event in the simulation timeline.

    Events are ordered by (timestamp, priority) so that `heapq` can
    maintain a proper min-heap.  Only *timestamp* and *priority* participate
    in comparison; event_type and payload are excluded.

    Attributes:
        timestamp:  Simulation time at which the event fires.
        priority:   Tie-breaker when timestamps are equal (lower = first).
                    Convention: SCHEDULE_CYCLE=0, POD_ARRIVAL=1, POD_COMPLETION=2.
        event_type: The kind of event (excluded from ordering).
        payload:    Arbitrary data attached to the event (e.g. a Pod object).
    """

    timestamp: float
    priority: int
    event_type: EventType = field(compare=False, default=EventType.POD_ARRIVAL)
    payload: Any = field(compare=False, default=None)


# Convenience constants for tie-breaking priorities.
PRIORITY_SCHEDULE_CYCLE = 0
PRIORITY_POD_ARRIVAL = 1
PRIORITY_POD_COMPLETION = 2
PRIORITY_NODE_FAILURE = 3
PRIORITY_NODE_RECOVERY = 4
