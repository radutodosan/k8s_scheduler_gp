"""EventQueue — min-heap wrapper for simulation events."""

from __future__ import annotations

import heapq
from typing import List

from simulator.event import Event


class EventQueue:
    """Priority queue of simulation events backed by a min-heap.

    Events are ordered by (timestamp, priority).  Provides O(log n) push
    and O(log n) pop.
    """

    def __init__(self) -> None:
        self._heap: List[Event] = []

    def push(self, event: Event) -> None:
        """Add an event to the queue."""
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        """Remove and return the earliest event.

        Raises IndexError if the queue is empty.
        """
        return heapq.heappop(self._heap)

    def peek(self) -> Event:
        """Return the earliest event without removing it.

        Raises IndexError if the queue is empty.
        """
        return self._heap[0]

    @property
    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)
