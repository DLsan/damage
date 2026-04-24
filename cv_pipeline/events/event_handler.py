"""
Event Handler – fires callbacks on damage-related events.
"""

from typing import Callable, Dict, List


class EventHandler:
    """
    Lightweight pub/sub event system for the CV pipeline.

    Supported events
    ----------------
    'new_damage'    – a new track ID is first detected
    'damage_lost'   – a tracked damage disappears
    'count_update'  – the per-class count changes
    """

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {
            "new_damage"    : [],
            "damage_lost"   : [],
            "count_update"  : [],
        }
        self._prev_track_ids: set = set()

    def subscribe(self, event: str, callback: Callable):
        """Register a callback for an event."""
        if event not in self._listeners:
            raise ValueError(f"Unknown event '{event}'")
        self._listeners[event].append(callback)

    def process(self, tracks: dict, counts: dict):
        """
        Compare current tracks against previous frame.
        Fire events as needed.
        """
        current_ids = set(tracks.keys())

        # New damage appeared
        for tid in current_ids - self._prev_track_ids:
            self._emit("new_damage", tracks[tid])

        # Damage disappeared
        for tid in self._prev_track_ids - current_ids:
            self._emit("damage_lost", tid)

        # Always emit count update
        self._emit("count_update", counts)

        self._prev_track_ids = current_ids

    # ── private ───────────────────────────────────────────────────────────────

    def _emit(self, event: str, payload):
        for cb in self._listeners[event]:
            try:
                cb(payload)
            except Exception as e:
                print(f"[EventHandler] Error in '{event}' callback: {e}")
