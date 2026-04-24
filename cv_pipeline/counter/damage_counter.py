"""
Damage Counter – tracks unique damage instances seen per class.
"""

from collections import defaultdict
from typing import Dict


class DamageCounter:
    """Counts unique damage detections per class using track IDs."""

    def __init__(self):
        # Set of track IDs already counted, per class
        self._counted: Dict[int, set] = defaultdict(set)

    def update(self, tracks: dict) -> Dict[str, int]:
        """
        Parameters
        ----------
        tracks : dict[int, Track]
            Active tracks from CentroidTracker.

        Returns
        -------
        dict[str_class_name, count]
        """
        for tid, track in tracks.items():
            self._counted[track.class_id].add(tid)

        return self.get_counts()

    def get_counts(self) -> Dict[int, int]:
        """Returns {class_id: count} of unique damage instances seen."""
        return {cid: len(ids) for cid, ids in self._counted.items()}

    def reset(self):
        self._counted.clear()
