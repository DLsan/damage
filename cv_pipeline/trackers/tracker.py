"""
Simple centroid-based multi-object tracker.
Each detected centroid is matched to the nearest existing track.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from core.config import MAX_DISAPPEARED, MAX_DISTANCE


@dataclass
class Track:
    track_id: int
    centroid: Tuple[int, int]
    class_id: int
    class_name: str
    disappeared: int = 0
    history: List[Tuple[int, int]] = field(default_factory=list)

    def update(self, centroid, class_id, class_name):
        self.centroid   = centroid
        self.class_id   = class_id
        self.class_name = class_name
        self.disappeared = 0
        self.history.append(centroid)


class CentroidTracker:
    """Lightweight centroid tracker – no external dependencies."""

    def __init__(self):
        self._next_id: int = 0
        self.tracks: Dict[int, Track] = OrderedDict()

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, detections) -> Dict[int, Track]:
        """
        Update tracker with a list of Detection objects.
        Returns the current dict of active Track objects.
        """
        if not detections:
            self._age_all()
            return self.tracks

        input_centroids = [d.centroid for d in detections]

        if not self.tracks:
            for d in detections:
                self._register(d)
            return self.tracks

        # Match detections → existing tracks by centroid distance
        track_ids    = list(self.tracks.keys())
        track_cents  = np.array([self.tracks[t].centroid for t in track_ids], dtype=float)
        input_cents  = np.array(input_centroids, dtype=float)

        # pairwise distance matrix
        D = np.linalg.norm(track_cents[:, None] - input_cents[None, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > MAX_DISTANCE:
                continue
            tid = track_ids[row]
            self.tracks[tid].update(
                detections[col].centroid,
                detections[col].class_id,
                detections[col].class_name,
            )
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched tracks → increment disappeared
        for row in range(len(track_ids)):
            if row not in used_rows:
                self.tracks[track_ids[row]].disappeared += 1

        # Unmatched detections → new tracks
        for col in range(len(detections)):
            if col not in used_cols:
                self._register(detections[col])

        # Drop lost tracks
        lost = [tid for tid, t in self.tracks.items() if t.disappeared > MAX_DISAPPEARED]
        for tid in lost:
            del self.tracks[tid]

        return self.tracks

    # ── private ───────────────────────────────────────────────────────────────

    def _register(self, detection):
        track = Track(
            track_id=self._next_id,
            centroid=detection.centroid,
            class_id=detection.class_id,
            class_name=detection.class_name,
        )
        track.history.append(detection.centroid)
        self.tracks[self._next_id] = track
        self._next_id += 1

    def _age_all(self):
        lost = []
        for tid, t in self.tracks.items():
            t.disappeared += 1
            if t.disappeared > MAX_DISAPPEARED:
                lost.append(tid)
        for tid in lost:
            del self.tracks[tid]
