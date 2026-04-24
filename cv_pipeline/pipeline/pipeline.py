"""
Pipeline – orchestrates detector → tracker → counter → events → processor.
"""

import time
import cv2
import numpy as np

from core.config import VIDEO_SOURCE, DISPLAY_WIDTH, DISPLAY_HEIGHT, SAVE_OUTPUT, OUTPUT_PATH
from models.loader import ModelLoader
from cv_pipeline.detectors.damage_detector import DamageDetector
from cv_pipeline.trackers.tracker import CentroidTracker
from cv_pipeline.counter.damage_counter import DamageCounter
from cv_pipeline.events.event_handler import EventHandler
from cv_pipeline.processors.frame_processor import FrameProcessor


class Pipeline:
    """End-to-end damage detection pipeline."""

    def __init__(self):
        print("[Pipeline] Initialising components …")
        self.model      = ModelLoader()
        self.detector   = DamageDetector()
        self.tracker    = CentroidTracker()
        self.counter    = DamageCounter()
        self.events     = EventHandler()
        self.processor  = FrameProcessor()

        # Register default event callbacks
        self.events.subscribe("new_damage",   self._on_new_damage)
        self.events.subscribe("damage_lost",  self._on_damage_lost)
        self.events.subscribe("count_update", self._on_count_update)

        self._writer   = None
        self._counts   = {}
        print("[Pipeline] Ready.")

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self, source=VIDEO_SOURCE):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"[Pipeline] Cannot open video source: {source}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

        if SAVE_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                OUTPUT_PATH, fourcc, 30,
                (DISPLAY_WIDTH, DISPLAY_HEIGHT)
            )

        print("[Pipeline] Running — press 'q' to quit, 'r' to reset counts.")
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Pipeline] Stream ended or frame not available.")
                break

            # ── FPS ──────────────────────────────────────────────────────────
            now      = time.time()
            fps      = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # ── Inference ────────────────────────────────────────────────────
            results    = self.model.predict(frame)
            detections = self.detector.parse(results)

            # ── Tracking ─────────────────────────────────────────────────────
            tracks = self.tracker.update(detections)

            # ── Counting & Events ─────────────────────────────────────────────
            self._counts = self.counter.update(tracks)
            self.events.process(tracks, self._counts)

            # ── Drawing ───────────────────────────────────────────────────────
            frame = self.processor.draw(frame, detections, tracks, self._counts, fps)

            if self._writer:
                self._writer.write(frame)

            cv2.imshow("DAMAGE Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.counter.reset()
                print("[Pipeline] Counts reset.")

        cap.release()
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()
        print("[Pipeline] Stopped.")

    # ── event callbacks ───────────────────────────────────────────────────────

    @staticmethod
    def _on_new_damage(track):
        print(f"[EVENT] New damage detected → ID #{track.track_id}  class='{track.class_name}'")

    @staticmethod
    def _on_damage_lost(track_id):
        print(f"[EVENT] Damage track lost  → ID #{track_id}")

    @staticmethod
    def _on_count_update(counts):
        pass   # counts are displayed on the HUD; add alerting logic here
