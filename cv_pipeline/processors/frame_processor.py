"""
Frame Processor – draws detections, tracks, and HUD onto frames.
"""

import cv2
import numpy as np
from typing import Dict

from core.config import CLASS_NAMES, CLASS_COLORS, SHOW_FPS


class FrameProcessor:
    """Handles all OpenCV drawing operations."""

    def __init__(self):
        self._fps_buffer = []

    # ── public API ────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, detections, tracks: dict,
             counts: dict, fps: float = 0.0) -> np.ndarray:
        """
        Draw bounding boxes, track IDs, and HUD.
        Returns the annotated frame.
        """
        frame = self._draw_detections(frame, detections)
        frame = self._draw_tracks(frame, tracks)
        frame = self._draw_hud(frame, counts, fps)
        return frame

    # ── drawing helpers ───────────────────────────────────────────────────────

    def _draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = CLASS_COLORS.get(det.class_id, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            self._put_label(frame, label, (x1, y1 - 6), color)
        return frame

    def _draw_tracks(self, frame, tracks):
        for tid, track in tracks.items():
            cx, cy = track.centroid
            color  = CLASS_COLORS.get(track.class_id, (200, 200, 200))
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, f"#{tid}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            # Draw centroid trail
            pts = track.history[-20:]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)
        return frame

    def _draw_hud(self, frame, counts: dict, fps: float):
        h, w = frame.shape[:2]

        # Semi-transparent dark panel on the left
        overlay = frame.copy()
        panel_w = 240
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        y = 30
        cv2.putText(frame, "DAMAGE DETECTION", (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
        cv2.line(frame, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
        y += 20

        for class_id, cnt in counts.items():
            name  = CLASS_NAMES.get(class_id, f"class_{class_id}")
            color = CLASS_COLORS.get(class_id, (200, 200, 200))
            text  = f"{name.upper():<14} {cnt:>3}"
            cv2.putText(frame, text, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
            y += 24

        if SHOW_FPS:
            cv2.putText(frame, f"FPS: {fps:.1f}", (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        return frame

    @staticmethod
    def _put_label(frame, text: str, pos, color):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = pos
        cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
        cv2.putText(frame, text, (x + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
