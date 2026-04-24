"""
Damage Detector – parses raw YOLO results into structured detections.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np

from core.config import CLASS_NAMES


@dataclass
class Detection:
    """Single detection result."""
    bbox: List[float]          # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    centroid: tuple = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))


class DamageDetector:
    """Converts YOLO result objects into a clean list of Detection instances."""

    def parse(self, yolo_results) -> List[Detection]:
        """
        Parameters
        ----------
        yolo_results : list[ultralytics.engine.results.Results]
            Raw output from ModelLoader.predict()

        Returns
        -------
        list[Detection]
        """
        detections: List[Detection] = []

        for result in yolo_results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf     = float(box.conf[0])
                class_id = int(box.cls[0])
                name     = CLASS_NAMES.get(class_id, f"class_{class_id}")

                detections.append(
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        class_id=class_id,
                        class_name=name,
                        confidence=conf,
                    )
                )

        return detections
