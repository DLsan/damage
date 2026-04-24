"""
Model loader – wraps an Ultralytics YOLOv8 model.
"""

from ultralytics import YOLO
from core.config import MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, IMGSZ, DEVICE


class ModelLoader:
    """Loads and holds the YOLOv8 weights for inference."""

    def __init__(self, model_path: str = MODEL_PATH):
        print(f"[ModelLoader] Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(DEVICE)
        print(f"[ModelLoader] Model ready on '{DEVICE}'")

    def predict(self, frame):
        """
        Run inference on a single BGR frame (numpy array).
        Returns a list of Ultralytics Result objects.
        """
        results = self.model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMGSZ,
            verbose=False,
        )
        return results
