"""
Core configuration for DAMAGE CV Pipeline
"""

import os

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "best.pt"
)

# ── Detection ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD        = 0.45
IMGSZ                = 640
DEVICE               = "cpu"   # change to "cuda" if GPU is available

# ── Class labels (must match the trained model in models/best.pt) ────────────
CLASS_NAMES = {
    0: "undamaged",
    1: "damaged",
}

# ── Colours per class (BGR) ───────────────────────────────────────────────────
CLASS_COLORS = {
    0: (0, 200, 0),     # green – undamaged
    1: (0, 0, 255),     # red   – damaged
}

# ── Tracking ──────────────────────────────────────────────────────────────────
MAX_DISAPPEARED = 30      # frames before a track is dropped
MAX_DISTANCE    = 120     # max centroid distance (px) to match track

# ── Video / Camera ────────────────────────────────────────────────────────────
VIDEO_SOURCE  = 0         # 0 = webcam, or path to video file
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# ── Output ────────────────────────────────────────────────────────────────────
SAVE_OUTPUT   = False
OUTPUT_PATH   = "output_damage.mp4"
SHOW_FPS      = True
