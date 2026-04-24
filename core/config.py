"""
Core configuration for DAMAGE CV Pipeline
"""

import os

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "runs", "detect", "train2", "weights", "best.pt"
)

# ── Detection ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD        = 0.45
IMGSZ                = 640
DEVICE               = "cpu"   # change to "cuda" if GPU is available

# ── Class labels (update to match your trained model) ─────────────────────────
CLASS_NAMES = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "deformation",
}

# ── Colours per class (BGR) ───────────────────────────────────────────────────
CLASS_COLORS = {
    0: (0,   165, 255),   # orange  – dent
    1: (0,   255, 0  ),   # green   – scratch
    2: (0,   0,   255),   # red     – crack
    3: (255, 0,   255),   # magenta – deformation
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
