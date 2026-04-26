"""
DAMAGE – JSON Detection Pipeline
=================================
Usage:
    python detect.py --source image.jpg
    python detect.py --source video.mp4
    python detect.py --source 0                      # webcam (Ctrl+C to stop)
    python detect.py --source image.jpg --out result.json
"""

import argparse
import json
import sys
import time

import cv2

from core import config
from models.loader import ModelLoader
from cv_pipeline.detectors.damage_detector import DamageDetector
from cv_pipeline.trackers.tracker import CentroidTracker
from cv_pipeline.counter.damage_counter import DamageCounter


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_frame(detections, tracks, counts, frame_idx=None, fps=None):
    out = {}
    if frame_idx is not None:
        out["frame"] = frame_idx
    if fps is not None:
        out["fps"] = round(fps, 1)

    out["detections"] = [
        {
            "track_id": next(
                (tid for tid, t in tracks.items() if t.centroid == d.centroid), None
            ),
            "class_id":   d.class_id,
            "class_name": d.class_name,
            "confidence": round(d.confidence, 3),
            "bbox":       [round(v, 1) for v in d.bbox],
            "centroid":   list(d.centroid),
        }
        for d in detections
    ]

    out["counts"] = {
        config.CLASS_NAMES.get(cid, f"class_{cid}"): cnt
        for cid, cnt in counts.items()
    }
    out["total"] = len(detections)
    return out


def _save(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Saved] {path}", file=sys.stderr)


# ── image ─────────────────────────────────────────────────────────────────────

def run_image(source, out_path):
    model    = ModelLoader()
    detector = DamageDetector()
    tracker  = CentroidTracker()
    counter  = DamageCounter()

    frame = cv2.imread(source)
    if frame is None:
        print(f"[ERROR] Cannot read image: {source}", file=sys.stderr)
        sys.exit(1)

    detections = detector.parse(model.predict(frame))
    tracks     = tracker.update(detections)
    counts     = counter.update(tracks)

    result = {"source": source, **_build_frame(detections, tracks, counts)}
    print(json.dumps(result, indent=2))
    if out_path:
        _save(result, out_path)


# ── video / webcam ────────────────────────────────────────────────────────────

def run_video(source, out_path):
    model    = ModelLoader()
    detector = DamageDetector()
    tracker  = CentroidTracker()
    counter  = DamageCounter()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}", file=sys.stderr)
        sys.exit(1)

    frames   = []
    idx      = 0
    prev     = time.time()

    print("[INFO] Processing … press Ctrl+C to stop early.", file=sys.stderr)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now  = time.time()
            fps  = 1.0 / max(now - prev, 1e-6)
            prev = now

            detections = detector.parse(model.predict(frame))
            tracks     = tracker.update(detections)
            counts     = counter.update(tracks)

            frames.append(_build_frame(detections, tracks, counts, idx, fps))
            idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    result = {
        "source":       str(source),
        "total_frames": idx,
        "final_counts": frames[-1]["counts"] if frames else {},
        "frames":       frames,
    }
    print(json.dumps(result, indent=2))
    if out_path:
        _save(result, out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DAMAGE JSON detection pipeline")
    p.add_argument("--source", default="0",
                   help="Image path, video path, or webcam index (default: 0)")
    p.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
                   help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})")
    p.add_argument("--device", default=config.DEVICE,
                   help="'cpu' or 'cuda'")
    p.add_argument("--out", default=None,
                   help="Optional path to save JSON output (e.g. result.json)")
    return p.parse_args()


def main():
    args = parse_args()
    config.CONFIDENCE_THRESHOLD = args.conf
    config.DEVICE               = args.device

    source = args.source
    if source.isdigit():
        source = int(source)

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    if isinstance(source, str) and source.lower().endswith(image_exts):
        run_image(source, args.out)
    else:
        run_video(source, args.out)


if __name__ == "__main__":
    main()
