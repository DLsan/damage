"""
DAMAGE – CV Pipeline for Damage Detection
==========================================
Usage:
    python main.py                         # webcam (default)
    python main.py --source video.mp4      # video file
    python main.py --source image.jpg      # single image
    python main.py --source 0 --save       # webcam + save output
"""

import argparse
import sys
import cv2
import os

from cv_pipeline.pipeline.pipeline import Pipeline
from core import config


def parse_args():
    parser = argparse.ArgumentParser(description="DAMAGE CV Detection Pipeline")
    parser.add_argument(
        "--source", default=str(config.VIDEO_SOURCE),
        help="Video source: webcam index (0,1,…), video file path, or image path"
    )
    parser.add_argument(
        "--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--device", default=config.DEVICE,
        help="Inference device: 'cpu' or 'cuda' (default: cpu)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save output video to disk"
    )
    parser.add_argument(
        "--output", default=config.OUTPUT_PATH,
        help=f"Output file path (default: {config.OUTPUT_PATH})"
    )
    return parser.parse_args()


def run_on_image(pipeline, image_path: str):
    """Run detection on a single image and show the result."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    results    = pipeline.model.predict(frame)
    detections = pipeline.detector.parse(results)
    tracks     = pipeline.tracker.update(detections)
    counts     = pipeline.counter.update(tracks)
    pipeline.events.process(tracks, counts)
    out = pipeline.processor.draw(frame, detections, tracks, counts, fps=0.0)

    cv2.imshow("DAMAGE Detection – Image", out)
    print("Press any key to close …")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if pipeline._writer is None and args.save:
        out_path = os.path.splitext(args.output)[0] + "_result.jpg"
        cv2.imwrite(out_path, out)
        print(f"[Saved] {out_path}")


def main():
    global args
    args = parse_args()

    # Apply CLI overrides to config
    config.CONFIDENCE_THRESHOLD = args.conf
    config.DEVICE               = args.device
    config.SAVE_OUTPUT          = args.save
    config.OUTPUT_PATH          = args.output

    # Convert numeric string source to int (webcam index)
    source = args.source
    if source.isdigit():
        source = int(source)

    pipeline = Pipeline()

    # Route: image vs video/webcam
    if isinstance(source, str) and source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    ):
        run_on_image(pipeline, source)
    else:
        pipeline.run(source=source)


if __name__ == "__main__":
    main()
