# 🔍 DAMAGE – CV Pipeline for Damage Detection

A modular, real-time computer vision pipeline for detecting and tracking physical damage (dents, scratches, cracks, deformations) using a fine-tuned **YOLOv8** model.

---

## 📸 Demo

| Detection on Image | Live HUD |
|---|---|
| Bounding boxes per damage type | Per-class count panel + FPS |

---

## 🗂️ Project Structure

```
DAMAGE/
│
├── core/
│   ├── __init__.py
│   └── config.py              # All tunable settings (model path, thresholds, etc.)
│
├── cv_pipeline/
│   ├── __init__.py
│   ├── counter/
│   │   └── damage_counter.py  # Counts unique damage instances per class
│   ├── detectors/
│   │   └── damage_detector.py # Parses YOLO results → structured Detection objects
│   ├── events/
│   │   └── event_handler.py   # Pub/sub events (new_damage, damage_lost, count_update)
│   ├── pipeline/
│   │   └── pipeline.py        # Orchestrates the full pipeline
│   ├── processors/
│   │   └── frame_processor.py # Draws boxes, track trails, and HUD onto frames
│   └── trackers/
│       └── tracker.py         # Centroid-based multi-object tracker
│
├── models/
│   ├── __init__.py
│   └── loader.py              # Loads YOLOv8 weights via Ultralytics
│
├── runs/
│   └── detect/train2/weights/
│       └── best.pt            # Trained YOLOv8 model weights
│
├── main.py                    # CLI entry point
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/DAMAGE.git
cd DAMAGE
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**
```powershell
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run on an image
```bash
python main.py --source "image.jpg"
```
Press **any key** to close the result window.

### Run on a video file
```bash
python main.py --source "video.mp4"
```
Press **`q`** to quit · **`r`** to reset damage counts.

### Run on webcam
```bash
python main.py --source 0
```
`0` = default webcam, `1` = second camera, etc.

### Save output video
```bash
python main.py --source "video.mp4" --save
```

### Adjust confidence threshold
```bash
python main.py --source "image.jpg" --conf 0.25
```

### Run on GPU (if available)
```bash
python main.py --source "video.mp4" --device cuda
```

---

## 🧩 Pipeline Architecture

```
Frame
  │
  ▼
ModelLoader ──► DamageDetector ──► CentroidTracker
                                        │
                              ┌─────────┴──────────┐
                              ▼                    ▼
                        DamageCounter         EventHandler
                              │                    │
                              └─────────┬──────────┘
                                        ▼
                                  FrameProcessor
                                        │
                                        ▼
                               Annotated Output
```

| Component | Role |
|---|---|
| `ModelLoader` | Loads `best.pt` and runs YOLOv8 inference |
| `DamageDetector` | Converts raw YOLO output → `Detection` dataclass list |
| `CentroidTracker` | Assigns persistent track IDs across frames |
| `DamageCounter` | Counts unique damage instances per class |
| `EventHandler` | Fires events on new/lost detections |
| `FrameProcessor` | Draws boxes, trails, and HUD panel |

---

## 🏷️ Damage Classes

| ID | Class | Colour |
|---|---|---|
| 0 | dent | 🟠 Orange |
| 1 | scratch | 🟢 Green |
| 2 | crack | 🔴 Red |
| 3 | deformation | 🟣 Magenta |

> Update `CLASS_NAMES` and `CLASS_COLORS` in `core/config.py` to match your trained model's labels.

---

## 🔧 Configuration

All settings live in **`core/config.py`**:

| Setting | Default | Description |
|---|---|---|
| `MODEL_PATH` | `runs/detect/train2/weights/best.pt` | Path to trained YOLOv8 weights |
| `CONFIDENCE_THRESHOLD` | `0.4` | Min detection confidence |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `VIDEO_SOURCE` | `0` | Default camera/video source |
| `MAX_DISAPPEARED` | `30` | Frames before track is dropped |
| `MAX_DISTANCE` | `120` | Max centroid distance (px) to match |
| `SAVE_OUTPUT` | `False` | Save output video to disk |

---

## 📦 Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.5.0
PyYAML>=6.0
scipy>=1.10.0
```

---

## 📄 License

MIT License © 2025

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
