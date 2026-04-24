import shutil
from pathlib import Path

# ==============================
# ✅ CONFIG (FIXED PATHS)
# ==============================

SOURCE_DATASETS = [
    Path("Damaged package.v1i.yolov8"),
    Path("Damaged package.v2i.yolov8"),
    Path("Package.v1i.yolov8"),
    Path("Package.v2i.yolov8"),
]

OUTPUT_DIR = Path("merged_dataset")
# YOLO format uses train / valid / test
SPLITS = ["train", "valid", "test"]

# ==============================
# 📁 CREATE OUTPUT FOLDERS
# ==============================

def create_dirs():
    for split in SPLITS:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# ==============================
# 📦 COPY DATA
# ==============================

def copy_data():
    counter = 0

    for dataset in SOURCE_DATASETS:
        print(f"\n📂 Processing: {dataset}")

        if not dataset.exists():
            print(f"❌ NOT FOUND: {dataset}")
            continue

        for split in SPLITS:
            img_dir = dataset / split / "images"
            lbl_dir = dataset / split / "labels"

            if not img_dir.exists():
                print(f"  ⚠️ Missing split: {split}")
                continue

            images = list(img_dir.glob("*.*"))
            print(f"  ✅ Found {len(images)} images in {split}")

            for img_file in images:
                # Keep extension (jpg/png)
                new_img_path = OUTPUT_DIR / "images" / split / f"{counter}{img_file.suffix}"
                shutil.copy(img_file, new_img_path)

                # Copy corresponding label
                label_file = lbl_dir / (img_file.stem + ".txt")
                if label_file.exists():
                    new_lbl_path = OUTPUT_DIR / "labels" / split / f"{counter}.txt"
                    shutil.copy(label_file, new_lbl_path)

                counter += 1

    print(f"\n🔥 Total files copied: {counter}")

# ==============================
# 📄 CREATE YAML FILE
# ==============================

def create_yaml():
    yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/valid
test: images/test

names:
  0: damaged
"""
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(yaml_content)
    print("📄 data.yaml created")

# ==============================
# 🚀 MAIN
# ==============================

def main():
    print("🚀 Starting dataset merge...")
    create_dirs()
    copy_data()
    create_yaml()
    print("\n✅ MERGE COMPLETE — READY FOR TRAINING")

if __name__ == "__main__":
    main()