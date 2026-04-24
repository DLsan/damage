import random
import shutil
from pathlib import Path

SRC = Path("final_dataset")
DST = Path("balanced_dataset")

SPLITS = ["train", "valid", "test"]

def create_dirs():
    for split in SPLITS:
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

def get_all_files():
    damaged = []
    undamaged = []

    for split in SPLITS:
        img_dir = SRC / "images" / split
        lbl_dir = SRC / "labels" / split

        for img in img_dir.glob("*.*"):
            label_file = lbl_dir / (img.stem + ".txt")

            if not label_file.exists():
                continue

            lines = label_file.read_text().splitlines()

            is_damaged = any(line.startswith("1") for line in lines)

            if is_damaged:
                damaged.append((img, label_file))
            else:
                undamaged.append((img, label_file))

    return damaged, undamaged

def rebalance():
    damaged, undamaged = get_all_files()

    print(f"Damaged: {len(damaged)}")
    print(f"Undamaged: {len(undamaged)}")

    # 🔥 Sample undamaged to match damaged
    undamaged_sample = random.sample(undamaged, len(damaged))

    combined = damaged + undamaged_sample
    random.shuffle(combined)

    # 🔥 Split
    train_split = int(0.7 * len(combined))
    val_split = int(0.2 * len(combined))

    splits = {
        "train": combined[:train_split],
        "valid": combined[train_split:train_split + val_split],
        "test": combined[train_split + val_split:]
    }

    counter = 0

    for split, data in splits.items():
        for img, lbl in data:
            new_img = DST / "images" / split / f"{counter}{img.suffix}"
            new_lbl = DST / "labels" / split / f"{counter}.txt"

            shutil.copy(img, new_img)
            shutil.copy(lbl, new_lbl)

            counter += 1

    print(f"✅ Balanced dataset created: {counter} samples")

def create_yaml():
    yaml = f"""path: {DST}
train: images/train
val: images/valid
test: images/test

names:
  0: undamaged
  1: damaged
"""
    (DST / "data.yaml").write_text(yaml)

if __name__ == "__main__":
    create_dirs()
    rebalance()
    create_yaml()