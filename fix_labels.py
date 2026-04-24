from pathlib import Path

LABEL_DIR = Path("merged_dataset/labels")

def fix_labels():
    total_files = 0

    for split in ["train", "valid", "test"]:
        split_dir = LABEL_DIR / split

        if not split_dir.exists():
            continue

        for file in split_dir.glob("*.txt"):
            lines = file.read_text().strip().split("\n")
            new_lines = []

            for line in lines:
                parts = line.split()
                if len(parts) == 0:
                    continue

                # FORCE CLASS = 0 (damaged)
                parts[0] = "0"
                new_lines.append(" ".join(parts))

            file.write_text("\n".join(new_lines))
            total_files += 1

    print(f"✅ Fixed {total_files} label files")

if __name__ == "__main__":
    fix_labels()