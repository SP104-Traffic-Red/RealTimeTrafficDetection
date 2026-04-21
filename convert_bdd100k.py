"""
convert_bdd100k.py
Converts BDD100K per-image JSON labels to YOLOv8 format (.txt files).

Handles the current BDD100K label format where each JSON file contains:
  { "name": "...", "frames": [{ "objects": [{ "category": "car", "box2d": {...} }] }] }

PREREQUISITES:
  1. Download from https://bdd-data.berkeley.edu:
     - "100K Images" → extract to datasets/bdd100k/images/100k/
     - "Labels"      → extract to datasets/bdd100k/labels/
  2. Your folder structure should look like:
       datasets/bdd100k/
       ├── images/100k/train/   (.jpg files)
       ├── images/100k/val/     (.jpg files)
       └── labels/
           ├── train/   (.json files, one per image)
           └── val/     (.json files, one per image)

USAGE:
  python convert_bdd100k.py
  python convert_bdd100k.py --max-images 20000   # use a subset for faster training
"""

import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

# BDD100K uses slightly different category names across releases.
# This map covers all known variants → YOLO class IDs.
CLASS_MAP = {
    "pedestrian": 0,
    "person": 0,       # some releases use "person" instead of "pedestrian"
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "motor": 6,         # alternate name in some releases
    "bicycle": 7,
    "bike": 7,          # alternate name in some releases
    "traffic light": 8,
    "traffic sign": 9,
}

# Categories to SKIP (lane markings, drivable areas, etc.)
# These appear in the JSON but are not object detection targets.
SKIP_PREFIXES = ("area/", "lane/")

# BDD100K image dimensions (all 100K images are 1280x720)
IMG_WIDTH = 1280
IMG_HEIGHT = 720


def box2d_to_yolo(box2d):
    """Convert a BDD100K box2d dict to YOLO normalized format."""
    x1 = max(0.0, min(float(box2d["x1"]), IMG_WIDTH))
    y1 = max(0.0, min(float(box2d["y1"]), IMG_HEIGHT))
    x2 = max(0.0, min(float(box2d["x2"]), IMG_WIDTH))
    y2 = max(0.0, min(float(box2d["y2"]), IMG_HEIGHT))

    if x2 <= x1 or y2 <= y1:
        return None

    x_center = ((x1 + x2) / 2) / IMG_WIDTH
    y_center = ((y1 + y2) / 2) / IMG_HEIGHT
    w = (x2 - x1) / IMG_WIDTH
    h = (y2 - y1) / IMG_HEIGHT
    return x_center, y_center, w, h


def convert_single_json(json_path: Path):
    """
    Convert one per-image JSON file to YOLO label lines.

    BDD100K JSON structure:
    {
      "name": "image_id",
      "frames": [{
        "timestamp": 10000,
        "objects": [{
          "category": "car",
          "box2d": { "x1": ..., "y1": ..., "x2": ..., "y2": ... }
        }, ...]
      }],
      "attributes": { "weather": "clear", "scene": "highway", "timeofday": "daytime" }
    }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    lines = []
    stats = defaultdict(int)

    # Navigate to the objects list
    frames = data.get("frames", [])
    for frame in frames:
        objects = frame.get("objects", [])
        for obj in objects:
            category = obj.get("category", "")

            # Skip non-detection categories (lanes, drivable areas, etc.)
            if any(category.startswith(prefix) for prefix in SKIP_PREFIXES):
                continue

            if category not in CLASS_MAP:
                continue

            box2d = obj.get("box2d")
            if box2d is None:
                continue

            result = box2d_to_yolo(box2d)
            if result is None:
                continue

            class_id = CLASS_MAP[category]
            stats[category] += 1
            x_c, y_c, w, h = result
            lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    return lines, stats


def convert_per_image_jsons(json_dir: Path, output_dir: Path, max_images: int = 0):
    """Convert all per-image JSON files in a directory to YOLO .txt files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total_stats = defaultdict(int)
    converted = 0

    json_files = sorted(json_dir.glob("*.json"))

    if max_images > 0:
        json_files = json_files[:max_images]

    total = len(json_files)
    for i, jf in enumerate(json_files):
        lines, stats = convert_single_json(jf)

        for cat, count in stats.items():
            total_stats[cat] += count

        # Write YOLO .txt file (same stem as the JSON / image)
        txt_file = output_dir / (jf.stem + ".txt")
        with open(txt_file, "w") as f:
            f.write("\n".join(lines))

        converted += 1

        if (i + 1) % 5000 == 0 or (i + 1) == total:
            print(f"    {i + 1}/{total} files converted...")

    return converted, total_stats


def copy_images(src_dir: Path, dst_dir: Path, label_dir: Path, max_images: int = 0):
    """Copy images that have corresponding YOLO label files."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    label_stems = {f.stem for f in label_dir.glob("*.txt")}
    copied = 0

    for img_path in sorted(src_dir.glob("*.jpg")):
        if max_images > 0 and copied >= max_images:
            break
        if img_path.stem in label_stems:
            dst = dst_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)
            copied += 1

            if copied % 5000 == 0:
                print(f"    {copied} images copied...")

    return copied


def create_data_yaml(output_dir: Path):
    """Create data.yaml for YOLOv8 training."""
    yaml_content = f"""# BDD100K Traffic Detection Dataset - YOLOv8 Format
# Source: Berkeley DeepDrive (https://bdd-data.berkeley.edu)
# Paper: Yu et al., "BDD100K: A Diverse Driving Dataset," CVPR 2020

path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Convert BDD100K labels to YOLOv8 format")
    parser.add_argument("--bdd-root", type=str, default="datasets/bdd100k",
                        help="Path to the BDD100K root folder")
    parser.add_argument("--output", type=str, default="datasets/traffic",
                        help="Output directory for YOLO-formatted dataset")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images per split (0 = use all)")
    parser.add_argument("--no-copy", action="store_true",
                        help="Skip image copying (just convert labels)")
    args = parser.parse_args()

    bdd_root = Path(args.bdd_root)
    output_dir = Path(args.output)

    # Validate structure
    train_labels = bdd_root / "labels" / "train"
    val_labels = bdd_root / "labels" / "val"
    train_images = bdd_root / "images" / "100k" / "train"
    val_images = bdd_root / "images" / "100k" / "val"

    if not train_labels.exists() or not any(train_labels.glob("*.json")):
        print("ERROR: No JSON label files found.")
        print(f"  Looked in: {train_labels}")
        print(f"\nExpected structure:")
        print(f"  {bdd_root}/labels/train/*.json")
        print(f"  {bdd_root}/labels/val/*.json")
        print(f"  {bdd_root}/images/100k/train/*.jpg")
        print(f"  {bdd_root}/images/100k/val/*.jpg")
        print(f"\nDownload 'Labels' and '100K Images' from: https://bdd-data.berkeley.edu")
        return

    n_train_json = len(list(train_labels.glob("*.json")))
    n_val_json = len(list(val_labels.glob("*.json"))) if val_labels.exists() else 0

    print("=" * 60)
    print("  BDD100K → YOLOv8 Format Conversion")
    print("=" * 60)
    print(f"  Train labels found: {n_train_json:,} JSON files")
    print(f"  Val labels found:   {n_val_json:,} JSON files")
    if args.max_images > 0:
        print(f"  Subset limit:       {args.max_images:,} per split")
    print()

    # Step 1: Convert training labels
    print("[1/4] Converting training labels...")
    train_yolo_dir = output_dir / "labels" / "train"
    train_conv, train_stats = convert_per_image_jsons(
        train_labels, train_yolo_dir, args.max_images
    )
    print(f"  Converted: {train_conv:,} files")
    for cls, count in sorted(train_stats.items(), key=lambda x: -x[1]):
        print(f"    {cls:20s}: {count:,}")

    # Step 2: Convert validation labels
    print(f"\n[2/4] Converting validation labels...")
    val_yolo_dir = output_dir / "labels" / "val"
    if val_labels.exists() and n_val_json > 0:
        val_conv, val_stats = convert_per_image_jsons(
            val_labels, val_yolo_dir, args.max_images
        )
        print(f"  Converted: {val_conv:,} files")
    else:
        print("  WARNING: No validation labels found, skipping")

    # Step 3-4: Copy images
    if not args.no_copy:
        print(f"\n[3/4] Copying training images...")
        if train_images.exists():
            n_train = copy_images(train_images, output_dir / "images" / "train",
                                  train_yolo_dir, args.max_images)
            print(f"  Copied {n_train:,} training images")
        else:
            print(f"  WARNING: {train_images} not found")

        print(f"\n[4/4] Copying validation images...")
        if val_images.exists():
            n_val = copy_images(val_images, output_dir / "images" / "val",
                                val_yolo_dir, args.max_images)
            print(f"  Copied {n_val:,} validation images")
        else:
            print(f"  WARNING: {val_images} not found")
    else:
        print("\n[3/4] Skipping image copy (--no-copy)")
        print("[4/4] Skipping image copy")

    # Create data.yaml
    yaml_path = create_data_yaml(output_dir)

    print("\n" + "=" * 60)
    print("  Conversion Complete!")
    print("=" * 60)
    print(f"  Output:    {output_dir}")
    print(f"  data.yaml: {yaml_path}")
    print(f"\n  Next step — train the model:")
    print(f"    python train.py --data {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
