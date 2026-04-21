"""
evaluate.py
Evaluates a trained YOLOv8 model on the validation set and prints mAP metrics.
This satisfies the project requirement for quantitative evaluation using mAP.

USAGE:
  python evaluate.py
  python evaluate.py --weights runs/detect/traffic_model/weights/best.pt --data datasets/traffic/data.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on validation data")
    parser.add_argument("--weights", type=str, default="runs/detect/traffic_model/weights/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--data", type=str, default="datasets/traffic/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for evaluation")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: Weights file not found: {weights_path}")
        print("Train the model first with: python train.py")
        return

    print("=" * 60)
    print("  SP-104 Red — Model Evaluation")
    print("=" * 60)
    print(f"  Weights : {args.weights}")
    print(f"  Dataset : {args.data}")
    print("=" * 60)

    # Load trained model
    model = YOLO(args.weights)

    # Run validation
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        verbose=True,
    )

    # Print key metrics
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"  mAP@0.5       : {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95  : {results.box.map:.4f}")
    print(f"  Precision      : {results.box.mp:.4f}")
    print(f"  Recall         : {results.box.mr:.4f}")

    # Per-class breakdown
    if hasattr(results.box, "maps") and results.box.maps is not None:
        print(f"\n  Per-Class mAP@0.5:")
        names = model.names
        for i, m in enumerate(results.box.maps):
            if i in names:
                print(f"    {names[i]:20s} : {m:.4f}")

    print("=" * 60)
    print(f"  Evaluation plots saved to: {results.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
