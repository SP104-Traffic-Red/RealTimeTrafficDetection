"""
train.py
Fine-tunes a pretrained YOLOv8s model on the BDD100K traffic detection dataset.

The pretrained model (yolov8s.pt) already knows 80 COCO classes including
cars, trucks, buses, people, traffic lights, and stop signs. Fine-tuning
adapts these weights to BDD100K's 10 traffic-specific classes for improved
accuracy on real-world road scenes.

USAGE:
  python train.py                           # defaults: 50 epochs, 640px, GPU 0
  python train.py --epochs 100 --imgsz 640  # custom settings
  python train.py --device cpu              # CPU fallback (slow)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on BDD100K traffic data")
    parser.add_argument("--data", type=str, default="datasets/traffic/data.yaml",
                        help="Path to data.yaml (default: datasets/traffic/data.yaml)")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="Pretrained model to fine-tune (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size, -1 for auto (default: -1)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU (default: 0)")
    parser.add_argument("--name", type=str, default="traffic_model",
                        help="Experiment name for the runs/ folder")
    args = parser.parse_args()

    # Validate data.yaml exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        print("Run the dataset setup first:")
        print("  1. Download BDD100K from https://bdd-data.berkeley.edu")
        print("  2. Extract to datasets/bdd100k/")
        print("  3. Run: python convert_bdd100k.py")
        return

    print("=" * 60)
    print("  SP-104 Red — YOLOv8 Training on BDD100K")
    print("=" * 60)
    print(f"  Base model  : {args.model}")
    print(f"  Dataset     : {args.data}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Image size  : {args.imgsz}")
    print(f"  Device      : {'GPU ' + args.device if args.device != 'cpu' else 'CPU'}")
    print("=" * 60)

    # Load pretrained YOLOv8 model
    model = YOLO(args.model)

    # Fine-tune on the BDD100K traffic dataset
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        plots=True,      # generate mAP/loss plots
        save=True,        # save checkpoints
        patience=20,      # early stopping if no improvement for 20 epochs
        pretrained=True,
        optimizer="auto",
        verbose=True,
    )

    # Print summary
    best_weights = Path("runs/detect") / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best weights saved to: {best_weights}")
    print(f"  Training plots saved to: runs/detect/{args.name}/")
    print(f"\n  To run inference:")
    print(f"    python detect.py --source your_video.mp4 --weights {best_weights}")
    print(f"\n  To evaluate mAP:")
    print(f"    python evaluate.py --weights {best_weights} --data {args.data}")
    print("=" * 60)


if __name__ == "__main__":
    main()
