# SP-104 Red: Real-Time Object Detection for Traffic Cameras

## Overview

This project implements a real-time traffic detection system using **YOLOv8** fine-tuned on the **BDD100K** (Berkeley DeepDrive 100K) dataset — the industry-standard benchmark for autonomous driving perception. The system detects and classifies vehicles, pedestrians, traffic lights, and road signs from video input, overlaying bounding boxes and confidence scores in real-time.

The model uses **transfer learning**: we start with YOLOv8s pretrained on COCO (80 general classes) and fine-tune it on BDD100K's 10 traffic-specific classes to achieve high detection accuracy on real driving footage.

## Project Team

- **Tate York:** Team Leader / ML Engineer
- **Parsa Rajabi:** ML Engineer
- **Ryan Booth:** Systems & Demo Integration

## Tech Stack

- **Language:** Python 3.9+
- **ML Framework:** PyTorch (CUDA 12.1)
- **Model:** YOLOv8s (Ultralytics)
- **Computer Vision:** OpenCV
- **Dataset:** BDD100K — 70K training / 10K validation images

## Detected Classes

The model detects **10 classes** from the BDD100K object detection benchmark:

| ID | Class | Description |
|---|---|---|
| 0 | `pedestrian` | People walking |
| 1 | `rider` | Cyclists and motorcyclists on their vehicle |
| 2 | `car` | Passenger vehicles |
| 3 | `truck` | Large freight vehicles |
| 4 | `bus` | Public transit buses |
| 5 | `train` | Trains and rail vehicles |
| 6 | `motorcycle` | Motorcycles |
| 7 | `bicycle` | Bicycles |
| 8 | `traffic light` | Traffic signals |
| 9 | `traffic sign` | Road signs |

## Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SP104-Traffic-Red/RealTimeTrafficDetection-SP104-Red-.git
   cd RealTimeTrafficDetection-SP104-Red-
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install PyTorch with CUDA 12.1 (for NVIDIA GPUs):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

This project uses the [BDD100K](https://bdd-data.berkeley.edu) dataset by Berkeley DeepDrive, the largest open driving video dataset with 100K annotated frames across diverse weather, lighting, and scene conditions.

1. Register at [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu) and download:
   - **100K Images** (`bdd100k_images_100k.zip`, ~6.5 GB)
   - **Detection 2020 Labels** (`bdd100k_labels_release.zip`)

2. Extract into the project:
   ```
   datasets/bdd100k/
   ├── images/100k/train/   (70K images)
   ├── images/100k/val/     (10K images)
   └── labels/
       ├── bdd100k_labels_images_train.json
       └── bdd100k_labels_images_val.json
   ```

3. Convert to YOLOv8 format:
   ```bash
   python convert_bdd100k.py
   ```

   For faster training with a subset (optional):
   ```bash
   python convert_bdd100k.py --max-images 20000
   ```

## Training the Model

Fine-tune the pretrained YOLOv8s model on BDD100K:

```bash
python train.py
```

Custom training options:

```bash
python train.py --epochs 50 --imgsz 640 --device 0
```

Training outputs are saved to `runs/detect/traffic_model/`. The best model weights are at `runs/detect/traffic_model/weights/best.pt`.

## Evaluating the Model

Run quantitative evaluation to generate mAP metrics on the validation set:

```bash
python evaluate.py
```

This prints mAP@0.5, mAP@0.5:0.95, precision, recall, and per-class breakdowns.

## Running Inference (The Demo)

Run the trained model on a video file:

```bash
python detect.py --source traffic_video.mp4
```

Other options:

```bash
python detect.py --source 0                                     # webcam
python detect.py --source traffic_video.mp4 --conf 0.50         # higher confidence
python detect.py --source traffic_video.mp4 --save              # save output video
```

> Press `q` to quit the video window.

## Project Structure

```
RealTimeTrafficDetection-SP104-Red-/
├── train.py               # Fine-tunes YOLOv8s on BDD100K
├── detect.py              # Real-time inference with bounding box visualization
├── evaluate.py            # Validation metrics (mAP, precision, recall)
├── convert_bdd100k.py     # Converts BDD100K JSON labels → YOLO .txt format
├── requirements.txt       # Python dependencies
├── .gitignore             # Excludes datasets, weights, and videos from Git
└── README.md
```

## References

- BDD100K Dataset: [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu)
- Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning," CVPR 2020
- Ultralytics YOLOv8: [docs.ultralytics.com](https://docs.ultralytics.com)
