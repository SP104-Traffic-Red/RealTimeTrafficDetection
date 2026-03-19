# SP-104 Red: Real-Time Object Detection for Traffic Cameras

## Overview

This project implements a real-time traffic detection system using **YOLOv8** to detect and classify road signs and vehicles from live camera feeds or recorded video. The long-term goal is to support traffic analytics use cases such as monitoring, vehicle counting, and congestion insights.

## Project Team

- **Tate York:** Team Leader / ML Engineer
- **Parsa Rajabi:** ML Engineer
- **Ryan Booth:** Systems & Demo Integration

## Goals Achieved

- [x] Detect and classify traffic elements from video input (live or pre-recorded).
- [x] Produce clear visual outputs (bounding boxes + confidence scores).
- [x] Evaluate performance using standard object detection metrics (mAP).
- [x] Keep the codebase modular for future extensions.

## Tech Stack

- **Language:** Python 3.9
- **ML Framework:** PyTorch (CUDA-enabled)
- **Computer Vision:** YOLOv8 (`ultralytics`), OpenCV, NumPy

## Repository Status

**Phase 1 Complete:** The baseline YOLOv8 model has been successfully fine-tuned on the Kaggle Road Sign Detection dataset. The inference pipeline is active and capable of processing real-time video feeds with overlaid bounding boxes and confidence scores.

## Detected Classes

This model is trained to detect exactly **4 classes**. It will not recognize or label anything outside of these:

| Class | Description |
|---|---|
| `trafficlight` | Traffic lights |
| `speedlimit` | Speed limit signs |
| `crosswalk` | Crosswalk signs |
| `stop` | Stop signs |

## Environment Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   ```

2. Create the virtual environment (ensure you have Python 3.9 installed):
   ```bash
   py -3.9 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   venv\Scripts\activate
   ```

4. Install PyTorch (CUDA 12.1 for NVIDIA GPUs):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. Install dependencies:
   ```bash
   pip install ultralytics opencv-python
   ```

## Running Inference (The Demo)

**CRITICAL:** You must activate the virtual environment before running the demo script.

1. Open your terminal in the project folder.

2. Activate the environment:
   ```bash
   venv\Scripts\activate
   ```

3. Run the object detection script on your video file:
   ```bash
   python detect.py --source <path_to_video.mp4> --weights runs/detect/train/weights/best.pt --conf 0.40
   ```

> Press the `q` key to quit the video window and end the script.

## Training the Model

To re-train or fine-tune the model, ensure your dataset is placed in the `datasets/` directory and mapped correctly in `data.yaml`.

Activate your virtual environment, then run:

```bash
python format_data.py  # Formats Kaggle XML annotations to YOLO TXT format
python train.py        # Initiates the 50-epoch training loop on the GPU
```
