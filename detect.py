"""
detect.py
Runs YOLOv8 inference on a video file, webcam feed, or stream URL.
Draws bounding boxes with class labels and confidence scores in real-time.

USAGE:
  python detect.py --source traffic_video.mp4                           # video file
  python detect.py --source 0                                           # webcam
  python detect.py --source https://example.com/stream.m3u8             # stream URL
  python detect.py --source traffic_video.mp4 --weights best.pt --conf 0.5
  python detect.py --source traffic_video.mp4 --save                    # save output video
"""

import cv2
import argparse
import time
from pathlib import Path
from ultralytics import YOLO


def run_inference(source, weights, conf_thres, save_output):
    # Load the model
    model = YOLO(weights)
    print(f"Model loaded: {weights}")
    print(f"Classes: {model.names}")
    print(f"Confidence threshold: {conf_thres}")

    # Determine source type
    if source.isdigit():
        source = int(source)
        print("Source: Webcam")
    else:
        print(f"Source: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source '{source}'")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate delay between frames to match original video speed
    frame_delay = max(1, int(1000 / fps))  # milliseconds per frame

    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")

    # Setup video writer if saving output
    writer = None
    if save_output:
        output_path = Path("runs/detect/output")
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = str(output_path / "result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
        print(f"Saving output to: {out_file}")

    print("\nRunning inference... Press 'q' to quit.\n")

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Run YOLOv8 inference
        results = model(frame, conf=conf_thres, verbose=False)

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Calculate and display FPS
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(
            annotated_frame,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Show detection count
        num_detections = len(results[0].boxes)
        cv2.putText(
            annotated_frame,
            f"Objects: {num_detections}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Display the frame
        cv2.imshow("SP-104 Red | Traffic Detection", annotated_frame)

        # Save frame if writer is active
        if writer is not None:
            writer.write(annotated_frame)

        # Break on 'q' key (wait matches original video frame rate)
        if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
            print("\nStopped by user.")
            break

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Print summary
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Traffic Detection Inference")
    parser.add_argument("--source", type=str, default="test_video.mp4",
                        help="Video path, webcam index (0), or stream URL")
    parser.add_argument("--weights", type=str, default="runs/detect/traffic_model/weights/best.pt",
                        help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.40,
                        help="Confidence threshold (default: 0.40)")
    parser.add_argument("--save", action="store_true",
                        help="Save the output video to runs/detect/output/")
    args = parser.parse_args()

    run_inference(args.source, args.weights, args.conf, args.save)
