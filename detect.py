import cv2
import argparse
from ultralytics import YOLO

def run_inference(source, weights, conf_thres, save_output=False):
    # Load the fine-tuned model
    model = YOLO(weights)

    # Open the video source (0 for webcam, or a string for a file/URL)
    cap = cv2.VideoCapture(source)

    # Calculate native delay based on video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 1

    # Setup video writer if saving output
    writer = None
    if save_output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output_old_model.mp4", fourcc, fps, (width, height))
        print(f"Saving output to: output_old_model.mp4")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame, conf=conf_thres, device=0)

        # Draw bounding boxes and labels on the frame
        annotated_frame = results[0].plot()

        # Save frame if writer is active
        if writer is not None:
            writer.write(annotated_frame)

        # Display the processed video
        cv2.imshow("SP-104 Red Traffic Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        # Native delay instead of a hardcoded '1'
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='test_video.mp4', help='Video path or 0 for webcam')
    parser.add_argument('--weights', type=str, default='runs/detect/train/weights/best.pt', help='Model path')
    parser.add_argument('--conf', type=float, default=0.60, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save output video')
    args = parser.parse_args()

    run_inference(args.source, args.weights, args.conf, args.save)