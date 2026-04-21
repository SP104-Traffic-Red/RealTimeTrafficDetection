from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 small model for speed efficiency
    model = YOLO('yolov8s.pt')

    # Train the model on your GPU
    results = model.train(
        data='data.yaml',
        epochs=50, # Number of times it loops through the dataset
        imgsz=640, # Resizes images to 640x640
        device=0,  # Forces the use of your RTX 4080 Super
        plots=True # Generates mAP performance graphs automatically
    )

if __name__ == '__main__':
    main()