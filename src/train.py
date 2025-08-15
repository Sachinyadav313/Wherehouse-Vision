from ultralytics import YOLO

def main():
    # Load a pre-trained model
    model = YOLO('yolov8n.pt')

    # Train the model using the simple path to data.yaml
    results = model.train(
        data='data/data.yaml',
        epochs=25,
        imgsz=640,
        project='models',
        name='hard_hat_detection_final'
    )
    print("✅ Training finished successfully.")  
    print(f"Model saved to: {results.save_dir}") 

if __name__ == '__main__':
    main() 