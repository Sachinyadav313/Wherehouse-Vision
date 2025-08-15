from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import os
import io
from PIL import Image
import json

# Initialize the FastAPI app
app = FastAPI(title="Warehouse Safety Detection API")

# --- MODEL LOADING ---
# Load your custom-trained YOLOv8 model
# We define the path relative to this script
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'hard_hat_detection_final4', 'weights', 'best.pt')
model = YOLO(model_path)
print("✅ Model loaded successfully.")


# --- API ENDPOINT DEFINITION ---
@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    This endpoint receives an image file, runs object detection on it,
    and returns the results as JSON.
    """
    # Read the image file from the upload
    contents = await file.read()
    
    # Convert the image bytes to a PIL Image object
    image = Image.open(io.BytesIO(contents))

    # Run the YOLO model on the image
    # We disable verbose output for a cleaner API response
    results = model(image, verbose=False)

    # Process the first result (as we're sending one image at a time)
    result = results[0]
    
    # Prepare a list to store detection data
    detections = []
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = result.names[class_id]
        confidence = float(box.conf)
        bounding_box = box.xyxy[0].tolist() # [x1, y1, x2, y2]

        detections.append({
            "class_name": class_name,
            "confidence": confidence,
            "bounding_box": bounding_box
        })

    return {"detections": detections}

# --- ROOT ENDPOINT FOR TESTING ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Warehouse Safety Detection API. Use the /docs endpoint for details."}