from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image
import json

app = FastAPI(title="Warehouse Safety Detection API")

# --- CORS MIDDLEWARE ---
# This allows your Vercel frontend to communicate with this backend.
origins = [
    "https://wherehouse-vision.vercel.app", # <-- Your live frontend URL
    "null",
]
# -------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------

# --- MODEL LOADING ---
try:
    model = YOLO("best.pt")
    print(" Model loaded successfully.")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None
# -------------------------

@app.get("/")
def read_root():
    return {"message": "Welcome to the Warehouse Safety Detection API. Use the /docs endpoint for details."}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    if not model:
        return {"error": "Model is not loaded."}

    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    # Perform inference with a lower confidence threshold to ensure detections appear.
    results = model(img, conf=0.10) # <-- THE KEY FIX IS HERE

    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            bounding_box = box.xyxy[0].tolist()

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bounding_box": bounding_box,
            })

    return {"detections": detections}