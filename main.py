from fastapi import FastAPI, File, UploadFile, Form
from enum import Enum
import numpy as np
import io
import cv2
import os
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
from fastapi.responses import HTMLResponse

# Environment settings to keep logs clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize the FastAPI app BEFORE using decorators like @app.get
app = FastAPI()

# --- 1. ROOT PATH GUIDE ---
# This shows a nice landing page instead of a "Not Found" error
@app.get("/", response_class=HTMLResponse)
async def root_guide():
    return """
    <html>
        <head>
            <title>Tomato AI - API Guide</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 40px; background-color: #f0f2f5; }
                .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); max-width: 800px; margin: auto; }
                h1 { color: #e74c3c; border-bottom: 2px solid #e74c3c; padding-bottom: 10px; }
                .btn { background: #e74c3c; color: white; padding: 12px 25px; text-decoration: none; border-radius: 8px; display: inline-block; margin-top: 20px; font-weight: bold; }
                .btn:hover { background: #c0392b; }
                code { background: #f8f9fa; padding: 2px 6px; border-radius: 4px; color: #d63384; border: 1px solid #dee2e6; }
                ul { padding-left: 20px; }
                li { margin-bottom: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🍅 Tomato AI Analysis System</h1>
                <p>Welcome to the <b>Final Year Project API</b> developed by <b>Wickrama Arachcilage Dasun Nirmal Weerawardhana</b>.</p>
                <hr>
                <h3>📌 How to use this API:</h3>
                <ul>
                    <li>Click the button below to go to the <b>Swagger UI</b>.</li>
                    <li>Look for the <code>POST /analyze</code> endpoint.</li>
                    <li>Click <b>"Try it out"</b> and upload a tomato plant image.</li>
                    <li>Select <code>leaf</code> to check for diseases or <code>fruit</code> to count tomatoes and check maturity.</li>
                </ul>
                <a href="/docs" class="btn">🚀 Launch Testing UI (Swagger)</a>
            </div>
        </body>
    </html>
    """

# Define scan categories
class ScanType(str, Enum):
    leaf = "leaf"
    fruit = "fruit"

# Global variables for models
disease_model = None
fruit_model = None

# --- MODEL LOADING LOGIC ---

def get_disease_model():
    """Load the CNN model only when needed (Lazy Loading)"""
    global disease_model
    if disease_model is None:
        print("[SYSTEM] Loading Tomato Disease Model...")
        # Load the .keras file
        disease_model = tf.keras.models.load_model("final_tomato_disease_model_optimized.keras", compile=False)
    return disease_model

def get_fruit_model():
    """Load the YOLO model for tomato detection"""
    global fruit_model
    if fruit_model is None:
        print("[SYSTEM] Loading YOLO Fruit Model...")
        fruit_model = YOLO("yolo_model.pt")
    return fruit_model

# Target labels for the 10 disease categories [cite: 1]
DISEASE_CLASSES = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

# --- IMAGE PROCESSING HELPERS ---

def is_it_green_leaf(pil_img):
    """Simple HSV filter to check if the image contains a green leaf"""
    cv_img = np.array(pil_img) 
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    ratio = np.count_nonzero(mask) / (pil_img.size[0] * pil_img.size[1])
    return ratio > 0.15

def is_ready_to_harvest(pil_img):
    """Simple HSV filter to check if the image contains red/orange fruits"""
    cv_img = np.array(pil_img)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)
    ratio = np.count_nonzero(red_mask) / (pil_img.size[0] * pil_img.size[1])
    return ratio > 0.10 

# --- THE MAIN API ENDPOINT ---

@app.post("/analyze")
async def analyze_plant(file: UploadFile = File(...), scan_type: ScanType = Form(...)):
    """Receives image and returns AI analysis based on scan_type"""
    
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')

    if scan_type.value == "leaf":
        model = get_disease_model()
        
        if not is_it_green_leaf(pil_image):
            return {"status": "Failed", "error": "The image does not look like a green leaf."}

        # Preprocessing: Resize to 224x224 and Normalize
        img_array = np.array(pil_image.resize((224, 224))) / 255.0 
        img_array = np.expand_dims(img_array, 0).astype(np.float32)

        # Predict
        preds = model.predict(img_array)
        conf = float(np.max(preds[0]))
        label = DISEASE_CLASSES[np.argmax(preds[0])]

        return {
            "analysis_type": "leaf",
            "status": label,
            "confidence": f"{conf*100:.1f}%",
            "is_healthy": "healthy" in label.lower()
        }

    elif scan_type.value == "fruit":
        model = get_fruit_model()
        results = model(pil_image)
        fruit_count = len(results[0].boxes) if results else 0
        harvest_ready = is_ready_to_harvest(pil_image)
        
        return {
            "analysis_type": "fruit",
            "total_fruits": fruit_count,
            "harvest_status": "Ready to Harvest" if harvest_ready else "Not Ready (Green)"
        }