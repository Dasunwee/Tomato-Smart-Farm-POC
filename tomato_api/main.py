import numpy as np
import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image

# Import Model 1 Tools (TensorFlow)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image

# Import Model 2 Tools (YOLO)
from ultralytics import YOLO

app = FastAPI()

# --- LOAD MODELS ---
# We load these once when the server starts to make predictions fast
print("Loading Disease Model...")
disease_model = load_model("model.keras")

print("Loading Fruit Model...")
fruit_model = YOLO("yolo_model.pt") 

# --- CLASS NAMES (Model 1: Disease) ---
# These must match the order from your training folder exactly
DISEASE_CLASSES = [
    'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

@app.get("/")
def home():
    return {"message": "Smart Farming AI is Active (Disease + Fruit Detection)"}

# --- MAIN ENDPOINT: ANALYZE PLANT ---
@app.post("/analyze")
async def analyze_plant(file: UploadFile = File(...)):
    """
    Runs BOTH models on the same image.
    1. Checks for Diseases (TensorFlow).
    2. Counts and Classifies Fruits (YOLO).
    """
    # A. Read Image Once
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    # --- TASK 1: DETECT DISEASE (Model 1) ---
    # Preprocess specifically for MobileNet (224x224, normalized)
    img_disease = pil_image.resize((224, 224))
    img_array = tf_image.img_to_array(img_disease)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    disease_pred = disease_model.predict(img_array)
    disease_index = np.argmax(disease_pred[0])
    disease_class = DISEASE_CLASSES[disease_index]
    disease_conf = float(np.max(disease_pred[0]))

    # --- TASK 2: DETECT FRUIT (Model 2) ---
    # YOLO handles PIL images directly!
    results = fruit_model(pil_image)
    
    # Process YOLO results
    detected_fruits = []
    fruit_summary = {"total": 0, "ripe": 0, "unripe": 0}
    
    # YOLO results are a list (one per image), we only sent one image
    for result in results:
        # result.boxes contains all the detections
        for box in result.boxes:
            # Get class ID and Confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get the name (e.g., 'b_fully_ripened')
            label_name = fruit_model.names[cls_id]
            
            detected_fruits.append({
                "type": label_name,
                "confidence": round(conf, 2)
            })
            
            # Update Summary Stats
            fruit_summary["total"] += 1
            # Check if label implies ripeness (based on your dataset names)
            if "fully_ripened" in label_name:
                fruit_summary["ripe"] += 1
            else:
                fruit_summary["unripe"] += 1

    # --- FINAL COMBINED REPORT ---
    return {
        "plant_health": {
            "status": disease_class,
            "confidence": f"{disease_conf*100:.1f}%",
            "is_healthy": "healthy" in disease_class
        },
        "harvest_info": {
            "total_fruits": fruit_summary["total"],
            "ripe_fruits": fruit_summary["ripe"],
            "unripe_fruits": fruit_summary["unripe"],
            "ready_to_harvest": fruit_summary["ripe"] > 0
        },
        "detailed_detections": detected_fruits
    }