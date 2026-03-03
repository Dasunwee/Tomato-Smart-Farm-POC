from fastapi import FastAPI, File, UploadFile, Form
from enum import Enum
import numpy as np
import io
import cv2 # OpenCV පාවිච්චි කරලා පින්තූරයේ පාට පරීක්ෂා කරමු
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

class ScanType(str, Enum):
    leaf = "leaf"
    fruit = "fruit"

app = FastAPI()

# --- MODELS ලෝඩ් කිරීම ---
disease_model = tf.keras.models.load_model("final_tomato_disease_model_optimized.keras")
fruit_model = YOLO("yolo_model.pt") 

DISEASE_CLASSES = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

# --- පින්තූරය තක්කාලි කොළයක්දැයි බැලීමට පාට පරීක්ෂා කරන Function එක ---
def is_it_green_leaf(pil_img):
    # OpenCV වලට පින්තූරය හරවා ගැනීම
    open_cv_image = np.array(pil_img) 
    img_hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2HSV)
    
    # කොළ පාට සඳහා Range එකක් ලබාගැනීම
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(mask) / (pil_img.size[0] * pil_img.size[1])
    
    # පින්තූරයෙන් 15% කට වඩා කොළ පාට නැත්නම් ඒක කොළයක් ලෙස නොගැනීම
    return green_ratio > 0.15

@app.post("/analyze")
async def analyze_plant(file: UploadFile = File(...), scan_type: ScanType = Form(...)):
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')

    if scan_type.value == "leaf":
        # 1. පියවර: පින්තූරයේ කොළ පාට තිබේදැයි බැලීම (New Filter)
        if not is_it_green_leaf(pil_image):
            return {
                "status": "Identification Failed",
                "error": "This is not a tomato leaf image.",
                "suggestion": "The image is missing natural green tones. Please upload a clear photo of a tomato leaf."
            }

        # 2. පියවර: Disease Prediction
        img_disease = pil_image.resize((224, 224))
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(np.array(img_disease), 0))
        disease_pred = disease_model.predict(img_array)
        disease_conf = float(np.max(disease_pred[0]))

        # 3. පියවර: Confidence එක 85% ට වඩා තිබිය යුතුය (Increased Threshold)
        if disease_conf < 0.85:
            return {
                "status": "Identification Failed",
                "error": "Uncertain prediction.",
                "suggestion": "AI is not 100% sure this is a tomato leaf. Please take a closer photo."
            }

        return {
            "analysis_type": "leaf",
            "status": DISEASE_CLASSES[np.argmax(disease_pred[0])],
            "confidence": f"{disease_conf*100:.1f}%",
            "is_healthy": "healthy" in DISEASE_CLASSES[np.argmax(disease_pred[0])]
        }

    elif scan_type.value == "fruit":
        # Fruit logic එක කලින් වගේමයි
        results = fruit_model(pil_image)
        fruit_count = len(results[0].boxes) if results else 0
        if fruit_count == 0:
            return {"error": "No tomatoes detected."}
        return {"analysis_type": "fruit", "total_fruits": fruit_count}