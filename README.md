# 🍅 Tomato AI Analysis System (Version 5)

Welcome to the **Tomato AI Analysis System (V5)**. This repository contains a production-ready, highly stable FastAPI-based backend designed for precision agriculture. The system leverages state-of-the-art Deep Learning models to analyze tomato plants via images, offering both **Disease Classification** (with tailored treatment guidelines) and **Fruit Detection/Ripeness Estimation**.

Developed as part of a Final Year Undergraduate Project by **Wickrama Arachcilage Dasun Nirmal Weerawardhana**.

---

## 📌 Core Features

The API exposes a highly dynamic `/analyze` endpoint that handles two primary operational modes using a custom type switch (`leaf` or `fruit`):

### 1. Leaf Scan (`scan_type: leaf`)
* **Advanced Architecture:** Uses a custom-reconstructed **MobileNetV2** structure to bypass traditional Keras object deserialization version conflicts.
* **Smart Filtering:** Utilizes an OpenCV HSV green-filter check (`is_it_green_leaf`) to confirm the uploaded image actually contains a leaf before running model inference.
* **10-Class Classification:** Accurately classifies the leaf into 9 specific disease groups or marks it as completely healthy.
* **Actionable Insights:** Returns targeted **Treatment Recommendations** based on the identified disease condition to support immediate field action.

### 2. Fruit Scan (`scan_type: fruit`)
* **Object Detection:** Integrates a **YOLOv8** model (`yolo_model.pt`) trained to detect individual tomato fruits and return absolute counts.
* **Ripeness & Harvest Estimation:** Evaluates color histograms using custom red/orange and yellow/orange HSV masks to categorize maturity status (`Ready to Harvest`, `Turning / Almost Ready`, or `Not Ready`) along with an estimated timeline in days.

---

## 🛠️ Tech Stack

* **Framework:** FastAPI (Python)
* **ASGI Server:** Uvicorn
* **Deep Learning Frameworks:** TensorFlow 2.15.0, Keras 2.15.0, Ultralytics (YOLOv8)
* **Image Processing:** OpenCV (cv2), Pillow (PIL)
* **Data Manipulation:** NumPy

---

## 📈 Evolution: What Changed in V4/V5?

Compared to previous baseline versions (V3), the current architecture features critical production level updates:
1. **The Deserialization Fix:** Solved the notorious Keras core error `TypeError: Could not deserialize class 'Functional'`. Instead of relying on a fragile `load_model()` call, the backend manually re-assembles the MobileNetV2 skeletal layout and directly remaps the weight layers using `.load_weights()`. This ensures **100% deployment compatibility** on local machines and remote servers alike (e.g., Hugging Face Spaces).
2. **Treatment Delivery:** Integrated real-world agricultural mitigation advice directly inside the output JSON payload under the new `"treatment"` field.
3. **Data Type Strictness:** Explicitly casts processed arrays to `np.float32` prior to execution, minimizing prediction variations across different host machine optimization setups.

---

## 🚀 Local Installation & Setup

Ensure you are working inside a clean virtual environment (`venv`).

### 1. Install Dependencies
Run the force-reinstall command to guarantee there are no conflicting Keras 3.x packages cached locally:
```bash
pip install --force-reinstall -r requirements.txt
