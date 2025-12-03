# 🍅 Smart Farming Assistant: AI-Powered Tomato Crop Monitoring

## 📌 Project Overview

This project is a cloud-based Artificial Intelligence system designed to assist tomato farmers in Sri Lanka. It automates two critical tasks:

1. **Disease Detection**: Identifying leaf diseases (e.g., Blight, Mold) to prevent crop loss.
2. **Harvest Monitoring**: Counting and classifying fruits by ripeness (Green vs. Ripe) to optimize harvest timing.

This repository contains the source code for the AI Backend, Cloud Deployment configuration, and a Proof-of-Concept Mobile Interface.

## 🏗️ System Architecture

The system follows a Microservices architecture deployed on Microsoft Azure:

- **AI Engine**:
  - Disease Model: TensorFlow/Keras (MobileNetV2) - 90.22% Accuracy.
  - Fruit Model: Ultralytics YOLOv8 (Nano) - 84.4% mAP.
- **Backend API**: FastAPI (Python 3.10) running in a Docker Container.
- **Cloud Infrastructure**: Azure App Service (Linux B1 Container).
- **Client**: Flutter Mobile App (for real-time field testing).

## 📂 Folder Structure

| Folder | Description |
|--------|-------------|
| `/tomato_api` | Contains the Python FastAPI backend, Dockerfile, and trained model weights (`model.keras`, `yolo_model.pt`). |
| `/tomato_smart_farm` | Contains the Flutter mobile application source code (Android). |
| `/documentation` | Contains training graphs, confusion matrices, and the final project report. |

## 📊 Model Performance

### Model 1: Leaf Disease Classification

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Training**: 19 Epochs (Early Stopping at Epoch 14)
- **Result**: 90.22% Accuracy on Test Set.
- **Key Capability**: 100% Precision on Yellow Leaf Curl Virus.

### Model 2: Fruit Detection

- **Architecture**: YOLOv8 Nano (Object Detection)
- **Training**: 50 Epochs on Tesla T4 GPU (Google Colab).
- **Result**: 84.4% mAP@50.
- **Key Capability**: High precision detection of 'Green' vs 'Fully Ripened' tomatoes for yield estimation.

## 🚀 How to Run (Local Testing)

### 1. Backend (API)

Navigate to the api folder and install dependencies:

```bash
cd tomato_api
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn main:app --reload
```

Access API Documentation at: `http://127.0.0.1:8000/docs`

### 2. Frontend (Mobile App)

Navigate to the app folder:

```bash
cd tomato_smart_farm
flutter run
```

## ☁️ Deployment

The API is currently deployed live on Microsoft Azure App Service.

- **Live Endpoint**: `https://tomato-farm-app-dasun.azurewebsites.net`
- **API Docs**: `https://tomato-farm-app-dasun.azurewebsites.net/docs`

## 👤 Author

- **Name**: [W.A.D.N Weerawardhana]
- **Student ID**: [ITBNM-2211-0194]
- **Intake**: 11
