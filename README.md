# 🍅 Tomato Leaf Disease Classification API (V4)

This repository contains the backend API and the highly optimized Convolutional Neural Network (CNN) model for the **Tomato Leaf Disease Classification System**. This system is designed to accurately identify diseases in tomato plants from leaf images.

## 🚀 What's New in V4 (Final Release)
* **Model Upgrade:** Integrated `best_tomato_model_V2.keras`.
* **Architecture Enhancements:** Implemented Custom Dense Blocks.
* **Class Imbalance Handling:** Applied Class Weights during training.
* **High Accuracy:** Achieved an overall model accuracy of **92.5%**.
* **API Optimization:** Fully updated routing and model path configurations to support the V2 model seamlessly.

## 📁 Repository Structure
* `best_tomato_model_V2.keras`: The optimized CNN model (92.5% Accuracy).
* `app.py` / `main.py`: The main API gateway.
* `requirements.txt`: Required Python dependencies.
* `README.md`: Project documentation.

## 🛠️ Tech Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Backend API:** Python (FastAPI / Flask)
* **Deployment:** Docker & Hugging Face (Backed up in V2 branches)

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Dasunwee/Tomato-Smart-Farm-POC.git](https://github.com/Dasunwee/Tomato-Smart-Farm-POC.git)
   cd Tomato-Smart-Farm-POC
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API:**
   ```bash
   python main.py  # Or 'uvicorn main:app --reload' if using FastAPI
   ```

---

## 👨‍💻 Author
**W.A.D.N Weerawardhana** Undergraduate - BSc (Hons) in Networking and Mobile Computing  
Horizon Campus  
**Student ID:** ITBNM-2211-0194
