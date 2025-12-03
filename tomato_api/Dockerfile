# 1. Use Python 3.10 Slim (Smaller starting point)
FROM python:3.10-slim

# 2. Set directory
WORKDIR /app

# 3. Install necessary system libs (Updated for Debian Bookworm)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements
COPY requirements.txt .

# 5. CRITICAL: Install CPU-ONLY PyTorch (Saves ~800MB)
# We do this BEFORE installing other things to ensure the lightweight version is used
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Install other libraries (Force tensorflow-cpu)
# Note: We remove 'ultralytics' from requirements.txt logic here and install it manually 
# to prevent it from reinstalling the GPU torch.
RUN pip install --no-cache-dir tensorflow-cpu ultralytics fastapi uvicorn python-multipart pillow opencv-python-headless numpy

# 7. Copy your code and models
COPY . .

# 8. Start the app
ENV PORT=8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]