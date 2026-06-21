# 1. Base image එකක් විදිහට Python පාවිච්චි කිරීම
FROM python:3.9-slim

# 2. වැඩ කරන folder එක සෑදීම
WORKDIR /app

# 3. පද්ධතියට අවශ්‍ය අතිරේක libraries (OpenCV සඳහා) install කිරීම
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. requirements.txt එක ඇතුළට copy කරලා libraries install කිරීම
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# 5. සියලුම code සහ models ඇතුළට copy කිරීම
COPY . .

# 6. API එක run කරන port එක විවෘත කිරීම
EXPOSE 8000

# 7. FastAPI run කරන විධානය
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]