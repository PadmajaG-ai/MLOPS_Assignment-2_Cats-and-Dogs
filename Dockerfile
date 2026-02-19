# ─────────────────────────────────────────────
# Dockerfile — Cats vs Dogs inference service
# M2: Model Packaging & Containerization
# Model is NOT baked into the image — it is
# mounted at runtime via docker-compose volume
# ─────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# OS-level dependencies needed by Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code only
COPY app.py .

# Expose the port uvicorn will listen on
EXPOSE 8000

# Model will be mounted via docker-compose volume at runtime
ENV MODEL_PATH=models/cats_dogs_cnn.pt

# Start the FastAPI service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
