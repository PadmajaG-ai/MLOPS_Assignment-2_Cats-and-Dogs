# ─────────────────────────────────────────────
# Dockerfile — Cats vs Dogs inference service
# M2: Model Packaging & Containerization
# ─────────────────────────────────────────────

# Use slim Python 3.11 base (matches our dev environment)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install OS-level dependencies needed by Pillow / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy the trained model artifact
COPY models/cats_dogs_cnn.pt models/cats_dogs_cnn.pt

# Expose the port uvicorn will listen on
EXPOSE 8000

# Set environment variable so app.py finds the model
ENV MODEL_PATH=models/cats_dogs_cnn.pt

# Start the FastAPI service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
