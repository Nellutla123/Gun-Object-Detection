# Stage 1: Download and cache model weights
FROM python:3.9-slim AS model-cache
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN python -c "import torch; import torchvision; torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')"

# Stage 2: Final application image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY docker-requirements.txt .

# Install Python dependencies and clean up in same layer
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r docker-requirements.txt && \
    pip cache purge

# Copy cached models from previous stage
COPY --from=model-cache /root/.cache/torch /root/.cache/torch

# Create necessary directories
RUN mkdir -p src artifacts/models static

# Copy specific source files
COPY src/model_architecture.py src/
COPY src/custom_exception.py src/
COPY src/logger.py src/
COPY src/data_processing.py src/
COPY static/ static/
COPY main.py .
COPY run_server.py .

# Copy the trained model
COPY artifacts/models/ artifacts/models/

# Create empty __init__.py files for Python modules
RUN touch src/__init__.py

# Expose port 8000 for FastAPI
EXPOSE 8000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Command to run the application
CMD ["python", "run_server.py"]

