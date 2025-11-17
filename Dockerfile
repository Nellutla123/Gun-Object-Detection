# Dockerfile (patched - robust apt + single pip layer)
FROM python:3.11-slim

# non-interactive for apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Install OS libs (robust names + no-install-recommends)
# non-interactive for apt and minimal libs
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements first for Docker cache (if you use requirements.txt)
COPY requirements.txt /app/requirements.txt

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install torch + torchvision (CPU wheels) and other python deps in one layer.
# Adjust index-url or package list as needed.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch" "torchvision" && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the app
COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

