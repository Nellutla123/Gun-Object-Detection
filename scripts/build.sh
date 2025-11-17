#!/bin/bash
# Build script for Docker image

set -e

echo "Building Docker image for Guns Object Detection API..."

docker build -t guns-detection-api:latest .

echo "Build complete!"
echo "To run the container: docker run -p 8000:8000 guns-detection-api:latest"
echo "Or use docker-compose: docker-compose up"

