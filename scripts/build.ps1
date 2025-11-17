# PowerShell build script for Docker image

Write-Host "Building Docker image for Guns Object Detection API..." -ForegroundColor Green

docker build -t guns-detection-api:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build complete!" -ForegroundColor Green
    Write-Host "To run the container: docker run -p 8000:8000 guns-detection-api:latest" -ForegroundColor Cyan
    Write-Host "Or use docker-compose: docker-compose up" -ForegroundColor Cyan
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

