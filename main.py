import base64
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from src.model_architecture import FasterRCNNModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your custom model architecture
model_instance = FasterRCNNModel(num_classes=2, device=device)
model = model_instance.model

# Load the trained weights from artifacts
model_path = "artifacts/models/fasterrcnn.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

GUN_LABEL_ID = 1  # dataset: 0 background, 1 gun
LABEL_MAP = {
    GUN_LABEL_ID: "Gun",
}

app = FastAPI(title="Guns Object Detection System", description="AI-powered weapon detection system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def predict_and_draw(image: Image.Image, score_threshold: float = 0.5):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)

    prediction = predictions[0]
    boxes = prediction.get("boxes", torch.empty((0, 4))).cpu().numpy()
    labels = prediction.get("labels", torch.empty((0,), dtype=torch.int64)).cpu().numpy()
    scores = prediction.get("scores", torch.empty((0,))).cpu().numpy()

    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)
    try:
        font = ImageFont.load_default()
    except Exception:  # fallback if system font missing
        font = None

    detections = []

    for box, score, label in zip(boxes, scores, labels):
        if float(score) < float(score_threshold):
            continue
        if int(label) != GUN_LABEL_ID:
            continue

        x_min, y_min, x_max, y_max = box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        label_text = f"{LABEL_MAP.get(int(label), f'class_{label}')} {score:.2f}"
        if font:
            draw.text((x_min + 4, max(0, y_min - 14)), label_text, fill="red", font=font)

        detections.append(
            {
                "label": LABEL_MAP.get(int(label), f"class_{label}"),
                "score": round(float(score), 3),
                "box": [float(x_min), float(y_min), float(x_max), float(y_max)],
            }
        )

    return img_rgb, detections

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict(file: UploadFile = File(...), score_threshold: float = 0.5):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    output_image, detections = predict_and_draw(image, score_threshold=score_threshold)
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(
        img_byte_arr,
        media_type="image/png",
        headers={"X-Detection-Count": str(len(detections))},
    )

@app.post("/predict/json/")
async def predict_json(file: UploadFile = File(...), score_threshold: float = 0.5):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    output_image, detections = predict_and_draw(image, score_threshold=score_threshold)
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format="PNG")
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    return {
        "count": len(detections),
        "detections": detections,
        "image": f"data:image/png;base64,{img_base64}",
    }