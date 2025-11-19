import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from src.model_architecture import FasterRCNNModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your custom model architecture
model_instance = FasterRCNNModel(num_classes=2, device=device)
model = model_instance.model

# Load the trained weights from artifacts
model_path = "artifacts/models/fasterrcnn.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])

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

def predict_and_draw(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)

    prediction = predictions[0]
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    for box,score in zip(boxes, scores):
        if score >0.7:
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    return img_rgb

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict(file:UploadFile=File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    output_image = predict_and_draw(image)
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")