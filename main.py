# main.py
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms, models

# --- model & device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# --- transform: returns CxHxW tensor (no batch dim) ---
transform = transforms.Compose([transforms.ToTensor()])

app = FastAPI()


def predict_and_draw(image: Image.Image, score_threshold: float = 0.7):
    """
    Run detection on a PIL image and draw boxes for detections >= score_threshold.
    Returns a PIL image with boxes drawn.
    """
    # ensure RGB
    img_rgb = image.convert("RGB")

    # prepare tensor (C,H,W) and move to device
    img_tensor = transform(img_rgb).to(device)  # shape: C x H x W

    # torchvision detection models expect a list of tensors
    try:
        with torch.no_grad():
            outputs = model([img_tensor])  # returns list[dict]
    except Exception as e:
        # surface model inference errors clearly
        raise RuntimeError(f"Model inference failed: {e}")

    if not outputs or len(outputs) == 0:
        # no outputs - return original image
        return img_rgb

    pred = outputs[0]  # correct variable (was typo in your code)

    # get boxes/scores/labels safely
    boxes = pred.get("boxes", torch.empty((0, 4))).cpu().numpy()
    scores = pred.get("scores", torch.empty((0,))).cpu().numpy()
    labels = pred.get("labels", torch.empty((0,), dtype=torch.int64)).cpu().numpy()

    # draw
    draw = ImageDraw.Draw(img_rgb)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box, score, label in zip(boxes, scores, labels):
        if float(score) >= float(score_threshold):
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            if font:
                text = f"{int(label)}: {score:.2f}"
                text_pos = (max(0, x_min), max(0, y_min - 10))
                draw.text(text_pos, text, fill="red", font=font)

    return img_rgb


@app.get("/")
def read_root():
    return {"message": "Welcome to the Object Detection API"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # read bytes and open PIL image
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # run inference and draw
    try:
        output_image = predict_and_draw(image, score_threshold=0.7)
    except RuntimeError as e:
        # model inference error
        raise HTTPException(status_code=500, detail=str(e))

    # return image as streaming response
    img_byte_array = io.BytesIO()
    output_image.save(img_byte_array, format="PNG")
    img_byte_array.seek(0)
    return StreamingResponse(img_byte_array, media_type="image/png")


            
    
    