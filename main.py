from fastapi import FastAPI, File, UploadFile
import torch
import io
import cv2
import numpy as np
import base64
from PIL import Image

app = FastAPI()

# Load the trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path="yolov5_model.pkl", source='local')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Convert to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run inference
    results = model(image)

    # Draw bounding boxes
    for *xyxy, conf, cls in results.xyxy[0].numpy():
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert image back to base64
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return {"image": encoded_image}
