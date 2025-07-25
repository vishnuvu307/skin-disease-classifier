"""
skin_disease_api.py
--------------------

This module exposes a simple HTTP API for running inference on the trained
skin‑disease classification model.  It uses FastAPI and Uvicorn to serve
requests.  Users can send a POST request with an image file; the server
will preprocess the image, run it through the model and return the top
predicted class with its probability.  If an ONNX model is provided, the
script will use onnxruntime for inference; otherwise it loads the PyTorch
weights into the same architecture defined in `skin_disease_model.py`.

IMPORTANT: This API is a demonstration.  It is not a medical device and
should not be used to diagnose or treat disease.  Always consult a
healthcare professional.
"""

import io
import json
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

import torch
from torchvision import models, transforms

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class SkinDiseasePredictor:
    def __init__(self, model_path: str, num_classes: int, class_names: List[str] = None):
        self.model_path = Path(model_path)
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [f"class_{i}" for i in range(num_classes)]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        if self.model_path.suffix == ".onnx" and ORT_AVAILABLE:
            # Load ONNX model for inference
            self.session = ort.InferenceSession(str(self.model_path))
            self.use_onnx = True
        else:
            # Load PyTorch weights into an EfficientNet model
            self.use_onnx = False
            model = models.efficientnet_b0(pretrained=False)
            in_features = model.classifier[1].in_features
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(in_features, num_classes),
            )
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            model.eval()
            self.model = model

    def predict(self, img: Image.Image) -> dict:
        input_tensor = self.transform(img).unsqueeze(0)
        if self.use_onnx:
            ort_inputs = {self.session.get_inputs()[0].name: input_tensor.numpy()}
            ort_outs = self.session.run(None, ort_inputs)
            scores = ort_outs[0][0]
        else:
            with torch.no_grad():
                scores = self.model(input_tensor)[0].numpy()
        # Convert to probabilities
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        top_idx = int(np.argmax(probs))
        return {
            "predicted_class": self.class_names[top_idx],
            "probability": float(probs[top_idx]),
            "all_probs": {self.class_names[i]: float(p) for i, p in enumerate(probs)},
        }


# Initialise FastAPI
app = FastAPI(title="Skin Disease Classifier API")
predictor = None


@app.on_event("startup")
async def startup_event():
    """Load the model at startup.  Modify the path and class_names as needed."""
    global predictor
    model_path = Path("skin_model.onnx") if Path("skin_model.onnx").exists() else Path("skin_model.pth")
    class_names = [
        "melanocytic_nevi",
        "melanoma",
        "benign_keratosis",
        "basal_cell_carcinoma",
        "actinic_keratoses",
        "vascular_lesions",
        "dermatofibroma",
    ]
    predictor = SkinDiseasePredictor(str(model_path), num_classes=len(class_names), class_names=class_names)
    print(f"Loaded model from {model_path}")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Endpoint to predict the class of a skin lesion image."""
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read image")
    result = predictor.predict(img)
    # Always include a disclaimer
    result["disclaimer"] = (
        "This prediction is for research/educational purposes only and is not "
        "a medical diagnosis. Consult a dermatologist for professional advice."
    )
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
