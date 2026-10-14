"""Reusable prediction logic. The backend can import predict_image() directly."""
import json
import numpy as np
import tensorflow as tf
from PIL import Image

import config
from src.advice import get_advice

CONFIDENCE_THRESHOLD = 0.60

_model = None
_class_names = None


def load_model(path=None):
    global _model, _class_names
    if _model is None:
        path = path or config.MODEL_DIR / "best_model.keras"
        _model = tf.keras.models.load_model(path)
        with open(config.MODEL_DIR / "class_names.json") as f:
            _class_names = json.load(f)
    return _model, _class_names


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(config.IMG_SIZE)
    return np.expand_dims(np.asarray(image, dtype=np.float32), 0)  # model rescales internally


def predict_image(image, top_k=3):
    """image: PIL.Image or file path. Returns a dict ready to send as JSON."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    model, class_names = load_model()
    probs = model.predict(preprocess(image), verbose=0)[0]
    order = np.argsort(probs)[::-1][:top_k]
    top = [{"label": class_names[i], "confidence": float(probs[i])} for i in order]
    best = top[0]
    return {
        "prediction": best["label"],
        "confidence": best["confidence"],
        "low_confidence": best["confidence"] < CONFIDENCE_THRESHOLD,
        "advice": get_advice(best["label"]),
        "top_k": top,
    }
