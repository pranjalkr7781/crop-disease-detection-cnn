"""Demo web app. Run: streamlit run app.py
Needs models/best_model.keras and models/class_names.json (from your Colab results.zip).
"""
import numpy as np
import streamlit as st
from PIL import Image

import config
from src.inference import load_model, predict_image
from src.gradcam import make_heatmap, overlay

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿")
st.title("🌿 Crop Disease Detection")
st.write("Upload a clear photo of a single leaf (tomato or potato) to check for disease.")


@st.cache_resource
def get_model():
    return load_model()


try:
    model, class_names = get_model()
except Exception as e:
    st.error("Model not found. Copy best_model.keras and class_names.json into the 'models' folder.")
    st.stop()

source = st.radio("Image source", ["Upload a file", "Use camera"], horizontal=True)
if source == "Upload a file":
    file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input("Take a photo of the leaf")

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Your image", use_container_width=True)

    with st.spinner("Analysing..."):
        result = predict_image(image)

    st.subheader(f"Result: {result['prediction'].replace('_', ' ')}")
    st.metric("Confidence", f"{result['confidence']:.1%}")

    if result["low_confidence"]:
        st.warning("Low confidence. Retake the photo: good light, one leaf, in focus, plain background.")

    st.info(f"**Suggested next step:** {result['advice']}")
    st.caption("This is a screening aid, not a replacement for an agronomist.")

    st.write("**Top predictions**")
    for r in result["top_k"]:
        st.write(f"{r['label'].replace('_', ' ')}: {r['confidence']:.1%}")
        st.progress(min(1.0, r["confidence"]))

    with st.expander("Show where the model looked (Grad-CAM)"):
        batch = np.expand_dims(
            np.asarray(image.resize(config.IMG_SIZE), dtype=np.float32), 0)
        cam, _ = make_heatmap(model, batch, class_names.index(result["prediction"]))
        st.image(overlay(image, cam), caption="Red = areas that influenced the decision",
                 use_container_width=True)
