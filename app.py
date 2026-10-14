import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

st.set_page_config(page_title="Image Classifier", page_icon="🔍", layout="wide")

SAMPLES_DIR = Path(__file__).parent / "samples"


@st.cache_resource
def load_model():
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights)
    model.eval()
    return model, weights.transforms(), weights.meta["categories"]


def predict(image, top_k=5):
    model, preprocess, labels = load_model()
    x = preprocess(image).unsqueeze(0)
    start = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)
    elapsed_ms = (time.perf_counter() - start) * 1000
    top_probs, top_idx = probs.topk(top_k)
    results = pd.DataFrame({
        "Class": [labels[i].title() for i in top_idx.tolist()],
        "Confidence (%)": [round(p * 100, 2) for p in top_probs.tolist()],
    })
    return results, elapsed_ms


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🔍 About")
    st.write(
        "This app uses a **pretrained CNN (MobileNetV3-Large)** trained on the "
        "**ImageNet** dataset to recognise **1000 object categories** such as "
        "animals, vehicles, food and everyday objects."
    )
    st.markdown(
        "**How it works**\n"
        "1. Image is resized to 224×224 and normalised\n"
        "2. CNN extracts features through convolution layers\n"
        "3. Softmax layer outputs a probability for each class"
    )
    top_k = st.slider("Number of predictions to show", 3, 10, 5)

# ---------------- Main page ----------------
st.title("Image Classification using CNN")
st.caption("Upload an image or pick a sample, and the model will predict what it contains.")

source = st.radio("Choose input", ["Upload image", "Use sample image", "Camera"], horizontal=True)

image = None
if source == "Upload image":
    file = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")
elif source == "Use sample image":
    samples = sorted(SAMPLES_DIR.glob("*.jpg"))
    choice = st.selectbox("Sample", samples, format_func=lambda p: p.stem.title())
    image = Image.open(choice).convert("RGB")
else:
    photo = st.camera_input("Take a photo")
    if photo:
        image = Image.open(photo).convert("RGB")

if image is not None:
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Input image", width="stretch")
    with right:
        with st.spinner("Classifying..."):
            results, elapsed_ms = predict(image, top_k)
        best = results.iloc[0]
        st.success(f"Prediction: **{best['Class']}**")
        c1, c2 = st.columns(2)
        c1.metric("Confidence", f"{best['Confidence (%)']:.1f}%")
        c2.metric("Inference time", f"{elapsed_ms:.0f} ms")
        st.subheader(f"Top {top_k} predictions")
        st.bar_chart(results, x="Class", y="Confidence (%)", horizontal=True, sort="-Confidence (%)")
        st.dataframe(results, hide_index=True, width="stretch")
else:
    st.info("👆 Provide an image to get started.")
