import html
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

st.set_page_config(page_title="Image Classifier", page_icon="🔍", layout="centered")

SAMPLES_DIR = Path(__file__).parent / "samples"
TOP_K = 5

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #f6f7fb;
  --card: #ffffff;
  --ink: #14151a;
  --muted: #6b7080;
  --line: #e6e8ef;
  --accent: #4f5bff;
  --accent-soft: #eceeff;
}

html, body, .stApp, p, span, div, label, button, input {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stApp { background: var(--bg); color: var(--ink); }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] * { color: #fff !important; }
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
.block-container { max-width: 980px; padding: 3rem 1.5rem 3rem; }

.title { font-size: clamp(2.2rem, 6vw, 3.2rem); font-weight: 800; letter-spacing: -0.03em;
  line-height: 1.05; color: var(--ink); margin: 0; }
.title span { color: var(--accent); }
.subtitle { color: var(--muted); font-size: 1rem; margin: 0.6rem 0 1.2rem; }
.tags { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 2.2rem; }
.tag { background: var(--card); border: 1px solid var(--line); border-radius: 999px;
  padding: 0.3rem 0.8rem; font-size: 0.8rem; color: var(--muted); font-weight: 500; }

/* input pills */
[data-testid="stButtonGroup"] button {
  border-radius: 999px !important; border: 1px solid var(--line) !important; background: var(--card) !important;
  color: var(--muted) !important; font-weight: 600 !important; padding: 0.45rem 1.1rem !important; margin-right: 0.4rem;
}
[data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important;
}
[data-testid="stFileUploaderDropzone"] { background: var(--card); border: 2px dashed var(--line); border-radius: 18px; }
[data-baseweb="select"] > div { border-radius: 12px !important; background: var(--card) !important; }
[data-testid="stImage"] img { border-radius: 18px; }

/* result card */
.card { background: var(--card); border: 1px solid var(--line); border-radius: 22px; padding: 1.6rem;
  box-shadow: 0 10px 30px rgba(20, 21, 26, 0.05); }
.label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.pred { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0.2rem 0 1rem; color: var(--ink); }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; margin-bottom: 1.4rem; }
.stat { background: var(--accent-soft); border-radius: 14px; padding: 0.8rem 1rem; }
.stat b { display: block; font-size: 1.4rem; font-weight: 800; color: var(--accent); }
.stat small { color: var(--muted); font-size: 0.78rem; font-weight: 500; }
.row { margin-top: 0.8rem; }
.row-top { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.3rem; }
.row-top span:last-child { color: var(--muted); font-weight: 600; }
.track { height: 8px; background: #f0f1f6; border-radius: 99px; overflow: hidden; }
.fill { height: 100%; background: #c7cbff; border-radius: 99px; }
.row.first .fill { background: var(--accent); }
.empty { background: var(--card); border: 2px dashed var(--line); border-radius: 22px; padding: 4rem 1.5rem;
  text-align: center; color: var(--muted); }
.empty div { font-size: 2.2rem; margin-bottom: 0.4rem; }
</style>
"""


@st.cache_resource
def load_model():
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights)
    model.eval()
    return model, weights.transforms(), weights.meta["categories"]


def predict(image, top_k=TOP_K):
    model, preprocess, labels = load_model()
    x = preprocess(image).unsqueeze(0)
    start = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)
    elapsed_ms = (time.perf_counter() - start) * 1000
    top_probs, top_idx = probs.topk(top_k)
    results = [(labels[i].title(), p * 100) for i, p in zip(top_idx.tolist(), top_probs.tolist())]
    return results, elapsed_ms


def result_card(results, elapsed_ms):
    name, conf = results[0]
    rows = "".join(
        f'<div class="row{" first" if i == 0 else ""}"><div class="row-top"><span>{html.escape(label)}</span>'
        f'<span>{p:.1f}%</span></div><div class="track"><div class="fill" style="width:{p:.2f}%"></div></div></div>'
        for i, (label, p) in enumerate(results)
    )
    return (
        f'<div class="card"><div class="label">Prediction</div><div class="pred">{html.escape(name)}</div>'
        f'<div class="stats"><div class="stat"><b>{conf:.1f}%</b><small>Confidence</small></div>'
        f'<div class="stat"><b>{elapsed_ms:.0f} ms</b><small>Inference time</small></div></div>'
        f'<div class="label">Top {len(results)}</div>{rows}</div>'
    )


st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="title">Image <span>Classifier</span></div>'
    '<div class="subtitle">Upload a photo and a CNN will tell you what it sees.</div>'
    '<div class="tags"><span class="tag">MobileNetV3</span><span class="tag">PyTorch</span>'
    '<span class="tag">ImageNet · 1000 classes</span></div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    source = st.segmented_control("Input", ["Upload", "Sample", "Camera"], default="Upload",
                                  label_visibility="collapsed")
    image = None
    if source == "Upload":
        file = st.file_uploader("Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if file:
            image = Image.open(file).convert("RGB")
    elif source == "Sample":
        samples = sorted(SAMPLES_DIR.glob("*.jpg"))
        choice = st.selectbox("Sample", samples, format_func=lambda p: p.stem.title(),
                              label_visibility="collapsed")
        image = Image.open(choice).convert("RGB")
    elif source == "Camera":
        photo = st.camera_input("Camera", label_visibility="collapsed")
        if photo:
            image = Image.open(photo).convert("RGB")

    if image is not None:
        st.image(image, width="stretch")

with right:
    if image is not None:
        with st.spinner("Classifying..."):
            results, elapsed_ms = predict(image)
        st.markdown(result_card(results, elapsed_ms), unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty"><div>🖼️</div>Your result will show up here</div>',
                    unsafe_allow_html=True)
