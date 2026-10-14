import html
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

st.set_page_config(page_title="Image Classifier", page_icon="🔍", layout="wide")

SAMPLES_DIR = Path(__file__).parent / "samples"
TOP_K = 5

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@400;500;600&display=swap');

:root {
  --bg: #0a0f0d;
  --card: #111815;
  --border: rgba(255,255,255,0.07);
  --text: #eef2ef;
  --muted: #8a968f;
  --accent: #3ee08f;
  --accent-2: #2bc4c4;
}

.stApp {
  background: radial-gradient(900px 500px at 50% -10%, rgba(62,224,143,0.12), transparent 70%), var(--bg);
  color: var(--text);
}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"] { display: none !important; }
.block-container { max-width: 1060px; padding: 3rem 1.5rem 3rem; }
.stApp, .stApp p, .stApp label, .stApp button { font-family: 'Manrope', sans-serif; }

/* header */
.top { text-align: center; margin-bottom: 2.2rem; }
.badge { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
  color: var(--accent); background: rgba(62,224,143,0.1); border: 1px solid rgba(62,224,143,0.25); }
.title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: clamp(2.4rem, 6vw, 3.6rem);
  letter-spacing: -0.03em; line-height: 1.05; margin: 1rem 0 0.5rem; color: var(--text); }
.title span { background: linear-gradient(90deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.sub { color: var(--muted); font-size: 1rem; }

/* cards */
.st-key-input_card, .st-key-result_card {
  background: var(--card); border: 1px solid var(--border); border-radius: 22px; padding: 1.4rem;
  box-shadow: 0 20px 50px -30px rgba(0,0,0,0.8);
}
.card-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; color: var(--text); margin-bottom: 0.2rem; }

/* inputs */
[data-testid="stButtonGroup"] button { border-radius: 12px !important; font-weight: 600 !important; }
[data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,0.02); border: 1.5px dashed rgba(255,255,255,0.12);
  border-radius: 16px; }
[data-baseweb="select"] > div { border-radius: 12px !important; }
[data-testid="stImage"] img { border-radius: 16px; }

/* result */
.result-head { display: flex; align-items: center; gap: 1.2rem; margin: 0.6rem 0 1.4rem; }
.ring { --p: 0; flex: 0 0 96px; height: 96px; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(var(--accent) calc(var(--p) * 1%), rgba(255,255,255,0.07) 0); }
.ring div { width: 78px; height: 78px; border-radius: 50%; background: var(--card); display: grid; place-items: center;
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.25rem; color: var(--text); }
.label-small { color: var(--muted); font-size: 0.82rem; font-weight: 500; }
.label-big { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.9rem; line-height: 1.1;
  letter-spacing: -0.02em; color: var(--text); margin: 0.15rem 0 0.4rem; }
.pill { display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
  color: var(--muted); background: rgba(255,255,255,0.05); }
.bar { margin: 0.75rem 0; }
.bar-top { display: flex; justify-content: space-between; font-size: 0.9rem; color: var(--text); margin-bottom: 0.35rem; }
.bar-top b { font-weight: 600; color: var(--muted); }
.bar-track { height: 8px; border-radius: 99px; background: rgba(255,255,255,0.06); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; background: rgba(62,224,143,0.35); }
.bar.first .bar-fill { background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.bar.first .bar-top b { color: var(--accent); }
.empty { text-align: center; padding: 4.5rem 1rem; color: var(--muted); }
.empty .icon { font-size: 2.4rem; margin-bottom: 0.6rem; opacity: 0.7; }

.foot { text-align: center; color: var(--muted); font-size: 0.82rem; margin-top: 2.2rem; }
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


def result_html(results, elapsed_ms):
    name, conf = results[0]
    parts = [
        '<div class="result-head">',
        f'<div class="ring" style="--p:{conf:.1f}"><div>{conf:.0f}%</div></div>',
        f'<div><div class="label-small">Prediction</div><div class="label-big">{html.escape(name)}</div>'
        f'<span class="pill">⚡ {elapsed_ms:.0f} ms</span></div>',
        "</div>",
    ]
    for rank, (label, p) in enumerate(results):
        cls = "bar first" if rank == 0 else "bar"
        parts.append(
            f'<div class="{cls}"><div class="bar-top"><span>{html.escape(label)}</span><b>{p:.1f}%</b></div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{p:.2f}%"></div></div></div>'
        )
    return "".join(parts)


st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="top"><span class="badge">CNN · MobileNetV3</span>'
    '<div class="title">Image <span>Classifier</span></div>'
    '<div class="sub">Drop in a photo. Get the top 5 guesses from 1000 classes.</div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="medium")

with left, st.container(key="input_card"):
    st.markdown('<div class="card-title">Image</div>', unsafe_allow_html=True)
    source = st.segmented_control("Input", ["Upload", "Sample", "Camera"], default="Upload",
                                  label_visibility="collapsed")
    image = None
    if source == "Upload":
        file = st.file_uploader("Image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if file:
            image = Image.open(file).convert("RGB")
    elif source == "Sample":
        samples = sorted(SAMPLES_DIR.glob("*.jpg"))
        choice = st.selectbox("Sample image", samples, format_func=lambda p: p.stem.title(),
                              label_visibility="collapsed")
        image = Image.open(choice).convert("RGB")
    elif source == "Camera":
        photo = st.camera_input("Camera", label_visibility="collapsed")
        if photo:
            image = Image.open(photo).convert("RGB")

    if image is not None:
        st.image(image, width="stretch")

with right, st.container(key="result_card"):
    st.markdown('<div class="card-title">Result</div>', unsafe_allow_html=True)
    if image is not None:
        with st.spinner("Classifying..."):
            results, elapsed_ms = predict(image)
        st.markdown(result_html(results, elapsed_ms), unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty"><div class="icon">🖼️</div>Add an image to see the prediction</div>',
                    unsafe_allow_html=True)

st.markdown('<div class="foot">PyTorch · Streamlit · Built by Vishnu</div>', unsafe_allow_html=True)
