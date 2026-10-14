import html
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

st.set_page_config(page_title="Image Classifier · CNN", page_icon="◐", layout="wide")

SAMPLES_DIR = Path(__file__).parent / "samples"
TOP_K = 5

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #090b10;
  --panel: #10131b;
  --line: rgba(255,255,255,0.08);
  --text: #e6e9f0;
  --muted: #8a90a0;
  --blue: #8fb8ff;
}

/* page */
.stApp {
  background:
    radial-gradient(600px 400px at 12% 8%, rgba(90,130,230,0.16), transparent 70%),
    radial-gradient(500px 500px at 90% 70%, rgba(90,130,230,0.08), transparent 70%),
    radial-gradient(rgba(255,255,255,0.035) 1px, transparent 1px) 0 0 / 28px 28px,
    var(--bg);
  color: var(--text);
  font-family: 'Inter', sans-serif;
}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"] { display: none !important; }
.block-container { max-width: 1100px; padding: 3.5rem 2rem 4rem; }
p, li, label, span, div { font-family: 'Inter', sans-serif; }

/* typography helpers */
.eyebrow {
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.22em;
  text-transform: uppercase; color: var(--muted); display: flex; align-items: center; gap: 0.9rem;
}
.eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--blue);
  box-shadow: 0 0 10px var(--blue); }
.eyebrow .rule { flex: 0 0 48px; height: 1px; background: var(--line); }
.hero-title {
  font-family: 'Instrument Serif', serif; font-weight: 400; color: var(--text);
  font-size: clamp(3.6rem, 9vw, 7rem); line-height: 0.92; letter-spacing: -0.02em; margin: 1.6rem 0 1.4rem;
}
.hero-title em { display: block; padding-left: 0.35em; }
.lede { color: #c2c7d2; font-size: 1.05rem; line-height: 1.65; max-width: 520px; font-weight: 300; }
.section-head { display: flex; align-items: baseline; gap: 1.4rem; margin: 4.5rem 0 1.6rem;
  padding-top: 2rem; border-top: 1px solid var(--line); }
.section-head .num { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--blue); }
.section-head .h2 { font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 2.6rem;
  color: var(--text); margin: 0; padding: 0; letter-spacing: -0.01em; }

/* input pills */
[data-testid="stSegmentedControl"] button, [data-testid="stButtonGroup"] button {
  border-radius: 999px !important; border: 1px solid var(--line) !important; background: transparent !important;
  color: var(--muted) !important; font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important; letter-spacing: 0.14em; text-transform: uppercase; padding: 0.5rem 1.1rem !important;
  margin-right: 0.5rem;
}
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] {
  background: var(--blue) !important; color: #090b10 !important; border-color: var(--blue) !important;
}
[data-testid="stWidgetLabel"] p { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem !important;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted) !important; }

/* uploader / selectbox / camera */
[data-testid="stFileUploaderDropzone"] { background: var(--panel); border: 1px dashed rgba(255,255,255,0.14);
  border-radius: 18px; padding: 2rem; }
[data-testid="stFileUploaderDropzone"] button { border-radius: 999px; }
[data-baseweb="select"] > div { background: var(--panel) !important; border-radius: 999px !important;
  border: 1px solid var(--line) !important; }
[data-testid="stCameraInput"] > div { border-radius: 18px; overflow: hidden; }

/* image */
[data-testid="stImage"] img { border-radius: 20px; border: 1px solid var(--line); }
[data-testid="stImageCaption"], [data-testid="caption"] { font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); }

/* result panel */
.result-label { font-family: 'Instrument Serif', serif; font-style: italic; font-size: clamp(2.4rem, 5vw, 3.6rem);
  line-height: 1; color: var(--text); margin: 0.9rem 0 1.8rem; }
.stat { display: flex; align-items: center; gap: 1rem; padding: 0.8rem 0; }
.stat .k { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--muted); white-space: nowrap; }
.stat .fill { flex: 1; height: 1px; background: var(--line); }
.stat .v { color: var(--text); font-size: 1rem; white-space: nowrap; }
.bars { margin-top: 2rem; }
.bar-row { display: grid; grid-template-columns: 1.6rem 1fr auto; gap: 0.9rem; align-items: center; padding: 0.55rem 0; }
.bar-row .rank { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--muted); }
.bar-row .name { color: var(--text); font-size: 0.95rem; }
.bar-row .pct { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--muted); }
.bar-track { grid-column: 2 / 4; height: 3px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; background: rgba(143,184,255,0.35); border-radius: 3px; }
.bar-row.top .bar-fill { background: var(--blue); box-shadow: 0 0 12px rgba(143,184,255,0.5); }
.bar-row.top .pct { color: var(--blue); }

.empty { border: 1px dashed var(--line); border-radius: 20px; padding: 3rem 2rem; text-align: center;
  color: var(--muted); font-size: 0.95rem; }

/* how it works */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }
.step { border: 1px solid var(--line); border-radius: 20px; padding: 1.5rem; background: rgba(16,19,27,0.6); }
.step .n { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--blue); letter-spacing: 0.2em; }
.step .h4 { font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 1.6rem; color: var(--text);
  margin: 0.8rem 0 0.4rem; padding: 0; }
.step p { color: var(--muted); font-size: 0.9rem; line-height: 1.6; margin: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.6rem; }
.chip { border: 1px solid var(--line); border-radius: 999px; padding: 0.45rem 1rem; font-size: 0.82rem; color: #c2c7d2; }

.foot { margin-top: 4rem; padding-top: 1.6rem; border-top: 1px solid var(--line); display: flex;
  justify-content: space-between; flex-wrap: wrap; gap: 1rem; font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); }

@media (max-width: 640px) {
  .block-container { padding: 2rem 1rem 3rem; }
  [data-testid="stSegmentedControl"] button, [data-testid="stButtonGroup"] button {
    padding: 0.45rem 0.8rem !important; letter-spacing: 0.08em; margin-right: 0.3rem; }
  .section-head { margin-top: 3rem; }
  .eyebrow { flex-wrap: wrap; gap: 0.5rem 0.9rem; letter-spacing: 0.16em; }
  .eyebrow .rule { display: none; }
}
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


def stat(key, value):
    return f'<div class="stat"><span class="k">{key}</span><span class="fill"></span><span class="v">{value}</span></div>'


def result_html(results, elapsed_ms):
    name, conf = results[0]
    parts = [
        '<div class="eyebrow"><span class="dot"></span>Prediction</div>',
        f'<div class="result-label">{html.escape(name)}</div>',
        stat("Confidence", f"{conf:.1f}%"),
        stat("Inference time", f"{elapsed_ms:.0f} ms"),
        stat("Model", "MobileNetV3-Large"),
        '<div class="bars"><div class="eyebrow">Top 5 predictions</div>',
    ]
    for rank, (label, p) in enumerate(results, start=1):
        cls = "bar-row top" if rank == 1 else "bar-row"
        parts.append(
            f'<div class="{cls}"><span class="rank">0{rank}</span>'
            f'<span class="name">{html.escape(label)}</span><span class="pct">{p:.1f}%</span>'
            f'<span></span><div class="bar-track"><div class="bar-fill" style="width:{p:.2f}%"></div></div></div>'
        )
    parts.append("</div>")
    return "".join(parts)


st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Hero ----------------
st.markdown(
    '<div class="eyebrow"><span class="dot"></span>Deep Learning · CNN<span class="rule"></span>1000 ImageNet classes</div>'
    '<div class="hero-title">Image <em>Classifier</em></div>'
    '<p class="lede">Upload any photo and a pretrained convolutional neural network will tell you what it sees, '
    "along with how confident it is about its top guesses.</p>",
    unsafe_allow_html=True,
)

# ---------------- Classifier ----------------
st.markdown('<div class="section-head"><span class="num">01</span><div class="h2">Try it</div></div>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    source = st.segmented_control("Input", ["Upload", "Sample", "Camera"], default="Upload")
    image = None
    if source == "Upload":
        file = st.file_uploader("Image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if file:
            image = Image.open(file).convert("RGB")
    elif source == "Sample":
        samples = sorted(SAMPLES_DIR.glob("*.jpg"))
        choice = st.selectbox("Sample image", samples, format_func=lambda p: p.stem.title())
        image = Image.open(choice).convert("RGB")
    elif source == "Camera":
        photo = st.camera_input("Camera", label_visibility="collapsed")
        if photo:
            image = Image.open(photo).convert("RGB")

    if image is not None:
        st.image(image, caption="Input image", width="stretch")

with right:
    if image is not None:
        with st.spinner("Classifying..."):
            results, elapsed_ms = predict(image)
        st.markdown(result_html(results, elapsed_ms), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty">Choose an image on the left.<br>The prediction will appear here.</div>',
            unsafe_allow_html=True,
        )

# ---------------- How it works ----------------
st.markdown('<div class="section-head"><span class="num">02</span><div class="h2">How it works</div></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="steps">'
    '<div class="step"><span class="n">01 / Preprocess</span><div class="h4">Resize &amp; normalise</div>'
    "<p>The image is resized to 224×224 pixels and normalised with ImageNet mean and standard deviation.</p></div>"
    '<div class="step"><span class="n">02 / Extract</span><div class="h4">Convolution layers</div>'
    "<p>MobileNetV3 passes the image through stacked convolution blocks that learn edges, textures and shapes.</p></div>"
    '<div class="step"><span class="n">03 / Classify</span><div class="h4">Softmax output</div>'
    "<p>A final dense layer with softmax gives a probability for each of the 1000 classes; the highest wins.</p></div>"
    "</div>"
    '<div class="chips"><span class="chip">Python</span><span class="chip">PyTorch</span>'
    '<span class="chip">Torchvision</span><span class="chip">MobileNetV3</span>'
    '<span class="chip">ImageNet</span><span class="chip">Streamlit</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="foot"><span>SIC · CH-8 Project</span><span>Built by Vishnu</span></div>',
    unsafe_allow_html=True,
)
