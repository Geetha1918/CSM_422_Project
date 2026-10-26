"""A custom presentation interface for the Smart Waste Classification project."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.data_loader import CLASS_NAMES
from src.neural_network import WasteNeuralNetwork

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@st.cache_resource
def load_model() -> WasteNeuralNetwork | None:
    weight_file = OUTPUTS / "model_weights.npz"
    if not weight_file.exists():
        return None
    saved = np.load(weight_file)
    model = WasteNeuralNetwork(saved["W1"].shape[0], (saved["W1"].shape[1], saved["W2"].shape[1]), saved["W3"].shape[1])
    model.parameters = {key: saved[key] for key in saved.files}
    return model


def prepare_image(contents: bytes) -> tuple[Image.Image, Image.Image, np.ndarray]:
    original = Image.open(BytesIO(contents)).convert("RGB")
    small = original.convert("L").resize((32, 32))
    pixels = np.asarray(small, dtype=np.float64).reshape(1, -1) / 255.0
    return original, small, pixels


def counts(split: str) -> list[int]:
    return [len([f for f in (ROOT / "data" / split / name).glob("*") if f.suffix.lower() in IMAGE_SUFFIXES]) for name in CLASS_NAMES]


def label(index: int) -> str:
    return CLASS_NAMES[index].replace("_", " ").title()


st.set_page_config(page_title="EcoSort Lab", page_icon="♻️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.stApp { background: #f4f7f2; color: #17332b; }
header[data-testid="stHeader"] { background: rgba(244,247,242,.92); }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1180px; }
.hero { background: linear-gradient(125deg,#0d4a3a,#16735d 62%,#d4a72c); padding:2.25rem 2.5rem; border-radius:28px; color:#fff; margin-bottom:1.5rem; }
.hero h1 { margin:0; font-size:2.7rem; letter-spacing:-.04em; }
.hero p { margin:.5rem 0 0; color:#dbf5e8; font-size:1.08rem; }
.tag { display:inline-block; margin-bottom:.65rem; padding:.28rem .75rem; border:1px solid #a7d9be; border-radius:999px; color:#dffbe8; font-size:.82rem; }
.panel { background:#fff; border:1px solid #d9e6de; border-radius:20px; padding:1.2rem 1.35rem; box-shadow:0 6px 20px rgba(27,70,55,.06); }
.small-note { color:#5b6b65; font-size:.9rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #d9e6de; border-radius:16px; padding:.9rem; }
div[data-testid="stMetricLabel"] p, div[data-testid="stMetricValue"] { color:#17332b !important; }
div[data-testid="stRadio"] label p { color:#17332b !important; font-weight:600; }
div[data-testid="stAlert"] p, div[data-testid="stAlert"] div { color:#17332b !important; }
div[data-testid="stAlert"] { border-radius:14px; }
div[data-testid="stFileUploader"] { background:#fff; border:2px dashed #75ac8a; border-radius:18px; padding:.45rem; }
</style>
<section class="hero"><div class="tag">INTELLIGENT WASTE SORTING • LIVE DEMO</div><h1>EcoSort Lab ♻</h1><p>Explore how a feedforward neural network helps classify everyday waste.</p></section>
""", unsafe_allow_html=True)

train_counts = counts("train") if (ROOT / "data" / "train").exists() else [0, 0, 0]
test_counts = counts("test") if (ROOT / "data" / "test").exists() else [0, 0, 0]
metrics = st.columns(4)
metrics[0].metric("Training images", f"{sum(train_counts):,}")
metrics[1].metric("Test images", f"{sum(test_counts):,}")
metrics[2].metric("Input vector", "1,024")
metrics[3].metric("Output classes", "3")

page = st.radio("Navigation", ["Classify an item", "Experiment results", "Neural network flow"], horizontal=True, label_visibility="collapsed")

if page == "Classify an item":
    st.subheader("Try a waste image")
    st.write("Upload one clear photo of an item against a plain background. JPG, JPEG and PNG are supported.")
    upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    model = load_model()
    if model is None:
        st.error("Model weights are not available. Run the training command before using this page.")
    elif upload is None:
        st.info("For a fair demo, use one photographed item. Transparent PNGs, watermarked images, dark photos, and multiple objects may be unreliable.")
    else:
        original, small, pixels = prepare_image(upload.getvalue())
        probabilities, _ = model.forward(pixels)
        probabilities = probabilities[0]
        result = int(np.argmax(probabilities))
        confidence = float(probabilities[result])
        image_columns = st.columns([1.3, .7])
        image_columns[0].image(original, caption="Uploaded item", use_container_width=True)
        image_columns[1].image(small.resize((256, 256)), caption="32 × 32 grayscale input", use_container_width=True)
        image_columns[1].caption("The model receives 1,024 normalized pixel values.")
        result_columns = st.columns([.55, .45])
        with result_columns[0]:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.subheader(label(result))
            st.metric("Model confidence", f"{confidence * 100:.1f}%")
            if confidence < .70:
                st.warning("Low-confidence result — manual verification is recommended.")
            else:
                st.success("High-confidence model result. Still check local waste rules.")
            st.markdown("</div>", unsafe_allow_html=True)
        with result_columns[1]:
            st.markdown("**Probability comparison**")
            st.bar_chart({label(i): float(value) for i, value in enumerate(probabilities)})
            st.caption("The bars are Softmax probabilities and add up to 1.")

elif page == "Experiment results":
    st.subheader("Dataset and model result")
    st.dataframe({"Waste category": [label(i) for i in range(3)], "Training images": train_counts, "Testing images": test_counts}, hide_index=True, use_container_width=True)
    st.caption("The non-recyclable class is smaller than the other categories, which can influence prediction quality.")
    history, confusion = OUTPUTS / "training_history.png", OUTPUTS / "confusion_matrix.png"
    if history.exists() and confusion.exists():
        chart_columns = st.columns(2)
        chart_columns[0].image(str(history), caption="Cross-entropy loss during training", use_container_width=True)
        chart_columns[1].image(str(confusion), caption="Test-set confusion matrix", use_container_width=True)
    else:
        st.info("Run model training to create result charts.")

else:
    st.subheader("From image to category")
    st.markdown("<div class='panel'><h3>32 × 32 image → 1,024 inputs → 64 neurons → 32 neurons → 3 probabilities</h3><p class='small-note'>Each batch is processed with matrix multiplication. ReLU is used in hidden layers and Softmax produces the final probabilities.</p></div>", unsafe_allow_html=True)
    flow = st.columns(4)
    flow[0].info("**1. Preprocess**\nResize and grayscale the image.")
    flow[1].info("**2. Feedforward**\nPass pixels through two hidden layers.")
    flow[2].info("**3. Softmax**\nConvert scores to probabilities.")
    flow[3].info("**4. Learn**\nBackpropagation updates weights.")
    st.warning("This is an educational prototype, not an authoritative disposal decision system.")
