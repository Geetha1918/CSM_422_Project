"""Streamlit interface for presenting the Smart Waste Classification project."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.data_loader import CLASS_NAMES
from src.neural_network import WasteNeuralNetwork


PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@st.cache_resource
def load_trained_model() -> WasteNeuralNetwork | None:
    """Load the weights produced by train.py, if training has been completed."""
    model_path = OUTPUT_DIR / "model_weights.npz"
    if not model_path.exists():
        return None
    saved = np.load(model_path)
    model = WasteNeuralNetwork(
        input_size=saved["W1"].shape[0],
        hidden_sizes=(saved["W1"].shape[1], saved["W2"].shape[1]),
        output_size=saved["W3"].shape[1],
    )
    model.parameters = {name: saved[name] for name in saved.files}
    return model


def prepare_image(image_bytes: bytes) -> tuple[Image.Image, Image.Image, np.ndarray]:
    """Display the original image and create the 32 x 32 grayscale model input."""
    original = Image.open(BytesIO(image_bytes)).convert("RGB")
    resized = original.convert("L").resize((32, 32))
    features = np.asarray(resized, dtype=np.float64).reshape(1, -1) / 255.0
    return original, resized, features


def count_images(data_dir: Path) -> dict[str, int]:
    return {
        class_name: len([file for file in (data_dir / class_name).glob("*") if file.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        for class_name in CLASS_NAMES
    }


st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️", layout="wide")
st.title("♻️ Smart Waste Classification")
st.caption("An educational deep-learning project for waste segregation")

st.sidebar.header("Project details")
st.sidebar.write("Model: Multilayer Perceptron (NumPy)")
st.sidebar.write("Input: 32 × 32 grayscale image = 1024 pixel values")
st.sidebar.write("Architecture: 1024 → 64 → 32 → 3")
st.sidebar.write("Output: Biodegradable, Recyclable, Non-recyclable")

predict_tab, results_tab, explanation_tab = st.tabs(["Predict Waste Type", "Training Results", "How It Works"])

with predict_tab:
    st.subheader("Upload a waste image")
    st.write("Choose a JPG or PNG image. The trained model will predict its disposal category.")
    upload = st.file_uploader("Waste image", type=["jpg", "jpeg", "png"])
    model = load_trained_model()
    if model is None:
        st.warning("No trained weights found. Run `python train.py --data-dir data ...` before using predictions.")
    elif upload is not None:
        original, resized, features = prepare_image(upload.getvalue())
        probabilities, _ = model.forward(features)
        probabilities = probabilities[0]
        predicted_index = int(np.argmax(probabilities))
        columns = st.columns(2)
        with columns[0]:
            st.image(original, caption="Uploaded image", use_container_width=True)
        with columns[1]:
            st.image(resized.resize((256, 256)), caption="Model input: 32 × 32 grayscale", use_container_width=True)
        st.success(f"Prediction: **{CLASS_NAMES[predicted_index].replace('_', ' ').title()}**")
        st.metric("Confidence", f"{probabilities[predicted_index] * 100:.1f}%")
        st.bar_chart({name.replace("_", " ").title(): float(probability) for name, probability in zip(CLASS_NAMES, probabilities)})
    elif model is not None:
        st.info("Upload an image to test the trained model.")

with results_tab:
    st.subheader("Dataset and training output")
    train_dir = PROJECT_ROOT / "data" / "train"
    test_dir = PROJECT_ROOT / "data" / "test"
    if train_dir.exists() and test_dir.exists():
        train_counts = count_images(train_dir)
        test_counts = count_images(test_dir)
        st.dataframe({"Class": [name.replace("_", " ").title() for name in CLASS_NAMES],
                      "Training images": [train_counts[name] for name in CLASS_NAMES],
                      "Testing images": [test_counts[name] for name in CLASS_NAMES]}, hide_index=True, use_container_width=True)
    else:
        st.info("Run `python download_dataset.py --data-dir data` to display full dataset counts.")

    history_path = OUTPUT_DIR / "training_history.png"
    confusion_path = OUTPUT_DIR / "confusion_matrix.png"
    if history_path.exists() and confusion_path.exists():
        chart_columns = st.columns(2)
        chart_columns[0].image(str(history_path), caption="Training loss by epoch", use_container_width=True)
        chart_columns[1].image(str(confusion_path), caption="Test confusion matrix", use_container_width=True)
    else:
        st.info("Training charts appear here after running train.py.")

with explanation_tab:
    st.subheader("Information flow in the neural network")
    st.code("Image (32 × 32) → flatten (1024) → Dense 64 + ReLU → Dense 32 + ReLU → Dense 3 + Softmax")
    st.markdown("""
1. The uploaded image is converted to grayscale and resized to **32 × 32**.
2. Its 1,024 pixel values form the input vector.
3. Two hidden layers learn patterns using **ReLU** activation.
4. The output layer uses **Softmax** to produce three probabilities.
5. During training, backpropagation calculates gradients and gradient descent updates weights and biases.
    """)
    st.warning("This is an academic prototype. Check local waste-management guidelines before disposing of any item.")
