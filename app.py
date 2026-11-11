"""Streamlit interface for the MNIST handwritten digit recognizer."""
from pathlib import Path
import json
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "mnist_cnn.keras"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Digit Recognizer | AI Lab", page_icon="🔢", layout="wide")

st.markdown("""
<style>
.main {background: #f6f8fc;}
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#101c3a,#2457a6); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; margin:0; font-size:2.1rem;}
.hero p {color:#dce8ff; margin:.4rem 0 0 0;}
.metric-card {background:white; padding:1rem; border-radius:14px; border:1px solid #e5eaf3;}
</style>
<div class="hero">
<h1>🔢 Handwritten Digit Recognizer</h1>
<p>Convolutional Neural Network • TensorFlow/Keras • MNIST</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

def prepare_image(image: Image.Image):
    """Convert a handwritten digit image to a centered 28x28 MNIST-like tensor."""
    img = ImageOps.grayscale(image)
    img = ImageOps.autocontrast(img)
    arr = np.asarray(img, dtype=np.uint8)

    # Convert dark ink on a light page to bright ink on a dark background.
    if float(arr.mean()) > 127:
        arr = 255 - arr
    img = Image.fromarray(arr)

    # Remove low-intensity background noise and find the digit's bounding box.
    img = img.point(lambda p: 255 if p > 45 else 0)
    bbox = img.getbbox()
    if bbox is None:
        raise ValueError("No clear digit was detected. Upload an image with one visible digit.")
    img = img.crop(bbox)

    # Preserve aspect ratio and fit the digit inside a 20x20 area.
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    x = (28 - img.width) // 2
    y = (28 - img.height) // 2
    canvas.paste(img, (x, y))

    # Center by intensity-weighted center of mass, similar to MNIST preprocessing.
    pixels = np.asarray(canvas, dtype=np.float32)
    total = pixels.sum()
    if total > 0:
        ys, xs = np.indices(pixels.shape)
        cx = float((xs * pixels).sum() / total)
        cy = float((ys * pixels).sum() / total)
        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))
        shifted = Image.new("L", (28, 28), 0)
        shifted.paste(canvas, (shift_x, shift_y))
        canvas = shifted

    tensor = np.asarray(canvas, dtype=np.float32) / 255.0
    return tensor.reshape(1, 28, 28, 1), canvas

if not MODEL_PATH.exists():
    st.error("The trained model is missing.")
    st.code("python train.py\nstreamlit run app.py", language="powershell")
    st.info("Run the training command first. The first run downloads MNIST and trains the CNN.")
    st.stop()

model = load_model()

left, right = st.columns([1.05, 1])
with left:
    st.subheader("Upload a handwritten digit")
    uploaded = st.file_uploader("Choose a PNG, JPG or JPEG image", type=["png", "jpg", "jpeg"])
    st.caption("Use one clearly written digit, preferably dark ink on a plain light background.")
    if uploaded:
        original = Image.open(uploaded).convert("RGB")
        st.image(original, caption="Uploaded image", use_container_width=True)
        try:
            batch, processed = prepare_image(original)
            probs = model.predict(batch, verbose=0)[0]
            digit = int(np.argmax(probs))
            confidence = float(probs[digit])
            with right:
                st.subheader("Prediction")
                st.markdown(f"<div class='metric-card'><div style='font-size:5rem;font-weight:800;text-align:center;color:#2457a6'>{digit}</div><div style='text-align:center'>Predicted digit</div><h3 style='text-align:center'>{confidence:.2%} confidence</h3></div>", unsafe_allow_html=True)
                st.image(processed.resize((224,224)), caption="Preprocessed 28 × 28 input", clamp=True)
                st.subheader("Class probabilities")
                chart_data = {str(i): float(probs[i]) for i in range(10)}
                st.bar_chart(chart_data)
        except Exception as exc:
            st.error(str(exc))
    else:
        with right:
            st.subheader("How to use")
            st.markdown("1. Write a single digit from **0 to 9**.\n2. Take a clear photo or scan it.\n3. Upload the image.\n4. Review the predicted digit and class probabilities.")
            st.info("This model is trained on MNIST. Images that differ significantly from MNIST handwriting may be misclassified.")

st.divider()
st.subheader("Model evaluation")
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{metrics['test_accuracy']:.2%}")
    c2.metric("Test loss", f"{metrics['test_loss']:.4f}")
    c3.metric("Test images", f"{metrics['test_samples']:,}")
    st.caption("Metrics are generated by train.py on the MNIST test split.")
else:
    st.info("Evaluation metrics appear here after training.")
st.caption("Educational internship project. Predictions are model estimates, not guaranteed results.")
