import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
from tensorflow.keras.models import load_model

# ==================== CONFIG ====================
MODEL_PATH = "hand-written-digit-v2.h5"   # <-- put your trained .h5 file next to this app.py
# ==================================================


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Preprocessing pipeline (same one we validated in the notebook)
# ---------------------------------------------------------------------------
def load_image_from_pil(pil_img: Image.Image) -> np.ndarray:
    """Handle transparency correctly, then convert to grayscale numpy array."""
    if pil_img.mode in ("RGBA", "LA") or (
        pil_img.mode == "P" and "transparency" in pil_img.info
    ):
        pil_img = pil_img.convert("RGBA")
        bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(bg, pil_img)

    pil_img = pil_img.convert("L")
    return np.array(pil_img)


def center_by_mass(img: np.ndarray) -> np.ndarray:
    cy, cx = np.array(img.shape) / 2.0
    m = cv2.moments(img)
    if m["m00"] == 0:
        return img
    shift_x = int(round(cx - (m["m10"] / m["m00"])))
    shift_y = int(round(cy - (m["m01"] / m["m00"])))
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def preprocess_for_mnist(img: np.ndarray, debug: bool = False):
    steps = {}
    steps["0_raw"] = img.copy()

    # Determine polarity using BORDER pixels
    border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    if border.mean() > 127:
        img = 255 - img
    steps["1_polarity_fixed"] = img.copy()

    # Otsu's threshold — adapts to actual contrast
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps["2_otsu_threshold"] = img.copy()

    # Morphological closing — fill small gaps in strokes
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    steps["3_morph_close"] = img.copy()

    # Bounding box crop via largest contour (ignores noise specks)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, steps
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    digit = img[y : y + h, x : x + w]
    steps["4_cropped"] = digit.copy()

    # Resize preserving aspect ratio, longest side -> 20px (MNIST convention)
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))
    interp = cv2.INTER_AREA if (w > new_w or h > new_h) else cv2.INTER_CUBIC
    digit = cv2.resize(digit, (new_w, new_h), interpolation=interp)

    # Pad to 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = digit
    steps["5_padded"] = canvas.copy()

    # Center by mass (real MNIST alignment)
    canvas = center_by_mass(canvas)
    steps["6_centered"] = canvas.copy()

    return canvas, steps


def predict(canvas: np.ndarray, model):
    model_input = canvas.astype("float32") / 255.0
    model_input = model_input.reshape(1, 28, 28, 1)
    preds = model.predict(model_input, verbose=0)[0]
    predicted_label = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)
    return predicted_label, confidence, preds


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="✏️", layout="centered")
st.title("✏️ Handwritten Digit Recognizer")
st.caption("Draw a digit or upload an image — the CNN was trained on MNIST (0-9).")

tab_draw, tab_upload = st.tabs(["🖌️ Draw", "📁 Upload Image"])

raw_gray_img = None  # will hold the grayscale numpy array to preprocess

with tab_draw:
    st.write("Draw a single digit below (thick strokes work best):")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict from drawing", use_container_width=True):
        if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
            pil_img = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
            raw_gray_img = load_image_from_pil(pil_img)
        else:
            st.warning("Please draw a digit first.")

with tab_upload:
    uploaded_file = st.file_uploader("Upload a PNG/JPG image of a digit", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None and st.button("Predict from upload", use_container_width=True):
        pil_img = Image.open(uploaded_file)
        raw_gray_img = load_image_from_pil(pil_img)

# ---------------------------------------------------------------------------
# Run prediction if we have an image
# ---------------------------------------------------------------------------
if raw_gray_img is not None:
    processed, steps = preprocess_for_mnist(raw_gray_img)

    if processed is None:
        st.error("No digit detected — try a clearer image with more contrast.")
    else:
        model = get_model()
        label, confidence, all_probs = predict(processed, model)

        col1, col2 = st.columns(2)
        with col1:
            st.image(raw_gray_img, caption="Original (grayscale)", width=200)
        with col2:
            st.image(processed, caption="Preprocessed (28x28, model input)", width=200)

        st.markdown(f"## Prediction: **{label}**")
        st.progress(min(int(confidence), 100))
        st.write(f"Confidence: **{confidence:.2f}%**")

        with st.expander("See probability for every digit"):
            st.bar_chart({str(i): float(p) for i, p in enumerate(all_probs)})

        with st.expander("See preprocessing steps (debug view)"):
            cols = st.columns(len(steps))
            for c, (title, im) in zip(cols, steps.items()):
                c.image(im, caption=title, use_container_width=True)

st.divider()
st.caption(
    "Model expects a clean 28x28 grayscale digit, white stroke on black background, "
    "normalized to 0-1 — matching the MNIST training distribution."
)
