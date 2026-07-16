# ✏️ Handwritten Digit Recognizer (MNIST + CNN + Streamlit)

A web app that recognizes hand-drawn or uploaded digit images (0–9) using a
Convolutional Neural Network trained on the MNIST dataset. Draw a digit on a
canvas or upload a photo/scan, and the app preprocesses it to match MNIST's
format before predicting.

## 🎯 Demo Flow
1. Draw a digit (or upload an image of one)
2. The image is cleaned, cropped, resized, and centered to match MNIST's format
3. The CNN predicts the digit (0–9) with a confidence score

## 🧠 Model
- **Architecture:** CNN (Conv2D → MaxPooling → Dense layers)
- **Dataset:** MNIST (60,000 training / 10,000 test images, 28×28 grayscale)
- **Input format:** 28×28 grayscale image, pixel values normalized to `[0, 1]`

## 📁 Project Structure
```
.
├── app.py                       # Streamlit app (UI + preprocessing + inference)
├── requirements.txt             # Python dependencies
├── hand-written-digit-v2.h5     # Trained model (add your own file here)
└── README.md
```

## ⚙️ Preprocessing Pipeline
Real-world images (photos, canvas drawings, scans) don't look like clean MNIST
data out of the box, so each input goes through:

1. **Transparency handling** — composites transparent PNGs onto a white background
2. **Polarity correction** — detects background vs. stroke color using border
   pixels, and inverts if needed (MNIST = white digit on black background)
3. **Otsu's thresholding** — adaptively binarizes the image based on its own
   contrast, instead of a fixed brightness cutoff
4. **Morphological closing** — fills small gaps in thin or broken strokes
5. **Largest-contour cropping** — isolates the digit and ignores noise specks
6. **Aspect-ratio-preserving resize** — scales the digit so its longest side
   is 20px (matching the MNIST convention)
7. **Padding to 28×28** — centers the resized digit on a black canvas
8. **Center-of-mass alignment** — shifts the digit so its "weight" is centered,
   exactly how the original MNIST dataset was constructed
9. **Normalization** — scales pixel values from `[0, 255]` to `[0, 1]`

## 🐛 The Bug That Broke Everything (and the fix)
Early versions of this project predicted the same wrong digit for every input,
regardless of preprocessing quality. Root cause: a **train/test normalization
mismatch**.

- **Training** fed the model raw pixel values (`0–255`), with no normalization step.
- **Inference** divided pixel values by 255 (`0–1`) before predicting, since that's
  the standard convention.

The model's learned weights only made sense for `0–255` inputs — feeding it
`0–1` values made every image look nearly blank to the network, collapsing
predictions to a single class.

**Fix:** retrain with matching normalization on both sides:
```python
x_train_norm = x_train.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_norm, y_train, epochs=10, validation_split=0.1)
model.save('hand-written-digit-v2.h5')
```
**Lesson:** whatever preprocessing you apply at inference time (resizing,
normalization, etc.) must exactly match what the model saw during training.

## 🚀 Run Locally
```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`).

> **Note:** place your trained `hand-written-digit-v2.h5` file in the project
> root before running. Update `MODEL_PATH` in `app.py` if you name or place
> it differently.

## ☁️ Deploy on Streamlit Community Cloud
1. Push this project (including the `.h5` model file) to a GitHub repo.
   - If the model file is larger than 25MB, use [Git LFS](https://git-lfs.com/)
     or host it externally (Hugging Face Hub / S3) and download it at startup.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo/branch, set the main file to `app.py`
4. Deploy — dependencies install automatically from `requirements.txt`

## 🛠️ Tech Stack
- **Streamlit** — web app framework
- **TensorFlow / Keras** — model training and inference
- **OpenCV** — image preprocessing (thresholding, contours, morphology)
- **Pillow** — image loading and transparency handling
- **NumPy** — array operations

## 📌 Requirements
```
streamlit>=1.32
streamlit-drawable-canvas>=0.9.3
tensorflow-cpu>=2.15
opencv-python-headless>=4.9
numpy>=1.26
pillow>=10.0
```

## 📄 License
MIT — free to use, modify, and distribute.
