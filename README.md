# MNIST Handwritten Digit Recognizer (Streamlit)

A Streamlit app that lets a user **draw** a digit on a canvas or **upload** an
image, runs it through the same preprocessing pipeline we validated in the
notebook (polarity fix → Otsu threshold → morphological close → largest-contour
crop → aspect-preserving resize → pad to 28x28 → center-of-mass alignment →
normalize), and predicts the digit with your trained CNN.

## Files
- `app.py` — the Streamlit app
- `requirements.txt` — pinned dependencies
- `hand-written-digit-v2.h5` — **you need to add this yourself** (your retrained,
  properly-normalized model). Place it in the same folder as `app.py`, or
  update `MODEL_PATH` in `app.py` if you name/place it differently.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploy on Streamlit Community Cloud
1. Push this folder (including your `.h5` model file) to a GitHub repo.
   - If the model file is large (>25MB), use [Git LFS](https://git-lfs.com/)
     or host it elsewhere (e.g. Hugging Face Hub / S3) and download it at
     startup inside `get_model()` instead of bundling it in the repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, pick your repo/branch, and set the main file
   path to `app.py`.
3. Deploy — it installs `requirements.txt` automatically and gives you a
   public URL.

## Deploy on other platforms
- **Hugging Face Spaces**: create a Space with SDK = Streamlit, push the same
  files.
- **Render / Railway / Fly.io**: any platform that can run
  `streamlit run app.py` behind their provided `$PORT` works; add a `Procfile`
  or start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Notes
- The model **must** have been trained on inputs normalized to `[0, 1]`
  (`x_train.astype('float32') / 255.0`). If you deploy the old, un-normalized
  model, predictions will be wrong regardless of how good the preprocessing is.
- The canvas defaults to a black background with white strokes (matching
  MNIST's white-digit-on-black-background convention), so drawings usually
  don't need inversion — the pipeline still auto-detects and fixes polarity
  either way, so uploaded photos of digits on paper work too.
