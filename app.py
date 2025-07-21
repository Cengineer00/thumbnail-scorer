import streamlit as st
import torch
from PIL import Image
import numpy as np
import joblib
import open_clip
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


st.set_page_config(page_title="YouTube Thumbnail Scorer", layout="wide")
st.title("📺 YouTube Thumbnail Scorer")
st.markdown("## [View on GitHub](https://github.com/Cengineer00/thumbnail-scorer)", unsafe_allow_html=True)

# Load models once, cache for performance
@st.cache_resource
def load_models():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
    model.to(device).eval()
    regressor = joblib.load("model.lgb")
    logger.info(f"regressor type: {type(regressor)}")
    logger.info(f"model path exists: {os.path.exists('model.lgb')}")
    return model, preprocess, regressor, device

model, preprocess, regressor, device = load_models()

def get_embedding_batch(images):
    tensors = [preprocess(img).to(device) for img in images]
    batch = torch.stack(tensors)
    with torch.no_grad():
        embeddings = model.encode_image(batch)
    return embeddings.cpu().numpy()

def predict_scores(embeddings):
    return regressor.predict(embeddings)

uploaded_files = st.file_uploader(
    "📂 Drag & drop or browse thumbnail images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    images = []
    filenames = []
    for file in uploaded_files:
        try:
            img = Image.open(file).convert("RGB")
            images.append(img)
            filenames.append(file.name)
        except Exception:
            st.warning(f"Skipping {file.name}, could not open image.")

    if images:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        batch_size = 32
        all_embeddings = []
        total = len(images)

        for i in range(0, total, batch_size):
            batch_imgs = images[i:i + batch_size]
            embeddings = get_embedding_batch(batch_imgs)
            all_embeddings.append(embeddings)
            progress_bar.progress(min((i + batch_size) / total, 1.0))
            progress_text.text(f"Processed {min(i + batch_size, total)} / {total} images")

        progress_bar.empty()
        progress_text.empty()

        all_embeddings = np.vstack(all_embeddings)
        logger.info(f"Total embeddings shape: {all_embeddings.shape}")
        if all_embeddings.shape[0] == 0:
            st.error("No valid images processed. Please check your uploads.")
            st.stop()
        scores = predict_scores(all_embeddings)

        # Build results list combining all necessary data
        results = []
        for i in range(len(images)):
            results.append({
                "filename": filenames[i],
                "score": round(float(scores[i]), 2),
                "image": images[i]
            })

        # Sort results by score
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

        # Prepare DataFrame
        df_sorted = pd.DataFrame([
            {"filename": r["filename"], "score": r["score"]}
            for r in results_sorted
        ])

        # Show thumbnails grid
        st.subheader("🖼️ Thumbnails Preview (Sorted by Score)")
        cols_per_row = 4
        rows = (len(results_sorted) + cols_per_row - 1) // cols_per_row

        for r in range(rows):
            cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                idx = r * cols_per_row + c
                if idx >= len(results_sorted):
                    break
                with cols[c]:
                    st.image(results_sorted[idx]["image"], use_container_width=True)
                    st.caption(f"{results_sorted[idx]['filename']}\nScore: {results_sorted[idx]['score']}")

        # CSV download
        csv = df_sorted.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download results as CSV",
            data=csv,
            file_name="thumbnail_scores.csv",
            mime="text/csv"
        )

        # Show metrics
        st.subheader("🔍 Insights")
        col1, col2 = st.columns(2)
        col1.metric("📈 Highest Score", f"{df_sorted['score'].max():.2f}")
        col2.metric("📊 Average Score", f"{df_sorted['score'].mean():.2f}")

        # Show results table
        st.subheader("📋 Score Table")
        st.dataframe(df_sorted)

else:
    st.info("Upload one or more thumbnail images to get started.")
