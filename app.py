import streamlit as st
import torch
from PIL import Image
import numpy as np
import joblib
import open_clip
import io
import pandas as pd

st.set_page_config(page_title="YouTube Thumbnail Scorer", layout="wide")
st.title("📺 YouTube Thumbnail Scorer")

# Load models once, cache for performance
@st.cache_resource
def load_models():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
    model.to(device).eval()
    regressor = joblib.load("model.lgb")
    return model, preprocess, regressor, device

model, preprocess, regressor, device = load_models()

def get_embedding_batch(images):
    # images: list of PIL Images
    tensors = [preprocess(img).to(device) for img in images]
    batch = torch.stack(tensors)
    with torch.no_grad():
        embeddings = model.encode_image(batch)
    return embeddings.cpu().numpy()

def predict_scores(embeddings):
    return regressor.predict(embeddings)

uploaded_files = st.file_uploader(
    "Upload thumbnail images (png, jpg, jpeg)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    images = []
    for file in uploaded_files:
        try:
            img = Image.open(file).convert("RGB")
            images.append(img)
        except Exception as e:
            st.warning(f"Skipping {file.name}, could not open image.")

    if images:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        batch_size = 32
        all_embeddings = []
        total = len(images)
        
        # Process in batches with progress bar
        for i in range(0, total, batch_size):
            batch_imgs = images[i:i+batch_size]
            embeddings = get_embedding_batch(batch_imgs)
            all_embeddings.append(embeddings)
            progress_bar.progress(min((i + batch_size) / total, 1.0))
            progress_text.text(f"Processed {min(i + batch_size, total)} / {total} images")

        progress_bar.empty()
        progress_text.empty()

        all_embeddings = np.vstack(all_embeddings)
        scores = predict_scores(all_embeddings)
        est_ratios = np.exp(scores)

        # Prepare data for table & CSV
        results = []
        for i, img in enumerate(images):
            results.append({
                "filename": uploaded_files[i].name,
                "log_score": round(float(scores[i]), 4),
                "views_per_subscriber_est": round(float(est_ratios[i]), 2)
            })

        df = pd.DataFrame(results)

        # Show table
        st.subheader("Results")
        st.dataframe(df)

        # CSV download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="thumbnail_scores.csv",
            mime='text/csv'
        )

        # Show thumbnails in grid with scores
        st.subheader("Thumbnails Preview")
        cols_per_row = 4
        rows = (len(images) + cols_per_row - 1) // cols_per_row
        for r in range(rows):
            cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                idx = r * cols_per_row + c
                if idx >= len(images):
                    break
                with cols[c]:
                    st.image(images[idx], use_container_width='always')
                    st.caption(f"{uploaded_files[idx].name}\nScore: {results[idx]['log_score']}\nEst. Views/Subs: {results[idx]['views_per_subscriber_est']}")

else:
    st.info("Upload one or more thumbnail images to get started.")
