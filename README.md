# 🖼️ YouTube Thumbnail Scorer
This project trains a machine learning model to predict the effectiveness of YouTube thumbnails using the ratio of **_views to average views of a channel_**. It leverages OpenAI’s CLIP model (ResNet50 backbone) for image embedding and trains a LightGBM regressor to estimate performance.

It is developed from the scratch at Noktalı Virgül live broadcast: [YouTube Broadcast Link](https://www.youtube.com/watch?v=Ahif9Zt3nBc)

## 🌐 Live Demo

🎉 The app is live! Try it out here:  
👉 [https://thumbnail-scorer.streamlit.app](https://thumbnail-scorer.streamlit.app)

Upload your own YouTube thumbnails and see how they score based on real-world YouTube performance data.


<img width="1876" height="955" alt="Screenshot 2025-07-21 at 19 47 53" src="https://github.com/user-attachments/assets/77520616-ea6c-46e7-b427-a74f497d7936" />
<img width="1876" height="955" alt="Screenshot 2025-07-21 at 19 49 25" src="https://github.com/user-attachments/assets/dcacb342-1817-4db5-88ce-b768d7842dd1" />

---

## 🛠️ Installation
1. Clone this repository:

```shell
git clone https://github.com/Cengineer00/thumbnail-scorer.git
cd thumbnail-scorer
```

2. Create a virtual environment and activate it:

```shell
conda create -n TScorer python=3.11
conda activate TScorer
```

3. Install dependencies:

```shell
pip install -r requirements.txt
```

---

## 🚀 How to Reproduce the Results

### 1. Scrape YouTube Metadata
Open and run `src/data_scraping/playground.ipynb` to retrieve metadata of videos from the desired YouTube channels.

- Make sure to provide your YouTube API Key inside the notebook.
- `You can use channel_id_conversions.ipynb` to help retrieve Channel IDs.

This step will generate a CSV file to:  
  `data/raw/total.csv`

### 2. Download Thumbnails
Run the image fetching script to download all thumbnails mentioned in the CSV.

```bash
cd src/data_scraping
python fetch_images.py
```

- Thumbnails will be saved to:  
  `src/data_scraping/thumbnails`

### 3. Train the Model
Run the complete `main.ipynb` notebook:

What it does:
- Loads a pre-trained **OpenAI CLIP ResNet50** model.
- Labels each thumbnail using:  
  `log(view_count / average_view_count)`
- Saves the labeled dataset to:  
  `data/raw/scored_metadata.csv`
- Extracts image embeddings and pairs them with scores.
- Trains a LightGBM regressor and saves it to:  
  `model.lgb`

You can test the model predictions at the end of the notebook.

### 4. Run the Streamlit App
Launch the app to score your own thumbnails interactively.

```bash
streamlit run app.py
```

- App will be hosted at:  
  `http://localhost:8501`
- Upload any number of thumbnails and view predicted scores.

---

## 🧠 Customization

- You can change the **channels** and **filtering steps** in  
  `src/data_scraping/playground.ipynb`
- **Model behavior is highly dependent on the `score_thumbnails()` function in `main.ipynb`.**  
- Try using different architectures or scoring logic for experiments.

---

## 🤝 Contributing

If you'd like to contribute, feel free to fork the repository and submit a pull request. Contributions, bug reports, and feature requests are welcome!

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/mlfoundations/open_clip)
- [LightGBM](https://github.com/microsoft/LightGBM)
- [YouTube Data API](https://developers.google.com/youtube/v3)
- [Streamlit](https://github.com/streamlit/streamlit)
