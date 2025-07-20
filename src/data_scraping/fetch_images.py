import csv
import os
import requests

# Input CSV path
data_file = '../../data/raw/total.csv'
# Directory to save thumbnails
output_dir = 'thumbnails'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

with open(data_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        video_id = row.get('video_id')
        url = row.get('thumbnail_maxres_url')
        if not video_id or not url:
            continue
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to download thumbnail for {video_id}: {e}")
            continue

        # Determine file extension from URL or default to .jpg
        ext = os.path.splitext(url)[1]
        if not ext:
            ext = '.jpg'

        filename = f"{video_id}{ext}"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as img_file:
            for chunk in response.iter_content(chunk_size=8192):
                img_file.write(chunk)
        print(f"Saved thumbnail for {video_id} to {filepath}")
