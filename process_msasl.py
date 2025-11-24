import json
import os
import cv2
import numpy as np
import yt_dlp
from ai_utils import extract_keypoints

# --- CONFIGURATION ---
JSON_FILE = "../data/MS-ASL/MSASL_train.json"  # Make sure this file is in your folder!
SAVE_DIR = "../ai/data/MSASL_Keypoints"  # Where to save the data

os.makedirs(SAVE_DIR, exist_ok=True)

def process_msasl():
    print(f"Reading {JSON_FILE}...")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    print(f"Found {len(data)} clips. Starting processing...")
    
    # --- RESUME CONFIGURATION ---
    START_INDEX = 11687  # <--- IT WILL START HERE
    for i, entry in enumerate(data):
        # 0. resume logic: skip videos we already passed
        if i < START_INDEX:
            continue
        
        # 1. Get Details
        url = entry['url']
        label = entry['text']
        start = entry['start_time']
        end = entry['end_time']
        
        # Create filenames
        clean_label = "".join([c for c in label if c.isalnum() or c in (' ','-')]).strip()
        video_id = url.split('v=')[-1]
        
        npy_filename = f"{clean_label}_{video_id}.npy"
        npy_path = os.path.join(SAVE_DIR, npy_filename)
        temp_mp4 = f"temp_{video_id}.mp4"

        # Skip if already done
        if os.path.exists(npy_path):
            continue

        print(f"[{i}/{len(data)}] Processing: {label} ({video_id})")

        # 2. Download Video Clip (Specific Start/End times)
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': temp_mp4,
            'download_ranges': yt_dlp.utils.download_range_func(None, [(start, end)]),
            'force_keyframes_at_cuts': True,
            'quiet': True,
            'ignoreerrors': True,
            'no_warnings': True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"  Download failed: {e}")
            continue

        if not os.path.exists(temp_mp4):
            print("  Video not found (deleted/private). Skipping.")
            continue

        # 3. Extract Keypoints using ai_utils
        cap = cv2.VideoCapture(temp_mp4)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(extract_keypoints(frame))
        cap.release()

        # 4. Save Data
        if len(frames) > 0:
            np.save(npy_path, np.array(frames))
            print(f"  SUCCESS: Saved {len(frames)} frames to {npy_filename}")
        else:
            print("  No frames extracted.")

        # 5. DELETE VIDEO (Storage Safety)
        if os.path.exists(temp_mp4):
            os.remove(temp_mp4)

if __name__ == "__main__":
    process_msasl()