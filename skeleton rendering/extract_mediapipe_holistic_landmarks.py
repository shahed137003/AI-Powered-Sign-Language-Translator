import os
import cv2
import numpy as np
import mediapipe as mp

VIDEOS_DIR = "videos"        # root folder with subfolders
OUTPUT_ROOT = "mediapipe_landmarks"  # where .npy files will go

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks(results):
    def get_pose():
        arr = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                arr.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            arr.extend([0]*132)
        return arr

    def get_face():
        arr = []
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark[:60]:  # keep 60 only
                arr.extend([lm.x, lm.y, lm.z])
        else:
            arr.extend([0]*180)
        return arr

    def get_hand(hand):
        arr = []
        if hand:
            for lm in hand.landmark:
                arr.extend([lm.x, lm.y, lm.z])
        else:
            arr.extend([0]*63)
        return arr

    pose = get_pose()
    face = get_face()
    lh = get_hand(results.left_hand_landmarks)
    rh = get_hand(results.right_hand_landmarks)

    return pose + face + lh + rh  # = 438

# Walk through subfolders
for root, dirs, files in os.walk(VIDEOS_DIR):
    # relative path from VIDEOS_DIR
    rel_path = os.path.relpath(root, VIDEOS_DIR)
    output_dir = os.path.join(OUTPUT_ROOT, rel_path)
    os.makedirs(output_dir, exist_ok=True)

    for video_file in files:
        if not video_file.endswith((".mp4", ".avi", ".mov")):
            continue

        video_path = os.path.join(root, video_file)
        name = os.path.splitext(video_file)[0]

        print(f"Processing {os.path.join(rel_path, video_file)}")

        cap = cv2.VideoCapture(video_path)
        sequence = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            keypoints = extract_landmarks(results)
            sequence.append(keypoints)
            frame_idx += 1

        cap.release()

        if len(sequence) == 0:
            print(f"Empty video: {video_file}")
            continue

        sequence = np.array(sequence)
        save_path = os.path.join(output_dir, f"{name}.npy")
        np.save(save_path, sequence)

        print(f"Saved: {save_path}")

holistic.close()
print("ALL VIDEOS DONE")