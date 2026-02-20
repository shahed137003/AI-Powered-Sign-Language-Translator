import os
import cv2
import numpy as np
import itertools
import mediapipe as mp


def main():
    # === NEW VIDEO FOLDER ===
    videos_folder = r"E:\ASL_Citizen\NEW\Top_Classes"
    output_folder = r"E:\ASL_Citizen\NEW\Top_Classes_Landmarks"
    os.makedirs(output_folder, exist_ok=True)

    # === MEDIAPIPE INITIALIZATION ===
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh

    FACEMESH_LIPS = set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS))
    FACEMESH_LEFT_EYEBROW = set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW))
    FACEMESH_RIGHT_EYEBROW = set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW))

    RELEVANT_FACE_INDICES = list(
        FACEMESH_LIPS | FACEMESH_LEFT_EYEBROW | FACEMESH_RIGHT_EYEBROW
    )
    RELEVANT_FACE_INDICES.sort()

    # Initialize holistic model
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # === PROCESS VIDEOS ===
    for video_file in os.listdir(videos_folder):
        if not video_file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue

        video_path = os.path.join(videos_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        base_name = os.path.splitext(video_file)[0]
        all_keypoints = []

        print(f"Processing {video_file}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)

            # 1) POSE (33 × 4)
            if results.pose_landmarks:
                pose = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in results.pose_landmarks.landmark]
                ).flatten()
            else:
                pose = np.zeros(33 * 4)

            # 2) HANDS (21 × 3 each)
            if results.left_hand_landmarks:
                lh = np.array(
                    [[lm.x, lm.y, lm.z]
                     for lm in results.left_hand_landmarks.landmark]
                ).flatten()
            else:
                lh = np.zeros(21 * 3)

            if results.right_hand_landmarks:
                rh = np.array(
                    [[lm.x, lm.y, lm.z]
                     for lm in results.right_hand_landmarks.landmark]
                ).flatten()
            else:
                rh = np.zeros(21 * 3)

            # 3) FACE (LIPS + EYEBROWS ONLY)
            if results.face_landmarks:
                relevant = [
                    results.face_landmarks.landmark[i]
                    for i in RELEVANT_FACE_INDICES
                ]
                face = np.array(
                    [[lm.x, lm.y, lm.z] for lm in relevant]
                ).flatten()
            else:
                face = np.zeros(len(RELEVANT_FACE_INDICES) * 3)

            # FINAL CONCATENATION
            final_kp = np.concatenate([pose, face, lh, rh])
            all_keypoints.append(final_kp)

        cap.release()

        # Convert to array and save
        all_keypoints = np.array(all_keypoints)
        save_path = os.path.join(output_folder, f"{base_name}.npy")
        np.save(save_path, all_keypoints)

        print(f"Saved {save_path} with shape {all_keypoints.shape}")

    holistic.close()
    print("All videos processed and converted successfully!")


if __name__ == "__main__":
    main()
