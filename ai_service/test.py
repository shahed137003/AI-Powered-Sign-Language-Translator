import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 4))

    lh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))

    rh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))

    face = np.array([[lm.x, lm.y, lm.z]
                     for lm in results.face_landmarks.landmark[:60]]) if results.face_landmarks else np.zeros((60, 3))

    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten(), face.flatten()])


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    print("🟢 Camera opened – press Q to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Logs
        print(
            "Pose:", bool(results.pose_landmarks),
            "LH:", bool(results.left_hand_landmarks),
            "RH:", bool(results.right_hand_landmarks),
            "Face:", bool(results.face_landmarks)
        )

        keypoints = extract_keypoints(results)
        print("Keypoints:", keypoints.shape, "Sum:", np.sum(keypoints))

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("Sign Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
