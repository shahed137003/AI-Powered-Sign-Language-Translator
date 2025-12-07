
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

import itertools

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

def draw_skeleton_hand(keypoints, img ,img_width=640, img_height=480, title="Hand Skeleton"):
    """
    Draws a hand skeleton using MediaPipe drawing utils on a blank image.
    
    keypoints: np.array of shape (21,3) with x,y in pixels (z optional)
    img_width, img_height: dimensions of the canvas image
    """
    keypoints = np.array(keypoints).reshape(-1, 3)
    
    # Create blank image
    # img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Convert keypoints to NormalizedLandmarkList
    landmarks = []
    for kp in keypoints:
        lm = landmark_pb2.NormalizedLandmark()
        lm.x = float(kp[0])  
        lm.y = float(kp[1]) 
        lm.z = float(kp[2])             
        landmarks.append(lm)

    landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=landmarks)

    # Draw hand skeleton
    mp_drawing.draw_landmarks(
    img,
    landmark_list,
    HAND_CONNECTIONS,
    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),  # vibrant magenta dots
    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 100, 255), thickness=2)  # lighter magenta lines
     )


    # Show image
    # cv2.imshow(title, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


mp_pose = mp.solutions.pose



def draw_skeleton_pose(keypoints, img, visibility_thresh=0.2):
    """
    Draws the full pose skeleton on an image, skipping low-visibility keypoints
    and removing face keypoints (indices 0-10).

    keypoints: np.array of shape (33,4) with (x, y, z, visibility)
    img: numpy array (H,W,3) where skeleton will be drawn
    visibility_thresh: minimum visibility to draw the point
    """
    keypoints = np.array(keypoints).reshape(-1, 4)  # ensure shape (33,4)

    # Exclude face keypoints (0-10)
    body_indices = list(range(11, 33))

    # Collect visible body landmarks
    landmarks = []
    index_map = {}  # map original index to new landmarks list index
    for i in body_indices:
        kp = keypoints[i]
        if kp[3] >= visibility_thresh:
            lm = landmark_pb2.NormalizedLandmark()
            lm.x = float(kp[0])
            lm.y = float(kp[1])
            lm.z = float(kp[2])
            landmarks.append(lm)
            index_map[i] = len(landmarks) - 1

    if not landmarks:
        return  # nothing visible

    landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=landmarks)

    # Filter connections to only include visible body landmarks
    connections = []
    for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
        if start_idx in index_map and end_idx in index_map:
            connections.append((index_map[start_idx], index_map[end_idx]))

    # Draw skeleton
    mp_drawing.draw_landmarks(
        img,
        landmark_list,
        connections,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
    )



mp_face_mesh = mp.solutions.face_mesh

# Define relevant facial features (lips + eyebrows)
FACEMESH_LIPS = set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS))
FACEMESH_LEFT_EYEBROW = set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW))
FACEMESH_RIGHT_EYEBROW = set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW))

RELEVANT_FACE_INDICES = list(FACEMESH_LIPS | FACEMESH_LEFT_EYEBROW | FACEMESH_RIGHT_EYEBROW)
RELEVANT_FACE_INDICES.sort()
# print("Number of relevant face indices:", len(RELEVANT_FACE_INDICES))


mp_face_mesh = mp.solutions.face_mesh

def draw_skeleton_face(keypoints, img, img_width=640, img_height=480):
    """
    Draws face skeleton for lips and eyebrows using MediaPipe drawing utils.
    
    keypoints: np.array of shape (60,3) in pixel coordinates
    img: numpy array where the skeleton will be drawn
    """
    keypoints = np.array(keypoints).reshape(-1, 3)  # (60,3)

    # Normalize x,y coordinates to [0,1] for MediaPipe
    landmarks = []
    for kp in keypoints:
        lm = landmark_pb2.NormalizedLandmark()
        lm.x = float(kp[0]) 
        lm.y = float(kp[1]) 
        lm.z = float(kp[2])
        landmarks.append(lm)

    landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=landmarks)

    # Prepare connections limited to the reduced 60 keypoints
    connections = []
    for start_idx, end_idx in list(mp_face_mesh.FACEMESH_LIPS) + \
                            list(mp_face_mesh.FACEMESH_LEFT_EYEBROW) + \
                            list(mp_face_mesh.FACEMESH_RIGHT_EYEBROW):
        if start_idx in RELEVANT_FACE_INDICES and end_idx in RELEVANT_FACE_INDICES:
            new_start = RELEVANT_FACE_INDICES.index(start_idx)
            new_end = RELEVANT_FACE_INDICES.index(end_idx)
            connections.append((new_start, new_end))

    # Draw landmarks and connections
    mp_drawing.draw_landmarks(
        img,
        landmark_list,
        connections,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=1)
    )



## draw the whole body skeleton
def draw_full_body_skeleton(keypoints, img_width=640, img_height=480):
    """
    keypoints = [pose, left_hand, right_hand, face]
    """
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # draw each part
    draw_skeleton_pose(keypoints[0], img)
    draw_skeleton_hand(keypoints[1], img)
    draw_skeleton_hand(keypoints[2], img)
    draw_skeleton_face(keypoints[3], img)

    return img

    
        


def get_keypoints_for_frame(frame):
    pose = frame[0:132]
    face = frame[132:312]
    left_hand = frame[312:375]
    right_hand = frame[375:438]
    return pose, left_hand, right_hand, face


def draw_video_skeleton(video, title="Video Skeleton"):
    """
    video: numpy array of shape (num_frames, 438)
    """
    for i in range(video.shape[0]):
        frame = video[i]  # shape (438,)
        pose, lh, rh, face = get_keypoints_for_frame(frame)
        keypoints = [pose, lh, rh, face]

        img = draw_full_body_skeleton(keypoints)
        cv2.imshow(title, img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# draw_video_skeleton(data[2800])