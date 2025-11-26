import numpy as np
import cv2

from .layout_438 import FACE_LANDMARKS


def draw_pose(img, pose, map_xy, pose_connections, pose_vis_thresh: float = 0.0, color=(0, 255, 0), thickness: int = 2):
    # Pose (use vis>pose_vis_thresh)
    vis = pose[:, 3]
    for s, e in pose_connections:
        if vis[s] <= pose_vis_thresh or vis[e] <= pose_vis_thresh:
            continue
        x1, y1 = map_xy(pose[s, 0], pose[s, 1])
        x2, y2 = map_xy(pose[e, 0], pose[e, 1])
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_hand(img, hand, map_xy, hand_connections, color, thickness: int = 2):
    # Hand (skip zero points)
    for s, e in hand_connections:
        if np.allclose(hand[s, :2], 0) or np.allclose(hand[e, :2], 0):
            continue
        x1, y1 = map_xy(hand[s, 0], hand[s, 1])
        x2, y2 = map_xy(hand[e, 0], hand[e, 1])
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_face(img, face, map_xy, color=(255, 0, 255), radius: int = 2):
    # Face points (skip zero)
    for i in range(FACE_LANDMARKS):
        if np.allclose(face[i, :2], 0):
            continue
        x, y = map_xy(face[i, 0], face[i, 1])
        cv2.circle(img, (x, y), radius, color, -1)
