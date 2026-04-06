### rendering of the skeleton 
import cv2
import numpy as np
import mediapipe as mp

mpHands = mp.solutions.hands
mpPose = mp.solutions.pose

def render_skeleton(data):
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    h, w = 480, 640

    # -------- Split data --------
    pose = data[0:132].reshape(33, 4)
    face = data[132:312].reshape(-1, 3)
    lh   = data[312:375].reshape(21, 3)
    rh   = data[375:438].reshape(21, 3)

    # -------- Helper --------
    def to_pixel(coord):
        return int(coord[0] * w), int(coord[1] * h)

    # -------- Draw Pose --------
    for connection in mpPose.POSE_CONNECTIONS:
        p1, p2 = connection

        if pose[p1][3] > 0.5 and pose[p2][3] > 0.5:  # visibility check
            x1, y1 = to_pixel(pose[p1])
            x2, y2 = to_pixel(pose[p2])
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for p in pose:
        if p[3] > 0.5:
            x, y = to_pixel(p)
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    # -------- Draw Hands --------
    def draw_hand(hand, color):
        for connection in mpHands.HAND_CONNECTIONS:
            p1, p2 = connection
            x1, y1 = to_pixel(hand[p1])
            x2, y2 = to_pixel(hand[p2])
            cv2.line(img, (x1, y1), (x2, y2), color, 2)

        for p in hand:
            x, y = to_pixel(p)
            cv2.circle(img, (x, y), 2, color, -1)

    draw_hand(lh, (255, 0, 0))   # left hand = blue
    draw_hand(rh, (0, 0, 255))   # right hand = red

    # -------- Draw Face (points only) --------
    for p in face:
        x, y = to_pixel(p)
        cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

    return img
def render_video(sequence, delay=30, loop=False):
    """
    sequence: numpy array of shape (num_frames, 438)
    delay: time between frames (ms)
    loop: replay video continuously
    """

    while True:
        for i, frame in enumerate(sequence):
            try:
                img = render_skeleton(frame)

                # 🟡 Add frame index (debugging)
                cv2.putText(img, f"Frame: {i}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)

                cv2.imshow("Skeleton Video", img)

                key = cv2.waitKey(delay) & 0xFF

                # ❌ Quit
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                
                # ⏸ Pause
                elif key == ord('p'):
                    cv2.waitKey(0)

            except Exception as e:
                print(f"❌ Error at frame {i}: {e}")

        if not loop:
            break

    cv2.destroyAllWindows()