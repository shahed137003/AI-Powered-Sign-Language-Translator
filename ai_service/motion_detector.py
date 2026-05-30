"""
Motion detection module for real-time sign language inference.
"""

import numpy as np

def detect_motion_realtime(hand_buffer, min_pts=8, velocity_thresh=0.015, pause_frames=5):
    """
    Detect if there's active signing motion in recent frames.
    
    Args:
        hand_buffer: List of recent hand landmarks [(21,3), (21,3), ...]
        min_pts: Minimum landmarks to consider hand valid
        velocity_thresh: Velocity threshold for motion detection
        pause_frames: Consecutive frames below threshold to consider "stopped"
    
    Returns:
        is_signing: True if actively signing, False if paused/idle
        motion_score: Average velocity (0 = stopped, higher = moving)
    """
    T = len(hand_buffer)
    if T < 5:
        return False, 0.0
    
    # Calculate hand centroids
    centroids = []
    for hand in hand_buffer:
        # Check if hand is valid (not all zeros)
        valid = np.any(np.abs(hand) > 1e-8, axis=1)
        if valid.sum() >= min_pts:
            centroid = hand[valid].mean(axis=0)[:2]  # Use only x,y
            centroids.append(centroid)
        else:
            centroids.append(None)
    
    # Calculate velocities
    velocities = []
    for i in range(1, len(centroids)):
        if centroids[i] is not None and centroids[i-1] is not None:
            vel = np.linalg.norm(centroids[i] - centroids[i-1])
            velocities.append(vel)
    
    if not velocities:
        return False, 0.0
    
    # Check for sustained motion
    recent_velocities = velocities[-pause_frames:] if len(velocities) >= pause_frames else velocities
    avg_velocity = np.mean(recent_velocities)
    
    # Count consecutive low-velocity frames
    low_count = sum(1 for v in recent_velocities if v < velocity_thresh)
    
    # Signing if most recent frames have motion
    is_signing = low_count < pause_frames
    
    return is_signing, avg_velocity


def extract_hands_from_features(feature_vector):
    """
    Extract left and right hands from (438,) feature vector.
    
    Returns:
        left_hand: (21, 3) or None
        right_hand: (21, 3) or None
    """
    # Indices based on your preprocessing constants
    POSE_SIZE = 132  # 33 * 4
    FACE_SIZE = 180  # 60 * 3
    HAND_SIZE = 63   # 21 * 3
    
    # Extract
    left_hand_flat = feature_vector[POSE_SIZE + FACE_SIZE : POSE_SIZE + FACE_SIZE + HAND_SIZE]
    right_hand_flat = feature_vector[POSE_SIZE + FACE_SIZE + HAND_SIZE :]
    
    left_hand = left_hand_flat.reshape(21, 3) if np.any(left_hand_flat != 0) else None
    right_hand = right_hand_flat.reshape(21, 3) if np.any(right_hand_flat != 0) else None
    
    return left_hand, right_hand


class SignStateDetector:
    """
    State machine for sign detection based on motion.
    """
    
    def __init__(self, buffer_size=30, velocity_thresh=0.015, pause_frames=5, min_sign_frames=10):
        self.buffer = []
        self.buffer_size = buffer_size
        self.velocity_thresh = velocity_thresh
        self.pause_frames = pause_frames
        self.min_sign_frames = min_sign_frames
        self.state = "IDLE"  # IDLE, SIGNING, STOPPED
        self.sign_buffer = []  # Frames that contain the sign
        
    def add_frame(self, feature_vector):
        """
        Process one frame. Call this for each frame from MediaPipe.
        
        Returns:
            status: "WAITING", "SIGNING", "SIGN_ENDED", or "PROCESS"
            sign_frames: List of feature vectors to process (only when SIGN_ENDED)
        """
        # Extract hands
        left_hand, right_hand = extract_hands_from_features(feature_vector)
        
        # Add to buffer
        self.buffer.append((left_hand, right_hand))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Check motion for each hand
        left_buffer = [lh for lh, _ in self.buffer if lh is not None]
        right_buffer = [rh for _, rh in self.buffer if rh is not None]
        
        left_signing = False
        right_signing = False
        
        if len(left_buffer) >= 5:
            left_signing, left_velocity = detect_motion_realtime(
                left_buffer, velocity_thresh=self.velocity_thresh, pause_frames=self.pause_frames
            )
        if len(right_buffer) >= 5:
            right_signing, right_velocity = detect_motion_realtime(
                right_buffer, velocity_thresh=self.velocity_thresh, pause_frames=self.pause_frames
            )
        
        is_signing = left_signing or right_signing
        
        # State machine
        if self.state == "IDLE":
            if is_signing:
                self.state = "SIGNING"
                self.sign_buffer = [feature_vector]
                return "SIGNING", None
            return "WAITING", None
            
        elif self.state == "SIGNING":
            if is_signing:
                self.sign_buffer.append(feature_vector)
                return "SIGNING", None
            else:
                # Motion stopped
                if len(self.sign_buffer) >= self.min_sign_frames:
                    self.state = "STOPPED"
                    frames_to_process = self.sign_buffer.copy()
                    self.sign_buffer = []
                    return "SIGN_ENDED", frames_to_process
                else:
                    # Too short, reset
                    self.state = "IDLE"
                    self.sign_buffer = []
                    return "WAITING", None
                    
        elif self.state == "STOPPED":
            self.state = "IDLE"
            return "WAITING", None
        
        return "WAITING", None
    
    def reset(self):
        """Reset the detector state"""
        self.buffer = []
        self.state = "IDLE"
        self.sign_buffer = []