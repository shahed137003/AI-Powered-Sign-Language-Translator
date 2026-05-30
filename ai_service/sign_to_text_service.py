import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Import preprocessing
from preprocessing.pipeline_v3 import preprocess_sequence_global
from preprocessing.constants import FEATURE_DIM
from motion_detector import SignStateDetector  # ADD THIS

# ============================================================
# 1. MODEL ARCHITECTURE (Matches main_inference)
# ============================================================
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        y = y[..., :x.size(2)]
        return y + self.res(x)

class TCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        channels = [128, 128, 128, 128] 
        layers = []
        for i, c in enumerate(channels):
            in_dim = input_dim if i == 0 else channels[i-1]
            layers.append(TemporalBlock(in_dim, c, dilation=2**i))
        self.tcn = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.tcn(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class SignToTextService:
    # Fixed parameters
    TARGET_LEN = 120
    MIN_FRAMES = 15

    def __init__(self, model, labels, device):
        self.model = model
        self.labels = labels
        self.device = device
        
        # Initialize motion detector
        self.detector = SignStateDetector(
            buffer_size=30,
            velocity_thresh=0.015,  # Adjust this for sensitivity
            pause_frames=5,         # Frames of no motion before stopping
            min_sign_frames=10      # Minimum frames for a valid sign
        )
        
        self.is_recording = False
        self.current_sign_frames = []

    def reset(self):
        self.detector.reset()
        self.is_recording = False
        self.current_sign_frames = []

    def add_keypoints(self, keypoints: list, hands_visible: bool = True):
        """
        Process keypoints with motion-based detection.
        """
        kp_array = np.array(keypoints)
        
        # Use motion detector to determine state
        status, frames = self.detector.add_frame(kp_array)
        
        if status == "SIGNING":
            # User is actively signing
            if not self.is_recording:
                self.is_recording = True
                print("🎬 Motion detected - Started recording sign")
                return {
                    "status": "collecting",
                    "frames_collected": len(self.detector.sign_buffer),
                    "progress": 0
                }
            else:
                # Update progress
                progress = min(100, int((len(self.detector.sign_buffer) / self.TARGET_LEN) * 100))
                return {
                    "status": "collecting",
                    "frames_collected": len(self.detector.sign_buffer),
                    "progress": progress
                }
                
        elif status == "SIGN_ENDED":
            # User stopped moving - time to predict
            if frames and len(frames) >= self.MIN_FRAMES:
                print(f"✋ Motion stopped - Buffer size: {len(frames)}")
                self.is_recording = False
                result = self.predict_sequence(frames)
                return result
            else:
                # Too short
                print(f"⚠️ Sign too short: {len(frames) if frames else 0} frames")
                self.is_recording = False
                return {
                    "status": "error",
                    "text": f"Sign too short ({len(frames) if frames else 0} frames)",
                    "confidence": 0.0
                }
                
        else:  # WAITING
            if self.is_recording:
                # This shouldn't happen normally, but just in case
                self.is_recording = False
            return {"status": "idle"}

    def predict_sequence(self, frames):
        """Run inference on collected frames"""
        try:
            # Convert to numpy array
            raw_sequence = np.array(frames)
            
            # Preprocess using v3 pipeline
            proc_seq = preprocess_sequence_global(raw_sequence)
            
            # Fixed length padding/trimming to TARGET_LEN (120)
            T = proc_seq.shape[0]
            if T > self.TARGET_LEN:
                final_input = proc_seq[:self.TARGET_LEN]
            else:
                final_input = np.concatenate([proc_seq, np.zeros((self.TARGET_LEN - T, FEATURE_DIM))])
            
            # Model expects (Batch, Channels, Time) -> (1, 438, 120)
            x = torch.from_numpy(final_input).float().transpose(0, 1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
            
            predicted_word = self.labels[idx.item()]
            confidence = float(conf.item()) * 100
            
            print(f"✅ Prediction: {predicted_word} ({confidence:.1f}%)")
            
            return {
                "text": predicted_word,
                "confidence": confidence / 100.0,
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                "text": "Error processing gesture",
                "confidence": 0.0,
                "status": "error"
            }
    
    def force_predict(self):
        """Force prediction on current sign (manual stop)"""
        if len(self.detector.sign_buffer) >= self.MIN_FRAMES:
            result = self.predict_sequence(self.detector.sign_buffer)
            self.detector.reset()
            self.is_recording = False
            return result
        else:
            frames = len(self.detector.sign_buffer)
            self.detector.reset()
            self.is_recording = False
            return {
                "text": f"Too short ({frames} frames, need {self.MIN_FRAMES})",
                "confidence": 0.0,
                "status": "error"
            }

