import numpy as np
import sys
import threading
import queue
from pathlib import Path

from .config import TARGET_LEN, MIN_FRAMES, COOLDOWN_FRAMES, CONF_THRESHOLD
from .model import ModelWrapper

# ── Preprocessing imports ─────────────────────────────────────────────────────
from .config import PREPROCESSING_PATH
sys.path.append(str(PREPROCESSING_PATH))

from preprocessing.pipeline_v3 import preprocess_sequence_global   # (T,438) → (T,438)
from preprocessing.constants import FEATURE_DIM                    # 438
from preprocessing.features.builder import build_features           # (T,438) → (T,928)


class ASLPipeline:
    def __init__(self, model, labels, llm):
        self.llm = llm
        self.model_wrapper = ModelWrapper(model, labels)

        # ── recording state ────────────────────────────────────────────────────
        self.buffer          = []
        self.recording       = False
        self.silence_counter = 0

        # ── sentence / LLM state ──────────────────────────────────────────────
        self.sentence_buffer  = []
        self.english_sentence = ""

        self.last_pred = "Waiting..."
        self.last_conf = 0.0

        # ── background LLM thread ─────────────────────────────────────────────
        self.llm_queue        = queue.Queue()
        self.llm_result_queue = queue.Queue()
        self.llm_busy         = False

        self.llm_thread = threading.Thread(target=self._llm_worker, daemon=True)
        self.llm_thread.start()

    # ──────────────────────────────────────────────────────────────────────────
    # LLM WORKER
    # ──────────────────────────────────────────────────────────────────────────
    def _llm_worker(self):
        while True:
            gloss = self.llm_queue.get()
            if gloss is None:
                break
            try:
                complete, result = self.llm.translate(gloss)
                self.llm_result_queue.put((complete, result))
            except Exception as e:
                print(f"LLM ERROR: {e}")
                self.llm_result_queue.put((False, ""))
            self.llm_busy = False
            self.llm_queue.task_done()

    # ──────────────────────────────────────────────────────────────────────────
    # FULL FEATURE PIPELINE
    # raw (T,438) → preprocess (T,438) → build_features (T,928) → pad (TARGET_LEN,928)
    # ──────────────────────────────────────────────────────────────────────────
    def _build_input(self, raw_frames: list) -> np.ndarray:
        """
        raw_frames : list of keypoint arrays, each shape (438,)
        returns    : np.ndarray  shape (TARGET_LEN, 928)
        """
        raw_seq = np.array(raw_frames, dtype=np.float32)     # (T, 438)
        print(f"  Raw shape        : {raw_seq.shape}")

        # Step 1 – normalise, clean, fill gaps  →  still (T, 438)
        proc_seq = preprocess_sequence_global(raw_seq)
        print(f"  After preprocess : {proc_seq.shape}")

        # Step 2 – feature engineering  438 → 928
        features = build_features(proc_seq)                   # (T, 928)
        print(f"  After features   : {features.shape}")

        # Step 3 – pad / trim to TARGET_LEN
        T = features.shape[0]
        if T >= TARGET_LEN:
            final = features[:TARGET_LEN]
        else:
            pad   = np.zeros((TARGET_LEN - T, features.shape[1]), dtype=np.float32)
            final = np.concatenate([features, pad], axis=0)

        print(f"  Final input      : {final.shape}")
        return final

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────
    def process_keypoints(self, keypoints: np.ndarray, hands_visible: bool):
        """
        keypoints    : (438,) raw landmark array for one frame
        hands_visible: True while hands are in frame

        Returns None while still collecting, or
                (pred, conf, sentence_buffer, english_sentence) when a
                gesture segment has been processed.
        """

        # ── RECORDING LOGIC ───────────────────────────────────────────────────
        if hands_visible:
            if not self.recording:
                self.recording = True
                self.buffer    = []
                print("🎬 Started recording")

            self.buffer.append(keypoints)
            self.silence_counter = 0
            return None   # still collecting

        elif self.recording:
            self.silence_counter += 1
            self.buffer.append(keypoints)

            if self.silence_counter > COOLDOWN_FRAMES:
                self.recording = False
                buf_len = len(self.buffer)
                print(f"✋ Stopped recording. Buffer size: {buf_len}")

                if buf_len > MIN_FRAMES:
                    try:
                        final_input = self._build_input(self.buffer)

                        pred, conf = self.model_wrapper.predict(final_input)
                        self.last_pred = pred
                        self.last_conf = conf
                        print(f"  ✅ Prediction: {pred} ({conf:.1f}%)")

                        if conf > CONF_THRESHOLD:
                            if not self.sentence_buffer or self.sentence_buffer[-1] != pred:
                                self.sentence_buffer.append(pred)
                                gloss_str = " ".join(self.sentence_buffer)
                                if not self.llm_busy:
                                    self.llm_busy = True
                                    self.llm_queue.put(gloss_str)
                                print(f"  📝 Added '{pred}', buffer: {self.sentence_buffer}")

                    except Exception as e:
                        print(f"  ❌ Prediction error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"⚠️  Gesture too short: {buf_len} frames (min {MIN_FRAMES})")

                self.buffer = []
                return (
                    self.last_pred,
                    self.last_conf,
                    self.sentence_buffer,
                    self.english_sentence,
                )

        # ── CHECK LLM RESULT (runs every frame, outside the recording block) ──
        if not self.llm_result_queue.empty():
            complete, result = self.llm_result_queue.get()
            if complete:
                self.english_sentence = result
                print(f"✨ LLM: '{' '.join(self.sentence_buffer)}' → '{result}'")

        return (
            self.last_pred,
            self.last_conf,
            self.sentence_buffer,
            self.english_sentence,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self):
        self.sentence_buffer  = []
        self.english_sentence = ""
        self.buffer           = []
        self.recording        = False
        self.silence_counter  = 0
        self.last_pred        = "Waiting..."
        self.last_conf        = 0.0