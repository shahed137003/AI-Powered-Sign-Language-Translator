import numpy as np
import cv2
import sys
import threading
import queue

from .config import *
from .model import ModelWrapper
from .keypoints import holistic, extract_keypoints

from preprocessing.pipeline_v3 import preprocess_sequence_global
from preprocessing.constants import FEATURE_DIM
from .config import TARGET_LEN, MIN_FRAMES, COOLDOWN_FRAMES, CONF_THRESHOLD

class ASLPipeline:
    def __init__(self, llm):
        self.llm = llm
        self.model = ModelWrapper()

        # =========================
        # STATE
        # =========================
        self.buffer = []
        self.recording = False
        self.silence_counter = 0

        self.sentence_buffer = []
        self.english_sentence = ""

        self.last_pred = "Waiting..."
        self.last_conf = 0.0

        # =========================
        # LLM THREAD SYSTEM
        # =========================
        self.llm_queue = queue.Queue()
        self.llm_result_queue = queue.Queue()
        self.llm_busy = False

        self.llm_thread = threading.Thread(
            target=self._llm_worker,
            daemon=True
        )
        self.llm_thread.start()

    # ============================================================
    # LLM WORKER THREAD
    # ============================================================
    def _llm_worker(self):
        while True:
            gloss = self.llm_queue.get()

            if gloss is None:
                break

            try:
                complete, result = self.llm.translate(gloss)
                self.llm_result_queue.put((complete, result))
            except Exception as e:
                print("LLM ERROR:", e)
                self.llm_result_queue.put((False, ""))

            self.llm_busy = False
            self.llm_queue.task_done()

    # ============================================================
    # FRAME PROCESSING
    # ============================================================
    def process_frame(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        hand_seen = results.left_hand_landmarks or results.right_hand_landmarks
        kp = extract_keypoints(results)

        # =========================
        # RECORDING LOGIC
        # =========================
        if hand_seen:
            if not self.recording:
                self.recording = True
                self.buffer = []

            self.buffer.append(kp)
            self.silence_counter = 0

        elif self.recording:
            self.silence_counter += 1
            self.buffer.append(kp)

            if self.silence_counter > COOLDOWN_FRAMES:
                self.recording = False

                if len(self.buffer) > MIN_FRAMES:

                    raw_seq = np.array(self.buffer)
                    proc_seq = preprocess_sequence_global(raw_seq)

                    T = proc_seq.shape[0]

                    if T >= TARGET_LEN:
                        final_input = proc_seq[:TARGET_LEN]
                    else:
                        pad = np.zeros((TARGET_LEN - T, proc_seq.shape[1]), dtype=np.float32)
                        final_input = np.concatenate([proc_seq, pad], axis=0)

                    pred, conf = self.model.predict(final_input)

                    self.last_pred = pred
                    self.last_conf = conf

                    # =========================
                    # BUILD SENTENCE
                    # =========================
                    if conf > CONF_THRESHOLD:
                        if not self.sentence_buffer or self.sentence_buffer[-1] != pred:
                            self.sentence_buffer.append(pred)

                            gloss = " ".join(self.sentence_buffer)

                            print("📤 Sending to LLM:", gloss)

                            # if not self.llm_busy:
                            #     self.llm_busy = True
                            #     self.llm_queue.put(gloss)

                    

                self.buffer = []

        # =========================
        # READ LLM RESULT (IMPORTANT: OUTSIDE BLOCK)
        # =========================
        if not self.llm_result_queue.empty():
            complete, result = self.llm_result_queue.get()

            if complete:
                self.english_sentence = result
                self.sentence_buffer = []
            else:
                self.english_sentence = "Waiting for more words..."

        return (
            self.last_pred,
            self.last_conf,
            self.sentence_buffer,
            self.english_sentence
        )
        
    def reset(self):
        self.sentence_buffer = []
        self.english_sentence = ""
        self.buffer = []
        self.recording = False
        self.silence_counter = 0
        self.last_pred = "Waiting..."
        self.last_conf = 0.0
    
    def process_keypoints(self, keypoints: np.ndarray, hands_visible: bool):
        """
        Process a single frame of keypoints (like process_frame but without MediaPipe).
        Returns (pred, conf, words, english_sentence) or None if still collecting.
        """
        # 1. Recording logic – same as in process_frame
        if hands_visible:
            if not self.recording:
                self.recording = True
                self.buffer = []
            self.buffer.append(keypoints)
            self.silence_counter = 0
            return None  # still collecting

        elif self.recording:
            self.silence_counter += 1
            self.buffer.append(keypoints)

            if self.silence_counter > COOLDOWN_FRAMES:
                self.recording = False
                if len(self.buffer) > MIN_FRAMES:
                    # --------------------------
                    # PREDICTION (only when enough frames)
                    # --------------------------
                    raw_seq = np.array(self.buffer)
                    proc_seq = preprocess_sequence_global(raw_seq)   # (T, FEATURE_DIM)
                    T = proc_seq.shape[0]
                    if T >= TARGET_LEN:
                        final_input = proc_seq[:TARGET_LEN]
                    else:
                        pad = np.zeros((TARGET_LEN - T, FEATURE_DIM), dtype=np.float32)
                        final_input = np.concatenate([proc_seq, pad], axis=0)

                    pred, conf = self.model.predict(final_input)   # (label, conf%)
                    self.last_pred = pred
                    self.last_conf = conf

                    # Add to sentence buffer if confidence is high enough
                    if conf > CONF_THRESHOLD:
                        if not self.sentence_buffer or self.sentence_buffer[-1] != pred:
                            self.sentence_buffer.append(pred)
                            gloss_str = " ".join(self.sentence_buffer)
                            # Send to LLM (non‑blocking)
                            if not self.llm_busy:
                                self.llm_busy = True
                                self.llm_queue.put(gloss_str)
                            print(f"📝 Added '{pred}', buffer: {self.sentence_buffer}")

                    # Log the prediction (only if prediction happened)
                    if len(self.buffer) > MIN_FRAMES:
                        print(f"Predicted: {pred} with confidence {conf:.1f}% (threshold {CONF_THRESHOLD})")
                else:
                    # Too short – just discard buffer
                    print(f"⚠️ Gesture too short: {len(self.buffer)} frames (min {MIN_FRAMES})")
                self.buffer = []
                # After finishing a gesture, return the current state (last word and sentence)
                return self.last_pred, self.last_conf, self.sentence_buffer, self.english_sentence

        # 2. Check if LLM finished translating
        if not self.llm_result_queue.empty():
            complete, result = self.llm_result_queue.get()
            if complete:
                self.english_sentence = result
                # Do NOT clear self.sentence_buffer – matches the Python script behaviour
                print(f"✨ LLM translation: {self.english_sentence}")
        return self.last_pred, self.last_conf, self.sentence_buffer, self.english_sentence