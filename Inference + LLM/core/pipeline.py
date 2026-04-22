import numpy as np
import cv2
import sys
import threading
import queue

from .config import *
from .model import ModelWrapper
from .keypoints import holistic, extract_keypoints

sys.path.append(str(PREPROCESSING_PATH))
from preprocessing.pipeline_v3 import preprocess_sequence_global


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

            print("LLM THREAD running on:", gloss)

            try:
                result = self.llm.translate(gloss)
                self.llm_result_queue.put(result)
            except Exception as e:
                print("LLM ERROR:", e)
                self.llm_result_queue.put("")

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

                    if T > TARGET_LEN:
                        final_input = proc_seq[:TARGET_LEN]
                    else:
                        final_input = np.concatenate(
                            [proc_seq, np.zeros((TARGET_LEN - T, proc_seq.shape[1]))]
                        )

                    pred, conf = self.model.predict(final_input)

                    self.last_pred = pred
                    self.last_conf = conf

                    # =========================
                    # BUILD SENTENCE
                    # =========================
                    if conf > CONF_THRESHOLD:
                        if not self.sentence_buffer or self.sentence_buffer[-1] != pred:
                            self.sentence_buffer.append(pred)

                    # =========================
                    # TRIGGER LLM (NON-BLOCKING)
                    # =========================
                    if len(self.sentence_buffer) >= 3:
                        gloss = " ".join(self.sentence_buffer)
                        self.sentence_buffer = []

                        print("📤 Sending to LLM:", gloss)

                        if not self.llm_busy:
                            self.llm_busy = True
                            self.llm_queue.put(gloss)

                self.buffer = []

        # =========================
        # READ LLM RESULT (IMPORTANT: OUTSIDE BLOCK)
        # =========================
        if not self.llm_result_queue.empty():
            self.english_sentence = self.llm_result_queue.get()
            print("English:", self.english_sentence)

        return (
            self.last_pred,
            self.last_conf,
            self.sentence_buffer,
            self.english_sentence
        )