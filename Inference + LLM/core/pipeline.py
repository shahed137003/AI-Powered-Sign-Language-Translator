import numpy as np
import cv2

from .config import *
from .model import ModelWrapper
from .keypoints import holistic, extract_keypoints
from .llm import LLMTranslator

import sys
sys.path.append(str(PREPROCESSING_PATH))
from preprocessing.pipeline_v3 import preprocess_sequence_global


class ASLPipeline:
    def __init__(self, llm):
        self.llm = llm
        self.model = ModelWrapper()
        

        self.buffer = [] 
        self.recording = False #initially there is no recording
        self.silence_counter = 0

        self.sentence_buffer = []#buffer to store the words in
        self.english_sentence = ""

        self.last_pred = "Waiting..."
        self.last_conf = 0.0

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb) #mediapipe

        hand_seen = results.left_hand_landmarks or results.right_hand_landmarks
        kp = extract_keypoints(results) #extract keypoints

        if hand_seen:#if we detect hands so start recording
            if not self.recording:
                self.recording = True
                self.buffer = []

            self.buffer.append(kp)
            self.silence_counter = 0

        elif self.recording: #detects silence
            self.silence_counter += 1
            self.buffer.append(kp)

            if self.silence_counter > COOLDOWN_FRAMES:
                self.recording = False

                if len(self.buffer) > MIN_FRAMES:
                    raw_seq = np.array(self.buffer)
                    proc_seq = preprocess_sequence_global(raw_seq)

                    T = proc_seq.shape[0]

                    if T > TARGET_LEN: #if longer than target frames so trim
                        final_input = proc_seq[:TARGET_LEN]
                    else: #if shorter than target frames so pad with 0
                        final_input = np.concatenate(
                            [proc_seq, np.zeros((TARGET_LEN - T, proc_seq.shape[1]))]
                        )

                    pred, conf = self.model.predict(final_input) #get the predicted and confidence from model

                    self.last_pred = pred
                    self.last_conf = conf

                    if conf > CONF_THRESHOLD: #if confidence < 60% we ignore that word
                        if not self.sentence_buffer or self.sentence_buffer[-1] != pred:
                            self.sentence_buffer.append(pred)

                    if len(self.sentence_buffer) >= 3: #after 3 words we call LLM
                        gloss = " ".join(self.sentence_buffer)
                        print("Calling LLM with:", gloss)

                        try:
                            self.english_sentence = self.llm.translate(gloss)
                            print("LLM OUTPUT:", self.english_sentence)
                        except Exception as e:
                            print("LLM ERROR:", e)
                        self.sentence_buffer = []

                self.buffer = []

        return self.last_pred, self.last_conf, self.sentence_buffer, self.english_sentence