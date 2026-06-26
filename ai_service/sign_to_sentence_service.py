import numpy as np
import sys
from pathlib import Path

# Import your working core modules
from InferenceLLM.core.pipeline import ASLPipeline
from InferenceLLM.core.llm import LLMTranslator

class SignToSentenceWebSocketService:
    """
    Wraps your exact working ASLPipeline to operate over WebSockets.
    It relies entirely on the pipeline's internal recording and LLM queue states.
    """
    def __init__(self, api_key: str = None):
        print("Initializing LLM Translator...")
        self.llm = LLMTranslator(api_key=api_key)
        print("Initializing ASL Pipeline...")
        self.pipeline = ASLPipeline(self.llm)
        
    def process_ws_keypoints(self, keypoints: list, hands_visible: bool):
        """
        Bridges the React WebSocket data into your existing ASLPipeline logic.
        """
        kp = np.array(keypoints)
        
        # 1. Execute the core pipeline logic directly using the new method we added earlier
        # (This handles the recording, silence counting, TCN prediction, and pushing to the LLM queue)
        result = self.pipeline.process_keypoints(kp, hands_visible)
        
        # 2. Return the exact state back to React
        if result is None:
            if self.pipeline.recording:
                return {
                    "status": "collecting",
                    "frames_collected": len(self.pipeline.buffer)
                }
            return {"status": "idle"}
            
        pred, conf, words, english_sentence = result
        
        if pred == "LLM_TRANSLATION":
            return {
                "status": "sentence",
                "sentence": english_sentence
            }
        # If the pipeline just made a prediction
        elif pred != "Waiting...":
            # Pass the current state to React
            return {
                "status": "success",
                "text": pred,
                "confidence": conf / 100.0,  # Send as decimal (0.0 to 1.0)
                "sentence_buffer": " ".join(words),
                "english_sentence": english_sentence
            }
            
        # If the LLM has updated the sentence but no new prediction was made
        elif english_sentence and english_sentence != "Waiting for more words...":
             return {
                 "status": "sentence",
                 "sentence": english_sentence
             }

        # Idle state
        return {"status": "idle"}

    def reset(self):
        """Manual override to clear buffers"""
        self.pipeline.reset()