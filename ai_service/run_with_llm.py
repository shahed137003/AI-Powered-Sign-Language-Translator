import cv2
from InferenceLLM.core import ASLPipeline
from InferenceLLM.core.llm import LLMTranslator

llm = LLMTranslator()
pipeline = ASLPipeline(llm)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    pred, conf, words, sentence = pipeline.process_frame(frame)

    h, w = frame.shape[:2]
    cv2.putText(frame, f"Gesture: {pred}", (20, h-120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Conf: {conf:.1f}%", (20, h-90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Words: {' '.join(words)}", (20, h-150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"English: {sentence}", (20, h-180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # ========== MANUAL OVERRIDE KEYS ==========
    cv2.imshow("ASL System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):          # press 'r' to reset the buffer (erase wrong signs)
        pipeline.reset()
        print("🔄 Buffer cleared")
    elif key == ord('t'):        # press 't' to test LLM with current words
        gloss = " ".join(pipeline.sentence_buffer)
        if gloss.strip():
            print(f"📤 Manually sending to LLM: '{gloss}'")
            complete, english = llm.translate(gloss)
            print(f"LLM -> complete={complete}, english='{english}'")
        else:
            print("No words in buffer")
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()