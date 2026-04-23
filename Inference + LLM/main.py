import cv2
from core.pipeline import ASLPipeline
from core.llm import LLMTranslator

llm = LLMTranslator()
pipeline = ASLPipeline(llm)

cap = cv2.VideoCapture(0) #open webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)#this is used to mirror view (natural)

    pred, conf, words, sentence = pipeline.process_frame(frame) #entire system runs in pipeline so just call it
    #now we will display the gesture, its confidence, words buffer, english sentence
    h, w, _ = frame.shape  

    cv2.putText(frame, f"Gesture: {pred}", (20, h - 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {conf:.1f}%", (20, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Words: {' '.join(words)}", (20, h - 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    display_text = sentence if sentence.strip() else "Waiting for more words..."

    cv2.putText(frame, f"English: {display_text}", (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("ASL System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #to press q to exit
        break

cap.release()
cv2.destroyAllWindows()