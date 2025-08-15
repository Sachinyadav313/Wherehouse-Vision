import cv2
from ultralytics import YOLO
import os
import threading
from gtts import gTTS
from playsound import playsound
import time

# --- NEW TEXT-TO-SPEECH (TTS) SETUP using gTTS ---
def speak_message(message, lang='en'):
    """Generates an audio file from text and plays it."""
    try:
        # Create the gTTS object
        tts = gTTS(text=message, lang=lang, slow=False)
        
        # Save the audio file
        audio_file = 'speech.mp3'
        tts.save(audio_file)
        
        # Play the audio file
        playsound(audio_file)
        
        # Optional: remove the file after playing
        os.remove(audio_file)
    except Exception as e:
        print(f"TTS Error: {e}")

def main():
    # --- MODEL LOADING ---
    model_path = os.path.join('models', 'hard_hat_detection_final4', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    model = YOLO(model_path)
    print("✅ Model loaded successfully.")

    # --- WEBCAM INITIALIZATION ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("✅ Webcam opened. Press 'q' to quit.")

    # --- MAIN LOOP VARIABLES ---
    last_known_violation_state = False
    speech_thread = None # To hold the speech thread

    while True:
        # --- FRAME CAPTURE ---
        success, frame = cap.read()
        if not success:
            break

        # --- MODEL INFERENCE ---
        results = model(frame, stream=True, verbose=False)

        # --- PROCESS DETECTIONS ---
        head_count = 0
        helmet_count = 0
        person_count = 0
        vest_count = 0
        annotated_frame = frame.copy()

        for r in results:
            annotated_frame = r.plot()
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                if class_name in ['person', 'head']:
                    person_count += 1
                if class_name == 'head':
                    head_count += 1
                if 'helmet' in class_name:
                    helmet_count += 1
                if 'vest' in class_name:
                    vest_count += 1
        
        # --- STATUS AND AUDIO LOGIC ---
        current_violation_state = head_count > 0 and head_count > helmet_count

        if current_violation_state != last_known_violation_state:
            message = "Warning. Safety violation detected." if current_violation_state else "Situation clear. All safe."
            
            # Ensure previous speech thread is finished before starting a new one
            if speech_thread is None or not speech_thread.is_alive():
                speech_thread = threading.Thread(target=speak_message, args=(message,))
                speech_thread.start()
        
        last_known_violation_state = current_violation_state

        # --- DRAW VISUAL INFO PANEL ---
        status_text = "VIOLATION" if current_violation_state else "SAFE"
        status_color = (0, 0, 255) if current_violation_state else (0, 255, 0)
        
        panel_width = 350
        panel_x_start = annotated_frame.shape[1] - panel_width
        
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (panel_x_start, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 0), -1)
        alpha = 0.6
        annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)

        cv2.putText(annotated_frame, "STATUS:", (panel_x_start + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, status_text, (panel_x_start + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

        cv2.putText(annotated_frame, "OBJECTS DETECTED:", (panel_x_start + 20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"- Persons: {person_count}", (panel_x_start + 20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"- Helmets: {helmet_count}", (panel_x_start + 20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"- Vests: {vest_count}", (panel_x_start + 20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # --- DISPLAY FRAME ---
        cv2.imshow("Live Safety Monitoring", annotated_frame)

        # --- EXIT CONDITION ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- CLEANUP ---
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Resources released.")

if __name__ == '__main__':
    main()