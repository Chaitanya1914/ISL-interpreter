import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
from tensorflow.keras.models import load_model
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Set up the Text-To-Speech function
def speak(text):
    # Initializes the TTS engine in a separate thread so video doesn't freeze!
    engine = pyttsx3.init()
    # Optional: adjust the speaking rate
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

def main():
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    
    try:
        model = load_model('isl_model.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print("Could not load 'isl_model.h5'. Did you run train_model.py first?")
        return

    # Real-time state tracking variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7

    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic
    
    print("Opening Application... Press 'q' to quit.")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Detection
            image, results = mediapipe_detection(frame, holistic)
            # Draw on screen
            draw_styled_landmarks(image, results)
            
            # 1. Real-time Prediction Logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:] # Always slice the last 30 frames
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                prediction_idx = np.argmax(res)
                predictions.append(prediction_idx)
                
                # Check for stability (ensure the model predicts the same thing consistently)
                if np.unique(predictions[-10:])[0] == prediction_idx:
                    if res[prediction_idx] > threshold:
                        predicted_word = actions[prediction_idx]
                        
                        if len(sentence) > 0: 
                            # If it's a NEW word, add it to the sentence and speak it!
                            if predicted_word != sentence[-1]:
                                sentence.append(predicted_word)
                                threading.Thread(target=speak, args=(predicted_word,), daemon=True).start()
                        else:
                            sentence.append(predicted_word)
                            threading.Thread(target=speak, args=(predicted_word,), daemon=True).start()

                # Keep the sentence array to 5 words max for the display
                if len(sentence) > 5: 
                    sentence = sentence[-5:]
            
            # 2. On-screen Graphics Display
            # Draw a banner at the top
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            # Draw Text
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show image
            cv2.imshow('Indian Sign Language & Text To Speech', image)
            
            # Exit clause
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
