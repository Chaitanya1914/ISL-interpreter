import cv2
import mediapipe as mp
import pickle
import time
import numpy as np
import threading
import collections
from collections import Counter
from gtts import gTTS
import os
# using pygame's mixer for reliable, local async audio playback without popping up media players
import pygame 

# Initialize pygame mixer for audio
pygame.mixer.init()

def normalise(landmarks):
    """
    Normalization to perfectly match training pipeline:
    - subtract wrist (landmark 0)
    - divide by distance perfectly matching landmark 9
    """
    lms = np.array(landmarks).reshape(21, 3)
    wrist = lms[0]
    lms_shifted = lms - wrist
    dist = np.linalg.norm(lms_shifted[9])
    if dist < 1e-6:
        dist = 1e-6
    return (lms_shifted / dist).flatten()

def play_audio(text):
    """
    Generate audio using gTTS and stream it purely from RAM
    without writing any temporary mp3 files to the hard drive.
    """
    import io
    def _play():
        try:
            # Generate the TTS
            tts = gTTS(text=text, lang='en')
            
            # Stream directly out of memory
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # Load and play the in-memory audio
            pygame.mixer.music.load(fp, 'mp3')
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"Audio playback error: {e}")

    threading.Thread(target=_play, daemon=True).start()

def main():
    # Model Loading & Inference
    try:
        with open('isl_model.pkl', 'rb') as f:
            model_dict = pickle.load(f)
            
        # Extract MLPClassifier and LabelEncoder
        model = model_dict['model']
        encoder = model_dict['encoder']
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'isl_model.pkl' is in the exact same directory.")
        # Returning here to avoid failing later, comment this out if testing without a model
        return

    # MediaPipe Detection Setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    # We use min_detection_confidence=0.7 as requested
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # Rolling Buffer & Majority Vote Setup
    buffer = collections.deque(maxlen=20)
    last_emit_time = time.time()
    confirmed_word = ""

    # Frame Capture Setup
    cap = cv2.VideoCapture(0)
    # Attempt to set the hardware capture to 30 FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Starting webcam... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame (convert to RGB for MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        
        predicted_sign = None
        max_prob = 0.0

        # Ensure the app doesn't crash if the hand leaves the frame
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the skeleton on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                extracted_landmarks = []
                for lm in hand_landmarks.landmark:
                    extracted_landmarks.append([lm.x, lm.y, lm.z])
                
                # Apply normalization function
                normalized_vector = normalise(extracted_landmarks)
                
                # Fallback safeguard in case normalise isn't implemented right away
                if normalized_vector is None:
                    normalized_vector = np.array(extracted_landmarks).flatten()
                else:
                    normalized_vector = np.array(normalized_vector).flatten()
                
                # Ensure the vector is shaped perfectly (63-feature normalized vector)
                if len(normalized_vector) == 63:
                    # Pass the vector to predict_proba
                    probas = model.predict_proba([normalized_vector])[0]
                    max_idx = np.argmax(probas)
                    max_prob = probas[max_idx]
                    
                    # Confidence Gate: > 0.85
                    if max_prob > 0.85:
                        predicted_sign = encoder.inverse_transform([max_idx])[0]

        # Add to Rolling Buffer if a prediction met the probability gate
        if predicted_sign:
            buffer.append(predicted_sign)
        
        current_time = time.time()
        
        # Pull the majority vote from the buffer
        if len(buffer) > 0:
            most_common_sign, vote_count = Counter(buffer).most_common(1)[0]
            
            # Debounce Logic: 
            # Check if this sign diff from the last confirmed word, or 
            # if we just want to ensure 500ms has passed since the last emission
            if most_common_sign != confirmed_word:
                if (current_time - last_emit_time) > 0.5:
                    confirmed_word = most_common_sign
                    last_emit_time = current_time
                    
                    # Output the confirmed sign aloud and print it
                    print(f"Confirmed Sign: {confirmed_word} (Prob: {max_prob:.2f}, Votes: {vote_count}/20)")
                    play_audio(confirmed_word)
                    
                    # Flush the buffer so we don't immediately double-read the same word next frame
                    buffer.clear()
        
        # Display overlay on the OpenCV window
        if confirmed_word:
            cv2.rectangle(frame, (0, 0), (640, 60), (245, 117, 16), -1)
            cv2.putText(frame, f"Sign: {confirmed_word}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        cv2.imshow("Real-time ISL Detection", frame)
        
        # Keyboard kill code (Press Q)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == '__main__':
    main()
