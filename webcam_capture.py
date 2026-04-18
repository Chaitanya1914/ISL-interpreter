import cv2
import mediapipe as mp

def main():
    # 1. Initialize MediaPipe Holistic model mapping (for hands, face, and pose)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # 2. Initialize the Webcam (0 is usually the default laptop camera)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("Webcam successfully opened. Press 'q' to quit.")

    # Initialize the Holistic model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            # 3. Read a frame from the webcam
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # 4. Preprocess the frame
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference. Also, MediaPipe needs RGB format, but OpenCV reads in BGR.
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 5. Process the frame to find face, pose, and hands
            results = holistic.process(frame_rgb)

            # 6. Post-process: Draw the landmarks on the original frame
            frame.flags.writeable = True

            # Draw Left Hand landmarks
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Draw Right Hand landmarks
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # (Optional) Draw Pose and Face landmarks
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # --- YOUR INFERENCE LOGIC GOES HERE ---
            # e.g., Extract landmark coordinates from results.left_hand_landmarks
            # Pass them to your trained ISL model, get a prediction (e.g., "Hello")
            prediction_text = "Waiting for prediction..." # Replace with actual logic
            
            # Add text to display the prediction on screen
            cv2.putText(frame, f'Prediction: {prediction_text}', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 7. Display the Output
            cv2.imshow('Indian Sign Language Recognition', frame)

            # 8. Graceful Exit on pressing 'q'
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # 9. Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()