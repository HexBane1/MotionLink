import cv2
import mediapipe as mp
import time

def main():
    # 1. Initialize MediaPipe Hands and Drawing Utils
    mp_hands = mp.solutions.hands
    # We set static_image_mode to False so it tracks across frames (faster)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # 2. Initialize the webcam (0 is usually the default laptop camera)
    cap = cv2.VideoCapture(0)

    # Variables for FPS calculation
    previous_time = 0

    print("Starting webcam feed. Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame. Check your webcam connection.")
            break

        # Flip the frame horizontally for a natural selfie-view display
        frame = cv2.flip(frame, 1)

        # 3. Convert BGR to RGB (OpenCV uses BGR, but MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 4. Process the frame to find hands
        results = hands.process(frame_rgb)

        # 5. Draw the 21 landmarks if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the dots and the connecting lines
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )

        # 6. Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time) if (current_time - previous_time) > 0 else 0
        previous_time = current_time
        
        cv2.putText(
            frame, 
            f'FPS: {int(fps)}', 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )

        # 7. Display the final image
        cv2.imshow("Hand Tracking Milestone", frame)

        # 8. Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()