import cv2
import mediapipe as mp
import time
import math
from collections import deque

def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    current_state = "unknown"
    wrist_x_history = deque(maxlen=15) 
    last_swipe_time = 0
    swipe_cooldown = 1.0 
    
    last_grab_time = 0
    grab_lockout_duration = 0.5 

    cap = cv2.VideoCapture(0)
    previous_time = 0
    print("Starting webcam feed. Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        gesture_event = None 

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark
                current_time = time.time()

                # --- A. Swipe Detection ---
                current_wrist_x = landmarks[0].x
                wrist_x_history.append(current_wrist_x)

                if len(wrist_x_history) == 15 and (current_time - last_swipe_time) > swipe_cooldown:
                    movement = wrist_x_history[-1] - wrist_x_history[0]
                    if movement > 0.15: 
                        gesture_event = "SWIPE RIGHT"
                        last_swipe_time = current_time
                    elif movement < -0.15:
                        gesture_event = "SWIPE LEFT"
                        last_swipe_time = current_time

                # --- B. Static Pose Detection ---
                if not gesture_event: 
                    # 1. Raw Data Collection
                    pinch_dist = get_distance(landmarks[4], landmarks[8])
                    # Relaxed threshold: increased from 0.05 to 0.08 to make pinching easier
                    is_pinch_distance = pinch_dist < 0.08 

                    fingers_up = [
                        landmarks[8].y < landmarks[6].y,   # Index
                        landmarks[12].y < landmarks[10].y, # Middle
                        landmarks[16].y < landmarks[14].y, # Ring
                        landmarks[20].y < landmarks[18].y  # Pinky
                    ]
                    open_fingers_count = sum(fingers_up)

                    # 2. Pose Priority Logic
                    detected_pose = "unknown"
                    
                    if open_fingers_count == 0:
                        detected_pose = "fist"
                        last_grab_time = current_time 
                        
                    # Relaxed Pinch: Now only requires 1 other finger to be open (prevents fist confusion)
                    elif is_pinch_distance and (current_time - last_grab_time) > grab_lockout_duration and open_fingers_count >= 1:
                        detected_pose = "pinch"
                        
                    elif open_fingers_count >= 3:
                        detected_pose = "open_palm"

                    # --- C. State Transitions (THE FIX) ---
                    # We ONLY update the state if the camera clearly sees a valid pose.
                    # This prevents "unknown" frame glitches from wiping out your held Grab!
                    if detected_pose != "unknown" and detected_pose != current_state:
                        if detected_pose == "fist":
                            gesture_event = "GRAB"
                        elif detected_pose == "open_palm" and current_state == "fist":
                            gesture_event = "RELEASE"
                        elif detected_pose == "pinch":
                            gesture_event = "PINCH"
                        elif detected_pose == "open_palm" and current_state == "pinch":
                            gesture_event = "UNPINCH"
                        current_state = detected_pose

        # --- Logging ---
        if gesture_event:
            timestamp = time.strftime("%H:%M:%S", time.localtime(current_time))
            print(f"[{timestamp}] - Gesture Detected: {gesture_event}")
            cv2.putText(frame, f'Event: {gesture_event}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Calculate FPS
        current_time_fps = time.time()
        fps = 1 / (current_time_fps - previous_time) if (current_time_fps - previous_time) > 0 else 0
        previous_time = current_time_fps
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking Milestone", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()