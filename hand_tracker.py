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
        max_num_hands=2, # Increased back to 2
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # --- THE FIX: Independent State Dictionaries for Left and Right ---
    hand_states = {
        "Left": {
            "current_state": "unknown",
            "wrist_x": deque(maxlen=15),
            "last_swipe_time": 0,
            "last_grab_time": 0,
            "display_event": "",
            "event_timer": 0
        },
        "Right": {
            "current_state": "unknown",
            "wrist_x": deque(maxlen=15),
            "last_swipe_time": 0,
            "last_grab_time": 0,
            "display_event": "",
            "event_timer": 0
        }
    }
    
    swipe_cooldown = 1.0 
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

        # We must check BOTH landmarks and handedness (L/R labels)
        if results.multi_hand_landmarks and results.multi_handedness:
            
            # Zip lets us loop through the hand skeletons and their L/R labels at the same time
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the raw label from MediaPipe
                raw_label = handedness.classification[0].label 
                
                # The label is already correct, no swapping needed!
                actual_hand = raw_label
                hand_tag = actual_hand[0] # Gets just 'L' or 'R'
                
                # Grab the specific memory state for this hand
                state = hand_states[actual_hand]

                landmarks = hand_landmarks.landmark
                current_time = time.time()
                gesture_event = None 

                # --- A. Swipe Detection ---
                current_wrist_x = landmarks[0].x
                state["wrist_x"].append(current_wrist_x)

                if len(state["wrist_x"]) == 15 and (current_time - state["last_swipe_time"]) > swipe_cooldown:
                    movement = state["wrist_x"][-1] - state["wrist_x"][0]
                    if movement > 0.15: 
                        gesture_event = f"SWIPE RIGHT {hand_tag}"
                        state["last_swipe_time"] = current_time
                    elif movement < -0.15:
                        gesture_event = f"SWIPE LEFT {hand_tag}"
                        state["last_swipe_time"] = current_time

                # --- B. Static Pose Detection ---
                if not gesture_event: 
                    pinch_dist = get_distance(landmarks[4], landmarks[8])
                    is_pinch_distance = pinch_dist < 0.08 

                    fingers_up = [
                        landmarks[8].y < landmarks[6].y,   
                        landmarks[12].y < landmarks[10].y, 
                        landmarks[16].y < landmarks[14].y, 
                        landmarks[20].y < landmarks[18].y  
                    ]
                    open_fingers_count = sum(fingers_up)

                    detected_pose = "unknown"
                    
                    if open_fingers_count == 0:
                        detected_pose = "fist"
                        state["last_grab_time"] = current_time 
                        
                    elif is_pinch_distance and (current_time - state["last_grab_time"]) > grab_lockout_duration and open_fingers_count >= 1:
                        detected_pose = "pinch"
                        
                    elif open_fingers_count >= 3:
                        detected_pose = "open_palm"

                    # --- C. State Transitions ---
                    if detected_pose != "unknown" and detected_pose != state["current_state"]:
                        if detected_pose == "fist":
                            gesture_event = f"GRAB {hand_tag}"
                        elif detected_pose == "open_palm" and state["current_state"] == "fist":
                            gesture_event = f"RELEASE {hand_tag}"
                        elif detected_pose == "pinch":
                            gesture_event = f"PINCH {hand_tag}"
                        elif detected_pose == "open_palm" and state["current_state"] == "pinch":
                            gesture_event = f"UNPINCH {hand_tag}"
                        
                        state["current_state"] = detected_pose

                # --- Update Event Display Data ---
                if gesture_event:
                    timestamp = time.strftime("%H:%M:%S", time.localtime(current_time))
                    print(f"[{timestamp}] - {gesture_event}")
                    state["display_event"] = gesture_event
                    state["event_timer"] = current_time

        # --- Draw UI on Screen ---
        # Calculate FPS
        current_time_fps = time.time()
        fps = 1 / (current_time_fps - previous_time) if (current_time_fps - previous_time) > 0 else 0
        previous_time = current_time_fps
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw Events (Keeps text on screen for 1.5 seconds so you can read it)
        y_position = 80
        for hand_name in ["Right", "Left"]:
            state = hand_states[hand_name]
            if current_time_fps - state["event_timer"] < 1.5:
                cv2.putText(frame, f'Event: {state["display_event"]}', (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                y_position += 45 # Shift down so left and right hands don't overlap

        cv2.imshow("Hand Tracking Milestone", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()