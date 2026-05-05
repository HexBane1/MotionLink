import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque


def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def get_palm_orientation(landmarks):
    """
    Raw single-frame palm orientation via Z-depth comparison (top-down camera).
    Middle knuckle (9) Z < wrist (0) Z  →  palm_up  (facing ceiling / camera)
    Middle knuckle (9) Z > wrist (0) Z  →  palm_down (facing desk / table)
    Call get_smoothed_palm() instead of this directly — it votes over 7 frames.
    """
    return "palm_up" if landmarks[9].z < landmarks[0].z else "palm_down"


def get_smoothed_palm(palm_history, landmarks):
    """
    Appends the current raw orientation to palm_history (deque maxlen=7) and
    returns the majority vote across all stored readings.
    This prevents rapid palm_up / palm_down flickering between frames.
    """
    palm_history.append(get_palm_orientation(landmarks))
    return "palm_up" if palm_history.count("palm_up") > len(palm_history) // 2 else "palm_down"


def get_hand_scale(landmarks):
    """
    Bounding box diagonal of all landmark XY positions.
    Larger value = hand is higher / closer to top-down camera.
    Used as a depth proxy in Unity (hand_scale → world Y height).
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def detect_circular_motion(motion_history):
    """
    Returns True if the last N wrist positions describe a roughly circular arc.
    Criteria:
      - Total angular sweep > 270°  (at least ¾ of a full circle)
      - Radius coefficient of variation < 0.5  (consistent radius = round, not random)
    Requires at least 15 points in history.
    """
    if len(motion_history) < 15:
        return False

    points = np.array(motion_history, dtype=np.float32)
    cx, cy = np.mean(points[:, 0]), np.mean(points[:, 1])

    radii = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    radius_mean = np.mean(radii)

    if radius_mean < 0.015:       # circle too small to be intentional
        return False

    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    total_rotation = abs(np.sum(np.diff(np.unwrap(angles))))
    radius_cv = np.std(radii) / radius_mean

    return total_rotation > (3 * math.pi / 2) and radius_cv < 0.5


def detect_chop(wrist_z_deque):
    """
    Detects a sharp downward drop along Z (hand raised then slammed to table).
    Top-down camera: hand lifted → more negative Z; hand dropped → more positive Z.
    Splits the deque in half and checks: max(second half) - min(first half) > 0.08.
    """
    if len(wrist_z_deque) < 8:
        return False

    zs = list(wrist_z_deque)
    half = len(zs) // 2
    return (max(zs[half:]) - min(zs[:half])) > 0.08


def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # ── Tuning constants ───────────────────────────────────────────────────────
    DEBOUNCE_FRAMES = 8     # frames a static pose must be held before it commits
    SWIPE_COOLDOWN  = 1.0   # seconds
    GRAB_LOCKOUT    = 0.5   # seconds — prevents pinch firing right after a fist
    CHOP_COOLDOWN   = 0.8   # seconds
    STIR_COOLDOWN   = 1.5   # seconds

    def _make_hand_state():
        return {
            # ── Confirmed state ────────────────────────────────────────────────
            "current_state": "unknown",

            # ── Debounce buffer ────────────────────────────────────────────────
            # A pose must match pose_candidate for DEBOUNCE_FRAMES consecutive
            # frames before current_state is updated and an event fires.
            "pose_candidate": "unknown",
            "pose_candidate_frames": 0,

            # ── Position history ───────────────────────────────────────────────
            "wrist_x": deque(maxlen=15),          # left/right
            "wrist_y": deque(maxlen=15),          # forward/back on table (top-down)
            "wrist_z": deque(maxlen=10),          # height above table
            "motion_history": deque(maxlen=20),   # (x, y) tuples for stir detection

            # ── Palm orientation smoothing ─────────────────────────────────────
            # Stores last 7 raw palm readings; majority vote used as final value.
            "palm_history": deque(maxlen=7),

            # ── Cooldown timestamps ────────────────────────────────────────────
            "last_swipe_time": 0,
            "last_grab_time":  0,
            "last_chop_time":  0,
            "last_stir_time":  0,

            # ── Display ────────────────────────────────────────────────────────
            "display_event": "",
            "event_timer":   0,
        }

    hand_states = {"Left": _make_hand_state(), "Right": _make_hand_state()}

    cap = cv2.VideoCapture(0)
    previous_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(frame_rgb)

        current_time = time.time()

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                actual_hand = handedness.classification[0].label   # "Left" / "Right"
                hand_tag    = actual_hand[0]                        # "L" / "R"
                state       = hand_states[actual_hand]
                landmarks   = hand_landmarks.landmark
                gesture_event = None

                # ── Update position history ────────────────────────────────────
                wrist = landmarks[0]
                state["wrist_x"].append(wrist.x)
                state["wrist_y"].append(wrist.y)
                state["wrist_z"].append(wrist.z)
                state["motion_history"].append((wrist.x, wrist.y))

                # ── Smoothed palm orientation (7-frame majority vote) ──────────
                palm       = get_smoothed_palm(state["palm_history"], landmarks)
                hand_scale = get_hand_scale(landmarks)

                # ── CHOP: sharp downward Z drop (motion — no debounce needed) ──
                if (current_time - state["last_chop_time"]) > CHOP_COOLDOWN:
                    if detect_chop(state["wrist_z"]):
                        gesture_event = f"CHOP {hand_tag}"
                        state["last_chop_time"] = current_time

                # ── STIR: circular wrist trajectory (motion — no debounce) ─────
                if not gesture_event and \
                        (current_time - state["last_stir_time"]) > STIR_COOLDOWN:
                    if detect_circular_motion(list(state["motion_history"])):
                        gesture_event = f"STIR {hand_tag}"
                        state["last_stir_time"] = current_time

                # ── SWIPE left / right (motion — no debounce) ─────────────────
                if not gesture_event and \
                        len(state["wrist_x"]) == 15 and \
                        (current_time - state["last_swipe_time"]) > SWIPE_COOLDOWN:
                    x_move = state["wrist_x"][-1] - state["wrist_x"][0]
                    if x_move > 0.15:
                        gesture_event = f"SWIPE RIGHT {hand_tag}"
                        state["last_swipe_time"] = current_time
                    elif x_move < -0.15:
                        gesture_event = f"SWIPE LEFT {hand_tag}"
                        state["last_swipe_time"] = current_time

                # ── Static Pose Detection (debounced) ─────────────────────────
                if not gesture_event:
                    pinch_dist        = get_distance(landmarks[4], landmarks[8])
                    is_pinch_distance = pinch_dist < 0.08

                    fingers_up = [
                        landmarks[8].y  < landmarks[6].y,    # index
                        landmarks[12].y < landmarks[10].y,   # middle
                        landmarks[16].y < landmarks[14].y,   # ring
                        landmarks[20].y < landmarks[18].y,   # pinky
                    ]
                    open_fingers_count = sum(fingers_up)

                    # Raw pose read for this frame
                    raw_pose = "unknown"

                    if open_fingers_count == 0:
                        raw_pose = "fist"

                    elif is_pinch_distance and open_fingers_count >= 1 and \
                            (current_time - state["last_grab_time"]) > GRAB_LOCKOUT:
                        raw_pose = "pinch_down" if palm == "palm_down" else "pinch"

                    elif open_fingers_count >= 3:
                        raw_pose = "open_palm_up" if palm == "palm_up" else "open_palm_down"

                    # ── Debounce logic ─────────────────────────────────────────
                    # Advance the candidate counter only when the same pose is
                    # seen on consecutive frames.  Any break resets to 1.
                    if raw_pose == state["pose_candidate"] and raw_pose != "unknown":
                        state["pose_candidate_frames"] += 1
                    else:
                        state["pose_candidate"]        = raw_pose
                        state["pose_candidate_frames"] = 1

                    # Commit only after DEBOUNCE_FRAMES consecutive matching frames
                    # and only if it's actually a new state.
                    if state["pose_candidate_frames"] >= DEBOUNCE_FRAMES and \
                            raw_pose != "unknown" and \
                            raw_pose != state["current_state"]:

                        # Update grab lockout when a fist is confirmed
                        if raw_pose == "fist":
                            state["last_grab_time"] = current_time

                        # Map confirmed pose → gesture event string
                        if raw_pose == "fist":
                            gesture_event = f"GRAB {hand_tag}"
                        elif raw_pose == "open_palm_down":
                            gesture_event = f"RELEASE {hand_tag}"
                        elif raw_pose == "open_palm_up":
                            gesture_event = f"HOLD {hand_tag}"
                        elif raw_pose == "pinch_down":
                            gesture_event = f"PINCH_DOWN {hand_tag}"
                        elif raw_pose == "pinch":
                            gesture_event = f"PINCH {hand_tag}"

                        state["current_state"]         = raw_pose
                        state["pose_candidate_frames"] = 0   # reset after commit

                # ── Fire event ────────────────────────────────────────────────
                if gesture_event:
                    ts = time.strftime("%H:%M:%S", time.localtime(current_time))
                    print(f"[{ts}] - {gesture_event}")
                    state["display_event"] = gesture_event
                    state["event_timer"]   = current_time

                # ── Per-hand debug overlay ────────────────────────────────────
                h, w   = frame.shape[:2]
                wx_px  = int(wrist.x * w)
                wy_px  = int(wrist.y * h)
                # Show: smoothed palm / debounce progress / confirmed state
                progress = f"{state['pose_candidate_frames']}/{DEBOUNCE_FRAMES}"
                label    = (f"{palm}  [{progress}]  "
                            f"{state['pose_candidate']} → {state['current_state']}")
                cv2.putText(frame, label, (wx_px - 80, wy_px - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

        # ── FPS counter ───────────────────────────────────────────────────────
        now = time.time()
        fps = 1.0 / (now - previous_time) if (now - previous_time) > 0 else 0
        previous_time = now
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ── Recent gesture banner ─────────────────────────────────────────────
        y_pos = 80
        for hand_name in ("Right", "Left"):
            s = hand_states[hand_name]
            if now - s["event_timer"] < 1.5:
                cv2.putText(frame, f"Event: {s['display_event']}",
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)
                y_pos += 45

        cv2.imshow("MotionLink - Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
