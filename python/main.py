"""Entry point for the MotionLink v2 vision pipeline.

Webcam -> MediaPipe Hands -> compute_hand_features -> per-hand
HandStateMachine -> per-hand VelocityDetectorBank -> UDPSender.

Position packets fire every frame for every detected hand. State-event
packets fire only when the FSM transitions. Action-event packets fire
only when a velocity detector matches its profile. This is the
"selective UDP" rule (efficiency E2): position is a continuous stream;
events are deliberate.

Usage:
    python main.py                  # zone gating ON (spec behaviour)
    python main.py --no-zone-gate   # let any zone fire FLIP/SQUEEZE/SEASON
                                    # (handy while Unity zones aren't built)
    python main.py --no-window      # headless; only console + UDP

Press q in the preview window (or Ctrl+C in this terminal) to quit.
To watch packets in another terminal:  python udp_sender.py listen
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from typing import Deque, Dict, Tuple

import cv2
import mediapipe as mp

import config
from gesture_classifier import GestureState, HandStateMachine
from gesture_primitives import HandFeatures, compute_hand_features
from udp_sender import UDPSender
from velocity_detectors import VelocityDetectorBank


_STATE_COLORS: Dict[GestureState, Tuple[int, int, int]] = {
    GestureState.IDLE:     (128, 128, 128),
    GestureState.TRACKING: (255, 255, 255),
    GestureState.GRAB:     (60,  220, 60),
    GestureState.RELEASE:  (60,  140, 255),
    GestureState.HOLD:     (60,  220, 220),
    GestureState.INSPECT:  (220, 60,  220),
}

_EVENT_BANNER_DURATION = 0.8  # seconds the on-screen event banner sticks


def main() -> int:
    parser = argparse.ArgumentParser(description="MotionLink v2 vision pipeline")
    parser.add_argument("--no-zone-gate", action="store_true",
                        help="disable zone gating on action detectors so "
                             "FLIP / SQUEEZE / SEASON can fire from any "
                             "zone (useful before Unity zones exist)")
    parser.add_argument("--no-window", action="store_true",
                        help="run headless without the OpenCV preview")
    args = parser.parse_args()

    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"error: cannot open webcam index {config.WEBCAM_INDEX}",
              file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # most-recent-frame mode (E3)

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    fsms: Dict[str, HandStateMachine] = {
        "L": HandStateMachine("L"),
        "R": HandStateMachine("R"),
    }
    banks: Dict[str, VelocityDetectorBank] = {
        "L": VelocityDetectorBank("L", enforce_zone=not args.no_zone_gate),
        "R": VelocityDetectorBank("R", enforce_zone=not args.no_zone_gate),
    }

    # Sliding window for an FPS estimate shown on the overlay
    frame_dts: Deque[float] = deque(maxlen=30)
    last_event_per_hand: Dict[str, Tuple[str, float]] = {
        "L": ("", 0.0), "R": ("", 0.0),
    }

    print(f"MotionLink v2 running. Sending to "
          f"{config.UDP_HOST}:{config.UDP_PORT}.")
    print(f"Zone gating: {'ON' if not args.no_zone_gate else 'OFF (test mode)'}")
    print("Press q in the video window to quit (or Ctrl+C here).\n")

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands, UDPSender() as sender:
        try:
            while True:
                t_frame_start = time.time()
                ok, frame = cap.read()
                if not ok:
                    continue

                if config.WEBCAM_FLIP_HORIZ:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)

                now = time.time()
                features_by_hand: Dict[str, HandFeatures] = {}

                # 1) primitives
                if result.multi_hand_landmarks and result.multi_handedness:
                    for lm_set, hd in zip(result.multi_hand_landmarks,
                                          result.multi_handedness):
                        pts = [(p.x, p.y, p.z) for p in lm_set.landmark]
                        raw_label = hd.classification[0].label
                        feats = compute_hand_features(pts, raw_label, now)
                        features_by_hand[feats.hand_label] = feats
                        if not args.no_window:
                            mp_draw.draw_landmarks(
                                frame, lm_set, mp_hands.HAND_CONNECTIONS)

                # 2) position packets (one per detected hand, every frame)
                for feats in features_by_hand.values():
                    sender.send_position(feats)

                # 3) state machine + velocity detectors per hand
                for label in ("L", "R"):
                    f = features_by_hand.get(label)
                    fsm = fsms[label]
                    transition = fsm.step(f, now)
                    if transition is not None:
                        sender.send_state_event(transition)
                        last_event_per_hand[label] = (
                            transition.new_state.value, now)
                        ts = time.strftime("%H:%M:%S", time.localtime(now))
                        print(f"[{ts}] {label}: "
                              f"{transition.prev_state.value} -> "
                              f"{transition.new_state.value}")
                    for action in banks[label].update(f, fsm.state, now):
                        sender.send_action_event(action)
                        last_event_per_hand[label] = (action.action, now)
                        ts = time.strftime("%H:%M:%S", time.localtime(now))
                        print(f"[{ts}] {label}: {action.action} "
                              f"@ {action.zone}")

                # 4) FPS bookkeeping + overlay
                frame_dts.append(time.time() - t_frame_start)
                if not args.no_window:
                    avg_dt = sum(frame_dts) / len(frame_dts) if frame_dts else 0
                    fps = (1.0 / avg_dt) if avg_dt > 0 else 0.0
                    _draw_overlay(frame, fsms, features_by_hand,
                                  last_event_per_hand, now, fps,
                                  sender.sent_count)
                    cv2.imshow("MotionLink v2 (q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            print("\ninterrupted")

    cap.release()
    cv2.destroyAllWindows()
    return 0


# --- overlay rendering ---------------------------------------------------

def _draw_overlay(frame, fsms: Dict[str, HandStateMachine],
                  features_by_hand: Dict[str, HandFeatures],
                  last_event: Dict[str, Tuple[str, float]],
                  now: float, fps: float, sent_count: int) -> None:
    h, w = frame.shape[:2]
    _draw_zone_grid(frame, w, h)

    cv2.putText(frame, f"FPS {fps:.0f}  pkt {sent_count}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 2, cv2.LINE_AA)

    y_text = 50
    for label in ("L", "R"):
        fsm = fsms[label]
        feature = features_by_hand.get(label)
        color = _STATE_COLORS[fsm.state]

        line = f"{label}: {fsm.state.value}"
        if feature is not None:
            line += (f"  zone={feature.zone}  open={feature.open_count}"
                     f"  palm_up={int(feature.palm_up)}")
        cv2.putText(frame, line, (8, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2, cv2.LINE_AA)
        y_text += 26

        evt, fired_at = last_event[label]
        if evt and (now - fired_at) < _EVENT_BANNER_DURATION:
            cv2.putText(frame, f"{label}: {evt}!", (8, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        color, 3, cv2.LINE_AA)
            y_text += 32


def _draw_zone_grid(frame, w: int, h: int) -> None:
    """Faint 3x2 zone grid + Z1..Z6 labels so the user can see which zone
    their wrist is in during testing."""
    x_left  = int(config.ZONE_X_LEFT  * w)
    x_right = int(config.ZONE_X_RIGHT * w)
    y_split = int(config.ZONE_Y_SPLIT * h)
    grid_color = (90, 90, 90)
    cv2.line(frame, (x_left, 0),  (x_left, h),  grid_color, 1, cv2.LINE_AA)
    cv2.line(frame, (x_right, 0), (x_right, h), grid_color, 1, cv2.LINE_AA)
    cv2.line(frame, (0, y_split), (w, y_split), grid_color, 1, cv2.LINE_AA)

    labels = [["Z1", "Z2", "Z3"], ["Z4", "Z5", "Z6"]]
    cell_xs = [x_left // 2, (x_left + x_right) // 2, (x_right + w) // 2]
    cell_ys = [y_split // 2, (y_split + h) // 2]
    for r, row in enumerate(labels):
        for c, name in enumerate(row):
            cv2.putText(frame, name, (cell_xs[c] - 14, cell_ys[r]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        grid_color, 2, cv2.LINE_AA)


if __name__ == "__main__":
    sys.exit(main())
