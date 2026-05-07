"""Velocity-based action detectors for MotionLink v2.

FLIP and combined SQUEEZE/SEASON are action sub-states of GRAB. Each detector:
        - only runs while the FSM is in GRAB (Lock 1, gated structurally)
        - uses hand-axis orientation to separate FLIP vs SQUEEZE/SEASON
        - keeps a TIME-based sliding window of wrist samples so behavior is
            consistent across 30fps / 60fps webcams
        - has its own per-hand cooldown
        - emits ActionEvent objects when its profile matches

Lock 2 (correct tool held) is NOT checked here -- it lives in Unity,
which is the only side that knows the player's inventory. Python sends
candidate FLIP / SQUEEZE / SEASON events; Unity's ToolGate decides
whether to apply gameplay effects.

Run this module directly to test the two detectors in isolation:
    python velocity_detectors.py
The standalone harness disables zone gating so you can fire any action
from any spot on screen as long as you're in GRAB.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import config
from gesture_classifier import GestureState, HandStateMachine
from gesture_primitives import HandFeatures


@dataclass
class ActionEvent:
    """Velocity-detector output. Sent to Unity for triple-lock validation."""
    hand_label: str
    action: str          # "FLIP" | "SQUEEZE" (combined squeeze/season)
    timestamp: float
    zone: str
    wrist_x: float
    wrist_y: float


# (wrist_x, wrist_y, timestamp)
Sample = Tuple[float, float, float]


def _prune(history: Deque[Sample], now: float, window_sec: float) -> None:
    """Drop samples older than window_sec from the left."""
    while history and (now - history[0][2]) > window_sec:
        history.popleft()


def _mirror_zone(zone: str) -> str:
    """Mirror a zone across the vertical axis (Z1<->Z3, Z4<->Z6)."""
    return {
        "Z1": "Z3",
        "Z2": "Z2",
        "Z3": "Z1",
        "Z4": "Z6",
        "Z5": "Z5",
        "Z6": "Z4",
    }.get(zone, zone)


def _flip_zone_for(hand_label: str) -> str:
    """Return the flip zone for this hand, accounting for mirrored preview."""
    if config.WEBCAM_FLIP_HORIZ and hand_label == "R":
        return _mirror_zone(config.ZONE_FLIP)
    return config.ZONE_FLIP


def _hand_axis(f: HandFeatures) -> str:
    """Return dominant hand axis: "x", "y", or "unknown"."""
    if len(f.landmarks) <= config.LM_MIDDLE_MCP:
        return "unknown"
    wrist = f.landmarks[config.LM_WRIST]
    middle_mcp = f.landmarks[config.LM_MIDDLE_MCP]
    dx = middle_mcp[0] - wrist[0]
    dy = middle_mcp[1] - wrist[1]
    adx = abs(dx)
    ady = abs(dy)
    if adx < 1e-6 and ady < 1e-6:
        return "unknown"
    ratio = config.HAND_AXIS_RATIO
    if adx >= ady * ratio:
        return "x"
    if ady >= adx * ratio:
        return "y"
    return "unknown"


# --- individual detectors -------------------------------------------------

class FlipDetector:
    """Y-axis back-and-forth with a velocity peak (spec: spatula flip)."""

    def __init__(self) -> None:
        self._history: Deque[Sample] = deque()
        self._last_fire: float = 0.0

    def reset(self) -> None:
        self._history.clear()

    def update(self, f: HandFeatures, now: float) -> bool:
        """Append a sample, return True iff a flip just fired."""
        self._history.append((f.wrist_x, f.wrist_y, now))
        _prune(self._history, now, config.FLIP_WINDOW_SEC)

        if now - self._last_fire < config.FLIP_COOLDOWN_SEC:
            return False
        if len(self._history) < config.FLIP_MIN_SAMPLES:
            return False

        peak_v = self._peak_y_velocity()
        ys = [s[1] for s in self._history]
        xs = [s[0] for s in self._history]
        dy_total = max(ys) - min(ys)
        dx_total = max(xs) - min(xs)
        changes = self._y_direction_changes()

        if (peak_v >= config.FLIP_PEAK_VELOCITY
                and dy_total >= config.FLIP_MIN_DISPLACEMENT
                and changes >= 1
                and dy_total >= config.FLIP_VERTICAL_RATIO * dx_total):
            self._last_fire = now
            self._history.clear()
            return True
        return False

    def _peak_y_velocity(self) -> float:
        max_v = 0.0
        prev = self._history[0]
        for s in list(self._history)[1:]:
            dt = s[2] - prev[2]
            if dt > 1e-6:
                v = abs((s[1] - prev[1]) / dt)
                if v > max_v:
                    max_v = v
            prev = s
        return max_v

    def _y_direction_changes(self) -> int:
        """Count Y-velocity sign flips (ignoring near-zero jitter)."""
        changes = 0
        prev_y = self._history[0][1]
        prev_sign = 0
        for s in list(self._history)[1:]:
            dy = s[1] - prev_y
            if abs(dy) < 1e-5:
                prev_y = s[1]
                continue
            sign = 1 if dy > 0 else -1
            if prev_sign and sign != prev_sign:
                changes += 1
            prev_sign = sign
            prev_y = s[1]
        return changes


class SqueezeDetector:
    """Gentle Y-axis back-and-forth (combined squeeze/season)."""

    def __init__(self) -> None:
        self._history: Deque[Sample] = deque()
        self._last_fire: float = 0.0

    def reset(self) -> None:
        self._history.clear()

    def update(self, f: HandFeatures, now: float) -> bool:
        self._history.append((f.wrist_x, f.wrist_y, now))
        _prune(self._history, now, config.SQUEEZE_WINDOW_SEC)

        if now - self._last_fire < config.SQUEEZE_COOLDOWN_SEC:
            return False
        if len(self._history) < config.SQUEEZE_MIN_SAMPLES:
            return False

        ys = [s[1] for s in self._history]
        xs = [s[0] for s in self._history]
        dy_total = max(ys) - min(ys)
        dx_total = max(xs) - min(xs)
        changes = self._y_direction_changes()

        if (dy_total >= config.SQUEEZE_MIN_Y_DROP
                and changes >= config.SEASON_MIN_DIR_CHANGES
                and dy_total >= config.SQUEEZE_VERTICAL_RATIO * dx_total):
            self._last_fire = now
            self._history.clear()
            return True
        return False

    def _y_direction_changes(self) -> int:
        """Count Y-velocity sign flips (ignoring small jitter)."""
        changes = 0
        prev_y = self._history[0][1]
        prev_sign = 0
        for s in list(self._history)[1:]:
            dy = s[1] - prev_y
            if abs(dy) < config.SEASON_PER_FRAME_DELTA:
                prev_y = s[1]
                continue
            sign = 1 if dy > 0 else -1
            if prev_sign and sign != prev_sign:
                changes += 1
            prev_sign = sign
            prev_y = s[1]
        return changes


# --- per-hand bank --------------------------------------------------------

class VelocityDetectorBank:
    """Per-hand bundle of action detectors with state and zone gating."""

    def __init__(self, hand_label: str, enforce_zone: bool = True) -> None:
        self.hand_label = hand_label
        self.enforce_zone = enforce_zone
        self.flip = FlipDetector()
        self.squeeze = SqueezeDetector()

    def update(self, features: Optional[HandFeatures],
               state: GestureState, now: float) -> List[ActionEvent]:
        """Advance all detectors. Returns the list of action events that
        fired this frame (usually 0 or 1).

        Detectors only run when the FSM is in GRAB. Leaving GRAB resets
        every detector's history so a fresh GRAB starts with no carry-over.
        """
        if features is None or state is not GestureState.GRAB:
            self._reset_all()
            return []

        events: List[ActionEvent] = []
        axis = _hand_axis(features)

        if axis in ("y", "unknown"):
            if self.flip.update(features, now):
                events.append(self._make_event("FLIP", features, now))
        else:
            self.flip.reset()

        if axis in ("x", "unknown"):
            if self.squeeze.update(features, now):
                events.append(self._make_event("SQUEEZE", features, now))
        else:
            self.squeeze.reset()

        return events

    def _reset_all(self) -> None:
        self.flip.reset()
        self.squeeze.reset()

    def _make_event(self, action: str, f: HandFeatures, now: float
                    ) -> ActionEvent:
        return ActionEvent(
            hand_label=self.hand_label,
            action=action,
            timestamp=now,
            zone=f.zone,
            wrist_x=f.wrist_x,
            wrist_y=f.wrist_y,
        )


# --- test harness ---------------------------------------------------------

_ACTION_COLORS = {
    "FLIP":    (60,  220, 60),    # green
    "SQUEEZE": (60,  140, 255),   # orange
}


def _run_webcam_test() -> None:
    import time
    import cv2
    import mediapipe as mp

    from gesture_primitives import compute_hand_features

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open webcam index {config.WEBCAM_INDEX}")

    fsms: Dict[str, HandStateMachine] = {
        "L": HandStateMachine("L"),
        "R": HandStateMachine("R"),
    }
    # Standalone testing -> ignore zone gating so any zone works. main.py
    # will instantiate these with enforce_zone=True.
    banks: Dict[str, VelocityDetectorBank] = {
        "L": VelocityDetectorBank("L", enforce_zone=False),
        "R": VelocityDetectorBank("R", enforce_zone=False),
    }
    last_action: Dict[str, Tuple[str, float]] = {"L": ("", 0.0), "R": ("", 0.0)}
    BANNER_DURATION = 0.8

    print("MotionLink velocity-detector test (enforce_zone=False).")
    print("Make a FIST (enter GRAB), then while staying in GRAB:")
    print("  FLIP            - move wrist up/down quickly (back-and-forth)")
    print("  SQUEEZE/SEASON  - gentle wrist up/down while in GRAB")
    print("Press q in the video window to quit.\n")

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
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
            if result.multi_hand_landmarks and result.multi_handedness:
                for lm_set, hd in zip(result.multi_hand_landmarks,
                                      result.multi_handedness):
                    pts = [(p.x, p.y, p.z) for p in lm_set.landmark]
                    raw_label = hd.classification[0].label
                    feats = compute_hand_features(pts, raw_label, now)
                    features_by_hand[feats.hand_label] = feats
                    mp_draw.draw_landmarks(frame, lm_set,
                                           mp_hands.HAND_CONNECTIONS)

            for label in ("L", "R"):
                f = features_by_hand.get(label)
                fsm = fsms[label]
                fsm.step(f, now)  # advance state; FSM transitions not the focus here
                events = banks[label].update(f, fsm.state, now)
                for ev in events:
                    ts = time.strftime("%H:%M:%S", time.localtime(now))
                    print(f"[{ts}] {ev.hand_label}: {ev.action} "
                          f"(zone={ev.zone}, x={ev.wrist_x:.2f}, "
                          f"y={ev.wrist_y:.2f})")
                    last_action[label] = (ev.action, now)

            # Overlay: per-hand state + recent action banner
            y_text = 28
            for label in ("L", "R"):
                fsm = fsms[label]
                feature = features_by_hand.get(label)
                line = f"{label}: {fsm.state.value}"
                if feature is not None:
                    line += f"  zone={feature.zone}"
                cv2.putText(frame, line, (8, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)
                y_text += 26
                action, fired_at = last_action[label]
                if action and (now - fired_at) < BANNER_DURATION:
                    color = _ACTION_COLORS.get(action, (0, 255, 255))
                    cv2.putText(frame, f"{label}: {action}!", (8, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                                color, 3, cv2.LINE_AA)
                    y_text += 36

            cv2.imshow("MotionLink velocity test (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_webcam_test()
