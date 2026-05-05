"""Hub-and-spoke state machine + static pose recognition for MotionLink v2.

Consumes the HandFeatures struct from gesture_primitives and emits state
transitions for one hand at a time. Implements the HARD RULE that all
transitions go through TRACKING -- e.g. GRAB -> HOLD must pass through
TRACKING first.

Static poses recognised here:
    GRAB    closed fist  (open_count == 0)
    RELEASE open palm facing DOWN
    HOLD    open palm facing UP
    INSPECT index + middle extended, ring + pinky curled

Action sub-states (FLIP, SQUEEZE, SEASON) are NOT handled here -- they live
in velocity_detectors.py (STEP 3) and gate off GRAB.

Hysteresis (spec rules A1, A2):
    Enter: POSE_ENTER_FRAMES consecutive strict matches from TRACKING.
    Exit:  POSE_EXIT_FRAMES consecutive "clear breach" frames.
A frame in the ambiguous middle (e.g. open_count == 1 while in GRAB)
neither advances entry nor counts toward exit -- this is what gives the
state its sticky feel without locking it.

Run this module directly to see live state transitions:
    python gesture_classifier.py
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Deque, Dict, Optional, Tuple

import config
from gesture_primitives import HandFeatures


class GestureState(Enum):
    IDLE     = "IDLE"
    TRACKING = "TRACKING"
    GRAB     = "GRAB"
    RELEASE  = "RELEASE"
    HOLD     = "HOLD"
    INSPECT  = "INSPECT"


# Static poses reachable from TRACKING.
STATIC_POSES: Tuple[GestureState, ...] = (
    GestureState.GRAB,
    GestureState.RELEASE,
    GestureState.HOLD,
    GestureState.INSPECT,
)


@dataclass
class StateTransition:
    """Emitted whenever the FSM changes state."""
    hand_label: str
    prev_state: GestureState
    new_state: GestureState
    timestamp: float
    features: Optional[HandFeatures]


# --- pose specifications -------------------------------------------------
# Each pose has two predicates against HandFeatures:
#   enter_match:  strict, used only from TRACKING to begin entering the pose
#   exit_breach:  obvious break from the pose, used while held to begin
#                 leaving. Frames that satisfy NEITHER are "ambiguous middle"
#                 and are ignored by the exit counter.
@dataclass(frozen=True)
class _PoseSpec:
    state: GestureState
    enter_match: Callable[[HandFeatures], bool]
    exit_breach: Callable[[HandFeatures], bool]


def _grab_enter(f: HandFeatures) -> bool:
    return f.open_count == 0


def _grab_breach(f: HandFeatures) -> bool:
    return f.open_count >= config.GRAB_EXIT_OPEN_COUNT


def _release_enter(f: HandFeatures) -> bool:
    return (not f.palm_up) and f.open_count >= config.STATIC_POSE_OPEN_MIN_ENTRY


def _release_breach(f: HandFeatures) -> bool:
    return f.palm_up or f.open_count <= config.STATIC_POSE_OPEN_MAX_EXIT


def _hold_enter(f: HandFeatures) -> bool:
    return f.palm_up and f.open_count >= config.STATIC_POSE_OPEN_MIN_ENTRY


def _hold_breach(f: HandFeatures) -> bool:
    return (not f.palm_up) or f.open_count <= config.STATIC_POSE_OPEN_MAX_EXIT


def _inspect_enter(f: HandFeatures) -> bool:
    return (f.index_open and f.middle_open
            and not f.ring_open and not f.pinky_open)


def _inspect_breach(f: HandFeatures) -> bool:
    # exit only when the index/middle pair breaks OR both ring AND pinky
    # come open (curling one of them is OK -- ambiguous middle).
    return ((not f.index_open) or (not f.middle_open)
            or (f.ring_open and f.pinky_open))


_POSE_SPECS: Dict[GestureState, _PoseSpec] = {
    GestureState.GRAB:    _PoseSpec(GestureState.GRAB,    _grab_enter,    _grab_breach),
    GestureState.RELEASE: _PoseSpec(GestureState.RELEASE, _release_enter, _release_breach),
    GestureState.HOLD:    _PoseSpec(GestureState.HOLD,    _hold_enter,    _hold_breach),
    GestureState.INSPECT: _PoseSpec(GestureState.INSPECT, _inspect_enter, _inspect_breach),
}


# Order matters: GRAB and INSPECT are evaluated first because their
# matching conditions are more specific than HOLD/RELEASE.
_ENTRY_ORDER: Tuple[GestureState, ...] = (
    GestureState.GRAB,
    GestureState.INSPECT,
    GestureState.HOLD,
    GestureState.RELEASE,
)


def _detect_entry_match(f: HandFeatures) -> Optional[GestureState]:
    """Return the first pose whose enter_match accepts this frame, or None."""
    for s in _ENTRY_ORDER:
        if _POSE_SPECS[s].enter_match(f):
            return s
    return None


# --- per-hand state machine ----------------------------------------------

class HandStateMachine:
    """FSM for one hand. Hub-and-spoke -- every static pose is reached via
    TRACKING. Hysteresis: N frames of strict match to enter, M frames of
    clear breach to leave."""

    def __init__(self, hand_label: str) -> None:
        self.hand_label = hand_label
        self.state: GestureState = GestureState.IDLE
        self._entry_counters: Dict[GestureState, int] = {p: 0 for p in STATIC_POSES}
        self._exit_counter: int = 0
        self._missing_frames: int = 0
        self._wrist_history: Deque[Tuple[float, float]] = deque(
            maxlen=config.IDLE_HISTORY_LEN)
        self._last_motion_time: float = 0.0

    # ----------------------------------------------------------------- API
    def step(self, features: Optional[HandFeatures], now: float
             ) -> Optional[StateTransition]:
        """Advance the FSM by one frame.

        Args:
            features: HandFeatures for this hand this frame, or None if
                MediaPipe did not detect this hand.
            now: current wall-clock timestamp (seconds since epoch).

        Returns:
            StateTransition if the state changed this frame, else None.
        """
        if features is None:
            return self._step_no_hand(now)
        return self._step_with_hand(features, now)

    # ---------------------------------------------------------- branches
    def _step_no_hand(self, now: float) -> Optional[StateTransition]:
        self._missing_frames += 1
        self._reset_counters()
        if (self.state is not GestureState.IDLE
                and self._missing_frames >= config.IDLE_MISSING_FRAMES):
            return self._transition(GestureState.IDLE, None, now)
        return None

    def _step_with_hand(self, f: HandFeatures, now: float
                        ) -> Optional[StateTransition]:
        self._missing_frames = 0
        self._track_motion(f, now)

        if self.state is GestureState.IDLE:
            return self._transition(GestureState.TRACKING, f, now)

        if self.state is GestureState.TRACKING:
            return self._tracking_step(f, now)

        return self._pose_step(f, now)

    def _tracking_step(self, f: HandFeatures, now: float
                       ) -> Optional[StateTransition]:
        # Motionless timeout in TRACKING -> IDLE.
        if (now - self._last_motion_time) >= config.IDLE_MOTIONLESS_SEC:
            return self._transition(GestureState.IDLE, f, now)

        matched = _detect_entry_match(f)
        for pose in STATIC_POSES:
            if pose is matched:
                self._entry_counters[pose] += 1
            else:
                self._entry_counters[pose] = 0

        if matched is not None and self._entry_counters[matched] >= config.POSE_ENTER_FRAMES:
            self._reset_counters()
            return self._transition(matched, f, now)
        return None

    def _pose_step(self, f: HandFeatures, now: float
                   ) -> Optional[StateTransition]:
        spec = _POSE_SPECS[self.state]
        if spec.exit_breach(f):
            self._exit_counter += 1
            if self._exit_counter >= config.POSE_EXIT_FRAMES:
                self._exit_counter = 0
                return self._transition(GestureState.TRACKING, f, now)
        else:
            # frame either still matches strictly or sits in the ambiguous
            # middle -- both reset the exit counter.
            self._exit_counter = 0
        return None

    # ---------------------------------------------------- bookkeeping
    def _track_motion(self, f: HandFeatures, now: float) -> None:
        self._wrist_history.append((f.wrist_x, f.wrist_y))
        if len(self._wrist_history) >= self._wrist_history.maxlen:
            x0, y0 = self._wrist_history[0]
            d = abs(f.wrist_x - x0) + abs(f.wrist_y - y0)  # cheap L1 distance
            if d > config.IDLE_MOTION_THRESHOLD:
                self._last_motion_time = now

    def _reset_counters(self) -> None:
        for k in self._entry_counters:
            self._entry_counters[k] = 0
        self._exit_counter = 0

    def _transition(self, new: GestureState, f: Optional[HandFeatures],
                    now: float) -> StateTransition:
        prev = self.state
        self.state = new
        if new is not GestureState.IDLE:
            # Reset the motion clock so a fresh TRACKING gets the full 2s.
            self._last_motion_time = now
        return StateTransition(
            hand_label=self.hand_label,
            prev_state=prev,
            new_state=new,
            timestamp=now,
            features=f,
        )


# --- test harness ---------------------------------------------------------

def _state_color(state: GestureState) -> Tuple[int, int, int]:
    return {
        GestureState.IDLE:     (128, 128, 128),
        GestureState.TRACKING: (255, 255, 255),
        GestureState.GRAB:     (60,  220, 60),
        GestureState.RELEASE:  (60,  140, 255),
        GestureState.HOLD:     (60,  220, 220),
        GestureState.INSPECT:  (220, 60,  220),
    }[state]


def _run_webcam_test() -> None:
    """Webcam loop that runs gesture_primitives + the FSM and prints every
    state transition. Press q in the video window to quit."""
    import time
    import cv2
    import mediapipe as mp

    from gesture_primitives import compute_hand_features

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open webcam index {config.WEBCAM_INDEX}")

    fsms = {"L": HandStateMachine("L"), "R": HandStateMachine("R")}

    print("MotionLink classifier test. Press q in the video window to quit.")
    print("Transitions will print to this console as they happen.\n")

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

            for label, fsm in fsms.items():
                t = fsm.step(features_by_hand.get(label), now)
                if t is not None:
                    ts = time.strftime("%H:%M:%S", time.localtime(now))
                    print(f"[{ts}] {t.hand_label}: "
                          f"{t.prev_state.value} -> {t.new_state.value}")

            # On-screen overlay: per-hand state and a feature digest
            y_text = 28
            for label in ("L", "R"):
                fsm = fsms[label]
                color = _state_color(fsm.state)
                text = f"{label}: {fsm.state.value}"
                f = features_by_hand.get(label)
                if f is not None:
                    text += (f"  open={f.open_count} palm_up={int(f.palm_up)}"
                             f" zone={f.zone}")
                cv2.putText(frame, text, (8, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            color, 2, cv2.LINE_AA)
                y_text += 26

            cv2.imshow("MotionLink classifier test (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_webcam_test()
