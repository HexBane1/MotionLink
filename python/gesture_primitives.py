"""Shared per-frame hand feature computation for MotionLink v2.

Computes finger states, palm orientation, hand scale, wrist position, and
zone ONCE per frame from raw MediaPipe Hands landmarks. Downstream modules
(gesture_classifier, velocity_detectors, udp_sender) consume the same
HandFeatures struct so we never re-derive the same primitives twice
(efficiency rule E1: single-pass landmark feature computation).

Coordinate system (top-down camera):
    x  in [0, 1] image left/right
    y  in [0, 1] image; y=0 is far from body, y=1 is near body
    z          MediaPipe relative depth, more negative = closer to lens
                 -> middle_mcp.z < wrist.z means knuckles are closer to the
                    camera than the wrist, i.e. palm faces UP toward the lens

Run this module directly to launch the test harness:
    python gesture_primitives.py
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config

Landmark = Tuple[float, float, float]


@dataclass
class HandFeatures:
    """Per-frame snapshot of one hand. Built once, shared by all detectors."""

    hand_label: str            # "L" or "R" (user's hand, after handedness fix)
    timestamp: float           # seconds since epoch

    wrist_x: float             # 0..1 image space
    wrist_y: float             # 0..1 image space (0 = far, 1 = near body)
    wrist_z: float             # MediaPipe relative depth (negative = closer)

    thumb_open: bool
    index_open: bool
    middle_open: bool
    ring_open: bool
    pinky_open: bool
    open_count: int            # 0..5

    palm_up: bool              # True if palm faces the camera
    palm_score: float          # signed palm-facing score; +1 = clearly UP, -1 = clearly DOWN
    hand_scale: float          # 3D wrist -> middle_mcp distance, ~0.15-0.25

    zone: str                  # "Z1".."Z6"

    landmarks: List[Landmark] = field(default_factory=list)


# --- helpers --------------------------------------------------------------

def _dist3(a: Landmark, b: Landmark) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _dist2(a: Landmark, b: Landmark) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _finger_extended(tip: Landmark, pip: Landmark, wrist: Landmark,
                     ratio: float) -> bool:
    """A non-thumb finger is extended when its tip is meaningfully farther
    from the wrist than its PIP joint."""
    return _dist3(tip, wrist) > _dist3(pip, wrist) * ratio


def _thumb_extended(thumb_tip: Landmark, index_mcp: Landmark,
                    hand_scale: float, scale_factor: float) -> bool:
    """Thumb is extended when its tip sits far from the index MCP in the
    palm plane. 2D distance is more stable than 3D here because thumb
    abduction is mostly an in-plane motion when the hand is on the desk."""
    return _dist2(thumb_tip, index_mcp) > hand_scale * scale_factor


def _palm_facing_score(wrist: Landmark, index_mcp: Landmark,
                       pinky_mcp: Landmark, raw_handedness: str) -> float:
    """Signed orientation score for the palm: +1 = clearly facing the
    camera (palm UP), -1 = clearly facing away (palm DOWN), 0 = edge-on.

    Method: take the 2D cross product of (index_mcp - wrist) and
    (pinky_mcp - wrist) in image coordinates, normalize by |u||v| so the
    result is sin(angle from u to v), then flip the sign for the user's
    left hand because the (index, pinky) pair sits on opposite sides of
    the palm depending on which hand it is. The sign correction uses
    MediaPipe's RAW handedness label (the topology in image coords),
    independent of any user-facing flip we apply later.

    Why not the old z-only check: middle_mcp.z vs wrist.z only carries
    signal when the hand is tilted relative to the camera. For a hand
    held perpendicular to a forward-facing webcam (the most common case),
    both landmarks sit at similar depth and the sign is noise. The cross
    product instead measures which side of the palm is facing the camera
    using the rotation direction of the wrist->index vs wrist->pinky pair,
    which is a structural property of the hand pose.
    """
    ux = index_mcp[0] - wrist[0]
    uy = index_mcp[1] - wrist[1]
    vx = pinky_mcp[0] - wrist[0]
    vy = pinky_mcp[1] - wrist[1]

    cross_z = ux * vy - uy * vx
    mag = math.sqrt((ux * ux + uy * uy) * (vx * vx + vy * vy))
    if mag < 1e-9:
        return 0.0

    score = cross_z / mag
    sign = 1.0 if raw_handedness[:1].upper() == "R" else -1.0
    return score * sign


def _zone_for(x: float, y: float) -> str:
    """Map normalized wrist (x, y) to one of Z1..Z6.

    Far half  (y < ZONE_Y_SPLIT):   Z1 left | Z2 center | Z3 right
    Near half (y >= ZONE_Y_SPLIT):  Z4 left | Z5 center | Z6 right
    """
    near = y >= config.ZONE_Y_SPLIT
    if x < config.ZONE_X_LEFT:
        col = 0
    elif x < config.ZONE_X_RIGHT:
        col = 1
    else:
        col = 2
    base = 4 if near else 1
    return f"Z{base + col}"


# --- main entry point -----------------------------------------------------

def compute_hand_features(
    landmarks: List[Landmark],
    raw_handedness: str,
    timestamp: Optional[float] = None,
    flip_handedness: Optional[bool] = None,
) -> HandFeatures:
    """Build the shared HandFeatures struct for one detected hand.

    Args:
        landmarks: 21-tuple list of (x, y, z) from MediaPipe Hands.
        raw_handedness: "Left" or "Right" as MediaPipe reports it.
        timestamp: seconds since epoch; defaults to time.time().
        flip_handedness: override config.FLIP_HANDEDNESS. With a mirrored
            webcam image, MediaPipe's "Right" is actually the user's left
            hand and vice versa, so we flip by default.

    Returns:
        HandFeatures populated for this frame.

    Raises:
        ValueError: if landmarks does not have exactly 21 entries.
    """
    if len(landmarks) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(landmarks)}")

    if timestamp is None:
        timestamp = time.time()
    if flip_handedness is None:
        flip_handedness = config.FLIP_HANDEDNESS

    wrist      = landmarks[config.LM_WRIST]
    middle_mcp = landmarks[config.LM_MIDDLE_MCP]
    index_mcp  = landmarks[config.LM_INDEX_MCP]
    pinky_mcp  = landmarks[config.LM_PINKY_MCP]

    hand_scale = _dist3(wrist, middle_mcp)
    if hand_scale < 1e-6:
        hand_scale = 1e-6  # downstream code may divide by this

    index_open  = _finger_extended(landmarks[config.LM_INDEX_TIP],
                                   landmarks[config.LM_INDEX_PIP],
                                   wrist, config.FINGER_EXTEND_RATIO)
    middle_open = _finger_extended(landmarks[config.LM_MIDDLE_TIP],
                                   landmarks[config.LM_MIDDLE_PIP],
                                   wrist, config.FINGER_EXTEND_RATIO)
    ring_open   = _finger_extended(landmarks[config.LM_RING_TIP],
                                   landmarks[config.LM_RING_PIP],
                                   wrist, config.FINGER_EXTEND_RATIO)
    pinky_open  = _finger_extended(landmarks[config.LM_PINKY_TIP],
                                   landmarks[config.LM_PINKY_PIP],
                                   wrist, config.FINGER_EXTEND_RATIO)
    thumb_open  = _thumb_extended(landmarks[config.LM_THUMB_TIP],
                                  index_mcp, hand_scale,
                                  config.THUMB_EXTEND_SCALE)

    open_count = (int(thumb_open) + int(index_open) + int(middle_open)
                  + int(ring_open) + int(pinky_open))

    palm_score = _palm_facing_score(wrist, index_mcp, pinky_mcp,
                                    raw_handedness)
    palm_up = palm_score > config.PALM_UP_SCORE_THRESHOLD

    label = raw_handedness[0].upper()
    if flip_handedness:
        label = "R" if label == "L" else "L"

    zone = _zone_for(wrist[0], wrist[1])

    return HandFeatures(
        hand_label=label,
        timestamp=timestamp,
        wrist_x=wrist[0],
        wrist_y=wrist[1],
        wrist_z=wrist[2],
        thumb_open=thumb_open,
        index_open=index_open,
        middle_open=middle_open,
        ring_open=ring_open,
        pinky_open=pinky_open,
        open_count=open_count,
        palm_up=palm_up,
        palm_score=palm_score,
        hand_scale=hand_scale,
        zone=zone,
        landmarks=list(landmarks),
    )


# --- test harness ---------------------------------------------------------

def _format_features(f: HandFeatures) -> str:
    return (
        f"[{f.hand_label}] zone={f.zone} "
        f"open={f.open_count} palm_up={int(f.palm_up)} "
        f"pscore={f.palm_score:+.2f} "
        f"scale={f.hand_scale:.3f} "
        f"wrist=({f.wrist_x:.2f},{f.wrist_y:.2f},{f.wrist_z:+.3f}) "
        f"fingers=T{int(f.thumb_open)}"
        f"I{int(f.index_open)}"
        f"M{int(f.middle_open)}"
        f"R{int(f.ring_open)}"
        f"P{int(f.pinky_open)}"
    )


def _run_webcam_test() -> None:
    """Webcam loop: print HandFeatures per frame. Press q to quit."""
    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open webcam index {config.WEBCAM_INDEX}")

    print("MotionLink primitives test. Press q in the video window to quit.")

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        last_print = 0.0
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
            features_this_frame: List[HandFeatures] = []

            if result.multi_hand_landmarks and result.multi_handedness:
                for lm_set, hd in zip(result.multi_hand_landmarks,
                                      result.multi_handedness):
                    pts = [(p.x, p.y, p.z) for p in lm_set.landmark]
                    raw_label = hd.classification[0].label
                    feats = compute_hand_features(pts, raw_label, now)
                    features_this_frame.append(feats)
                    mp_draw.draw_landmarks(frame, lm_set,
                                           mp_hands.HAND_CONNECTIONS)

            # throttle prints to ~5 Hz so the console stays readable
            if features_this_frame and now - last_print > 0.2:
                for f in features_this_frame:
                    print(_format_features(f))
                last_print = now

            # on-screen overlay: the most recent feature line per hand
            h, _ = frame.shape[:2]
            y_text = 24
            for f in features_this_frame:
                cv2.putText(frame, _format_features(f), (8, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 0), 1, cv2.LINE_AA)
                y_text += 22

            cv2.imshow("MotionLink primitives test (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_webcam_test()
