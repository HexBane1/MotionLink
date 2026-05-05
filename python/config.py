"""Centralized thresholds and constants for MotionLink v2.

All magic numbers live here. Detectors and primitives import from this module
so tuning happens in one place.
"""

# --- MediaPipe Hands landmark indices ------------------------------------
LM_WRIST       = 0
LM_THUMB_CMC   = 1
LM_THUMB_MCP   = 2
LM_THUMB_IP    = 3
LM_THUMB_TIP   = 4
LM_INDEX_MCP   = 5
LM_INDEX_PIP   = 6
LM_INDEX_DIP   = 7
LM_INDEX_TIP   = 8
LM_MIDDLE_MCP  = 9
LM_MIDDLE_PIP  = 10
LM_MIDDLE_DIP  = 11
LM_MIDDLE_TIP  = 12
LM_RING_MCP    = 13
LM_RING_PIP    = 14
LM_RING_DIP    = 15
LM_RING_TIP    = 16
LM_PINKY_MCP   = 17
LM_PINKY_PIP   = 18
LM_PINKY_DIP   = 19
LM_PINKY_TIP   = 20

# --- finger-extension thresholds (gesture_primitives) --------------------
# A non-thumb finger counts as extended when the 3D distance from wrist to
# TIP is at least this multiple of the wrist-to-PIP distance. When the
# finger is curled into a fist the tip comes back toward the palm and the
# ratio drops below 1; fully extended fingers sit around 1.7-2.0.
FINGER_EXTEND_RATIO = 1.30

# Thumb is extended when the 2D distance from thumb tip to index MCP
# exceeds this multiple of hand_scale. Tucked thumb sits at ~0.4 * scale,
# splayed thumb at ~1.3 * scale, so 0.55 is a safe midpoint.
THUMB_EXTEND_SCALE  = 0.55

# Palm-up detection. We compute the palm normal from the
# (index_mcp, wrist, pinky_mcp) triangle and project it onto the camera
# axis, normalized so the result is sin(angle) of the wrist->index vs
# wrist->pinky pair (range -1..+1) with a per-hand sign correction. A
# clear palm-toward-camera reads ~+0.4-0.6; clear palm-away reads ~-0.4
# to -0.6; edge-on reads ~0. Threshold 0.15 gives a small dead zone where
# the orientation is ambiguous and we default to palm_up=False (the
# stricter HOLD branch won't fire there, which is the safer failure mode).
PALM_UP_SCORE_THRESHOLD = 0.15

# --- zone layout (top-down camera, Y=0 far edge, Y=1 near body) ----------
ZONE_Y_SPLIT  = 0.50   # Y < this  -> far half (Z1-Z3); else near half (Z4-Z6)
ZONE_X_LEFT   = 0.33   # X < this  -> left column
ZONE_X_RIGHT  = 0.66   # X >= this -> right column

# --- gesture classifier / state machine (gesture_classifier.py) ----------
# Hub-and-spoke FSM. From TRACKING we need POSE_ENTER_FRAMES consecutive
# strict matches to enter a static pose (A1: multi-frame consistency).
# To leave a held pose we need POSE_EXIT_FRAMES consecutive "clear breach"
# frames -- a frame that neither matches the held pose strictly nor sits in
# the ambiguous middle (A2: strict to enter, loose to maintain).
POSE_ENTER_FRAMES    = 8
POSE_EXIT_FRAMES     = 4

# Open-finger thresholds for HOLD / RELEASE entry vs. exit. Entry requires
# the strict count; exit fires when the count drops to or below the loose
# count (so open_count == 3 keeps you in HOLD without progress toward exit).
STATIC_POSE_OPEN_MIN_ENTRY = 4
STATIC_POSE_OPEN_MAX_EXIT  = 2

# GRAB: closed-fist breach when at least this many fingers go open.
GRAB_EXIT_OPEN_COUNT = 2

# --- IDLE detection ------------------------------------------------------
IDLE_MISSING_FRAMES   = 15    # frames without a hand detection -> IDLE
IDLE_MOTIONLESS_SEC   = 2.0   # seconds of stillness in TRACKING -> IDLE
IDLE_MOTION_THRESHOLD = 0.02  # normalized wrist drift counted as motion
IDLE_HISTORY_LEN      = 15    # wrist samples in the motion window (~0.5s)

# --- velocity detectors (velocity_detectors.py) --------------------------
# All three action detectors only run while the FSM is in GRAB (Lock 1)
# and -- when enforce_zone is on -- only in their own zone (Lock 3). Each
# detector keeps a TIME-based sliding window of wrist samples so behavior
# is the same at 30fps and 60fps webcams.

# FLIP: an X-axis swipe in Z6 (grill).
FLIP_WINDOW_SEC          = 0.40   # samples kept in the sliding window
FLIP_MIN_SAMPLES         = 5      # minimum samples before evaluating
FLIP_PEAK_VELOCITY       = 1.5    # min peak |dx/dt| in normalized units / sec
FLIP_MIN_DISPLACEMENT    = 0.15   # min |x_last - x_first| over the window
FLIP_HORIZONTAL_RATIO    = 2.0    # |dx| must exceed this * |dy| (mostly horizontal)
FLIP_COOLDOWN_SEC        = 1.0

# SQUEEZE: a brief downward Y press in Z5 (assembly).
SQUEEZE_WINDOW_SEC       = 0.40
SQUEEZE_MIN_SAMPLES      = 6
SQUEEZE_MIN_Y_DROP       = 0.06   # min total Y range over the window
SQUEEZE_VERTICAL_RATIO   = 1.3    # |dy| must exceed this * |dx| (mostly vertical)
SQUEEZE_COOLDOWN_SEC     = 0.5

# SEASON: rapid X-axis oscillation in Z5 (assembly).
SEASON_WINDOW_SEC        = 0.50
SEASON_MIN_SAMPLES       = 8
SEASON_MIN_DIR_CHANGES   = 3      # number of X-velocity sign flips in window
SEASON_MIN_X_AMPLITUDE   = 0.04   # min total X range over the window
SEASON_PER_FRAME_DELTA   = 0.005  # min |dx| between consecutive frames to count
SEASON_COOLDOWN_SEC      = 0.3

# Zone gating for the action-tier detectors. enforce_zone=True in main.py.
ZONE_FLIP    = "Z6"   # grill
ZONE_SQUEEZE = "Z5"   # assembly
ZONE_SEASON  = "Z5"   # assembly

# --- UDP transport (udp_sender.py) ---------------------------------------
# Unity listens; Python sends. Default to loopback so we never accidentally
# leak hand-tracking telemetry off-machine.
UDP_HOST   = "127.0.0.1"
UDP_PORT   = 5052
UDP_FLOAT_PRECISION = 4   # decimal places when serializing floats

# --- webcam (test harnesses only; main.py will own this in STEP 5) -------
WEBCAM_INDEX        = 0
WEBCAM_FLIP_HORIZ   = True   # mirror image so user sees a selfie view
FLIP_HANDEDNESS     = True   # mirrored image flips MediaPipe's L/R label
