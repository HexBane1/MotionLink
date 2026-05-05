"""UDP packet sender for MotionLink v2.

Sends two kinds of packets to Unity over loopback (config.UDP_HOST:PORT):

POSITION (every frame, ~30Hz):
    {"type": "position", "hand": "R", "x", "y", "z", "hand_scale",
     "palm_up", "zone", "timestamp"}

EVENT (only on state change OR action fire):
    {"type": "event", "hand", "event", "prev_state", "new_state",
     "x", "y", "zone", "timestamp"}

Selective transmission rule (efficiency E2): the caller decides when to
send each kind. UDPSender does not deduplicate, throttle, or buffer --
it just serializes and ships. Position packets are expected every frame;
event packets only on FSM transitions and velocity-detector fires.

Run this module directly to verify the packet format end-to-end:
    python udp_sender.py            # self-test: spawn a listener, send one
                                    # of each packet kind, print results
    python udp_sender.py listen     # passive listener; useful while a
                                    # second terminal runs main.py later
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from typing import Any, Dict, Optional

import config
from gesture_classifier import GestureState, StateTransition
from gesture_primitives import HandFeatures
from velocity_detectors import ActionEvent


def _r(value: float) -> float:
    """Round a float to the configured precision; reduces packet size and
    avoids leaking float-printer noise into Unity."""
    return round(value, config.UDP_FLOAT_PRECISION)


class UDPSender:
    """Thin wrapper over a UDP datagram socket. Fire-and-forget."""

    def __init__(self,
                 host: str = config.UDP_HOST,
                 port: int = config.UDP_PORT) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)
        self._sent_count = 0

    @property
    def sent_count(self) -> int:
        """Total packets successfully sent. Useful for the HUD / debug overlay."""
        return self._sent_count

    def send_position(self, features: HandFeatures) -> None:
        """Position packet for the per-frame stream. Caller invokes every
        frame, even if the hand hasn't moved -- Unity uses the steady
        cadence to drive cursor lerp."""
        payload: Dict[str, Any] = {
            "type":       "position",
            "hand":       features.hand_label,
            "x":          _r(features.wrist_x),
            "y":          _r(features.wrist_y),
            "z":          _r(features.wrist_z),
            "hand_scale": _r(features.hand_scale),
            "palm_up":    bool(features.palm_up),
            "zone":       features.zone,
            "timestamp":  features.timestamp,
        }
        self._send(payload)

    def send_state_event(self, transition: StateTransition) -> None:
        """Event packet for an FSM state transition (e.g. TRACKING -> GRAB).
        Caller invokes only when HandStateMachine.step returns a transition."""
        f = transition.features
        payload: Dict[str, Any] = {
            "type":       "event",
            "hand":       transition.hand_label,
            "event":      transition.new_state.value,
            "prev_state": transition.prev_state.value,
            "new_state":  transition.new_state.value,
            "x":          _r(f.wrist_x) if f is not None else None,
            "y":          _r(f.wrist_y) if f is not None else None,
            "zone":       f.zone if f is not None else None,
            "timestamp":  transition.timestamp,
        }
        self._send(payload)

    def send_action_event(self, action: ActionEvent) -> None:
        """Event packet for a velocity-detector fire (FLIP / SQUEEZE / SEASON).
        Action events do not change FSM state, so prev_state == new_state == GRAB."""
        payload: Dict[str, Any] = {
            "type":       "event",
            "hand":       action.hand_label,
            "event":      action.action,
            "prev_state": GestureState.GRAB.value,
            "new_state":  GestureState.GRAB.value,
            "x":          _r(action.wrist_x),
            "y":          _r(action.wrist_y),
            "zone":       action.zone,
            "timestamp":  action.timestamp,
        }
        self._send(payload)

    def close(self) -> None:
        """Release the socket. Safe to call multiple times."""
        try:
            self._socket.close()
        except OSError:
            pass

    def __enter__(self) -> "UDPSender":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _send(self, payload: Dict[str, Any]) -> None:
        try:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._socket.sendto(data, self._addr)
            self._sent_count += 1
        except OSError:
            # Loopback target may not be listening yet (Unity not running).
            # UDP is fire-and-forget; swallow the error so the camera loop
            # doesn't crash. Real network errors at this boundary are rare
            # and surface as Unity simply not seeing packets.
            pass


# --- test harness ---------------------------------------------------------

def _run_self_test() -> None:
    """Spawn a short-lived listener, send one of each packet kind, print
    everything that arrived. Verifies wire format and round-trip."""
    import threading
    import time

    received: list = []
    listener_ready = threading.Event()

    def listen() -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((config.UDP_HOST, config.UDP_PORT))
        sock.settimeout(0.25)
        listener_ready.set()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                data, _ = sock.recvfrom(4096)
                received.append(data.decode("utf-8"))
            except socket.timeout:
                continue
        sock.close()

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()
    if not listener_ready.wait(timeout=2.0):
        print("listener thread failed to bind; is port in use?")
        return

    fake_features = HandFeatures(
        hand_label="R",
        timestamp=time.time(),
        wrist_x=0.5123, wrist_y=0.3001, wrist_z=-0.0876,
        thumb_open=False, index_open=False, middle_open=False,
        ring_open=False, pinky_open=False, open_count=0,
        palm_up=True, palm_score=0.42, hand_scale=0.1834,
        zone="Z5", landmarks=[],
    )
    fake_transition = StateTransition(
        hand_label="R",
        prev_state=GestureState.TRACKING,
        new_state=GestureState.GRAB,
        timestamp=time.time(),
        features=fake_features,
    )
    fake_action = ActionEvent(
        hand_label="R",
        action="FLIP",
        timestamp=time.time(),
        zone="Z6",
        wrist_x=0.6612, wrist_y=0.7344,
    )

    with UDPSender() as sender:
        sender.send_position(fake_features)
        sender.send_state_event(fake_transition)
        sender.send_action_event(fake_action)
        sent = sender.sent_count

    listener.join(timeout=3.0)

    print(f"sent {sent} packets, listener received {len(received)}:\n")
    for raw in received:
        try:
            decoded = json.loads(raw)
            print(json.dumps(decoded, indent=2))
            print()
        except json.JSONDecodeError:
            print(f"<undecodable> {raw!r}")

    if sent != len(received):
        print(f"WARNING: {sent} sent vs {len(received)} received -- "
              "possible packet loss on loopback or Windows firewall prompt")
        sys.exit(1)


def _run_listener_loop() -> None:
    """Long-running passive listener. Print every packet that arrives.
    Useful in a second terminal while main.py runs."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((config.UDP_HOST, config.UDP_PORT))
    print(f"listening on {config.UDP_HOST}:{config.UDP_PORT}  (Ctrl+C to quit)")
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            try:
                decoded = json.loads(data.decode("utf-8"))
                kind = decoded.get("type", "?")
                if kind == "position":
                    print(f"POS  hand={decoded['hand']} "
                          f"x={decoded['x']:.3f} y={decoded['y']:.3f} "
                          f"zone={decoded['zone']} "
                          f"palm_up={decoded['palm_up']}")
                elif kind == "event":
                    print(f"EVT  hand={decoded['hand']} "
                          f"event={decoded['event']} "
                          f"({decoded['prev_state']} -> {decoded['new_state']}) "
                          f"zone={decoded['zone']}")
                else:
                    print(f"???  {decoded}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"<bad packet> {data!r}: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MotionLink UDP sender")
    parser.add_argument("mode", nargs="?", choices=("test", "listen"),
                        default="test",
                        help="test: round-trip self-check (default). "
                             "listen: long-running listener that prints "
                             "incoming packets from another process.")
    args = parser.parse_args()
    if args.mode == "listen":
        _run_listener_loop()
    else:
        _run_self_test()
