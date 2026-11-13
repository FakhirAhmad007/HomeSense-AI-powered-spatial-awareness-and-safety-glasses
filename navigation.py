"""Turns the target's bounding box into a spoken direction. No distances in metres are
ever computed: only (a) horizontal offset from image centre and (b) how much of the frame
the box fills, which is a relative-size proxy for 'closer / farther'."""
from enum import Enum
from typing import Optional


class Guidance(Enum):
    LEFT = "Move a little to the left."
    RIGHT = "Move a little to the right."
    FORWARD = "Move slightly forward."
    BACK = "Move slightly backward."
    ARRIVED = "The object is directly in front of you and is nearby."


class NavigationGuide:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._off = None
        self._fill = None
        self._centered = None
        self._arrived = False
        self._reset_arrival()

    def _reset_arrival(self) -> None:
        self._arr_start = None
        self._arr_count = 0

    def update(self, box, frame_w: int, frame_h: int, ts: float, label: str) -> Optional[Guidance]:
        """Returns a Guidance, or None meaning 'nothing new to say' (target not visible this
        frame, or arrival still being validated)."""
        cfg = self.cfg
        if box is None:
            self._reset_arrival()
            return None
        off = (box.cx - frame_w / 2) / frame_w                 # -0.5 (far left) .. +0.5 (far right)
        fill = max(box.w / frame_w, box.h / frame_h)           # relative size of target in view
        a = cfg.smoothing_alpha
        self._off = off if self._off is None else a * off + (1 - a) * self._off
        self._fill = fill if self._fill is None else a * fill + (1 - a) * self._fill
        off, fill = self._off, self._fill

        # Hysteresis: easy to stay centred once centred, so guidance does not flip left/right.
        if self._centered is None:
            self._centered = abs(off) <= cfg.center_exit
        elif self._centered and abs(off) > cfg.center_exit:
            self._centered = False
        elif not self._centered and abs(off) <= cfg.center_enter:
            self._centered = True

        if not self._centered:
            self._reset_arrival()
            self._arrived = False
            return Guidance.LEFT if off < 0 else Guidance.RIGHT

        near = cfg.near_fill.get(label, cfg.near_fill_default)
        too_close = min(0.95, near * cfg.too_close_factor)
        # Once arrived, the object must get clearly smaller before we say "forward" again (hysteresis),
        # otherwise tiny size changes would make the guidance flip between "nearby" and "forward".
        leave_near = near * 0.85 if self._arrived else near
        if fill < leave_near:
            self._reset_arrival()
            self._arrived = False
            return Guidance.FORWARD
        if fill > too_close:
            self._reset_arrival()
            self._arrived = False
            return Guidance.BACK

        # near AND centred: only announce arrival after consistent observations
        if self._arr_start is None:
            self._arr_start = ts
        self._arr_count += 1
        if self._arr_count >= cfg.arrival_frames and ts - self._arr_start >= cfg.arrival_min_s:
            self._arrived = True
            return Guidance.ARRIVED      # returned on EVERY frame while near + centred; the controller paces the speech
        return None
