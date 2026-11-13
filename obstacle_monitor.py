"""Decides whether a NON-target detection actually blocks the intended path.

An object is an obstacle only if ALL are true:
  * confident, not the target class, not on the ignore list
  * big enough (not a distant speck) and its base is in the lower part of the frame
    (i.e. near the floor in front of us, not a clock on a far wall)
  * it overlaps the 'corridor' between straight-ahead and the direction of the target
  * it is not a support surface the target rests on (table/bed/couch/bench, target base in its upper 75%)
  * the conflict persists (3 of last 5 frames) -- and clears only after 4 clean frames.

LIMITATION: a YOLO model only sees classes it was trained on. Walls, glass doors, stairs,
cables, etc. are invisible to it. This is an assistive hint, NOT a safety system."""
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class ObstacleStatus:
    blocked: bool
    newly_blocked: bool
    cleared: bool
    obstacle: Optional[object]
    suggestion: Optional[str]      # "left" | "right" | None  (None -> tell the user to stop)


class ObstacleMonitor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._hist = deque(maxlen=self.cfg.obstacle_window)
        self._clear_run = 0
        self.blocked = False
        self._current = None

    # ---- geometry helpers ----
    def _candidates(self, dets, target_label, target_box, W, H):
        c = self.cfg
        out = []
        for d in dets:
            if d.label == target_label or d.label in c.ignore_obstacle_classes or d.conf < c.obstacle_conf:
                continue
            if d.area / (W * H) < c.obstacle_min_area or d.y2 / H < c.obstacle_min_bottom:
                continue
            if (d.label in c.support_surface_classes and target_box is not None
                    and d.x1 <= target_box.cx <= d.x2 and d.y1 <= target_box.y2 <= d.y1 + 0.75 * d.h):
                continue        # the target rests on this surface (e.g. cup on a table), so it is not in the way
            out.append(d)
        return out

    def _find_blocker(self, cands, target_cx_norm, W):
        tx = 0.5 if target_cx_norm is None else target_cx_norm
        lo, hi = min(0.5, tx) - self.cfg.corridor_margin, max(0.5, tx) + self.cfg.corridor_margin
        hits = []
        for d in cands:
            x1, x2 = d.x1 / W, d.x2 / W
            inside = max(0.0, min(x2, hi) - max(x1, lo))
            # share of the SMALLER of (obstacle, corridor) that overlaps: a wall-to-wall obstacle must count
            if inside / max(min(x2 - x1, hi - lo), 1e-6) >= self.cfg.obstacle_overlap:
                hits.append(d)
        return max(hits, key=lambda d: d.y2) if hits else None     # lowest base = closest

    def _suggest(self, obstacle, cands, target_cx_norm, W) -> Optional[str]:
        """Sidestep direction ONLY if a wide gap with no detected objects exists beside the obstacle."""
        if not self.cfg.allow_avoid_direction or obstacle is None:
            return None
        spans = [(d.x1 / W, d.x2 / W) for d in cands]
        bx1, bx2 = obstacle.x1 / W, obstacle.x2 / W
        changed = True
        while changed:                           # merge every box that overlaps the blocked span
            changed = False
            for x1, x2 in spans:
                if x1 < bx2 and x2 > bx1 and (x1 < bx1 or x2 > bx2):
                    bx1, bx2, changed = min(bx1, x1), max(bx2, x2), True
        if bx2 - bx1 > 0.85:
            return None
        left_edge = max([x2 for x1, x2 in spans if x2 <= bx1] + [0.0])
        right_edge = min([x1 for x1, x2 in spans if x1 >= bx2] + [1.0])
        gaps = {"left": bx1 - left_edge, "right": right_edge - bx2}
        ok = {s: g for s, g in gaps.items() if g >= self.cfg.min_free_gap}
        if not ok:
            return None
        if target_cx_norm is not None and target_cx_norm != (bx1 + bx2) / 2:
            pref = "left" if target_cx_norm < (bx1 + bx2) / 2 else "right"
            if pref in ok:
                return pref
        return max(ok, key=ok.get)

    def update(self, dets, target_box, target_label, target_cx_norm, W, H) -> ObstacleStatus:
        cands = self._candidates(dets, target_label, target_box, W, H)
        blocker = self._find_blocker(cands, target_cx_norm, W)
        self._hist.append(blocker is not None)
        newly = cleared = False
        if not self.blocked:
            if blocker is not None and sum(self._hist) >= self.cfg.obstacle_onset_frames:
                self.blocked, newly, self._clear_run = True, True, 0
        elif blocker is None:
            self._clear_run += 1
            if self._clear_run >= self.cfg.obstacle_clear_frames:
                self.blocked, cleared = False, True
                self._hist.clear()
        else:
            self._clear_run = 0
        if blocker is not None:
            self._current = blocker
        suggestion = self._suggest(self._current, cands, target_cx_norm, W) if self.blocked else None
        return ObstacleStatus(self.blocked, newly, cleared, self._current if self.blocked else None, suggestion)
