"""Turns spoken text into a command, resolves the requested object against the model's
REAL class list, and confirms the target reliably across consecutive frames.

Key honesty rule: an object is only 'supported' if its canonical name is in
model.names. 'keys' is in ALIASES so we can say 'the model cannot find keys' instead
of silently failing -- but with a COCO model it resolves as UNSUPPORTED."""
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

# canonical class name (must match model.names to be supported) -> spoken aliases
ALIASES: Dict[str, List[str]] = {
    # ---- present in standard COCO models ----
    "cell phone": ["phone", "mobile", "mobile phone", "cell phone", "cellphone", "smartphone"],
    "remote": ["remote", "tv remote", "remote control"],
    "cup": ["cup", "mug", "coffee cup", "tea cup"],
    "bottle": ["bottle", "water bottle"],
    "wine glass": ["wine glass"],
    "bowl": ["bowl"], "knife": ["knife"], "fork": ["fork"], "spoon": ["spoon"],
    "laptop": ["laptop", "computer", "notebook"],
    "keyboard": ["keyboard"], "mouse": ["mouse", "computer mouse"],
    "book": ["book", "books"], "scissors": ["scissors"], "clock": ["clock"], "vase": ["vase"],
    "toothbrush": ["toothbrush"], "teddy bear": ["teddy bear", "teddy"],
    "backpack": ["backpack", "bag", "rucksack"], "handbag": ["handbag", "purse"],
    "umbrella": ["umbrella"],
    "chair": ["chair"], "couch": ["couch", "sofa"], "bed": ["bed"], "tv": ["tv", "television"],
    "dining table": ["table", "dining table"], "potted plant": ["plant", "potted plant"],
    "microwave": ["microwave"], "refrigerator": ["fridge", "refrigerator"], "sink": ["sink"],
    "toilet": ["toilet"], "door": ["door"],
    # ---- NOT in COCO: resolve as unsupported unless a custom model has the class ----
    "keys": ["keys", "key", "key chain", "keychain"],
    "wallet": ["wallet"], "glasses": ["glasses", "spectacles", "eyeglasses"],
    "charger": ["charger"], "pen": ["pen"], "watch": ["watch"],
    "headphones": ["headphones", "earphones", "earbuds"], "medicine": ["medicine", "pills"],
}

FILLER = {"find", "my", "the", "a", "an", "please", "locate", "where", "is", "are", "search",
          "for", "look", "can", "you", "help", "me", "get", "to", "i", "want", "need", "it", "found"}
FIND_VERBS = {"find", "locate", "search", "where", "look", "get"}
STOP_WORDS = {"stop", "cancel", "abort", "nevermind"}
QUIT_WORDS = {"quit", "exit", "goodbye"}


class Intent(Enum):
    FIND = "find"
    STOP = "stop"
    FOUND = "found"
    QUIT = "quit"
    UNKNOWN = "unknown"


@dataclass
class Command:
    intent: Intent
    text: str
    phrase: Optional[str] = None       # what the user called the object
    canonical: Optional[str] = None    # class name from ALIASES, if recognised


@dataclass
class Resolution:
    status: str                        # "ok" | "unsupported" | "unknown"
    phrase: str
    class_name: Optional[str]
    message: str


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z ]", " ", text.lower()).split())


class TargetIdentifier:
    def __init__(self, model_class_names: List[str]):
        self.model_classes = set(model_class_names)
        pairs = [(alias, canon) for canon, al in ALIASES.items() for alias in al]
        # longest alias first so "mobile phone" beats "phone"
        self._pairs = sorted(pairs, key=lambda p: -len(p[0]))

    @property
    def supported_objects(self) -> List[str]:
        return sorted(c for c in ALIASES if c in self.model_classes)

    def vocabulary(self) -> List[str]:
        """Word list handed to the speech recogniser: restricting it to our domain greatly
        improves accuracy for a small offline model."""
        words = set(FILLER) | STOP_WORDS | QUIT_WORDS | FIND_VERBS | {"never", "mind", "shut", "down"}
        for al in ALIASES.values():
            for a in al:
                words.update(a.split())
        return sorted(words) + ["[unk]"]

    def parse(self, text: str) -> Command:
        t = _norm(text)
        words = set(t.split())
        if not t:
            return Command(Intent.UNKNOWN, text)
        if words & STOP_WORDS or "never mind" in t:      # stop wins over everything: safety first
            return Command(Intent.STOP, t)
        if "found" in words:
            return Command(Intent.FOUND, t)
        if words & QUIT_WORDS or "shut down" in t:
            return Command(Intent.QUIT, t)
        padded = f" {t} "
        for alias, canon in self._pairs:
            if f" {alias} " in padded:
                return Command(Intent.FIND, t, phrase=alias, canonical=canon)
        if words & FIND_VERBS:
            rest = " ".join(w for w in t.split() if w not in FILLER)
            return Command(Intent.FIND, t, phrase=rest or None)
        return Command(Intent.UNKNOWN, t)

    def resolve(self, cmd: Command) -> Resolution:
        if not cmd.phrase:
            return Resolution("unknown", "", None, "What would you like to find?")
        if cmd.canonical is None:
            return Resolution("unknown", cmd.phrase, None, f"Sorry, I do not know what {cmd.phrase} is.")
        if cmd.canonical not in self.model_classes:
            return Resolution("unsupported", cmd.phrase, None,
                              f"Sorry, I cannot find {cmd.phrase} yet. My current model was not trained to recognise it.")
        return Resolution("ok", cmd.phrase, cmd.canonical, f"Looking for {cmd.phrase}.")


@dataclass
class TrackState:
    box: Optional[object]          # target Detection seen THIS frame (None if not seen now)
    confirmed: bool
    just_confirmed: bool
    lost: bool
    first_hit_ts: Optional[float]
    last_cx_norm: Optional[float]  # last known horizontal position (0..1), kept during short dropouts


class TargetTracker:
    """Confirms a target only after N consecutive confident frames, then tolerates short dropouts."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._hits = 0
        self._confirmed = False
        self._last = None
        self._last_seen = None
        self._first_hit_ts = None

    def update(self, dets, label: str, frame_w: int, ts: float) -> TrackState:
        cfg = self.cfg
        cands = [d for d in dets if d.label == label and d.conf >= cfg.target_conf]
        box = None
        if cands:
            if self._last is None:
                box = max(cands, key=lambda d: d.conf)
            else:
                nearest = min(cands, key=lambda d: abs(d.cx - self._last.cx))
                if abs(nearest.cx - self._last.cx) <= cfg.target_max_jump * frame_w:
                    box = nearest
                elif self._confirmed:
                    # Already confirmed: a big jump is most likely the wearer turning their head
                    # quickly, so keep following the nearest instance instead of losing the target.
                    box = nearest
                else:
                    # Not yet confirmed and the object 'teleported': probably a different object or
                    # a false positive, so restart the consecutive-frame count.
                    self._hits = 0
                    self._first_hit_ts = None
                    box = max(cands, key=lambda d: d.conf)
        just, lost = False, False
        if box is not None:
            self._hits += 1
            self._last, self._last_seen = box, ts
            if self._first_hit_ts is None:
                self._first_hit_ts = ts
            if not self._confirmed and self._hits >= cfg.target_confirm_frames:
                self._confirmed, just = True, True
        elif not self._confirmed:
            self._hits, self._last, self._first_hit_ts = 0, None, None
        elif ts - self._last_seen > cfg.target_lost_timeout_s:
            lost = True
        cx = self._last.cx / frame_w if self._last is not None else None
        state = TrackState(box, self._confirmed, just, lost, self._first_hit_ts, cx)
        if lost:
            self.reset()
        return state
