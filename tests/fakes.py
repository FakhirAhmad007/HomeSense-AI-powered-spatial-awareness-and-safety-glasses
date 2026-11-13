"""Fake devices so the COMPLETE workflow can be tested without a camera, mic, speaker or model.
These test the logic; they do NOT replace testing on the real hardware."""
import time
import numpy as np
from camera_manager import Frame
from config import Config
from object_detection import Detection
from voice_input import VoiceResult

W, H = 640, 480


def det(label, cx, cy, w, h, conf=0.85):
    return Detection(label, conf, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def cup(cx, h=70, conf=0.85):
    return det("cup", cx, 240, h * 0.8, h, conf)


def chair(x1=250, x2=390):
    return Detection("chair", 0.9, x1, 200, x2, 460)


def seq(scene, n):
    return [list(scene) for _ in range(n)]


def make_config(**kw):
    c = Config()
    c.min_instruction_gap_s = 0.3
    c.instruction_repeat_s = 1.0
    c.arrival_min_s = 0.3
    c.obstacle_repeat_s = 1.0
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class FakeCamera:
    """Plays one scene (list of Detections) per frame, 0.1 s apart. `failed` after the last frame."""
    def __init__(self, scenes, fail_after=True):
        self.scenes, self.i, self.failed, self.fail_after = scenes, 0, False, fail_after
        self.current = []
    def get_latest(self, after_seq=-1, timeout=0.3):
        if self.i >= len(self.scenes):
            if self.fail_after:
                self.failed = True
            return None
        self.current = self.scenes[self.i]
        self.i += 1
        return Frame(np.zeros((H, W, 3), np.uint8), self.i * 0.1, self.i)


class FakeDetector:
    def __init__(self, camera, class_names=None, raise_on_detect=False):
        self.camera, self.raise_on_detect = camera, raise_on_detect
        self.class_names = class_names or ["cup", "chair", "cell phone", "bottle", "dining table", "person"]
    def detect(self, image):
        if self.raise_on_detect:
            raise RuntimeError("model exploded")
        return self.camera.current


class FakeAudio:
    def __init__(self):
        self.said = []
    def say(self, text, priority=0, cooldown=0.0):
        if cooldown and self.said and self.said[-1][0] == text:
            return False
        self.said.append((text, priority))
        return True
    def wait_idle(self, t=0): pass
    @property
    def texts(self):
        return [t for t, _ in self.said]
    def spoke(self, fragment):
        return any(fragment in t for t in self.texts)


class FakeListener:
    """script: list of (condition(audio, camera) -> bool, spoken_text). Released in order."""
    failed = False
    def __init__(self, script, audio, camera):
        self.script, self.audio, self.camera = list(script), audio, camera
    def start(self): pass
    def poll(self, timeout=0.0):
        time.sleep(0.001)
        if self.script and self.script[0][0](self.audio, self.camera):
            return VoiceResult(self.script.pop(0)[1], time.monotonic())
        return None
    def stop(self): pass
