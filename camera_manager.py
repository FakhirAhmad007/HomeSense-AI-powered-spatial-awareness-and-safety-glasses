"""Webcam capture on its own thread. Only the NEWEST frame is kept, so the queue is
bounded (size 1) and the AI never processes stale frames.

To move to the Raspberry Pi, write another class with the same three members
(start / get_latest / stop + .failed) -- e.g. one wrapping picamera2 -- and nothing
else in the application has to change."""
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger("homesense.camera")


class CameraError(RuntimeError):
    pass


@dataclass
class Frame:
    image: Any
    ts: float
    seq: int


class CameraManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.failed = False
        self._cap = None
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread = None
        self._times = deque(maxlen=60)

    def _open(self):
        import cv2
        api = {"auto": cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY,
               "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2,
               "any": cv2.CAP_ANY}[self.cfg.camera_backend]
        cap = cv2.VideoCapture(self.cfg.camera_index, api)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Cannot open camera index {self.cfg.camera_index}. Is it in use or blocked by Windows privacy settings?")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, img = cap.read()
        if not ok or img is None:
            cap.release()
            raise CameraError("Camera opened but returned no frames.")
        return cap

    def start(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError as e:
            raise CameraError("opencv-python is not installed. Run: pip install opencv-python") from e
        self._cap = self._open()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _publish(self, img) -> None:
        now = time.monotonic()
        with self._cond:
            self._seq += 1
            self._frame = Frame(img, now, self._seq)
            self._times.append(now)
            self._cond.notify_all()

    def _loop(self) -> None:
        fails = 0
        while not self._stop.is_set():
            ok, img = self._cap.read()
            if ok and img is not None:
                fails = 0
                self._publish(img)
                continue
            fails += 1
            time.sleep(0.05)
            if fails >= 30:
                log.warning("Camera stopped delivering frames; trying to reopen")
                if self._reopen():
                    fails = 0
                else:
                    self.failed = True
                    with self._cond:
                        self._cond.notify_all()
                    return

    def _reopen(self) -> bool:
        try:
            self._cap.release()
        except Exception:
            pass
        for _ in range(self.cfg.camera_max_reopen):
            try:
                self._cap = self._open()
                return True
            except CameraError:
                time.sleep(1.0)
        return False

    def get_latest(self, after_seq: int = -1, timeout: float = 0.5) -> Optional[Frame]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq or self.failed or self._stop.is_set(), timeout)
            if self._frame is not None and self._seq > after_seq:
                return self._frame
        return None

    @property
    def capture_fps(self) -> float:
        t = list(self._times)
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
