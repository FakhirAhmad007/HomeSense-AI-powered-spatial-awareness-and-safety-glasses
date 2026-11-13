"""Non-blocking text-to-speech.

* say() returns immediately; a worker thread does the (blocking) speaking, so camera and
  AI processing never wait for audio.
* Only ONE message can be pending: a newer message replaces an older unspoken one, so the
  user never hears stale directions. A lower-priority message never replaces a higher one.
* Identical text is suppressed for `cooldown` seconds (no repetitive chatter).
* URGENT messages try to cut off the utterance currently being spoken.

To use another engine on the Raspberry Pi (espeak-ng, Piper), write a backend with
init() / speak(text) / stop() and pass it in."""
import logging
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("homesense.audio")

NORMAL, HIGH, URGENT = 0, 1, 2


class ConsoleBackend:
    """Fallback when no speech engine works: prints instead of speaking."""
    def init(self): pass
    def speak(self, text): print(f"[SPEAK] {text}", flush=True)
    def stop(self): pass


class Pyttsx3Backend:
    """Windows SAPI5 via pyttsx3. The engine MUST be created and used in the worker thread."""
    def __init__(self, rate=175, volume=1.0):
        self.rate, self.volume, self.engine = rate, volume, None

    def init(self):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", self.rate)
        self.engine.setProperty("volume", self.volume)

    def speak(self, text):
        print(f"[SPEAK] {text}", flush=True)
        self.engine.say(text)
        self.engine.runAndWait()

    def stop(self):
        try:
            self.engine.stop()
        except Exception:
            pass


class AudioOutput:
    def __init__(self, cfg, backend=None, on_start: Optional[Callable[[str, float], None]] = None):
        self.cfg = cfg
        self._backend = backend if backend is not None else Pyttsx3Backend(cfg.tts_rate, cfg.tts_volume)
        self.on_start = on_start          # called as on_start(text, time_it_was_queued)
        self.degraded = False             # True if we fell back to console output
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._pending = None
        self._last_said = {}
        self._speaking = False
        self._cur_prio = NORMAL
        self._tail_until = 0.0
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=8)

    def _run(self) -> None:
        try:
            self._backend.init()
        except Exception as e:
            log.warning("Speech engine unavailable (%s). Falling back to console output.", e)
            self._backend, self.degraded = ConsoleBackend(), True
        self._ready.set()
        while not self._stop.is_set():
            self._wake.wait(0.2)
            with self._lock:
                item, self._pending = self._pending, None
                self._wake.clear()
            if item is None:
                continue
            prio, text, queued = item
            self._speaking, self._cur_prio = True, prio
            try:
                if self.on_start:
                    self.on_start(text, queued)
                self._backend.speak(text)
            except Exception as e:
                log.warning("Speech failed: %s", e)
            finally:
                self._speaking = False
                self._tail_until = time.monotonic() + 0.3     # let the room echo die before the mic listens

    def say(self, text: str, priority: int = NORMAL, cooldown: float = 0.0) -> bool:
        now = time.monotonic()
        with self._lock:
            last = self._last_said.get(text)
            if cooldown and last is not None and now - last < cooldown:
                return False
            if self._pending is not None and self._pending[0] > priority:
                return False
            self._pending = (priority, text, now)
            self._last_said[text] = now
            interrupt = priority >= URGENT and self._speaking and self._cur_prio < URGENT
        self._wake.set()
        if interrupt:
            self._backend.stop()
        return True

    @property
    def is_speaking(self) -> bool:
        return self._speaking or self._pending is not None or time.monotonic() < self._tail_until

    def wait_idle(self, timeout: float = 5.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end and (self._speaking or self._pending is not None):
            time.sleep(0.05)

    def stop(self) -> None:
        self.wait_idle(3.0)
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2)
