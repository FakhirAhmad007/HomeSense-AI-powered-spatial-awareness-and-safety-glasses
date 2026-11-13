"""Non-blocking text-to-speech.

* say() returns immediately; a worker thread does the (blocking) speaking, so camera and
  AI processing never wait for audio.
* Only ONE message can be pending: a newer message replaces an older unspoken one.
* Identical text is suppressed for `cooldown` seconds (no repetitive chatter).
* URGENT messages / silence() cut off the sentence being spoken.
* Engines are tried in order (Windows SAPI -> pyttsx3 -> console text). If the active engine
  fails twice in a row, the next one is activated automatically.
"""
import logging
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("homesense.audio")

NORMAL, HIGH, URGENT = 0, 1, 2


class ConsoleBackend:
    """Last resort when no speech engine works: prints instead of speaking."""
    def init(self): pass
    def speak(self, text): print(f"[SPEAK] {text}", flush=True)
    def stop(self): pass


class SapiBackend:
    """Windows SAPI5 through COM (needs pywin32). Created and used only in the speech thread."""
    def __init__(self, rate=175, volume=1.0):
        self.rate = max(-10, min(10, round((rate - 175) / 15)))     # SAPI scale: -10..10
        self.volume = int(max(0.0, min(1.0, volume)) * 100)
        self.voice = None
        self._cancel = threading.Event()

    def init(self):
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        self.voice = win32com.client.Dispatch("SAPI.SpVoice")
        self.voice.Rate = self.rate
        self.voice.Volume = self.volume

    def speak(self, text):
        self._cancel.clear()
        print(f"[SPEAK] {text}", flush=True)
        self.voice.Speak(text, 1)                       # 1 = asynchronous
        while not self.voice.WaitUntilDone(40):         # poll so a cancel request is noticed quickly
            if self._cancel.is_set():
                self.voice.Speak("", 3)                 # 3 = async + purge: cuts the current sentence
                break

    def stop(self):
        self._cancel.set()                              # actual purge happens in the speech thread


class Pyttsx3Backend:
    """Fallback. A fresh engine per sentence avoids pyttsx3's 'speaks only once' problem."""
    def __init__(self, rate=175, volume=1.0):
        self.rate, self.volume = rate, volume

    def init(self):
        import pyttsx3  # noqa: F401

    def speak(self, text):
        import pyttsx3
        print(f"[SPEAK] {text}", flush=True)
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        engine.say(text)
        engine.runAndWait()
        del engine

    def stop(self):
        pass


class AudioOutput:
    def __init__(self, cfg, backend=None, on_start: Optional[Callable[[str, float], None]] = None):
        self.cfg = cfg
        if backend is None:
            cands = [SapiBackend(cfg.tts_rate, cfg.tts_volume), Pyttsx3Backend(cfg.tts_rate, cfg.tts_volume)]
        elif isinstance(backend, (list, tuple)):
            cands = list(backend)
        else:
            cands = [backend]
        if not any(isinstance(c, ConsoleBackend) for c in cands):
            cands.append(ConsoleBackend())
        self._candidates, self._idx, self._backend, self._fails = cands, -1, None, 0
        self.on_start = on_start          # called as on_start(text, time_it_was_queued)
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

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__ if self._backend else "none"

    @property
    def degraded(self) -> bool:
        """True when speech is only being printed, not spoken."""
        return isinstance(self._backend, ConsoleBackend)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    def _activate_next(self) -> None:
        """Initialise the next working engine (must run in the speech thread)."""
        while self._idx + 1 < len(self._candidates):
            self._idx += 1
            cand = self._candidates[self._idx]
            try:
                cand.init()
                self._backend = cand
                log.info("Speech engine: %s", type(cand).__name__)
                return
            except Exception as e:
                log.warning("Speech engine %s unavailable (%s: %s)", type(cand).__name__, type(e).__name__, e)
        self._backend = ConsoleBackend()
        log.error("No speech engine works. Speech will only be printed as text.")

    def _run(self) -> None:
        self._activate_next()
        self._ready.set()
        while not self._stop.is_set():
            self._wake.wait(0.2)
            with self._lock:
                item, self._pending = self._pending, None
                self._wake.clear()
                if item is not None:
                    self._speaking, self._cur_prio = True, item[0]
            if item is None:
                continue
            prio, text, queued = item
            try:
                if self.on_start:
                    self.on_start(text, queued)
                self._backend.speak(text)
                self._fails = 0
            except Exception as e:
                self._fails += 1
                log.error("Speech failed (%s: %s) while saying %r", type(e).__name__, e, text)
                if self._fails >= 2 and not isinstance(self._backend, ConsoleBackend):
                    log.warning("Switching to the next speech engine")
                    self._fails = 0
                    self._activate_next()
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
        if interrupt and self._backend is not None:
            self._backend.stop()
        return True

    def silence(self) -> None:
        """Discard anything queued and cut off the sentence being spoken (used by 'stop')."""
        with self._lock:
            self._pending = None
            speaking = self._speaking
        if speaking and self._backend is not None:
            self._backend.stop()

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