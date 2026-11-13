"""Offline speech recognition (Vosk) from the microphone, plus a typed-command fallback.

Both classes expose the same interface: start(), poll(timeout) -> VoiceResult|None,
stop(), .failed  -- so a Raspberry Pi microphone just needs the same four members.

Why Vosk: it runs fully offline (matches the project requirement), works on Windows and
on Raspberry Pi, and accepts a restricted vocabulary which makes a small model accurate."""
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

log = logging.getLogger("homesense.voice")

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


class VoiceInputError(RuntimeError):
    pass


@dataclass
class VoiceResult:
    text: str
    ts: float


class VoiceInput:
    def __init__(self, cfg, vocabulary: List[str], is_muted: Callable[[], bool] = lambda: False, metrics=None):
        self.cfg, self.vocab, self.is_muted, self.metrics = cfg, vocabulary, is_muted, metrics
        self.failed = False
        self._audio: "queue.Queue[bytes]" = queue.Queue(maxsize=40)    # bounded: old audio is dropped, not queued forever
        self._results: "queue.Queue[VoiceResult]" = queue.Queue()
        self._stop = threading.Event()
        self._stream = None
        self._thread = None

    def start(self) -> None:
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except (ImportError, OSError) as e:      # OSError: PortAudio library missing
            raise VoiceInputError(f"Speech packages unavailable ({e}). Run: pip install vosk sounddevice") from e
        if not os.path.isdir(self.cfg.vosk_model_path):
            raise VoiceInputError(f"Vosk model not found at '{self.cfg.vosk_model_path}'. Download {MODEL_URL} and unzip it into models/.")
        try:
            sd.query_devices(self.cfg.mic_device, "input")
        except Exception as e:
            raise VoiceInputError(f"No usable microphone: {e}") from e
        SetLogLevel(-1)
        try:
            model = Model(self.cfg.vosk_model_path)
            self._rec = KaldiRecognizer(model, self.cfg.sample_rate, json.dumps(self.vocab))
            self._stream = sd.RawInputStream(samplerate=self.cfg.sample_rate, blocksize=self.cfg.mic_block,
                                             device=self.cfg.mic_device, dtype="int16", channels=1,
                                             callback=self._on_audio)
            self._stream.start()
        except Exception as e:
            raise VoiceInputError(f"Could not start microphone/recogniser: {e}") from e
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
        self._thread.start()

    def _on_audio(self, indata, frames, time_info, status):
        try:
            self._audio.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _run(self) -> None:
        last_partial, last_change = "", None
        while not self._stop.is_set():
            try:
                chunk = self._audio.get(timeout=0.3)
            except queue.Empty:
                if self._stream is not None and not self._stream.active:
                    log.error("Microphone stream stopped")
                    self.failed = True
                    return
                continue
            if self.cfg.mute_mic_while_speaking and self.is_muted():
                try:
                    self._rec.Reset()
                except Exception:
                    pass
                last_partial, last_change = "", None
                continue
            t0 = time.perf_counter()
            done = self._rec.AcceptWaveform(chunk)
            if self.metrics:
                self.metrics.record("stt_chunk_decode", time.perf_counter() - t0)
            now = time.monotonic()
            if done:
                text = " ".join(w for w in json.loads(self._rec.Result()).get("text", "").split() if w != "[unk]")
                if text:
                    if self.metrics and last_change is not None:
                        self.metrics.record("stt_end_of_speech_to_text", now - last_change)
                    self._results.put(VoiceResult(text, now))
                last_partial, last_change = "", None
            else:
                partial = json.loads(self._rec.PartialResult()).get("partial", "")
                if partial and partial != last_partial:
                    last_partial, last_change = partial, now

    def poll(self, timeout: float = 0.0) -> Optional[VoiceResult]:
        try:
            return self._results.get(timeout=timeout) if timeout > 0 else self._results.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass


class TerminalInput:
    """Typed commands: used with --no-voice, or automatically if the microphone/model fails."""
    failed = False

    def __init__(self):
        self._results: "queue.Queue[VoiceResult]" = queue.Queue()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        print("Type commands and press Enter (e.g. 'find my cup', 'stop', 'i found it', 'quit').", flush=True)
        threading.Thread(target=self._read, name="terminal-input", daemon=True).start()

    def _read(self) -> None:
        for line in sys.stdin:
            if line.strip():
                self._results.put(VoiceResult(line.strip(), time.monotonic()))

    def poll(self, timeout: float = 0.0) -> Optional[VoiceResult]:
        try:
            return self._results.get(timeout=timeout) if timeout > 0 else self._results.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        pass
