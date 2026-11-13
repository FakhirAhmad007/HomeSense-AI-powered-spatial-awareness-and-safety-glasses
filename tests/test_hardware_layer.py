import threading, time
import pytest
from audio_output import AudioOutput, HIGH, NORMAL, URGENT
from camera_manager import CameraError, CameraManager
from voice_input import VoiceInput, VoiceInputError
from fakes import make_config


class RecordingBackend:
    def __init__(self, delay=0.15):
        self.spoken, self.delay, self.stopped = [], delay, 0
    def init(self): pass
    def speak(self, t): self.spoken.append(t); time.sleep(self.delay)
    def stop(self): self.stopped += 1


class BrokenBackend:
    def init(self): raise RuntimeError("no sound card")
    def speak(self, t): raise AssertionError("should not be used")
    def stop(self): pass


def make(backend):
    a = AudioOutput(make_config(), backend=backend); a.start(); return a


def test_say_never_blocks_the_caller():
    a = make(RecordingBackend(delay=0.5))
    t = time.perf_counter()
    for i in range(20):
        a.say(f"msg {i}")
    assert time.perf_counter() - t < 0.1
    a.stop()


def test_only_latest_pending_message_is_spoken():
    b = RecordingBackend(delay=0.3); a = make(b)
    a.say("first"); time.sleep(0.05)                 # "first" is now being spoken
    for t in ["stale 1", "stale 2", "newest"]:
        a.say(t)
    a.wait_idle(3)
    assert b.spoken == ["first", "newest"]
    a.stop()


def test_cooldown_suppresses_repeats():
    b = RecordingBackend(delay=0.01); a = make(b)
    assert a.say("Move forward.", cooldown=1.0)
    assert not a.say("Move forward.", cooldown=1.0)
    a.wait_idle(2); a.stop()
    assert b.spoken == ["Move forward."]


def test_lower_priority_cannot_replace_urgent_and_urgent_interrupts():
    b = RecordingBackend(delay=0.4); a = make(b)
    a.say("long normal sentence"); time.sleep(0.05)
    a.say("STOP", URGENT)
    assert not a.say("chatter", NORMAL)              # must not displace the pending warning
    a.wait_idle(3)
    assert b.stopped >= 1 and b.spoken[-1] == "STOP" and "chatter" not in b.spoken
    a.stop()


def test_broken_speech_engine_falls_back_instead_of_crashing(capsys):
    a = make(BrokenBackend())
    assert a.degraded
    a.say("still reports in text"); a.wait_idle(2); a.stop()
    assert "still reports in text" in capsys.readouterr().out


def test_microphone_unavailable_raises_clean_error():
    cfg = make_config(vosk_model_path="models/does-not-exist")
    with pytest.raises(VoiceInputError):
        VoiceInput(cfg, ["find"]).start()


def test_camera_unavailable_raises_clean_error():
    cfg = make_config(camera_index=97)
    with pytest.raises(CameraError):
        CameraManager(cfg).start()
