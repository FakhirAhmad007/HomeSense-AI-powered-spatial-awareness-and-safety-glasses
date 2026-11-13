"""End-to-end tests of the REAL controller + tracker + navigation + obstacle code,
driven by scripted fake camera/voice/audio. Spoken output is the observable result."""
from fakes import *
from navigation import NavigationGuide
from obstacle_monitor import ObstacleMonitor
from system_controller import State, SystemController
from target_identifier import TargetIdentifier, TargetTracker
from metrics import Metrics


def build(scenes, script, cfg=None, detector_kw=None, fail_after=True):
    cfg = cfg or make_config()
    cam = FakeCamera(scenes, fail_after=fail_after)
    audio = FakeAudio()
    detector = FakeDetector(cam, **(detector_kw or {}))
    ident = TargetIdentifier(detector.class_names)
    listener = FakeListener(script, audio, cam)
    ctl = SystemController(cfg, cam, detector, listener, audio, ident, TargetTracker(cfg),
                           NavigationGuide(cfg), ObstacleMonitor(cfg), Metrics())
    return ctl, audio, cam


ready = lambda a, c: a.spoke("HomeSense is ready")
now = lambda a, c: True
spoke = lambda frag: (lambda a, c: a.spoke(frag))


def order(audio, *frags):
    """Assert fragments were spoken in this order."""
    texts, pos = audio.texts, 0
    for f in frags:
        idx = next((i for i in range(pos, len(texts)) if f in texts[i]), None)
        assert idx is not None, f"'{f}' not spoken after position {pos}. Spoken: {texts}"
        pos = idx + 1


def test_full_happy_path_with_obstacle_and_resume():
    scenes = (seq([], 5)                       # searching, nothing visible
              + seq([cup(100)], 2) + seq([], 2)  # 2-frame flicker must NOT confirm
              + seq([cup(100)], 5)             # real detection on the left
              + seq([cup(320)], 6)             # centred but far -> forward
              + seq([chair(250, 390), cup(320)], 6)   # chair in the path
              + seq([cup(320)], 7)             # chair gone -> resume
              + seq([cup(320, h=150)], 10))    # near + centred -> arrival
    script = [(ready, "find my cup"),
              (lambda a, c: a.spoke("in front of you"), "I found it"),
              (spoke("Search complete"), "quit")]
    ctl, audio, cam = build(scenes, script)
    ctl.run()
    order(audio, "HomeSense is ready", "Looking for cup", "Object detected. Starting navigation.",
          "Move slightly left.", "Move forward.", "Chair ahead.", "Path is clear. Continuing.",
          "The object is in front of you.", "Search complete", "Goodbye")
    assert audio.texts.count("Object detected. Starting navigation.") == 1   # flicker did not confirm early
    assert ctl.state is State.READY


def test_obstacle_with_no_evidence_of_free_space_says_stop():
    scenes = seq([cup(100)], 4) + seq([chair(40, 600), cup(320)], 8)
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    assert audio.spoke("Chair ahead. Stop.")
    assert not any("Chair ahead. Move" in t for t in audio.texts)


def test_obstacle_with_free_gap_suggests_a_side():
    scenes = seq([cup(320)], 4) + seq([chair(250, 390), cup(320)], 8)
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    assert any(t.startswith("Chair ahead. Move slightly") for t in audio.texts)


def test_no_navigation_instructions_while_blocked():
    scenes = seq([cup(320)], 4) + seq([chair(40, 600), cup(320)], 15)
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    i = audio.texts.index("Chair ahead. Stop.")
    assert not any(t in ("Move forward.", "Move slightly left.", "Move slightly right.") for t in audio.texts[i + 1:])


def test_temporary_loss_is_tolerated_then_resumes():
    scenes = seq([cup(320)], 5) + seq([], 8) + seq([cup(320)], 6)      # 0.8 s gap < 1.5 s timeout
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    assert not audio.spoke("lost sight")
    assert audio.texts.count("Object detected. Starting navigation.") == 1


def test_long_loss_returns_to_searching_and_reacquires():
    scenes = seq([cup(320)], 5) + seq([], 25) + seq([cup(320)], 6)
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    order(audio, "Object detected", "lost sight", "Object detected")


def test_stop_cancels_search_and_returns_to_ready():
    scenes = seq([cup(100)], 40)
    script = [(ready, "find my cup"), (spoke("Move slightly left"), "stop"), (spoke("Search cancelled"), "quit")]
    ctl, audio, _ = build(scenes, script)
    ctl.run()
    order(audio, "Move slightly left", "Search cancelled", "Goodbye")


def test_unsupported_object_is_rejected_honestly_then_supported_works():
    scenes = seq([cup(320)], 6)
    script = [(ready, "find my keys"), (spoke("cannot find keys"), "find my cup")]
    ctl, audio, _ = build(scenes, script)
    ctl.run()
    order(audio, "cannot find keys", "Looking for cup", "Object detected")
    assert ctl.target == "cup"


def test_unknown_words_in_ready_get_a_prompt_not_a_search():
    ctl, audio, _ = build([], [(ready, "blah blah"), (spoke("did not catch"), "quit")])
    ctl.run()
    assert audio.spoke("did not catch") and ctl.state is State.READY and ctl.target is None


def test_search_timeout_returns_to_ready():
    cfg = make_config(search_timeout_s=2.0, search_reminder_s=1.0)
    ctl, audio, _ = build(seq([], 40), [(ready, "find my cup"), (spoke("Search ended"), "quit")], cfg=cfg)
    ctl.run()
    order(audio, "Still searching", "Search ended", "Goodbye")


def test_found_it_during_navigation_completes():
    scenes = seq([cup(100)], 30)
    script = [(ready, "find my cup"), (spoke("Move slightly left"), "i found it"), (spoke("Search complete"), "quit")]
    ctl, audio, _ = build(scenes, script)
    ctl.run()
    order(audio, "Move slightly left", "Search complete", "Goodbye")


def test_instructions_are_not_spammed():
    scenes = seq([cup(100)], 60)               # 6 s of the same situation
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    lefts = audio.texts.count("Move slightly left.")
    assert 1 <= lefts <= 6                      # repeat interval is 1 s in the test config -> never one per frame


def test_right_side_target():
    ctl, audio, _ = build(seq([cup(560)], 10), [(ready, "find my cup")])
    ctl.run()
    assert audio.spoke("Move slightly right.") and not audio.spoke("Move slightly left.")


def test_quit_refused_mid_search():
    scenes = seq([], 30)
    script = [(ready, "find my cup"), (spoke("Looking for"), "quit"), (spoke("Say stop first"), "stop"), (spoke("cancelled"), "quit")]
    ctl, audio, _ = build(scenes, script)
    ctl.run()
    order(audio, "Say stop first", "Search cancelled", "Goodbye")


# ---------------- failure handling ----------------
def test_camera_failure_is_announced_and_stops_cleanly():
    ctl, audio, cam = build(seq([], 3), [(ready, "find my cup")])      # frames run out -> camera "fails"
    ctl.run()
    assert audio.spoke("camera has stopped working")


def test_detector_failure_is_announced_and_stops_cleanly():
    ctl, audio, _ = build(seq([cup(100)], 30), [(ready, "find my cup")], detector_kw={"raise_on_detect": True})
    ctl.run()
    assert audio.spoke("Detection has failed")


def test_microphone_failure_falls_back_to_typed_input():
    scenes = seq([cup(320)], 6)
    ctl, audio, cam = build(scenes, [])
    typed = FakeListener([(ready, "find my cup"), (spoke("Object detected"), "quit")], audio, cam)
    typed.failed = False
    ctl.listener.failed = True
    ctl.fallback_listener = typed
    ctl.run()
    order(audio, "microphone stopped working", "Looking for cup")
