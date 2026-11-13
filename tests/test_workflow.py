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


ready = lambda a, c: a.spoke("Welcome to HomeSense")
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
              (spoke("Stopped."), "quit")]
    ctl, audio, cam = build(scenes, script)
    ctl.run()
    order(audio, "Welcome to HomeSense", "Looking for cup", "Object found.",
          "Move a little to the left.", "Move slightly forward.", "Chair ahead.", "Path is clear. Continuing.",
          "The object is directly in front of you and is nearby.", "Stopped.", "Goodbye")
    assert audio.texts.count("Object found.") == 1   # flicker did not confirm early
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
    assert any(t.startswith("Chair ahead. Move a little") for t in audio.texts)


def test_no_navigation_instructions_while_blocked():
    scenes = seq([cup(320)], 4) + seq([chair(40, 600), cup(320)], 15)
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    i = audio.texts.index("Chair ahead. Stop.")
    assert not any(t in ("Move slightly forward.", "Move a little to the left.", "Move a little to the right.") for t in audio.texts[i + 1:])


def test_temporary_loss_is_tolerated_then_resumes():
    scenes = seq([cup(320)], 5) + seq([], 8) + seq([cup(320)], 6)      # 0.8 s gap < 1.5 s timeout
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    assert not audio.spoke("lost sight")
    assert audio.texts.count("Object found.") == 1


def test_long_loss_returns_to_searching_and_reacquires():
    scenes = seq([cup(320)], 5) + seq([], 25) + seq([cup(320)], 6)
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    order(audio, "Object found", "lost sight", "Object found")


def test_stop_cancels_search_and_returns_to_ready():
    scenes = seq([cup(100)], 40)
    script = [(ready, "find my cup"), (spoke("Move a little to the left"), "stop"), (spoke("Stopped."), "quit")]
    ctl, audio, _ = build(scenes, script)
    ctl.run()
    order(audio, "Move a little to the left", "Stopped.", "Goodbye")


def test_unsupported_object_is_rejected_honestly_then_supported_works():
    scenes = seq([cup(320)], 6)
    script = [(ready, "find my keys"), (spoke("cannot find keys"), "find my cup")]
    ctl, audio, _ = build(scenes, script)
    ctl.run()
    order(audio, "cannot find keys", "Looking for cup", "Object found")
    assert ctl.target == "cup"


def test_unknown_words_in_ready_get_a_prompt_not_a_search():
    ctl, audio, _ = build([], [(ready, "blah blah"), (spoke("did not catch"), "quit")])
    ctl.run()
    assert audio.spoke("did not catch") and ctl.state is State.READY and ctl.target is None


def test_search_timeout_returns_to_ready():
    cfg = make_config(search_timeout_s=2.0, search_reminder_s=1.0)
    ctl, audio, _ = build(seq([], 40), [(ready, "find my cup"), (spoke("Search ended"), "quit")], cfg=cfg)
    ctl.run()
    order(audio, "Move your head slowly", "Search ended", "Goodbye")


def test_found_it_during_navigation_completes():
    scenes = seq([cup(100)], 30)
    script = [(ready, "find my cup"), (spoke("Move a little to the left"), "i found it"), (spoke("Stopped."), "quit")]
    ctl, audio, _ = build(scenes, script)
    ctl.run()
    order(audio, "Move a little to the left", "Stopped.", "Goodbye")


def test_instructions_are_not_spammed():
    scenes = seq([cup(100)], 60)               # 6 s of the same situation
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    lefts = audio.texts.count("Move a little to the left.")
    assert 1 <= lefts <= 6                      # repeat interval is 1 s in the test config -> never one per frame


def test_right_side_target():
    ctl, audio, _ = build(seq([cup(560)], 10), [(ready, "find my cup")])
    ctl.run()
    assert audio.spoke("Move a little to the right.") and not audio.spoke("Move a little to the left.")


def test_quit_refused_mid_search():
    scenes = seq([], 30)
    script = [(ready, "find my cup"), (spoke("Looking for"), "quit"), (spoke("Say stop first"), "stop"), (spoke("Stopped."), "quit")]
    ctl, audio, _ = build(scenes, script)
    ctl.run()
    order(audio, "Say stop first", "Stopped.", "Goodbye")


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
    typed = FakeListener([(ready, "find my cup"), (spoke("Object found"), "quit")], audio, cam)
    typed.failed = False
    ctl.listener.failed = True
    ctl.fallback_listener = typed
    ctl.run()
    order(audio, "microphone stopped working", "Looking for cup")


# ---------------- new behaviour requested after first real-world test ----------------
def test_welcome_message_is_exact():
    ctl, audio, _ = build([], [(ready, "quit")])
    ctl.run()
    assert audio.texts[0] == "Welcome to HomeSense. What object would you like to find?"


def test_scan_prompt_while_object_not_visible_then_object_found():
    cfg = make_config(scan_prompt_first_s=0.5, scan_prompt_interval_s=1.5)
    scenes = seq([], 40) + seq([cup(320)], 6)                     # 4 s with nothing, then the cup appears
    ctl, audio, _ = build(scenes, [(ready, "find my cup")], cfg=cfg)
    ctl.run()
    assert audio.texts.count("Move your head slowly to scan your surroundings.") >= 2    # repeated while searching
    order(audio, "Move your head slowly to scan your surroundings.", "Object found.")
    first_found = audio.texts.index("Object found.")
    assert "Move your head slowly to scan your surroundings." not in audio.texts[first_found:]   # no scan prompts once found


def test_cell_phone_full_flow_exact_phrases():
    ph = lambda cx, h=70: det("cell phone", cx, 240, h * 0.5, h)
    scenes = (seq([], 25) + seq([ph(100)], 5) + seq([ph(560)], 5) + seq([ph(320)], 6) + seq([ph(320, h=120)], 12))
    script = [(ready, "find my cell phone"), (spoke("directly in front of you"), "stop"), (spoke("Stopped."), "quit")]
    ctl, audio, _ = build(scenes, script)
    ctl.run()
    order(audio, "Looking for cell phone", "Move your head slowly to scan your surroundings.", "Object found.",
          "Move a little to the left.", "Move a little to the right.", "Move slightly forward.",
          "The object is directly in front of you and is nearby.", "Stopped.")


def test_move_slightly_backward_when_too_close():
    scenes = seq([cup(320)], 4) + seq([cup(320, h=400)], 10)
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    assert audio.spoke("Move slightly backward.")


def test_stop_ends_everything_and_gives_no_further_instructions():
    scenes = seq([cup(100)], 80)                                     # situation that would keep generating guidance
    script = [(ready, "find my cup"), (spoke("Move a little to the left"), "stop"), (spoke("Stopped."), "quit")]
    ctl, audio, _ = build(scenes, script)
    ctl.run()
    i = audio.texts.index("Stopped.")
    assert "<silenced>" in audio.texts[:i]                            # queued guidance was discarded
    assert audio.texts[i + 1:] == ["Goodbye."]                        # no instructions after stopping (only our own quit reply)
    assert ctl.state is State.READY and ctl.target is None


def test_stop_after_arrival_is_silent_and_can_start_a_new_search():
    scenes = seq([cup(320, h=150)], 12) + seq([cup(320)], 8)
    script = [(ready, "find my cup"), (spoke("directly in front"), "stop"), (spoke("Stopped."), "find my cup")]
    ctl, audio, _ = build(scenes, script)
    ctl.run()
    order(audio, "directly in front", "Stopped.", "Looking for cup")


def test_lost_target_restarts_scan_prompts_without_crashing():
    scenes = seq([cup(320)], 5) + seq([], 60)
    ctl, audio, _ = build(scenes, [(ready, "find my cup")])
    ctl.run()
    order(audio, "Object found.", "lost sight", "Move your head slowly")
