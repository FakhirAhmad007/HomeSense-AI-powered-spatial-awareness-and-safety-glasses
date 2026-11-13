from fakes import *
from navigation import Guidance, NavigationGuide
from obstacle_monitor import ObstacleMonitor
from target_identifier import Intent, TargetIdentifier, TargetTracker

COCO_LIKE = ["cup", "chair", "cell phone", "bottle", "laptop", "remote"]


def test_parse_and_resolve_supported():
    ti = TargetIdentifier(COCO_LIKE)
    c = ti.parse("Find my mobile phone please")
    assert c.intent is Intent.FIND and c.canonical == "cell phone"
    assert ti.resolve(c).status == "ok"


def test_unsupported_keys_is_not_faked():
    ti = TargetIdentifier(COCO_LIKE)           # no 'keys' class, exactly like COCO
    r = ti.resolve(ti.parse("find my keys"))
    assert r.status == "unsupported" and "keys" in r.message


def test_custom_model_with_keys_class_is_supported():
    ti = TargetIdentifier(COCO_LIKE + ["keys"])
    assert ti.resolve(ti.parse("find my keys")).status == "ok"


def test_unknown_object_and_intents():
    ti = TargetIdentifier(COCO_LIKE)
    assert ti.resolve(ti.parse("find the flux capacitor")).status == "unknown"
    assert ti.parse("stop").intent is Intent.STOP
    assert ti.parse("never mind").intent is Intent.STOP
    assert ti.parse("I found it").intent is Intent.FOUND
    assert ti.parse("find my cup stop").intent is Intent.STOP       # stop always wins
    assert ti.parse("blah blah").intent is Intent.UNKNOWN
    assert "[unk]" in ti.vocabulary()


def test_tracker_needs_consecutive_frames_and_tolerates_dropout():
    cfg = make_config(); tr = TargetTracker(cfg)
    s = [tr.update([cup(100)], "cup", W, t * 0.1) for t in range(2)]
    assert not any(x.just_confirmed for x in s)
    tr.update([], "cup", W, 0.2)                          # miss resets the count
    s = [tr.update([cup(100)], "cup", W, 0.3 + t * 0.1) for t in range(3)]
    assert s[-1].just_confirmed and not s[0].just_confirmed
    assert not tr.update([], "cup", W, 1.0).lost            # short dropout tolerated
    assert tr.update([], "cup", W, 3.0).lost                # long dropout -> lost


def test_tracker_ignores_low_confidence():
    tr = TargetTracker(make_config())
    for t in range(10):
        assert not tr.update([cup(100, conf=0.3)], "cup", W, t * 0.1).confirmed


def test_left_right_centre_and_hysteresis():
    g = NavigationGuide(make_config(smoothing_alpha=1.0))
    assert g.update(cup(100), W, H, 0, "cup") is Guidance.LEFT
    g.reset(); assert g.update(cup(560), W, H, 0, "cup") is Guidance.RIGHT
    g.reset(); assert g.update(cup(330), W, H, 0, "cup") is Guidance.FORWARD
    # hysteresis: once guiding LEFT, being only slightly off-centre keeps saying LEFT
    g.reset(); g.update(cup(100), W, H, 0, "cup")
    assert g.update(cup(320 - 0.12 * 640), W, H, 0.1, "cup") is Guidance.LEFT
    assert g.update(cup(320 - 0.03 * 640), W, H, 0.2, "cup") is Guidance.FORWARD


def test_too_close_says_back_and_arrival_needs_consistency():
    g = NavigationGuide(make_config(smoothing_alpha=1.0))
    assert g.update(cup(320, h=300), W, H, 0, "cup") is Guidance.BACK
    g.reset(); out = [g.update(cup(320, h=150), W, H, t * 0.1, "cup") for t in range(8)]
    assert out[0] is None and Guidance.ARRIVED in out and out.index(Guidance.ARRIVED) >= 4


def test_one_near_frame_is_not_arrival():
    g = NavigationGuide(make_config(smoothing_alpha=1.0))
    assert g.update(cup(320, h=150), W, H, 0, "cup") is None


def test_obstacle_relevance_rules():
    cfg = make_config(); m = ObstacleMonitor(cfg)
    far_clock = det("clock", 320, 100, 80, 80)                   # high on the wall -> ignored
    tiny = det("bottle", 320, 450, 20, 30)                        # tiny -> ignored
    side = chair(0, 80)                                           # far left, target straight ahead
    for _ in range(6):
        st = m.update([far_clock, tiny, side], cup(320), "cup", 0.5, W, H)
    assert not st.blocked
    m.reset()
    for _ in range(6):
        st = m.update([chair()], cup(320), "cup", 0.5, W, H)
    assert st.blocked


def test_target_support_surface_is_not_an_obstacle():
    m = ObstacleMonitor(make_config())
    table = Detection("dining table", 0.9, 100, 150, 540, 470)    # cup is ON the table
    for _ in range(6):
        st = m.update([table], cup(320), "cup", 0.5, W, H)
    assert not st.blocked


def test_obstacle_debounce_and_clear():
    m = ObstacleMonitor(make_config())
    assert not m.update([chair()], cup(320), "cup", 0.5, W, H).blocked   # 1 frame is not enough
    m.update([], cup(320), "cup", 0.5, W, H)
    assert not m.update([chair()], cup(320), "cup", 0.5, W, H).blocked
    m.reset()
    for _ in range(3):
        st = m.update([chair()], cup(320), "cup", 0.5, W, H)
    assert st.blocked and st.newly_blocked
    for _ in range(3):
        assert m.update([], cup(320), "cup", 0.5, W, H).blocked           # not yet cleared
    assert m.update([], cup(320), "cup", 0.5, W, H).cleared


def test_avoidance_direction_only_with_free_gap():
    m = ObstacleMonitor(make_config())
    for _ in range(3):
        st = m.update([chair(250, 390)], cup(320), "cup", 0.5, W, H)
    assert st.suggestion in ("left", "right")                    # wide free gaps on both sides
    m = ObstacleMonitor(make_config())
    for _ in range(3):
        st = m.update([chair(40, 600)], cup(320), "cup", 0.5, W, H)
    assert st.blocked and st.suggestion is None                  # no evidence of free space -> stop
    m = ObstacleMonitor(make_config())
    other = Detection("person", 0.9, 0, 150, 230, 470)           # left side occupied by a person
    for _ in range(3):
        st = m.update([chair(250, 390), other], cup(320), "cup", 0.5, W, H)
    assert st.suggestion == "right"


def test_direction_prefers_side_of_target():
    m = ObstacleMonitor(make_config())
    for _ in range(3):
        st = m.update([chair(250, 390)], cup(500), "cup", 500 / W, W, H)
    assert st.blocked and st.suggestion == "right"


def test_confirmed_target_survives_a_fast_head_turn():
    cfg = make_config(); tr = TargetTracker(cfg)
    for t in range(3):
        tr.update([cup(100)], "cup", W, t * 0.1)
    s = tr.update([cup(540)], "cup", W, 0.4)            # 440 px jump in one frame
    assert s.confirmed and s.box is not None and not s.lost


def test_unconfirmed_teleporting_detection_restarts_count():
    tr = TargetTracker(make_config())
    tr.update([cup(100)], "cup", W, 0.0); tr.update([cup(100)], "cup", W, 0.1)
    s = tr.update([cup(560)], "cup", W, 0.2)            # a different object, not the same one
    assert not s.just_confirmed
