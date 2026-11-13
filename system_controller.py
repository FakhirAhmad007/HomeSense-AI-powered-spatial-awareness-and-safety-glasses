"""The workflow state machine:  READY -> SEARCHING -> NAVIGATING <-> OBSTACLE -> ARRIVED -> READY.

All hardware/AI parts are injected, so the same controller runs on the laptop, on a
Raspberry Pi, and in the automated tests with fake devices."""
import logging
import time
from enum import Enum

from audio_output import HIGH, NORMAL, URGENT
from navigation import Guidance
from target_identifier import Intent

log = logging.getLogger("homesense.controller")


class State(Enum):
    READY = "READY"
    SEARCHING = "SEARCHING"
    NAVIGATING = "NAVIGATING"
    OBSTACLE = "OBSTACLE"
    ARRIVED = "ARRIVED"


class SystemController:
    def __init__(self, cfg, camera, detector, listener, audio, identifier, tracker, guide,
                 obstacles, metrics, fallback_listener=None, frame_hook=None, clock=time.monotonic):
        self.cfg, self.camera, self.detector, self.listener = cfg, camera, detector, listener
        self.audio, self.identifier, self.tracker = audio, identifier, tracker
        self.guide, self.obstacles, self.metrics = guide, obstacles, metrics
        self.fallback_listener, self.frame_hook, self.clock = fallback_listener, frame_hook, clock
        self.state = State.READY
        self.target = None
        self.target_phrase = ""
        self._quit = False
        self._cmd_ts = None
        self._det_errors = 0
        self._last_dets = []
        self._last_stats = 0.0
        self._reset_pacing()

    # ------------------------------------------------------------ helpers
    def _reset_pacing(self):
        self._last_guidance = None
        self._last_guidance_ts = -1e9
        self._last_obstacle_say = -1e9
        self._last_search_prompt = 0.0
        self._search_started = None
        self._last_reminder = 0.0

    def _set_state(self, s):
        if s is not self.state:
            log.info("STATE %s -> %s", self.state.value, s.value)
            self.state = s

    def _say(self, text, priority=HIGH, cooldown=0.0, frame=None):
        if self.audio.say(text, priority, cooldown) and frame is not None:
            self.metrics.record("frame_to_instruction", self.clock() - frame.ts)

    def on_speech_start(self, text, queued_ts):
        """Hooked to AudioOutput.on_start to measure command -> first spoken response."""
        if self._cmd_ts is not None:
            self.metrics.record("command_to_speech_start", time.monotonic() - self._cmd_ts)
            self._cmd_ts = None

    def _begin_search(self, now=None):
        self.tracker.reset(); self.guide.reset(); self.obstacles.reset(); self._reset_pacing()
        self._search_started = self._last_search_prompt = now     # None -> set from the first frame's timestamp
        self._set_state(State.SEARCHING)

    def _to_ready(self, text):
        self.target, self.target_phrase = None, ""
        self.tracker.reset(); self.guide.reset(); self.obstacles.reset()
        self._set_state(State.READY)
        self.audio.say(text, HIGH)

    # ------------------------------------------------------------ main loop
    def run(self):
        self.audio.say("HomeSense is ready. What would you like to find?", HIGH)
        seq = -1
        try:
            while not self._quit:
                if not self._hardware_ok():
                    break
                cmd = self.listener.poll(0.1 if self.state is State.READY else 0.0)
                if cmd:
                    self._handle_command(cmd)
                if self.state is State.READY:
                    continue
                frame = self.camera.get_latest(after_seq=seq, timeout=0.3)
                if frame is None:
                    continue
                seq = frame.seq
                try:
                    self._on_frame(frame)
                    self._det_errors = 0
                except Exception as e:                      # detector/runtime fault: never crash silently
                    self._det_errors += 1
                    log.exception("Frame processing error (%d)", self._det_errors)
                    if self._det_errors >= self.cfg.max_detector_errors:
                        self.audio.say("Detection has failed. HomeSense is stopping.", URGENT)
                        break
                if self.frame_hook and self.frame_hook(frame.image, self._last_dets, self.state.value):
                    break
        finally:
            self.audio.wait_idle(3.0)

    def _hardware_ok(self) -> bool:
        if getattr(self.camera, "failed", False):
            log.error("Camera failure")
            self.audio.say("The camera has stopped working. HomeSense is stopping.", URGENT)
            return False
        if getattr(self.listener, "failed", False):
            if self.fallback_listener is not None:
                log.error("Microphone failure -> typed input")
                self.audio.say("The microphone stopped working. Type commands in the terminal.", URGENT)
                self.listener, self.fallback_listener = self.fallback_listener, None
                self.listener.start()
            else:
                self.audio.say("The microphone has stopped working. HomeSense is stopping.", URGENT)
                return False
        return True

    # ------------------------------------------------------------ commands
    def _handle_command(self, vr):
        cmd = self.identifier.parse(vr.text)
        log.info("HEARD %r -> %s", vr.text, cmd.intent.value)
        self._cmd_ts = vr.ts
        s = self.state
        if cmd.intent is Intent.STOP:
            if s is State.ARRIVED:
                self._to_ready("Search complete. HomeSense is ready. What would you like to find?")
            elif s is State.READY:
                self.audio.say("Okay. What would you like to find?", HIGH, cooldown=3)
            else:
                self._to_ready("Search cancelled. HomeSense is ready. What would you like to find?")
        elif cmd.intent is Intent.FOUND:
            if s is not State.READY:
                self._to_ready("Great. Search complete. HomeSense is ready. What would you like to find?")
        elif cmd.intent is Intent.QUIT:
            if s is State.READY:
                self.audio.say("Goodbye.", HIGH)
                self._quit = True
            else:
                self.audio.say("Say stop first, then quit.", HIGH, cooldown=3)
        elif cmd.intent is Intent.FIND:
            if s is not State.READY:
                self.audio.say("Say stop to cancel the current search first.", HIGH, cooldown=3)
                return
            res = self.identifier.resolve(cmd)
            log.info("RESOLVE %s phrase=%r class=%s", res.status, res.phrase, res.class_name)
            if res.status != "ok":
                if res.status == "unsupported":
                    log.warning("Model cannot detect %r. Supported: %s", res.phrase, ", ".join(self.identifier.supported_objects))
                self.audio.say(res.message, HIGH)
                return
            self.target, self.target_phrase = res.class_name, res.phrase
            self.audio.say(res.message, HIGH)
            self._begin_search()
        else:
            if s is State.READY:
                self.audio.say("Sorry, I did not catch that. What would you like to find?", NORMAL, cooldown=5)

    # ------------------------------------------------------------ per-frame processing
    def _on_frame(self, frame):
        now = frame.ts
        if self.state is State.ARRIVED:
            self._last_dets = []
            if now - self._last_reminder >= self.cfg.arrival_reminder_s:
                self._last_reminder = now
                self.audio.say("The object is in front of you. Say stop or I found it.", NORMAL)
            return
        self.metrics.mark_frame()
        self.metrics.sample_resources()
        if self.cfg.stats_interval_s and now - self._last_stats >= self.cfg.stats_interval_s:
            self._last_stats = now
            log.info(self.metrics.format())
        H, W = frame.image.shape[:2]
        t0 = time.perf_counter()
        dets = self.detector.detect(frame.image)
        self.metrics.record("inference", time.perf_counter() - t0)
        self.metrics.record("capture_to_result", self.clock() - frame.ts)
        self._last_dets = dets
        if self.state is State.SEARCHING:
            self._searching(dets, frame, W, now)
        else:
            self._navigating(dets, frame, W, H, now)

    def _searching(self, dets, frame, W, now):
        if self._search_started is None:
            self._search_started = self._last_search_prompt = now
        ts = self.tracker.update(dets, self.target, W, now)
        if ts.just_confirmed:
            self.metrics.record("confirm_latency", now - ts.first_hit_ts)
            self.guide.reset(); self.obstacles.reset(); self._reset_pacing()
            self._set_state(State.NAVIGATING)
            self._say("Object detected. Starting navigation.", HIGH, frame=frame)
        elif now - self._search_started > self.cfg.search_timeout_s:
            self._to_ready(f"I could not find {self.target_phrase}. Search ended. What would you like to find?")
        elif now - self._last_search_prompt >= self.cfg.search_reminder_s:
            self._last_search_prompt = now
            self.audio.say(f"Still searching for {self.target_phrase}. Please turn slowly.", NORMAL)

    def _navigating(self, dets, frame, W, H, now):
        ts = self.tracker.update(dets, self.target, W, now)
        if ts.lost:
            self._say("I lost sight of the object. Searching again.", HIGH, frame=frame)
            self._begin_search(now)
            return

        st = self.obstacles.update(dets, ts.box, self.target, ts.last_cx_norm, W, H)
        if st.blocked:
            self._set_state(State.OBSTACLE)
            if st.newly_blocked or now - self._last_obstacle_say >= self.cfg.obstacle_repeat_s:
                name = st.obstacle.label.capitalize() if st.obstacle else "Obstacle"
                text = f"{name} ahead. Move slightly {st.suggestion}." if st.suggestion else f"{name} ahead. Stop."
                self._last_obstacle_say = now
                self._say(text, URGENT, frame=frame)
            return
        if self.state is State.OBSTACLE:                     # path is clear again -> resume
            self._set_state(State.NAVIGATING)
            self._last_guidance, self._last_guidance_ts = None, -1e9
            self.guide.reset()
            self._say("Path is clear. Continuing.", HIGH, frame=frame)

        g = self.guide.update(ts.box, W, H, now, self.target)
        if g is None:
            return
        if g is Guidance.ARRIVED:
            self._set_state(State.ARRIVED)
            self._last_reminder = now
            self._say(g.value, URGENT, frame=frame)
            return
        if now - self._last_guidance_ts < self.cfg.min_instruction_gap_s:
            return
        if g is self._last_guidance and now - self._last_guidance_ts < self.cfg.instruction_repeat_s:
            return
        self._last_guidance, self._last_guidance_ts = g, now
        self._say(g.value, HIGH, frame=frame)
