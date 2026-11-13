"""HomeSense entry point:  python main.py

Useful flags:
  --list-classes    print what the loaded model can actually recognise, then exit
  --check-hardware  test camera, microphone, speaker and model one by one, then exit
  --no-voice        type commands in the terminal instead of speaking them
  --show            optional OpenCV debug window (boxes + state); press q to quit
  --stats           print performance metrics every 10 s (always saved on exit)"""
import argparse
import logging
import os
import sys
import time

from config import Config


def parse_args():
    p = argparse.ArgumentParser(description="HomeSense - voice-controlled object finding and navigation")
    p.add_argument("--camera", type=int, help="camera index (default 0)")
    p.add_argument("--model", help="path to a YOLO .pt file (default yolo11n.pt)")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--device", help="cpu, 0 (first GPU), ...")
    p.add_argument("--vosk-model", help="folder of the Vosk speech model")
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--silent", action="store_true", help="print speech instead of speaking it")
    p.add_argument("--headphones", action="store_true", help="keep the mic open while speaking (use only with headphones)")
    p.add_argument("--show", action="store_true")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--list-classes", action="store_true")
    p.add_argument("--check-hardware", action="store_true")
    p.add_argument("--test-speech", action="store_true", help="speak the HomeSense phrases in a row, then test interruption")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def setup_logging(debug):
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/homesense.log", encoding="utf-8")])
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def make_debug_view():
    import cv2
    def hook(image, dets, state):
        img = image.copy()
        for d in dets:
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{d.label} {d.conf:.2f}", (int(d.x1), max(15, int(d.y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, state, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("HomeSense debug (q = quit)", img)
        return (cv2.waitKey(1) & 0xFF) == ord("q")
    return hook

def test_speech(cfg):
    import time
    from audio_output import AudioOutput, HIGH, NORMAL, URGENT
    audio = AudioOutput(cfg); audio.start()
    print("Speech engine in use:", audio.backend_name, "(FALLBACK: text only)" if audio.degraded else "")
    phrases = ["Welcome to HomeSense. What object would you like to find?",
               "Move your head slowly to scan your surroundings.",
               "Object found.", "Move a little to the left.", "Move a little to the right.",
               "Move slightly forward.", "Move slightly backward.",
               "The object is directly in front of you and is nearby."]
    for i, t in enumerate(phrases, 1):
        print(f"{i}/{len(phrases)}"); audio.say(t, HIGH); audio.wait_idle(12); time.sleep(0.3)
    print("Interrupt test: the long sentence should be cut off, then you should hear 'Stopped.'")
    audio.say("This is a long sentence that should be cut off before it reaches the end.", NORMAL)
    time.sleep(1.2); audio.silence(); audio.say("Stopped.", URGENT); audio.wait_idle(8)
    print("Afterwards: did you hear ALL 8 sentences and 'Stopped.'?")
    audio.stop()

def check_hardware(cfg, args):
    from audio_output import AudioOutput, HIGH
    from camera_manager import CameraError, CameraManager
    from object_detection import DetectorError, ObjectDetector
    from target_identifier import TargetIdentifier
    from voice_input import VoiceInput, VoiceInputError
    print("\n== 1. Speaker ==")
    audio = AudioOutput(cfg); audio.start()
    print("TTS engine:", audio.backend_name, "- FALLBACK (text only)" if audio.degraded else "- OK")
    audio.say("HomeSense speaker test. If you can hear this, the speaker works.", HIGH); audio.wait_idle(8)
    print("\n== 2. Model ==")
    try:
        det = ObjectDetector(cfg); print("OK,", len(det.class_names), "classes:", ", ".join(det.class_names))
    except DetectorError as e:
        print("FAILED:", e); det = None
    print("\n== 3. Camera ==")
    cam = CameraManager(cfg)
    try:
        cam.start(); time.sleep(1.5); f = cam.get_latest(timeout=2)
        print("OK" if f is not None else "FAILED: no frame", f"- capture {cam.capture_fps:.1f} fps, frame {f.image.shape[1]}x{f.image.shape[0]}" if f else "")
        if f is not None and det is not None:
            det.warmup(); t = time.perf_counter(); out = det.detect(f.image)
            print(f"One inference: {(time.perf_counter() - t) * 1000:.0f} ms, saw: {sorted({d.label for d in out}) or 'nothing'}")
    except CameraError as e:
        print("FAILED:", e)
    finally:
        cam.stop()
    print("\n== 4. Microphone (say 'find my cup' within 6 s) ==")
    ident = TargetIdentifier(det.class_names if det else [])
    v = VoiceInput(cfg, ident.vocabulary())
    try:
        v.start(); r = v.poll(timeout=6)
        print("Heard:", repr(r.text) if r else "nothing (check mic level / model)")
    except VoiceInputError as e:
        print("FAILED:", e)
    finally:
        v.stop()
    audio.stop()


def main():
    args = parse_args()
    setup_logging(args.debug)
    log = logging.getLogger("homesense.main")
    cfg = Config()
    if args.camera is not None: cfg.camera_index = args.camera
    if args.model: cfg.model_path = args.model
    if args.imgsz: cfg.imgsz = args.imgsz
    if args.device: cfg.device = args.device
    if args.vosk_model: cfg.vosk_model_path = args.vosk_model
    if args.headphones: cfg.mute_mic_while_speaking = False
    if args.stats: cfg.stats_interval_s = 10.0

    if args.test_speech:
        return test_speech(cfg)

    if args.check_hardware:
        return check_hardware(cfg, args)

    from audio_output import AudioOutput, ConsoleBackend, HIGH, URGENT
    from camera_manager import CameraError, CameraManager
    from metrics import Metrics
    from navigation import NavigationGuide
    from object_detection import DetectorError, ObjectDetector
    from obstacle_monitor import ObstacleMonitor
    from system_controller import SystemController
    from target_identifier import TargetIdentifier, TargetTracker
    from voice_input import TerminalInput, VoiceInput, VoiceInputError

    metrics = Metrics()
    audio = AudioOutput(cfg, backend=ConsoleBackend() if args.silent else None)
    audio.start()                                    # never fatal: falls back to console text

    try:
        detector = ObjectDetector(cfg)
    except DetectorError as e:
        log.error("%s", e); audio.say("The AI model could not be loaded.", URGENT); audio.stop(); return 1
    log.info("Model can recognise: %s", ", ".join(detector.class_names))
    identifier = TargetIdentifier(detector.class_names)
    log.info("Objects you can ask for: %s", ", ".join(identifier.supported_objects))
    if args.list_classes:
        audio.stop(); return 0

    camera = CameraManager(cfg)
    try:
        camera.start()
    except CameraError as e:
        log.error("%s", e); audio.say("The camera could not be started.", URGENT); audio.stop(); return 1

    terminal = TerminalInput()
    fallback = None if args.no_voice else terminal
    if args.no_voice:
        listener = terminal
    else:
        listener = VoiceInput(cfg, identifier.vocabulary(), is_muted=lambda: audio.is_speaking, metrics=metrics)
    try:
        listener.start()
    except VoiceInputError as e:
        log.error("%s", e)
        audio.say("The microphone is not available. Type commands in the terminal instead.", HIGH)
        listener, fallback = terminal, None
        listener.start()

    detector.warmup((cfg.frame_height, cfg.frame_width, 3))
    controller = SystemController(cfg, camera, detector, listener, audio, identifier, TargetTracker(cfg),
                                  NavigationGuide(cfg), ObstacleMonitor(cfg), metrics,
                                  fallback_listener=fallback, frame_hook=make_debug_view() if args.show else None)
    audio.on_start = controller.on_speech_start
    try:
        controller.run()
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        listener.stop(); camera.stop(); audio.stop()
        path = metrics.save()
        print(metrics.format()); print("Metrics saved to", path)
        try:
            import cv2; cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
