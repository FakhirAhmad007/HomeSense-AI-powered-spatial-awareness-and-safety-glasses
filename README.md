# HomeSense (laptop build)

Voice-controlled indoor object finding and navigation. Terminal only: `python main.py`.

## Install (Windows, PowerShell, Python 3.10-3.12)
```
cd homesense
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# offline speech model (about 40 MB)
Invoke-WebRequest https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -OutFile vosk.zip
Expand-Archive vosk.zip models
Remove-Item vosk.zip
```
The YOLO weights (`yolo11n.pt`) download automatically on first run.

## Run, in this order
```
python main.py --list-classes      # what the model can REALLY recognise (no camera/mic needed)
python main.py --check-hardware    # speaker, model, camera + one inference, microphone, one by one
python main.py --no-voice          # full workflow with typed commands (isolates mic problems)
python main.py                     # the real thing
python main.py --show --stats      # optional debug window (q quits) + live metrics
```
Use headphones if possible; otherwise the mic is muted while HomeSense speaks (so it cannot hear itself),
which means "stop" is ignored for the length of one short sentence. With headphones add `--headphones`.

## Voice commands
`find my cell phone` / `where is my cup` | `stop` (any time: ends search and navigation, discards queued guidance, says "Stopped." once, then stays silent) | `I found it/cancel` (same as stop), | `quit` (only when idle)

Spoken phrases are defined at the top of `system_controller.py` and in `navigation.py` (the `Guidance` enum). Scan-prompt timing: `scan_prompt_first_s` / `scan_prompt_interval_s` in `config.py`.

## Tests
```
cd tests
python -m pytest -q
```
These drive the real controller, tracker, navigation and obstacle code with a fake camera, microphone,
speaker and model. They prove the LOGIC. They do not replace testing on the real devices (see below).

## Design decisions (for your evaluation)
| Decision | Why |
|---|---|
| Latest-frame-only camera thread | Queue size 1: AI never processes stale frames, memory is bounded |
| Speech on its own thread, one pending message, newest wins | Camera/AI never block on audio; user never hears outdated directions |
| Offline Vosk with a restricted vocabulary | Matches the offline requirement; small model is accurate for ~100 words |
| Target confirmed after N consecutive confident frames, tolerated dropout of 1.5 s | Prevents false positives without losing the target on a blink |
| Direction from box centre vs image centre, with hysteresis | No flip-flopping between left and right |
| Closeness = fraction of the frame the box fills, per class | No metre estimates; needs calibration (config.near_fill) |
| Arrival needs near + centred for 5 frames over 0.6 s | One lucky frame cannot announce arrival |
| Obstacle = confident, large enough, base low in frame, inside the corridor toward the target, persistent | Not every object is an obstacle |
| Sidestep advice only if a >=30 % wide gap with no detected objects exists beside the obstacle | Otherwise: "Stop" |
| Hardware behind 3 tiny interfaces (camera, listener, audio) | Pi camera/mic/TTS replace them without touching the core |

## Known limitations (state these honestly)
* **Keys are not supported by COCO.** The app says so. Fix: collect ~200-500 labelled images of your keys in
  your home, train a custom YOLO (`yolo detect train data=keys.yaml model=yolo11n.pt`), pass it with
  `--model best.pt`; the class name `keys` is already in `target_identifier.ALIASES`.
* Obstacle detection only sees classes the model knows. Walls, glass, stairs, cables, wet floors, and
  objects outside the camera view are invisible. This is an assistive hint, NOT collision avoidance and NOT
  a certified mobility aid.
* `near_fill` thresholds are starting guesses. Calibrate per object at your real arm's-reach distance.
* "Move slightly left" assumes the wearer turns/steps toward the target side. Glasses-mounted camera angle
  changes how well this works; test while walking.
* Urgent warnings can wait up to one short sentence if speech is already playing (pyttsx3 interrupt is best effort).
* Not yet tested on real hardware: microphone accuracy, TTS interruption, real FPS, lighting, partial occlusion.

## Hardware test checklist (do this on the laptop and record results)
1. `--check-hardware` all four sections OK.
2. Say "find my cup" 20 times at normal volume and at 1 m: count correct recognitions.
3. Cup in view: confirm detection announced, guidance matches reality when you move left/right/forward.
4. Place a chair in the path: warning appears; remove it: "Path is clear".
5. Cover the camera for 3 s mid-navigation: "I lost sight", then re-acquire.
6. Unplug the webcam mid-run: spoken camera failure, clean exit. Same for the microphone: typed fallback.
7. Dim room, backlit window, cup half hidden: record how detection confidence and false alarms change.
8. Run with `--stats`; the saved `logs/metrics_*.json` holds FPS, inference latency, speech and command latencies, CPU and RAM.

## Raspberry Pi 5 later
Camera: write `PiCameraManager` with `start()/get_latest(after_seq, timeout)/stop()/failed` (picamera2).
Speech out: an espeak-ng or Piper backend with `init()/speak()/stop()`. Speech in: Vosk + sounddevice already works on the Pi.
For speed, export the model (`yolo export model=yolo11n.pt format=ncnn`) and load it via `--model`.
