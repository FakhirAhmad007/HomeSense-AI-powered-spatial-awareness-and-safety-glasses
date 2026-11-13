"""All tunable settings in one place.

Every threshold the system uses lives here so it can be calibrated on the real
laptop (and later on the Raspberry Pi) without touching the logic modules.
Values marked CALIBRATE are starting guesses, NOT measured results.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


@dataclass
class Config:
    # ---------------- Devices (hardware layer) ----------------
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    camera_backend: str = "auto"        # auto | dshow | msmf | v4l2 | any
    camera_max_reopen: int = 3          # reconnect attempts before declaring failure

    mic_device: Optional[int] = None    # None = system default input
    vosk_model_path: str = "models/vosk-model-small-en-us-0.15"
    sample_rate: int = 16000
    mic_block: int = 4000               # 0.25 s of audio per chunk
    mute_mic_while_speaking: bool = True  # stops the mic hearing our own TTS; set False with headphones

    tts_rate: int = 175
    tts_volume: float = 1.0

    # ---------------- Model ----------------
    model_path: str = "yolo11n.pt"      # swap for a custom-trained .pt later (e.g. one with a 'keys' class)
    imgsz: int = 416
    device: Optional[str] = None        # None = let Ultralytics choose (GPU if available); "cpu" to force
    detect_conf: float = 0.30           # global floor; per-role thresholds below are stricter

    # ---------------- Target confirmation ----------------
    target_conf: float = 0.45           # min confidence for a target detection to count
    target_confirm_frames: int = 3      # consecutive frames needed before "Object detected"
    target_lost_timeout_s: float = 1.5  # target may vanish this long before we call it lost
    target_max_jump: float = 0.30       # max centre jump between frames (fraction of width) to be "the same" object
    scan_prompt_first_s: float = 2.0    # first 'move your head' prompt this long after the search starts
    scan_prompt_interval_s: float = 8.0  # then repeated this often while the object is still not visible
    search_timeout_s: float = 120.0

    # ---------------- Navigation ----------------
    center_exit: float = 0.15           # leave "centred" when |offset| exceeds this (fraction of width)
    center_enter: float = 0.08          # become "centred" again only inside this (hysteresis)
    smoothing_alpha: float = 0.5        # EMA on offset/size to stop jitter
    arrival_frames: int = 5             # consecutive "near + centred" frames required for arrival
    arrival_min_s: float = 0.6          # ...spread over at least this long (fps independent)
    too_close_factor: float = 1.8       # fill > near * this  ->  "Move slightly back"
    near_fill_default: float = 0.40     # CALIBRATE: box fill ratio that means "within arm's reach"
    near_fill: Dict[str, float] = field(default_factory=lambda: {   # CALIBRATE per class
        "cell phone": 0.22, "remote": 0.25, "cup": 0.30, "bottle": 0.35, "book": 0.30,
        "mouse": 0.15, "keyboard": 0.40, "laptop": 0.45, "scissors": 0.20, "toothbrush": 0.15,
        "bowl": 0.30, "vase": 0.30, "clock": 0.25, "backpack": 0.45, "handbag": 0.40,
        "umbrella": 0.50, "chair": 0.60, "couch": 0.70, "bed": 0.70, "tv": 0.50,
    })

    # ---------------- Obstacles ----------------
    obstacle_conf: float = 0.40
    obstacle_near_fill: float = 0.55    # CALIBRATE: box must fill >= 55% of frame height or width to count as 'quite near'
    obstacle_min_area: float = 0.04     # Ignore boxes smaller than 4% of the frame
    obstacle_min_bottom: float = 0.55   # Box base must be in the lower 45% of the frame 
    corridor_margin: float = 0.15       # extra width around the line between "straight ahead" and the target
    obstacle_overlap: float = 0.40      # fraction of obstacle width inside the corridor to count as a conflict
    obstacle_window: int = 5
    obstacle_onset_frames: int = 3      # conflict in 3 of the last 5 frames -> blocked
    obstacle_clear_frames: int = 4      # 4 clean frames in a row -> path clear
    min_free_gap: float = 0.30          # side gap (fraction of width) needed before suggesting a sidestep
    allow_avoid_direction: bool = True
    ignore_obstacle_classes: Set[str] = field(default_factory=set)
    # Things objects commonly sit ON. A target resting on one of these is not blocked by it.
    # Anything else that covers the target in the image (e.g. a chair) is treated as an obstacle.
    support_surface_classes: Set[str] = field(default_factory=lambda: {"dining table", "bed", "couch", "bench"})

    # ---------------- Speech pacing ----------------
    min_instruction_gap_s: float = 1.2  # min time between any two guidance phrases
    instruction_repeat_s: float = 3.0   # same guidance phrase repeated no faster than this
    obstacle_repeat_s: float = 3.0
    arrival_repeat_s: float = 6.0       # while still near the object, repeat the 'nearby' message this often
    max_detector_errors: int = 5

    # ---------------- Debug ----------------
    show_window: bool = False
    stats_interval_s: float = 0.0       # >0 prints metrics periodically
