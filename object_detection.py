"""YOLO wrapper. Heavy imports are lazy so the rest of the app (and the tests) import fast."""
import logging
from dataclasses import dataclass
from typing import List

log = logging.getLogger("homesense.detect")


class DetectorError(RuntimeError):
    pass


@dataclass(frozen=True)
class Detection:
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


class ObjectDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise DetectorError("ultralytics is not installed. Run: pip install ultralytics") from e
        try:
            self.model = YOLO(cfg.model_path)
        except Exception as e:
            raise DetectorError(f"Could not load model '{cfg.model_path}': {e}") from e
        names = self.model.names
        self.names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        log.info("Loaded %s with %d classes", cfg.model_path, len(self.names))

    @property
    def class_names(self) -> List[str]:
        return [self.names[i] for i in sorted(self.names)]

    def warmup(self, shape=(480, 640, 3)) -> None:
        import numpy as np
        self.detect(np.zeros(shape, dtype=np.uint8))

    def detect(self, image) -> List[Detection]:
        res = self.model.predict(image, imgsz=self.cfg.imgsz, conf=self.cfg.detect_conf,
                                 device=self.cfg.device, verbose=False)[0]
        out: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            out.append(Detection(self.names[int(k)], float(c), float(x1), float(y1), float(x2), float(y2)))
        return out
