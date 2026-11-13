"""Runtime measurements. Nothing here is a promised figure: it records what actually
happened on this machine so you can quote real numbers in the evaluation."""
import json
import os
import time
from collections import defaultdict, deque

try:
    import psutil
except ImportError:      # metrics still work without CPU/RAM
    psutil = None


class Metrics:
    def __init__(self):
        self._series = defaultdict(lambda: deque(maxlen=5000))
        self._frames = deque(maxlen=300)
        self._cpu_proc, self._cpu_sys, self._rss = [], [], []
        self._last_sample = 0.0
        self._proc = psutil.Process(os.getpid()) if psutil else None
        if self._proc:
            self._proc.cpu_percent(None)
            psutil.cpu_percent(None)

    def record(self, name: str, seconds: float) -> None:
        self._series[name].append(seconds)

    def mark_frame(self) -> None:
        self._frames.append(time.monotonic())

    def sample_resources(self, every: float = 2.0) -> None:
        now = time.monotonic()
        if not self._proc or now - self._last_sample < every:
            return
        self._last_sample = now
        self._cpu_proc.append(self._proc.cpu_percent(None))      # % of ONE core; can exceed 100
        self._cpu_sys.append(psutil.cpu_percent(None))
        self._rss.append(self._proc.memory_info().rss / 1e6)

    @property
    def fps(self) -> float:
        f = list(self._frames)
        return (len(f) - 1) / (f[-1] - f[0]) if len(f) > 1 and f[-1] > f[0] else 0.0

    def snapshot(self) -> dict:
        out = {"processing_fps_recent": round(self.fps, 2)}
        for name, vals in self._series.items():
            v = sorted(vals)
            out[name + "_ms"] = {"n": len(v), "mean": round(1000 * sum(v) / len(v), 1),
                                 "p95": round(1000 * v[int(0.95 * (len(v) - 1))], 1), "max": round(1000 * v[-1], 1)}
        if self._rss:
            out["process_cpu_pct_mean"] = round(sum(self._cpu_proc) / len(self._cpu_proc), 1)
            out["system_cpu_pct_mean"] = round(sum(self._cpu_sys) / len(self._cpu_sys), 1)
            out["ram_mb_mean"] = round(sum(self._rss) / len(self._rss), 1)
            out["ram_mb_max"] = round(max(self._rss), 1)
        return out

    def format(self) -> str:
        return "METRICS " + json.dumps(self.snapshot(), indent=2)

    def save(self, directory="logs") -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, time.strftime("metrics_%Y%m%d_%H%M%S.json"))
        with open(path, "w") as f:
            json.dump(self.snapshot(), f, indent=2)
        return path
