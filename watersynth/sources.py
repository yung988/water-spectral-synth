"""Zdroje obrazu: simulace, kamera (na Macu i iPhone po kabelu -
Continuity Camera se hlásí jako běžné video zařízení) a video soubor.

Každý zdroj vrací z read() aktuální RGB snímek (uint8, HxWx3) nebo None.
Kamera a video čtou ve vlastním vlákně, aby neblokovaly audio ani UI.
"""

import sys
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:  # simulace funguje i bez OpenCV
    cv2 = None

from .sim import WaterSim


def to_square_gray(rgb: np.ndarray, n: int) -> np.ndarray:
    """Střední čtvercový výřez -> NxN šedotón float32 0..1."""
    h, w = rgb.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = rgb[y0:y0 + s, x0:x0 + s]
    if cv2 is not None:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (n, n), interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32) / 255.0
    # nouzová cesta bez OpenCV: podvzorkování
    idx = (np.arange(n) * s // n)
    crop = crop[idx][:, idx].astype(np.float32)
    return (crop @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)) / 255.0


class SimSource:
    name = "Simulace vody"

    def __init__(self, size: int = 128):
        self.sim = WaterSim(size)
        self._last = time.monotonic()

    def read(self):
        now = time.monotonic()
        dt = min(0.1, now - self._last)
        self._last = now
        gray = self.sim.step(dt)
        v = (gray * 255).astype(np.uint8)
        # modravý nádech, ať náhled vypadá jako voda
        rgb = np.stack([(v * 0.45).astype(np.uint8),
                        (v * 0.75).astype(np.uint8), v], axis=-1)
        return rgb

    def close(self):
        pass


class _ThreadedReader(threading.Thread):
    """Společný základ: čte snímky ve vlákně, drží poslední."""

    def __init__(self):
        super().__init__(daemon=True)
        self.frame = None
        self.error = None
        self._stop = threading.Event()

    def read(self):
        return self.frame

    def close(self):
        self._stop.set()


class CameraSource(_ThreadedReader):
    def __init__(self, index: int = 0):
        super().__init__()
        if cv2 is None:
            raise RuntimeError("Kamera vyžaduje opencv-python (pip install opencv-python).")
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Kameru {index} se nepodařilo otevřít. Zkus jiný --camera index; "
                "na Macu také zkontroluj oprávnění ke kameře pro Terminál "
                "(Nastavení systému -> Soukromí a zabezpečení -> Kamera).")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.name = f"Kamera {index}"
        self.start()

    def run(self):
        while not self._stop.is_set():
            ok, bgr = self.cap.read()
            if ok:
                self.frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                time.sleep(0.05)
        self.cap.release()


class VideoFileSource(_ThreadedReader):
    def __init__(self, path: str):
        super().__init__()
        if cv2 is None:
            raise RuntimeError("Přehrávání videa vyžaduje opencv-python.")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video {path} nejde otevřít.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.name = f"Video: {path}"
        self.start()

    def run(self):
        period = 1.0 / max(1.0, self.fps)
        while not self._stop.is_set():
            t0 = time.monotonic()
            ok, bgr = self.cap.read()
            if not ok:  # smyčka od začátku
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
        self.cap.release()
