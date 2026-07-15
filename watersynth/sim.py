"""Procedurální simulace vodní hladiny - zdroj obrazu pro hraní a testování
bez kamery. Kliknutím do náhledu se přidá kapka."""

import random

import numpy as np

TAU = 2 * np.pi


class WaterSim:
    C = 0.22        # rychlost šíření vlnky (jednotek/s)
    K = TAU * 14    # prostorová frekvence vlnek kapky

    def __init__(self, size: int = 128):
        self.n = size
        y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
        self.x, self.y = x, y
        self.drops = []           # (x, y, t0, amp)
        self.t = 0.0
        self.wind = 0.35          # síla okolního vlnění 0..1
        self.auto = True          # automatické náhodné kapky
        self._next_auto = 1.5

    def add_drop(self, x: float, y: float, amp: float = 1.0):
        self.drops.append((x, y, self.t, amp))
        if len(self.drops) > 12:
            self.drops.pop(0)

    def step(self, dt: float) -> np.ndarray:
        """Vrátí šedotónový snímek NxN float32 v rozsahu 0..1."""
        self.t += dt
        if self.auto and self.t >= self._next_auto:
            self.add_drop(0.15 + random.random() * 0.7,
                          0.15 + random.random() * 0.7,
                          0.6 + random.random() * 0.6)
            self._next_auto = self.t + 1.2 + random.random() * 2.5
        self.drops = [d for d in self.drops if self.t - d[2] < 6.0]

        t, x, y = self.t, self.x, self.y
        # okolní vlnění: tři šikmé postupné vlny
        h = self.wind * (
            0.5 * np.sin(TAU * (2.8 * x + 1.6 * y) + 0.7 * t)
            + 0.35 * np.sin(TAU * (1.1 * x - 2.4 * y) + 0.45 * t)
            + 0.2 * np.sin(TAU * (5.2 * x + 0.5 * y) + 1.3 * t)
        )
        for dx0, dy0, t0, amp in self.drops:
            age = t - t0
            r = np.hypot(x - dx0, y - dy0)
            front = self.C * age
            # kruhová vlnka: sin(k·r - w·t), tlumená vzdáleností i časem,
            # s měkkou náběžnou hranou u čela vlny
            edge = np.clip((front - r) * 25, 0, 1)
            h += (amp * edge * np.sin(self.K * r - self.K * self.C * age)
                  * np.exp(-2.5 * r) * np.exp(-age / 2.2) * 0.9)

        return np.clip(0.5 + 0.41 * h, 0.0, 1.0).astype(np.float32)
