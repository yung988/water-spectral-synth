"""Aditivní syntéza: banka sinusových oscilátorů řízená spektrem obrazu.

Levý a pravý kanál mají vlastní amplitudy (podle orientace struktur na
hladině); pravý kanál lze jemně rozladit ("spread" z diagonální energie).
Render běží po blocích v audio callbacku, parametry chodí z hlavního vlákna
pod zámkem.
"""

import threading

import numpy as np


def midi_to_freq(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


class AdditiveSynth:
    def __init__(self, samplerate: int = 48000, harmonics: int = 96):
        self.sr = samplerate
        self.h = h = harmonics
        self._lock = threading.Lock()

        self.target_l = np.zeros(h, dtype=np.float64)
        self.target_r = np.zeros(h, dtype=np.float64)
        self.amp_l = np.zeros(h, dtype=np.float64)
        self.amp_r = np.zeros(h, dtype=np.float64)
        self.phase_l = np.zeros(h, dtype=np.float64)
        self.phase_r = np.zeros(h, dtype=np.float64)
        self.k = np.arange(1, h + 1, dtype=np.float64)

        # deterministické rozladění pravého kanálu pro každou harmonickou
        rng = np.random.default_rng(1234)
        self.det = rng.random(h) * 2 - 1

        self.f0 = 110.0
        self.f0_target = 110.0
        self.gate = 0.0
        self.vel = 1.0
        self.env = 0.0
        self.spread = 0.0
        self.spread_target = 0.0

        self.attack = 0.02    # náběh amplitud harmonických (s)
        self.release = 0.25   # dozvuk amplitud harmonických (s)
        self.glide = 0.03     # klouzání výšky tónu (s)
        self.gain = 0.5

    # ---- řízení z hlavního vlákna --------------------------------------

    def set_frame(self, amps_l, amps_r, spread: float):
        with self._lock:
            np.copyto(self.target_l, amps_l)
            np.copyto(self.target_r, amps_r)
            self.spread_target = float(spread)

    def note_on(self, midi: float, vel: float = 0.9):
        with self._lock:
            self.f0_target = midi_to_freq(midi)
            self.gate = 1.0
            self.vel = float(vel)

    def note_off(self):
        with self._lock:
            self.gate = 0.0

    def set_freq(self, f0: float):
        with self._lock:
            self.f0_target = float(f0)

    def set_params(self, attack=None, release=None, glide=None, gain=None):
        with self._lock:
            if attack is not None:
                self.attack = max(0.001, float(attack))
            if release is not None:
                self.release = max(0.01, float(release))
            if glide is not None:
                self.glide = max(0.001, float(glide))
            if gain is not None:
                self.gain = float(gain)

    # ---- audio vlákno ---------------------------------------------------

    def render(self, n: int) -> np.ndarray:
        """Vrátí blok (n, 2) float32 v rozsahu -1..1."""
        with self._lock:
            target_l = self.target_l.copy()
            target_r = self.target_r.copy()
            gate, vel = self.gate, self.vel
            attack, release, glide, gain = self.attack, self.release, self.glide, self.gain
            spread_target = self.spread_target
            f0_target = self.f0_target

        sr = self.sr

        # klouzání výšky tónu a spread na úrovni bloku
        self.f0 += (f0_target - self.f0) * (1 - np.exp(-n / (glide * sr)))
        self.spread += (spread_target - self.spread) * 0.2

        # obálka noty: rychlý náběh, uvolnění dle release (uzavřený tvar
        # exponenciály pro konstantní cíl v rámci bloku)
        env_target = gate * vel
        tau = 0.008 if env_target > self.env else release
        env = env_target + (self.env - env_target) * np.exp(-1.0 / (tau * sr)) ** np.arange(1, n + 1)
        self.env = float(env[-1])

        # vyhlazení amplitud: jednopólový filtr na úrovni bloku, uvnitř bloku
        # lineární rampa od staré hodnoty k nové (žádné lupání)
        up = 1 - np.exp(-n / (attack * sr))
        down = 1 - np.exp(-n / (release * sr))
        start_l = self.amp_l.copy()
        start_r = self.amp_r.copy()
        self.amp_l += (target_l - self.amp_l) * np.where(target_l >= self.amp_l, up, down)
        self.amp_r += (target_r - self.amp_r) * np.where(target_r >= self.amp_r, up, down)

        out = np.zeros((n, 2), dtype=np.float32)
        if self.env < 1e-5 and env_target == 0.0:
            return out

        # jen harmonické pod ~45 % Nyquista (žádný aliasing)
        max_k = int(min(self.h, np.floor(0.45 * sr / max(self.f0, 1.0))))
        if max_k < 1:
            return out

        k = self.k[:max_k]
        t = np.arange(1, n + 1, dtype=np.float64)
        ramp = (t / n)[None, :]
        amps_l = start_l[:max_k, None] + (self.amp_l[:max_k] - start_l[:max_k])[:, None] * ramp
        amps_r = start_r[:max_k, None] + (self.amp_r[:max_k] - start_r[:max_k])[:, None] * ramp

        inc_l = self.f0 * k / sr
        inc_r = self.f0 * k * (1 + self.det[:max_k] * self.spread * 0.004) / sr
        sines_l = np.sin(2 * np.pi * (self.phase_l[:max_k, None] + inc_l[:, None] * t[None, :]))
        sines_r = np.sin(2 * np.pi * (self.phase_r[:max_k, None] + inc_r[:, None] * t[None, :]))
        self.phase_l[:max_k] = (self.phase_l[:max_k] + inc_l * n) % 1.0
        self.phase_r[:max_k] = (self.phase_r[:max_k] + inc_r * n) % 1.0

        sum_l = np.einsum("kn,kn->n", amps_l, sines_l)
        sum_r = np.einsum("kn,kn->n", amps_r, sines_r)

        # normalizace podle skutečné energie amplitud: řídké spektrum
        # (klidná hladina) nezeslabujeme, husté nepřebudí výstup
        energy = 0.5 * (float(np.square(self.amp_l).sum())
                        + float(np.square(self.amp_r).sum()))
        scale = gain / max(1.0, np.sqrt(energy))
        out[:, 0] = np.tanh(sum_l * env * scale)
        out[:, 1] = np.tanh(sum_r * env * scale)
        return out
