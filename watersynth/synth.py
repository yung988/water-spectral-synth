"""Aditivní syntéza: banka sinusových oscilátorů řízená spektrem obrazu.

Levý a pravý kanál mají vlastní amplitudy (podle orientace struktur na
hladině); pravý kanál lze jemně rozladit ("spread" z diagonální energie).
Render běží po blocích v audio callbacku, parametry chodí z hlavního vlákna
pod zámkem.
"""

import threading
from dataclasses import dataclass, field

import numpy as np


def midi_to_freq(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


# ---- datové struktury padového syntezátoru (fáze 1) ------------------------

@dataclass
class PadParams:
    """Parametry společné všem hlasům; mění je slidery z hlavního vlákna."""
    attack: float = 0.8         # náběh obálky hlasu (s) - pomalý, padový
    release: float = 4.0        # dozvuk obálky hlasu (s) - dlouhý, rituální
    tilt: float = 0.6           # spektrální sklon 0..1: 1 = silné tlumení
                                #   vysokých harmonických (teplý, tmavý zvuk)
    unison: int = 3             # počet rozladěných vrstev na hlas
    detune: float = 0.006       # max. relativní rozladění vrstev (+-)
    spread: float = 0.5         # stereo rozprostření vrstev 0..1
    reverb_mix: float = 0.45    # podíl reverbu ve výstupu 0..1
    gain: float = 0.5           # celková hlasitost


@dataclass
class PadVoice:
    """Jeden znějící tón: vlastní banka oscilátorů (unison x harmonické)
    a vlastní obálka. Hlas žije od note_on do konce release."""
    midi: float = 0.0
    freq: float = 0.0           # základní frekvence (Hz)
    vel: float = 0.0            # velocity 0..1 (síla vlny)
    gate: float = 0.0           # 1 = drží, 0 = uvolněn (release)
    env: float = 0.0            # aktuální hodnota obálky 0..1
    age: int = 0                # pořadí spuštění (pro voice stealing)
    dur: float = 0.0            # plánovaná délka noty (s) - určuje ji voda
    t_on: float = 0.0           # čas spuštění (monotonic)
    phases: np.ndarray = field(default=None)   # fáze (unison, harmonické) 0..1
    det_ratio: np.ndarray = field(default=None)  # rozladění vrstev (unison,)
    pan: np.ndarray = field(default=None)        # panorama vrstev (unison,) 0..1


@dataclass
class PadSynthState:
    """Stav polyfonního padového syntezátoru (samotná třída PadSynth
    dostane rozhraní ve fázi 2)."""
    samplerate: int = 48000
    harmonics: int = 64         # harmonických na hlas (pad nepotřebuje 96)
    max_voices: int = 6
    voices: list = field(default_factory=list)   # aktivní PadVoice
    params: "PadParams" = field(default_factory=lambda: PadParams())
    # sdílené spektrum z vody (po aplikaci tilt), cíl + vyhlazený stav
    target_l: np.ndarray = field(default=None)   # (harmonics,)
    target_r: np.ndarray = field(default=None)
    amp_l: np.ndarray = field(default=None)
    amp_r: np.ndarray = field(default=None)
    reverb: "ReverbState" = field(default=None)
    age_counter: int = 0        # roste s každým note_on


@dataclass
class ReverbState:
    """Schroederův reverb: 4 comb filtry + 2 allpass, stereo."""
    comb_buf: list = field(default=None)     # [np.ndarray (delay, 2)] x4
    comb_idx: list = field(default=None)     # zápisové indexy combů
    comb_damp: list = field(default=None)    # stav tlumení výšek v combu (2,) x4
    ap_buf: list = field(default=None)       # [np.ndarray (delay, 2)] x2
    ap_idx: list = field(default=None)       # zápisové indexy allpassů
    feedback: float = 0.84                   # delka dozvuku
    damp: float = 0.4                        # tlumení výšek v ocasu 0..1


class Reverb:
    """Schroederův reverb vektorizovaný po blocích: všechna zpoždění jsou
    delší než audio blok, takže čtení a zápis do kruhových bufferů se
    v rámci bloku nepřekrývají a jde je dělat najednou (žádná smyčka
    po vzorcích)."""

    COMB_DELAYS = [1557, 1617, 1491, 1422]   # vzorky při 44.1 kHz (Freeverb)
    AP_DELAYS = [556, 441]

    def __init__(self, samplerate: int = 48000, feedback: float = 0.84,
                 damp: float = 0.4):
        s = samplerate / 44100.0
        self.feedback = feedback
        self.damp = damp
        self.combs = []
        for i, d in enumerate(self.COMB_DELAYS):
            n = int(d * s) + i * 13   # mírně rozladit délky
            self.combs.append({
                "buf": np.zeros((n, 2), dtype=np.float64),
                "idx": 0,
                "last": np.zeros(2, dtype=np.float64),
            })
        self.aps = []
        for d in self.AP_DELAYS:
            n = int(d * s)
            self.aps.append({"buf": np.zeros((n, 2), dtype=np.float64), "idx": 0})
        self.max_block = min(min(len(c["buf"]) for c in self.combs),
                             min(len(a["buf"]) for a in self.aps))

    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n > self.max_block:
            return np.vstack([self.process(x[i:i + self.max_block])
                              for i in range(0, n, self.max_block)])
        wet = np.zeros_like(x)
        for c in self.combs:
            buf = c["buf"]
            idxs = (c["idx"] + np.arange(n)) % len(buf)
            out = buf[idxs]
            # tlumení výšek: jemné FIR vyhlazení zpětné vazby; každý průchod
            # smyčkou ho aplikuje znovu, výšky tak doznívají rychleji
            prev = np.vstack([c["last"][None, :], out[:-1]])
            buf[idxs] = x + self.feedback * ((1 - self.damp) * out + self.damp * prev)
            c["last"] = out[-1].copy()
            c["idx"] = (c["idx"] + n) % len(buf)
            wet += out
        wet *= 0.25
        for a in self.aps:
            buf = a["buf"]
            idxs = (a["idx"] + np.arange(n)) % len(buf)
            r = buf[idxs]
            buf[idxs] = wet + 0.5 * r
            wet = r - 0.5 * wet
            a["idx"] = (a["idx"] + n) % len(buf)
        return wet


class PadSynth:
    """Polyfonní padová aditivní syntéza: každý hlas je banka sinusových
    oscilátorů (unison vrstvy x harmonické) s pomalou obálkou; barvu všech
    hlasů tvaruje spektrum vody (po spektrálním sklonu tilt). Součet jde
    do reverbu. Render běží v audio callbacku, řízení pod zámkem."""

    def __init__(self, samplerate: int = 48000, harmonics: int = 64,
                 max_voices: int = 6):
        self.sr = samplerate
        self.h = h = harmonics
        self.max_voices = max_voices
        self._lock = threading.Lock()
        self.params = PadParams()

        self.k = np.arange(1, h + 1, dtype=np.float64)
        self.target_l = np.zeros(h, dtype=np.float64)
        self.target_r = np.zeros(h, dtype=np.float64)
        self.amp_l = np.zeros(h, dtype=np.float64)
        self.amp_r = np.zeros(h, dtype=np.float64)

        self.voices: list[PadVoice] = []
        self._age = 0
        self._rng = np.random.default_rng(7)
        self.reverb = Reverb(samplerate)

    # ---- řízení z hlavního vlákna --------------------------------------

    def set_frame(self, amps_l, amps_r, spread: float = 0.0):
        with self._lock:
            np.copyto(self.target_l, amps_l[:self.h])
            np.copyto(self.target_r, amps_r[:self.h])

    def set_params(self, **kw):
        with self._lock:
            for name, val in kw.items():
                if val is not None and hasattr(self.params, name):
                    setattr(self.params, name, float(val) if name != "unison" else int(val))

    def note_on(self, midi: float, vel: float = 0.8, dur: float = 0.0):
        with self._lock:
            for v in self.voices:
                if v.midi == midi:
                    v.gate = 1.0
                    v.vel = max(v.vel, float(vel))
                    v.dur = dur
                    return
            if len(self.voices) >= self.max_voices:
                released = [v for v in self.voices if v.gate == 0.0]
                victim = min(released or self.voices, key=lambda v: (v.env, v.age))
                self.voices.remove(victim)
            u = max(1, self.params.unison)
            v = PadVoice(
                midi=float(midi), freq=midi_to_freq(midi), vel=float(vel),
                gate=1.0, env=0.0, age=self._age, dur=dur,
                phases=self._rng.random((u, self.h)),
                det_ratio=1.0 + (np.linspace(-1, 1, u) if u > 1 else np.zeros(1))
                          * self.params.detune,
                pan=np.linspace(0.0, 1.0, u) if u > 1 else np.array([0.5]),
            )
            self._age += 1
            self.voices.append(v)

    def note_off(self, midi: float):
        with self._lock:
            for v in self.voices:
                if v.midi == midi:
                    v.gate = 0.0

    def all_off(self):
        with self._lock:
            for v in self.voices:
                v.gate = 0.0

    def active_notes(self):
        with self._lock:
            return {int(v.midi) for v in self.voices if v.gate > 0.0}

    # ---- audio vlákno ---------------------------------------------------

    def render(self, n: int) -> np.ndarray:
        with self._lock:
            p = self.params
            attack, release = p.attack, p.release
            tilt, gain, mix = p.tilt, p.gain, p.reverb_mix
            target_l = self.target_l.copy()
            target_r = self.target_r.copy()
            voices = list(self.voices)

        sr = self.sr
        out = np.zeros((n, 2), dtype=np.float64)
        t = np.arange(1, n + 1, dtype=np.float64)

        # spektrální sklon: potlačení vysokých harmonických (teplý pad)
        curve = self.k ** (-1.7 * tilt)
        # vyhlazení barvy (~120 ms) - timbre se mění plynule, ne skokem
        coef = 1 - np.exp(-n / (0.12 * sr))
        self.amp_l += (target_l * curve - self.amp_l) * coef
        self.amp_r += (target_r * curve - self.amp_r) * coef

        dead = []
        for v in voices:
            env_target = v.gate * v.vel
            tau = attack if env_target > v.env else release
            env_ramp = env_target + (v.env - env_target) * np.exp(-t / (tau * sr))
            v.env = float(env_ramp[-1])
            if v.env < 1e-4 and v.gate == 0.0:
                dead.append(v)
                continue

            max_k = int(min(self.h, np.floor(0.45 * sr / max(v.freq, 1.0))))
            if max_k < 1:
                continue
            k = self.k[:max_k]
            for u in range(len(v.det_ratio)):
                inc = v.freq * v.det_ratio[u] * k / sr
                ph = v.phases[u, :max_k]
                sines = np.sin(2 * np.pi * (ph[:, None] + inc[:, None] * t[None, :]))
                v.phases[u, :max_k] = (ph + inc * n) % 1.0
                pan = float(v.pan[u])
                amps = self.amp_l[:max_k] * (1 - pan) + self.amp_r[:max_k] * pan
                mono = (amps @ sines) * env_ramp
                out[:, 0] += mono * np.cos(pan * np.pi / 2)
                out[:, 1] += mono * np.sin(pan * np.pi / 2)

        if dead:
            with self._lock:
                for v in dead:
                    if v in self.voices:
                        self.voices.remove(v)

        # normalizace podle energie spektra a počtu unison vrstev
        energy = 0.5 * (float(np.square(self.amp_l).sum())
                        + float(np.square(self.amp_r).sum()))
        u = max(1, len(voices[0].det_ratio) if voices else 1)
        scale = gain / max(1.0, np.sqrt(energy * u))
        out *= scale

        wet = self.reverb.process(out)
        out = out * (1.0 - 0.7 * mix) + wet * (1.6 * mix)
        return np.tanh(out).astype(np.float32)


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
