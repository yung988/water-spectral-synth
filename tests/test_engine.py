"""Testy jádra (bez kamery, zvuku i GUI): python3 tests/test_engine.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from watersynth.analysis import SpectralAnalyzer
from watersynth.sim import WaterSim
from watersynth.synth import AdditiveSynth, midi_to_freq

failures = 0


def check(name, cond, detail=""):
    global failures
    if cond:
        print("  ok –", name)
    else:
        failures += 1
        print("  FAIL –", name, detail)


N, H = 128, 96
xx, yy = np.meshgrid(np.arange(N), np.arange(N))

# --- analýza: orientace struktur -> stereo kanály --------------------------

an = SpectralAnalyzer(size=N, harmonics=H)
vertical = (0.5 + 0.4 * np.sin(2 * np.pi * 10 * xx / N)).astype(np.float32)   # |||
f = an.analyze(vertical)
check("svislé pruhy -> pravý kanál", f.amps_r.max() > 5 * max(f.amps_l.max(), 1e-6),
      f"L={f.amps_l.max():.3f} R={f.amps_r.max():.3f}")

an = SpectralAnalyzer(size=N, harmonics=H)
horizontal = (0.5 + 0.4 * np.sin(2 * np.pi * 10 * yy / N)).astype(np.float32)  # ===
f = an.analyze(horizontal)
check("vodorovné pruhy -> levý kanál", f.amps_l.max() > 5 * max(f.amps_r.max(), 1e-6),
      f"L={f.amps_l.max():.3f} R={f.amps_r.max():.3f}")

an = SpectralAnalyzer(size=N, harmonics=H)
diagonal = (0.5 + 0.4 * np.sin(2 * np.pi * 10 * (xx + yy) / N)).astype(np.float32)
f = an.analyze(diagonal)
check("diagonální pruhy -> vysoký diag_ratio", f.diag_ratio > 0.5, f"diag={f.diag_ratio:.3f}")

# hrubší struktura -> nižší harmonické (nižší těžiště)
an = SpectralAnalyzer(size=N, harmonics=H)
coarse = (0.5 + 0.4 * np.sin(2 * np.pi * 4 * xx / N)).astype(np.float32)
c_coarse = an.analyze(coarse).centroid
an = SpectralAnalyzer(size=N, harmonics=H)
fine = (0.5 + 0.4 * np.sin(2 * np.pi * 40 * xx / N)).astype(np.float32)
c_fine = an.analyze(fine).centroid
check("jemnější vlnky -> vyšší těžiště spektra", c_fine > 2 * c_coarse,
      f"hrubé={c_coarse:.1f} jemné={c_fine:.1f}")

# --- diff režim: statická scéna mlčí, změna zní ----------------------------

an = SpectralAnalyzer(size=N, harmonics=H)
an.analyze(vertical, diff_mix=1.0)                 # první snímek naplní prev
still = an.analyze(vertical, diff_mix=1.0)         # beze změny
moved = np.roll(vertical, 3, axis=1)
moving = an.analyze(moved, diff_mix=1.0)           # posunutá vlna
check("diff: statická scéna téměř mlčí, pohyb zní",
      moving.energy > 5 * max(still.energy, 1e-6),
      f"klid={still.energy:.4f} pohyb={moving.energy:.4f}")

# --- simulace: kapka otevře spektrum ---------------------------------------

sim = WaterSim(size=N)
sim.auto = False
sim.wind = 0.3
an = SpectralAnalyzer(size=N, harmonics=H)
for _ in range(10):
    calm_frame = sim.step(1 / 30)
calm = an.analyze(calm_frame)
calm_high = float(calm.amps_l[15:].sum() + calm.amps_r[15:].sum())
sim.add_drop(0.5, 0.5, 1.0)
for _ in range(10):
    drop_frame = sim.step(1 / 30)
drop = an.analyze(drop_frame)
drop_high = float(drop.amps_l[15:].sum() + drop.amps_r[15:].sum())
check("kapka přidá vysoké harmonické", drop_high > 2 * max(calm_high, 1e-6),
      f"klid={calm_high:.3f} kapka={drop_high:.3f}")

# --- syntéza ----------------------------------------------------------------

sr = 48000
synth = AdditiveSynth(samplerate=sr, harmonics=H)
amps = np.zeros(H)
amps[0] = 1.0  # jen základní tón
synth.set_frame(amps, amps, 0.0)
synth.note_on(57)  # A3 = 220 Hz
audio = np.concatenate([synth.render(256) for _ in range(40)])  # ~0.2 s
check("výstup není NaN a je v rozsahu -1..1",
      np.isfinite(audio).all() and np.abs(audio).max() <= 1.0)
check("při gate=1 zní zvuk", float(np.abs(audio[-2000:]).max()) > 0.05,
      f"max={np.abs(audio[-2000:]).max():.4f}")

# dominantní frekvence odpovídá notě
tail = audio[-8192:, 0] * np.hanning(8192)
spec = np.abs(np.fft.rfft(tail))
peak_hz = float(np.argmax(spec) * sr / 8192)
check("dominantní frekvence ~220 Hz", abs(peak_hz - midi_to_freq(57)) < 8,
      f"peak={peak_hz:.1f} Hz")

# vyšší harmonické se objeví, když je zapneme
amps2 = np.zeros(H)
amps2[0] = 1.0
amps2[4] = 1.0  # 5. harmonická = 1100 Hz
synth.set_frame(amps2, amps2, 0.0)
audio2 = np.concatenate([synth.render(256) for _ in range(40)])
tail2 = audio2[-8192:, 0] * np.hanning(8192)
spec2 = np.abs(np.fft.rfft(tail2))
bin5 = int(round(5 * midi_to_freq(57) * 8192 / sr))
check("5. harmonická se objeví ve zvuku",
      float(spec2[bin5 - 2:bin5 + 3].max()) > 10 * float(spec[bin5 - 2:bin5 + 3].max() + 1e-9),
      f"před={spec[bin5-2:bin5+3].max():.2f} po={spec2[bin5-2:bin5+3].max():.2f}")

# note_off -> zvuk dozní do ticha
synth.note_off()
audio3 = np.concatenate([synth.render(256) for _ in range(400)])  # ~2.1 s
check("po note_off dozní do ticha", float(np.abs(audio3[-1000:]).max()) < 1e-3,
      f"zbytek={np.abs(audio3[-1000:]).max():.5f}")

# žádný aliasing: vysoká nota omezí počet harmonických
synth2 = AdditiveSynth(samplerate=sr, harmonics=H)
full = np.ones(H)
synth2.set_frame(full, full, 0.0)
synth2.note_on(96)  # C7 ~ 2093 Hz; 96 harmonických by letělo přes 200 kHz
audio4 = np.concatenate([synth2.render(256) for _ in range(30)])
check("vysoká nota: výstup stále čistý (bez NaN/přebuzení)",
      np.isfinite(audio4).all() and np.abs(audio4).max() <= 1.0)

# výkon: render musí být rychlejší než realtime
import time
synth3 = AdditiveSynth(samplerate=sr, harmonics=H)
synth3.set_frame(np.ones(H), np.ones(H), 0.5)
synth3.note_on(36)  # nízká nota -> hraje všech 96 harmonických
t0 = time.perf_counter()
blocks = 400  # ~2.1 s zvuku
for _ in range(blocks):
    synth3.render(256)
elapsed = time.perf_counter() - t0
realtime = blocks * 256 / sr
check(f"render rychlejší než realtime ({elapsed:.2f}s na {realtime:.2f}s zvuku)",
      elapsed < realtime * 0.5)

print()
if failures:
    print(failures, "testů selhalo")
    sys.exit(1)
print("Všechny testy enginu prošly.")
