"""Dirigent: pohyb vody -> noty.

Nástroj hraje POUZE když se hladina hýbe. Klidná voda = ticho (jen
doznívající release už spuštěných padů). Vlna spustí notu ze stupnice;
síla vlny určí velocity a délku noty, rozložení energie ve spektru
(jemné vs. hrubé vlnky) určí rejstřík. Klaviatura jen přelaďuje
kořen/tóninu, ve které voda hraje.
"""

import random
from dataclasses import dataclass, field


# stupnice pro rituální charakter (intervaly od kořene)
SCALES = {
    "pentatonika moll": [0, 3, 5, 7, 10],
    "pentatonika dur": [0, 2, 4, 7, 9],
    "lydická": [0, 2, 4, 6, 7, 9, 11],
    "dórská": [0, 2, 3, 5, 7, 9, 10],
}


@dataclass
class NoteEvent:
    """Jedna událost pro syntezátor."""
    kind: str = "on"            # "on" | "off"
    midi: int = 0
    vel: float = 0.0            # 0..1, ze síly vlny
    dur: float = 0.0            # plánovaná délka drženi noty (s), z doznívání vlny


@dataclass
class ConductorParams:
    """Nastavení chování; mění je slidery/klaviatura."""
    root: int = 45              # kořen (MIDI, A2) - přelaďuje klaviatura
    scale: str = "pentatonika moll"
    threshold: float = 0.06     # práh onsetu: slabší pohyb noty nespouští
    density: float = 0.5        # hustota not 0..1 (mapuje se na min. rozestup)
    min_gap: float = 0.18       # min. čas mezi dvěma notami (s), odvozen z density
    note_cooldown: float = 1.2  # tatáž nota se neopakuje dřív než za (s)
    register_span: int = 3      # kolik oktáv nad kořenem voda pokrývá
    dur_min: float = 0.8        # nejkratší nota (s) - slabá vlnka
    dur_max: float = 6.0        # nejdelší nota (s) - velká vlna
    silence_release: float = 0.4  # klesne-li motion pod tento podíl prahu,
                                  #   držené noty se pouštějí (voda "domluvila")


@dataclass
class ConductorState:
    """Vnitřní paměť dirigenta mezi snímky."""
    held: dict = field(default_factory=dict)   # midi -> čas plánovaného note_off
    last_note_time: float = -1e9               # čas poslední spuštěné noty
    last_note_midi: dict = field(default_factory=dict)  # midi -> čas posledního spuštění


class WaterConductor:
    def __init__(self, params: ConductorParams = None):
        self.p = params or ConductorParams()
        self.s = ConductorState()

    def set_root(self, midi: int):
        """Kořen z klaviatury; složí se do rozumného spodního rejstříku."""
        midi = int(midi)
        while midi > 57:
            midi -= 12
        while midi < 33:
            midi += 12
        self.p.root = midi

    def held_notes(self) -> set:
        return set(self.s.held)

    def release_all(self) -> list:
        events = [NoteEvent("off", midi) for midi in self.s.held]
        self.s.held.clear()
        return events

    def update(self, feat, now: float) -> list:
        """Zavolat na každý analyzovaný snímek; vrací NoteEvent list."""
        p, st = self.p, self.s
        events = []

        # noty, kterým vypršela délka určená vodou
        for midi, t_off in list(st.held.items()):
            if now >= t_off:
                events.append(NoteEvent("off", midi))
                del st.held[midi]

        # voda se uklidnila -> pustit vše, ať jen doznívá
        if feat.motion < p.threshold * p.silence_release and st.held:
            for midi in list(st.held):
                events.append(NoteEvent("off", midi))
            st.held.clear()

        # spouštění: jen skutečný nárůst pohybu (onset), s rozestupem podle density
        min_gap = 0.10 + (1.0 - p.density) * 0.8
        if feat.onset >= p.threshold and now - st.last_note_time >= min_gap:
            vel = min(1.0, 0.25 + (feat.onset - p.threshold)
                      / max(1e-6, 1.0 - p.threshold) * 1.4)
            scale = SCALES[p.scale]
            span = len(scale) * p.register_span

            # jemné vlnky (energie ve vysokém pásmu) -> vyšší polohy
            if feat.band_energy is not None:
                pos = float(0.10 * feat.band_energy[0]
                            + 0.50 * feat.band_energy[1]
                            + 0.90 * feat.band_energy[2])
            else:
                pos = 0.4
            pos = min(1.0, max(0.0, pos + random.uniform(-0.12, 0.12)))
            idx = min(span - 1, int(pos * span))

            octv, deg = divmod(idx, len(scale))
            midi = p.root + 12 * octv + scale[deg]
            # tatáž nota příliš brzy -> zkus sousední stupeň
            if now - st.last_note_midi.get(midi, -1e9) < p.note_cooldown:
                octv, deg = divmod((idx + 1) % span, len(scale))
                midi = p.root + 12 * octv + scale[deg]

            if now - st.last_note_midi.get(midi, -1e9) >= p.note_cooldown:
                dur = p.dur_min + (p.dur_max - p.dur_min) * vel
                events.append(NoteEvent("on", midi, vel, dur))
                st.held[midi] = now + dur
                st.last_note_time = now
                st.last_note_midi[midi] = now
                # silná vlna přidá dlouhý základ o oktávu pod kořenem
                low = p.root - 12
                if vel > 0.72 and low not in st.held:
                    events.append(NoteEvent("on", low, vel * 0.8, dur * 1.5))
                    st.held[low] = now + dur * 1.5
                    st.last_note_midi[low] = now
        return events
