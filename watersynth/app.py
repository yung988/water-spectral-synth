"""Hlavní aplikace: pygame okno, propojení zdroje obrazu, analýzy,
syntézy, klaviatury (obrazovka / klávesnice / MIDI) a vizualizací."""

import argparse
import sys
import time

import numpy as np
import pygame

from . import ui
from .analysis import SpectralAnalyzer
from .midiin import MidiInput
from .sources import SimSource, to_square_gray
from .synth import AdditiveSynth, midi_to_freq

ANALYSIS_FPS = 30
PENTATONIC = [0, 3, 5, 7, 10]  # mollová pentatonika pro výšku z obrazu

# fyzická řada kláves jako klaviatura (A = C, W = C#, ...)
KEY_TO_SEMI = {
    pygame.K_a: 0, pygame.K_w: 1, pygame.K_s: 2, pygame.K_e: 3, pygame.K_d: 4,
    pygame.K_f: 5, pygame.K_t: 6, pygame.K_g: 7, pygame.K_y: 8, pygame.K_h: 9,
    pygame.K_u: 10, pygame.K_j: 11, pygame.K_k: 12, pygame.K_o: 13,
    pygame.K_l: 14, pygame.K_p: 15, pygame.K_SEMICOLON: 16,
}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "H"]


def note_name(midi: int) -> str:
    return NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


class AudioOut:
    def __init__(self, synth: AdditiveSynth, device=None):
        import sounddevice as sd
        self.stream = sd.OutputStream(
            samplerate=synth.sr, channels=2, dtype="float32",
            blocksize=256, latency="low", device=device,
            callback=self._callback)
        self.synth = synth
        self.stream.start()

    def _callback(self, outdata, frames, _time, _status):
        outdata[:] = self.synth.render(frames)

    def close(self):
        self.stream.stop()
        self.stream.close()


class App:
    W, H = 1120, 700

    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption("Water Spectral Synth")
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.font = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        self.font_big = pygame.font.Font(None, 30)
        self.clock = pygame.time.Clock()

        self.analyzer = SpectralAnalyzer(size=args.size, harmonics=args.harmonics)
        self.synth = AdditiveSynth(samplerate=args.samplerate, harmonics=args.harmonics)
        self.midi = MidiInput()

        self.audio = None
        self.audio_error = None
        if not args.no_audio:
            try:
                self.audio = AudioOut(self.synth, device=args.device)
            except Exception as e:
                self.audio_error = f"Audio nedostupné: {e}"

        # zdroje obrazu
        self.args = args
        self.source = SimSource(size=args.size)
        self.source_error = None

        # widgety
        self.sl_diff = ui.Slider("Statický obraz <-> Změna (diff)", 0.0, 1.0, 0.0)
        self.sl_gamma = ui.Slider("Kontrast spektra (gamma)", 0.4, 3.0, 1.4)
        self.sl_spread = ui.Slider("Prostor z diagonál (spread)", 0.0, 1.0, 0.5)
        self.sl_attack = ui.Slider("Náběh harmonických (s)", 0.005, 0.5, 0.02, "{:.3f}")
        self.sl_release = ui.Slider("Dozvuk harmonických (s)", 0.02, 2.0, 0.25)
        self.sl_gain = ui.Slider("Hlasitost", 0.0, 1.0, 0.5)
        self.sl_wind = ui.Slider("Vlnění simulace", 0.0, 1.0, 0.35)
        self.sliders = [self.sl_diff, self.sl_gamma, self.sl_spread,
                        self.sl_attack, self.sl_release, self.sl_gain, self.sl_wind]
        self.tg_drone = ui.Toggle("Drone (držet tón)", False, "mezerník")
        self.tg_auto = ui.Toggle("Náhodné kapky v simulaci", True)
        self.tg_image_pitch = ui.Toggle("Výška tónu z obrazu (pentatonika)", False, "M")
        self.toggles = [self.tg_drone, self.tg_auto, self.tg_image_pitch]

        self.piano = ui.Piano(octaves=2)
        self.piano.base_midi = 48  # C3

        # rozložení
        x = 700
        y = 64
        for s in self.sliders:
            s.set_rect(x, y, 404)
            y += 44
        y += 6
        for t in self.toggles:
            t.set_rect(x, y, 404)
            y += 30
        self.status_y = y + 6
        self.preview_rect = pygame.Rect(16, 64, 320, 320)
        self.spectrum_rect = pygame.Rect(352, 64, 320, 320)
        self.bars_rect = pygame.Rect(16, 412, 656, 118)
        self.piano.set_rect(16, 548, 1088, 118)

        # stav hraní
        self.held = []            # zásobník (zdroj, midi, vel); poslední hraje
        self.mouse_note = None
        self.image_note = None
        self.image_candidate = None
        self.image_stable = 0
        self.last_frame = None
        self.last_analysis = 0.0
        self.running = True

    # ---- noty -----------------------------------------------------------

    def _update_gate(self):
        if self.tg_image_pitch.value:
            return  # výšku řídí obraz
        if self.held:
            _, midi, vel = self.held[-1]
            self.synth.note_on(midi, vel)
            self.piano.active = {m for _, m, _ in self.held}
        elif self.tg_drone.value:
            self.synth.gate = 1.0
            self.piano.active = set()
        else:
            self.synth.note_off()
            self.piano.active = set()

    def note_on(self, origin, midi, vel=0.9):
        self.held = [h for h in self.held if not (h[0] == origin and h[1] == midi)]
        self.held.append((origin, midi, vel))
        self._update_gate()

    def note_off(self, origin, midi):
        self.held = [h for h in self.held if not (h[0] == origin and h[1] == midi)]
        self._update_gate()

    def _image_pitch(self, frame):
        """Výška tónu z těžiště spektra, kvantovaná do pentatoniky."""
        if frame.energy < 0.01:
            return
        norm = min(1.0, np.log2(max(1.0, frame.centroid)) / np.log2(self.analyzer.h))
        degree = min(9, int(norm * 10))
        midi = self.piano.base_midi + PENTATONIC[degree % 5] + 12 * (degree // 5)
        if midi == self.image_candidate:
            self.image_stable += 1
        else:
            self.image_candidate = midi
            self.image_stable = 0
        # nota se změní až po pár stabilních snímcích, aby tón necukal
        if self.image_stable >= 2 and midi != self.image_note:
            self.image_note = midi
            self.synth.note_on(midi, 0.9)
            self.piano.active = {midi}

    # ---- zdroje ----------------------------------------------------------

    def switch_source(self, kind):
        from .sources import CameraSource, VideoFileSource
        if not isinstance(self.source, SimSource):
            self.source.close()
        self.source_error = None
        try:
            if kind == "camera":
                self.source = CameraSource(self.args.camera)
            elif kind == "video" and self.args.video:
                self.source = VideoFileSource(self.args.video)
            else:
                self.source = SimSource(size=self.args.size)
        except Exception as e:
            self.source_error = str(e)
            self.source = SimSource(size=self.args.size)
        self.analyzer.reset()

    # ---- hlavní smyčka ----------------------------------------------------

    def run(self):
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)

            for kind, note, vel in self.midi.poll():
                if kind == "on":
                    self.note_on("midi", note, vel)
                else:
                    self.note_off("midi", note)

            if isinstance(self.source, SimSource):
                self.source.sim.wind = self.sl_wind.value
                self.source.sim.auto = self.tg_auto.value

            rgb = self.source.read()
            now = time.monotonic()
            if rgb is not None and now - self.last_analysis >= 1.0 / ANALYSIS_FPS:
                self.last_analysis = now
                gray = to_square_gray(rgb, self.analyzer.n)
                frame = self.analyzer.analyze(
                    gray, diff_mix=self.sl_diff.value, gamma=self.sl_gamma.value)
                self.last_frame = frame
                self.synth.set_frame(frame.amps_l, frame.amps_r,
                                     frame.diag_ratio * self.sl_spread.value)
                self.synth.set_params(attack=self.sl_attack.value,
                                      release=self.sl_release.value,
                                      gain=self.sl_gain.value)
                if self.tg_image_pitch.value:
                    self._image_pitch(frame)

            self.draw(rgb)
            self.clock.tick(60)

        self.shutdown()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return

        for s in self.sliders:
            s.handle(event)
        for t in self.toggles:
            if t.handle(event):
                if t is self.tg_drone or t is self.tg_image_pitch:
                    self._on_mode_change()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            midi = self.piano.note_at(event.pos)
            if midi is not None:
                self.mouse_note = midi
                self.note_on("mouse", midi)
            elif self.preview_rect.collidepoint(event.pos) and isinstance(self.source, SimSource):
                rx = (event.pos[0] - self.preview_rect.x) / self.preview_rect.w
                ry = (event.pos[1] - self.preview_rect.y) / self.preview_rect.h
                self.source.sim.add_drop(rx, ry)
        if event.type == pygame.MOUSEBUTTONUP and self.mouse_note is not None:
            self.note_off("mouse", self.mouse_note)
            self.mouse_note = None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key in KEY_TO_SEMI:
                self.note_on("key", self.piano.base_midi + KEY_TO_SEMI[event.key])
            elif event.key == pygame.K_z:
                self.shift_octave(-12)
            elif event.key == pygame.K_x:
                self.shift_octave(12)
            elif event.key == pygame.K_SPACE:
                self.tg_drone.value = not self.tg_drone.value
                self._on_mode_change()
            elif event.key == pygame.K_m:
                self.tg_image_pitch.value = not self.tg_image_pitch.value
                self._on_mode_change()
            elif event.key == pygame.K_1:
                self.switch_source("sim")
            elif event.key == pygame.K_2:
                self.switch_source("camera")
            elif event.key == pygame.K_3:
                self.switch_source("video")
        if event.type == pygame.KEYUP and event.key in KEY_TO_SEMI:
            self.note_off("key", self.piano.base_midi + KEY_TO_SEMI[event.key])

    def _on_mode_change(self):
        self.image_note = None
        if self.tg_image_pitch.value:
            self.synth.gate = 1.0
        else:
            self._update_gate()

    def shift_octave(self, d):
        self.piano.base_midi = max(24, min(72, self.piano.base_midi + d))
        # držené noty z klávesnice by po posunu visely - pustit
        self.held = [h for h in self.held if h[0] != "key"]
        self._update_gate()

    # ---- kreslení ----------------------------------------------------------

    def draw(self, rgb):
        s = self.screen
        s.fill(ui.BG)
        title = self.font_big.render("Water Spectral Synth", True, ui.ACCENT)
        s.blit(title, (16, 14))
        sub = self.font_small.render(
            "2D FFT hladiny -> aditivní syntéza · noty: A W S E D F T G Y H U J · "
            "Z/X oktáva · mezerník drone · M výška z obrazu · 1 simulace / 2 kamera / 3 video",
            True, ui.DIM)
        s.blit(sub, (16, 42))

        # náhled zdroje
        if rgb is not None:
            surf = ui.rgb_surface(rgb)
            # střední čtvercový výřez jako v analýze
            side = min(surf.get_width(), surf.get_height())
            crop = pygame.Rect((surf.get_width() - side) // 2,
                               (surf.get_height() - side) // 2, side, side)
            surf = surf.subsurface(crop)
            s.blit(pygame.transform.smoothscale(surf, self.preview_rect.size),
                   self.preview_rect)
        pygame.draw.rect(s, ui.BORDER, self.preview_rect, 1, border_radius=6)

        # 2D spektrum
        if self.last_frame is not None and self.last_frame.spectrum_img is not None:
            spec = ui.spectrum_surface(self.last_frame.spectrum_img)
            s.blit(pygame.transform.scale(spec, self.spectrum_rect.size),
                   self.spectrum_rect)
        pygame.draw.rect(s, ui.BORDER, self.spectrum_rect, 1, border_radius=6)

        cap1 = self.font_small.render(
            getattr(self.source, "name", "?") + " (klik = kapka)", True, ui.DIM)
        s.blit(cap1, (self.preview_rect.x, self.preview_rect.bottom + 6))
        cap2 = self.font_small.render(
            "2D spektrum (střed = nízké prostorové frekvence)", True, ui.DIM)
        s.blit(cap2, (self.spectrum_rect.x, self.spectrum_rect.bottom + 6))

        # harmonické
        if self.last_frame is not None:
            ui.draw_bars(s, self.bars_rect, self.last_frame.amps_l, self.last_frame.amps_r)
        cap3 = self.font_small.render(
            "harmonické: levý kanál nahoru (azurová) / pravý dolů (oranžová)",
            True, ui.DIM)
        s.blit(cap3, (self.bars_rect.x, self.bars_rect.bottom + 4))

        for sl in self.sliders:
            sl.draw(s, self.font_small)
        for t in self.toggles:
            t.draw(s, self.font_small)

        # stavové řádky
        y = self.status_y
        lines = [self.midi.status]
        if self.audio_error:
            lines.append(self.audio_error)
        if self.source_error:
            lines.append("Zdroj: " + self.source_error)
        note = None
        if self.tg_image_pitch.value:
            note = self.image_note
        elif self.held:
            note = self.held[-1][1]
        lines.append("Tón: " + (note_name(note) if note is not None else
                                ("drone" if self.tg_drone.value else "—")))
        lines.append(f"{self.clock.get_fps():.0f} fps")
        for line in lines:
            s.blit(self.font_small.render(str(line), True, ui.DIM), (700, y))
            y += 20

        self.piano.draw(s, self.font_small)
        pygame.display.flip()

    def shutdown(self):
        if self.audio:
            self.audio.close()
        self.source.close()
        self.midi.close()
        pygame.quit()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="watersynth",
        description="Water Spectral Synth - 2D FFT vodní hladiny řídí aditivní syntézu.")
    p.add_argument("--camera", type=int, default=0,
                   help="index kamery (iPhone po kabelu bývá 0 nebo 1)")
    p.add_argument("--video", type=str, default=None, help="cesta k video souboru")
    p.add_argument("--size", type=int, default=128,
                   help="rozlišení analýzy (mocnina dvou, výchozí 128)")
    p.add_argument("--harmonics", type=int, default=96, help="počet harmonických")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--device", type=str, default=None, help="audio výstup (sounddevice)")
    p.add_argument("--list-devices", action="store_true", help="vypsat audio zařízení")
    p.add_argument("--no-audio", action="store_true", help="běžet bez zvuku (ladění)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    app = App(args)
    if args.video:
        app.switch_source("video")
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
