"""Hlavní aplikace: pygame okno, propojení zdroje obrazu, analýzy,
syntézy, klaviatury (obrazovka / klávesnice / MIDI) a vizualizací."""

import argparse
import sys
import threading
import time

import pygame

from . import ui
from .analysis import SpectralAnalyzer
from .conductor import WaterConductor
from .midiin import MidiInput
from .sources import SimSource, to_square_gray
from .synth import PadSynth
from .updater import Updater

ANALYSIS_FPS = 30

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
    """Zvuk přes blokující zápis z vlastního vlákna místo callbacku:
    velký buffer (~250 ms) vyhladí zákmity GILu z UI/kamery/FFT - callback
    s deadlinem 5-20 ms se v Pythonu sekal. Pro pomalé pady latence nevadí."""

    BLOCK = 2048

    def __init__(self, synth, device=None):
        import sounddevice as sd
        self.synth = synth
        self.stream = sd.OutputStream(
            samplerate=synth.sr, channels=2, dtype="float32",
            blocksize=self.BLOCK, latency=0.25, device=device)
        self.stream.start()
        self.underruns = 0
        self.render_ms = 0.0     # klouzavý průměr času renderu na blok
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        budget_ms = 1000.0 * self.BLOCK / self.synth.sr
        while not self._stop.is_set():
            t0 = time.perf_counter()
            block = self.synth.render(self.BLOCK)
            dt = (time.perf_counter() - t0) * 1000.0
            self.render_ms += (dt - self.render_ms) * 0.1
            try:
                if self.stream.write(block):   # True = podtečení
                    self.underruns += 1
                    print(f"[audio] podtečení #{self.underruns} "
                          f"(render {self.render_ms:.1f} ms / limit {budget_ms:.0f} ms)",
                          flush=True)
            except Exception:
                break

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
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
        self.synth = PadSynth(samplerate=args.samplerate,
                              harmonics=min(64, args.harmonics))
        self.conductor = WaterConductor()
        self.midi = MidiInput()
        self.updater = Updater()

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
        self.camera_index = None

        # widgety
        self.sl_sens = ui.Slider("Citlivost na vlny", 0.01, 0.30, 0.06, "{:.3f}")
        self.sl_density = ui.Slider("Hustota not", 0.0, 1.0, 0.5)
        self.sl_length = ui.Slider("Délka not z vody (s)", 1.0, 10.0, 6.0)
        self.sl_release = ui.Slider("Dozvuk pádů (s)", 1.0, 8.0, 4.0)
        self.sl_tilt = ui.Slider("Jas (vysoké harmonické)", 0.0, 1.0, 0.35)
        self.sl_reverb = ui.Slider("Reverb", 0.0, 1.0, 0.45)
        self.sl_gain = ui.Slider("Hlasitost", 0.0, 1.0, 0.6)
        self.sl_wind = ui.Slider("Vlnění simulace", 0.0, 1.0, 0.35)
        self.sliders = [self.sl_sens, self.sl_density, self.sl_length,
                        self.sl_release, self.sl_tilt, self.sl_reverb,
                        self.sl_gain, self.sl_wind]
        self.tg_water = ui.Toggle("Voda hraje (rituál)", True, "M")
        # kapky vypnuté: aplikace po startu mlčí, dokud se něco nepohne
        self.tg_auto = ui.Toggle("Náhodné kapky v simulaci", False)
        self.toggles = [self.tg_water, self.tg_auto]

        # tlačítka zdroje obrazu (viditelná volba místo tajných kláves)
        self.src_buttons = [
            (ui.Button("Simulace"), "sim", None),
            (ui.Button("Kamera 0"), "camera", 0),
            (ui.Button("Kamera 1"), "camera", 1),
            (ui.Button("Kamera 2"), "camera", 2),
        ]

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
        bx = 16
        for btn, _, _ in self.src_buttons:
            btn.set_rect(bx, 410, 156)
            bx += 168
        self.bars_rect = pygame.Rect(16, 448, 656, 66)
        self.piano.set_rect(16, 548, 1088, 118)

        # stav hraní
        self.manual_notes = set()   # noty držené klaviaturou/MIDI/myší
        self.mouse_note = None
        self.last_frame = None
        self.last_analysis = 0.0
        self.running = True

    # ---- noty -----------------------------------------------------------

    def note_on(self, origin, midi, vel=0.85):
        """Ruční hraní: nota zní jako pad a zároveň přelaďuje kořen,
        ve kterém voda hraje."""
        self.synth.note_on(midi, vel)
        self.manual_notes.add(midi)
        self.conductor.set_root(midi)

    def note_off(self, origin, midi):
        self.synth.note_off(midi)
        self.manual_notes.discard(midi)

    # ---- zdroje ----------------------------------------------------------

    def switch_source(self, kind, cam_index=None):
        from .sources import CameraSource, VideoFileSource
        if not isinstance(self.source, SimSource):
            self.source.close()
        self.source_error = None
        self.camera_index = None
        try:
            if kind == "camera":
                idx = cam_index if cam_index is not None else (
                    self.args.camera if self.args.camera is not None else 0)
                self.source = CameraSource(idx)
                self.camera_index = idx
            elif kind == "video" and self.args.video:
                self.source = VideoFileSource(self.args.video)
            else:
                self.source = SimSource(size=self.args.size)
        except Exception as e:
            self.source_error = str(e)
            self.source = SimSource(size=self.args.size)
        self.analyzer.reset()

    def try_auto_camera(self):
        """Při startu bez parametrů zkusí najít kameru (iPhone/webka);
        když žádná není, zůstane simulace a napíše se to do stavu."""
        for idx in (0, 1, 2):
            self.switch_source("camera", cam_index=idx)
            if self.source_error is None:
                return
        self.switch_source("sim")
        self.source_error = ("Kamera nenalezena/nepovolená - běží simulace. "
                             "Povol kameru v Nastavení a klikni na Kamera 0/1/2.")

    # ---- hlavní smyčka ----------------------------------------------------

    def run(self):
        while self.running:
            if self.updater.state == "restarting":
                self.running = False   # nová verze se právě spouští
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
                frame = self.analyzer.analyze(gray, diff_mix=0.25, gamma=1.4)
                self.last_frame = frame
                self.synth.set_frame(frame.amps_l, frame.amps_r)
                self.synth.set_params(release=self.sl_release.value,
                                      tilt=1.0 - self.sl_tilt.value,
                                      reverb_mix=self.sl_reverb.value,
                                      gain=self.sl_gain.value)
                cp = self.conductor.p
                cp.threshold = self.sl_sens.value
                cp.density = self.sl_density.value
                cp.dur_max = self.sl_length.value
                if self.tg_water.value:
                    for ev in self.conductor.update(frame, now):
                        if ev.kind == "on":
                            self.synth.note_on(ev.midi, ev.vel, ev.dur)
                        else:
                            self.synth.note_off(ev.midi)
                self.piano.active = self.manual_notes | self.conductor.held_notes()

            self.draw(rgb)
            # 30 fps stačí (analýza běží na 30) a nechává CPU audio vláknu
            self.clock.tick(30)

        self.shutdown()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return

        for s in self.sliders:
            s.handle(event)
        for t in self.toggles:
            if t.handle(event):
                if t is self.tg_water and not t.value:
                    self._silence_water()
        for btn, kind, idx in self.src_buttons:
            if btn.handle(event):
                self.switch_source(kind, cam_index=idx)

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
            elif event.key == pygame.K_m:
                self.tg_water.value = not self.tg_water.value
                if not self.tg_water.value:
                    self._silence_water()
            elif event.key == pygame.K_i:
                self.updater.install()
            elif event.key == pygame.K_1:
                self.switch_source("sim")
            elif event.key == pygame.K_2:
                self.switch_source("camera")
            elif event.key == pygame.K_3:
                self.switch_source("video")
        if event.type == pygame.KEYUP and event.key in KEY_TO_SEMI:
            self.note_off("key", self.piano.base_midi + KEY_TO_SEMI[event.key])

    def _silence_water(self):
        for ev in self.conductor.release_all():
            self.synth.note_off(ev.midi)

    def shift_octave(self, d):
        self.piano.base_midi = max(24, min(72, self.piano.base_midi + d))
        # noty držené z klávesnice by po posunu visely - pustit
        for midi in list(self.manual_notes):
            self.synth.note_off(midi)
        self.manual_notes.clear()

    # ---- kreslení ----------------------------------------------------------

    def draw(self, rgb):
        s = self.screen
        s.fill(ui.BG)
        title = self.font_big.render("Water Spectral Synth", True, ui.ACCENT)
        s.blit(title, (16, 14))
        sub = self.font_small.render(
            "voda hraje pady - vlna spustí notu, síla vlny určí délku · "
            "klaviatura přelaďuje tóninu · Z/X oktáva · M voda hraje zap/vyp · "
            "1 simulace / 2 kamera / 3 video",
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
            # obyčejné scale: smoothscale na HD snímku bral CPU audio vláknu
            s.blit(pygame.transform.scale(surf, self.preview_rect.size),
                   self.preview_rect)
        pygame.draw.rect(s, ui.BORDER, self.preview_rect, 1, border_radius=6)
        if self.source_error:
            msg = self.font_small.render(self.source_error[:52], True, ui.ACCENT_R)
            s.blit(msg, (self.preview_rect.x + 8, self.preview_rect.y + 8))

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

        # tlačítka zdroje
        for btn, kind, idx in self.src_buttons:
            active = (kind == "sim" and isinstance(self.source, SimSource)
                      or kind == "camera" and self.camera_index == idx)
            btn.draw(s, self.font_small, active)

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
        if self.last_frame is not None:
            lines.append(f"pohyb {self.last_frame.motion:.2f}  "
                         f"vlna {self.last_frame.onset:.2f}")
        water = sorted(self.conductor.held_notes())
        lines.append("Tónina: " + note_name(self.conductor.p.root)
                     + "  ·  voda hraje: "
                     + (" ".join(note_name(m) for m in water) if water else "—"))
        lines.append(f"{self.clock.get_fps():.0f} fps"
                     + (f" · audio {self.audio.render_ms:.0f}/"
                        f"{1000 * self.audio.BLOCK / self.synth.sr:.0f} ms"
                        f" · podtečení {self.audio.underruns}"
                        if self.audio else ""))
        for line in lines:
            s.blit(self.font_small.render(str(line), True, ui.DIM), (700, y))
            y += 20
        update_line = self.updater.status_line()
        if update_line:
            s.blit(self.font_small.render(update_line, True, ui.ACCENT), (700, y))

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
    p.add_argument("--camera", type=int, default=None,
                   help="index kamery (iPhone po kabelu bývá 0 nebo 1); "
                        "s tímto přepínačem se startuje rovnou z kamery")
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
    elif args.camera is not None:
        app.switch_source("camera")
    else:
        app.try_auto_camera()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
