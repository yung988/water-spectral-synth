"""Jednoduché pygame widgety (slider, přepínač, klaviatura) a kreslení
vizualizací. Žádný GUI framework - jen obdélníky a text."""

import numpy as np
import pygame

BG = (6, 11, 22)
PANEL = (11, 18, 32)
BORDER = (28, 42, 68)
TEXT = (215, 228, 245)
DIM = (125, 144, 171)
ACCENT = (87, 212, 255)
ACCENT_R = (255, 170, 80)


class Slider:
    HEIGHT = 34

    def __init__(self, label, vmin, vmax, value, fmt="{:.2f}"):
        self.label = label
        self.vmin, self.vmax = vmin, vmax
        self.value = value
        self.fmt = fmt
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.dragging = False

    def set_rect(self, x, y, w):
        self.rect = pygame.Rect(x, y, w, self.HEIGHT)

    @property
    def _track(self):
        r = self.rect
        return pygame.Rect(r.x, r.y + 20, r.w, 8)

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self._track.inflate(0, 10).collidepoint(event.pos):
            self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if self.dragging and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            t = (event.pos[0] - self._track.x) / max(1, self._track.w)
            self.value = self.vmin + min(1.0, max(0.0, t)) * (self.vmax - self.vmin)
            return True
        return False

    def draw(self, screen, font):
        r = self.rect
        screen.blit(font.render(self.label, True, DIM), (r.x, r.y))
        val = font.render(self.fmt.format(self.value), True, TEXT)
        screen.blit(val, (r.right - val.get_width(), r.y))
        tr = self._track
        pygame.draw.rect(screen, BORDER, tr, border_radius=4)
        t = (self.value - self.vmin) / (self.vmax - self.vmin)
        fill = pygame.Rect(tr.x, tr.y, int(tr.w * t), tr.h)
        pygame.draw.rect(screen, ACCENT, fill, border_radius=4)
        pygame.draw.circle(screen, TEXT, (tr.x + int(tr.w * t), tr.centery), 7)


class Toggle:
    HEIGHT = 24

    def __init__(self, label, value=False, key_hint=""):
        self.label = label
        self.value = value
        self.key_hint = key_hint
        self.rect = pygame.Rect(0, 0, 0, 0)

    def set_rect(self, x, y, w):
        self.rect = pygame.Rect(x, y, w, self.HEIGHT)

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.value = not self.value
            return True
        return False

    def draw(self, screen, font):
        r = self.rect
        box = pygame.Rect(r.x, r.y + 3, 16, 16)
        pygame.draw.rect(screen, BORDER, box, border_radius=4)
        if self.value:
            pygame.draw.rect(screen, ACCENT, box.inflate(-6, -6), border_radius=2)
        text = self.label + (f"   [{self.key_hint}]" if self.key_hint else "")
        screen.blit(font.render(text, True, TEXT), (r.x + 24, r.y + 3))


class Button:
    HEIGHT = 28

    def __init__(self, label):
        self.label = label
        self.rect = pygame.Rect(0, 0, 0, 0)

    def set_rect(self, x, y, w):
        self.rect = pygame.Rect(x, y, w, self.HEIGHT)

    def handle(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))

    def draw(self, screen, font, active=False):
        pygame.draw.rect(screen, (26, 60, 84) if active else PANEL,
                         self.rect, border_radius=6)
        pygame.draw.rect(screen, ACCENT if active else BORDER,
                         self.rect, 1, border_radius=6)
        text = font.render(self.label, True, TEXT if active else DIM)
        screen.blit(text, (self.rect.centerx - text.get_width() // 2,
                           self.rect.centery - text.get_height() // 2))


class Piano:
    """Dvouoktávová klaviatura: kreslení, klikání, zvýraznění noty."""

    WHITE_SEMIS = [0, 2, 4, 5, 7, 9, 11]
    BLACK_SEMIS = {0: 1, 1: 3, 3: 6, 4: 8, 5: 10}  # index bílé -> půltón za ní
    NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "H"]

    def __init__(self, octaves=2):
        self.octaves = octaves
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.base_midi = 48  # C3
        self.active = set()
        self._mouse_note = None

    def set_rect(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def note_at(self, pos):
        r = self.rect
        if not r.collidepoint(pos):
            return None
        n_white = 7 * self.octaves
        ww = r.w / n_white
        xi = (pos[0] - r.x) / ww
        wi = int(min(n_white - 1, xi))
        # nejdřív černé klávesy (jsou navrchu)
        if pos[1] < r.y + r.h * 0.62:
            for cand in (wi - 1, wi):
                oct_i, pos_i = divmod(cand, 7)
                if 0 <= cand < n_white and pos_i in self.BLACK_SEMIS:
                    bx = r.x + (cand + 1) * ww  # hrana mezi bílými
                    if abs(pos[0] - bx) < ww * 0.32:
                        return self.base_midi + 12 * oct_i + self.BLACK_SEMIS[pos_i]
        oct_i, pos_i = divmod(wi, 7)
        return self.base_midi + 12 * oct_i + self.WHITE_SEMIS[pos_i]

    def draw(self, screen, font):
        r = self.rect
        n_white = 7 * self.octaves
        ww = r.w / n_white
        for i in range(n_white):
            oct_i, pos_i = divmod(i, 7)
            midi = self.base_midi + 12 * oct_i + self.WHITE_SEMIS[pos_i]
            rect = pygame.Rect(int(r.x + i * ww), r.y, int(ww) - 1, r.h)
            color = ACCENT if midi in self.active else (232, 238, 247)
            pygame.draw.rect(screen, color, rect, border_radius=4)
            if pos_i == 0:
                label = font.render("C" + str(midi // 12 - 1), True, (86, 105, 138))
                screen.blit(label, (rect.x + 4, rect.bottom - 20))
        bh = int(r.h * 0.62)
        bw = int(ww * 0.62)
        for i in range(n_white):
            oct_i, pos_i = divmod(i, 7)
            if pos_i not in self.BLACK_SEMIS:
                continue
            midi = self.base_midi + 12 * oct_i + self.BLACK_SEMIS[pos_i]
            bx = int(r.x + (i + 1) * ww - bw / 2)
            rect = pygame.Rect(bx, r.y, bw, bh)
            color = (43, 140, 184) if midi in self.active else (22, 32, 47)
            pygame.draw.rect(screen, color, rect, border_radius=3)


def gray_surface(gray: np.ndarray, tint=(0.45, 0.75, 1.0)) -> pygame.Surface:
    """NxN float 0..1 -> pygame Surface s vodním nádechem."""
    v = np.clip(gray * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([(v * tint[0]).astype(np.uint8),
                    (v * tint[1]).astype(np.uint8),
                    (v * tint[2]).astype(np.uint8)], axis=-1)
    return pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))


def rgb_surface(rgb: np.ndarray) -> pygame.Surface:
    return pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))


def spectrum_surface(mag: np.ndarray) -> pygame.Surface:
    """Log-magnituda 0..1 (fftshift) -> barevná mapa tmavě modrá -> azurová -> bílá."""
    v = np.clip(mag, 0, 1)
    r = (255 * v * v).astype(np.uint8)
    g = (255 * v ** 1.4).astype(np.uint8)
    b = (60 + 195 * v).astype(np.uint8)
    return pygame.surfarray.make_surface(np.swapaxes(np.stack([r, g, b], -1), 0, 1))


def draw_bars(screen, rect, amps_l, amps_r):
    """Harmonické: levý kanál nahoru (azurová), pravý dolů (oranžová)."""
    pygame.draw.rect(screen, PANEL, rect, border_radius=8)
    mid = rect.centery
    n = len(amps_l)
    bw = rect.w / n
    half = rect.h / 2 - 4
    for i in range(n):
        x = int(rect.x + i * bw)
        w = max(1, int(bw) - 1)
        l = int(amps_l[i] * half)
        r = int(amps_r[i] * half)
        if l > 0:
            pygame.draw.rect(screen, ACCENT, (x, mid - l, w, l))
        if r > 0:
            pygame.draw.rect(screen, ACCENT_R, (x, mid, w, r))
    pygame.draw.line(screen, (255, 255, 255, 60), (rect.x, mid), (rect.right, mid))
