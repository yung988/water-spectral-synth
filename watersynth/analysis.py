"""Spektrální analýza obrazu: snímek -> 2D FFT -> amplitudy harmonických.

Mapování:
  - radiální vzdálenost ve spektru (prostorová frekvence) -> index harmonické;
    prostorová frekvence k se stává k-tou harmonickou základního tónu
  - orientace energie -> stereo: vodorovné struktury (===) mají energii na
    svislé ose spektra -> levý kanál; svislé struktury (|||) -> pravý kanál;
    diagonály -> oba kanály + míra "spread" pro rozšíření sterea
  - volitelně se analyzuje rozdíl snímků obraz(t) - obraz(t-1), takže nástroj
    reaguje na změnu hladiny, ne na její statický tvar
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameFeatures:
    amps_l: np.ndarray          # amplitudy harmonických, levý kanál, 0..1
    amps_r: np.ndarray          # pravý kanál
    energy: float               # celková energie 0..1 (po normalizaci)
    centroid: float             # těžiště spektra v indexech harmonických (1..H)
    diag_ratio: float           # podíl energie na diagonálách 0..1
    spectrum_img: np.ndarray = field(default=None)  # log-magnituda NxN, fftshift, 0..1


class SpectralAnalyzer:
    def __init__(self, size: int = 128, harmonics: int = 96):
        self.n = n = size
        self.h = h = harmonics

        # 2D Hannovo okno proti únikům na okrajích snímku
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
        self.window = np.outer(w, w).astype(np.float32)

        # centrované souřadnice frekvenčních binů
        ky = np.fft.fftfreq(n, d=1.0 / n)[:, None]   # svislá osa spektra
        kx = np.fft.fftfreq(n, d=1.0 / n)[None, :]
        r = np.hypot(kx, ky)
        half = n / 2

        # přiřazení binu k harmonické 1..H podle radiální vzdálenosti
        hidx = np.rint(r / half * h).astype(np.int64)
        self.valid = (r >= 0.5) & (r <= half) & (hidx >= 1) & (hidx <= h)
        self.hidx = hidx[self.valid] - 1  # 0-based

        # úhel 0 = energie na ose kx (svislé pruhy), pi/2 = osa ky (vodorovné)
        ang = np.arctan2(np.abs(ky), np.abs(kx))[self.valid]
        self.w_l = (np.sin(ang) ** 2).astype(np.float32)   # vodorovné struktury -> L
        self.w_r = (1.0 - self.w_l).astype(np.float32)     # svislé -> R
        self.w_diag = (np.sin(2 * ang) ** 2).astype(np.float32)  # 1 na diagonále

        # počet binů na harmonickou (vyšší poloměr jich má víc) pro normalizaci
        counts = np.bincount(self.hidx, minlength=h).astype(np.float32)
        self.bin_count = np.maximum(counts, 1.0)

        self.prev = None
        self.agc = 0.05  # pomalé automatické vyrovnání úrovně

    def reset(self):
        self.prev = None

    def analyze(self, gray: np.ndarray, diff_mix: float = 0.0, gamma: float = 1.4,
                want_spectrum_img: bool = True) -> FrameFeatures:
        """gray: NxN float32 v rozsahu 0..1"""
        n, h = self.n, self.h
        gray = gray.astype(np.float32, copy=False)

        # mix statického obrazu (bez střední hodnoty) a rozdílu snímků;
        # rozdíl zesilujeme, protože mezisnímkové změny jsou malé
        static = gray - float(gray.mean())
        if self.prev is not None:
            diff = (gray - self.prev) * 4.0
        else:
            diff = np.zeros_like(gray)
        self.prev = gray.copy()
        work = (1.0 - diff_mix) * static + diff_mix * diff

        spec = np.fft.fft2(work * self.window)
        mag = np.abs(spec).astype(np.float32)

        flat = mag.reshape(-1)[self.valid.reshape(-1)]
        amps_l = np.bincount(self.hidx, weights=flat * self.w_l, minlength=h)
        amps_r = np.bincount(self.hidx, weights=flat * self.w_r, minlength=h)
        amps_l = (amps_l / self.bin_count).astype(np.float32)
        amps_r = (amps_r / self.bin_count).astype(np.float32)

        total = float(flat.sum())
        diag_ratio = float((flat * self.w_diag).sum() / total) if total > 1e-9 else 0.0

        # AGC: špička roste okamžitě, klesá zvolna - zachová dynamiku,
        # ale zabrání přebuzení při prudkých změnách jasu
        peak = float(max(amps_l.max(), amps_r.max()))
        self.agc = max(peak, self.agc * 0.995, 0.005)
        amps_l = np.clip(amps_l / self.agc, 0.0, 1.0) ** gamma
        amps_r = np.clip(amps_r / self.agc, 0.0, 1.0) ** gamma

        s = amps_l + amps_r
        den = float(s.sum())
        centroid = float((np.arange(1, h + 1) * s).sum() / den) if den > 1e-6 else 0.0
        energy = den / (2 * h)

        spectrum_img = None
        if want_spectrum_img:
            logmag = np.log1p(mag)
            m = float(logmag.max())
            if m > 1e-9:
                logmag /= m
            spectrum_img = np.fft.fftshift(logmag)

        return FrameFeatures(
            amps_l=amps_l, amps_r=amps_r, energy=energy,
            centroid=centroid, diag_ratio=diag_ratio, spectrum_img=spectrum_img,
        )
