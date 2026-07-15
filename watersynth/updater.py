"""Kontrola a instalace aktualizací z GitHub release "latest".

Aplikace při startu na pozadí zjistí, jestli na GitHubu není novější
build (porovnává krátké SHA commitu vložené při balení). Aktualizace
stáhne zip, vymění WaterSynth.app na místě a spustí novou verzi.
Funguje jen v zabalené aplikaci (PyInstaller); ve vývoji se nehlásí.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

from ._buildinfo import BUILD_SHA

RELEASE_API = ("https://api.github.com/repos/yung988/"
               "water-spectral-synth/releases/tags/latest")


class Updater:
    """Stavy: idle -> available -> downloading -> restarting (nebo error)."""

    def __init__(self):
        self.state = "idle"
        self.remote_sha = None
        self.asset_url = None
        self.error = ""
        self.frozen = bool(getattr(sys, "frozen", False))
        if self.frozen:
            self._cleanup_old()
        threading.Thread(target=self._check, daemon=True).start()

    def _app_path(self) -> Path:
        # Contents/MacOS/WaterSynth -> WaterSynth.app
        return Path(sys.executable).resolve().parents[2]

    def _cleanup_old(self):
        """Smaže zálohu předchozí verze z minulé aktualizace."""
        try:
            for old in self._app_path().parent.glob("*.app.old"):
                shutil.rmtree(old, ignore_errors=True)
        except Exception:
            pass

    def _check(self):
        if BUILD_SHA == "dev":
            return
        try:
            req = urllib.request.Request(
                RELEASE_API, headers={"Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=6) as r:
                rel = json.load(r)
            m = re.search(r"commitu ([0-9a-f]{7,40})", rel.get("body") or "")
            sha = m.group(1)[:7] if m else None
            for a in rel.get("assets", []):
                if a["name"].endswith(".zip"):
                    self.asset_url = a["browser_download_url"]
            if sha and self.asset_url and sha != BUILD_SHA:
                self.remote_sha = sha
                self.state = "available"
        except Exception:
            pass  # bez internetu apod. - prostě nenabízíme

    def install(self):
        if self.state != "available":
            return
        self.state = "downloading"
        threading.Thread(target=self._install, daemon=True).start()

    def _install(self):
        try:
            if not self.frozen:
                raise RuntimeError("aktualizace funguje jen v zabalené .app")
            app = self._app_path()
            tmp = Path(tempfile.mkdtemp(prefix="watersynth-update-"))
            zip_path = tmp / "app.zip"
            urllib.request.urlretrieve(self.asset_url, zip_path)
            subprocess.run(["ditto", "-xk", str(zip_path), str(tmp)], check=True)
            new_app = tmp / app.name
            if not new_app.exists():
                raise RuntimeError("v zipu chybí " + app.name)
            old = app.with_name(app.name + ".old")
            app.rename(old)
            shutil.move(str(new_app), str(app))
            subprocess.Popen(["open", str(app)])
            self.state = "restarting"   # aplikace se sama ukončí
        except Exception as e:
            self.state = "error"
            self.error = str(e)

    def status_line(self):
        if self.state == "available":
            return f"Nová verze ({self.remote_sha}) - stiskni I pro aktualizaci"
        if self.state == "downloading":
            return "Stahuji aktualizaci..."
        if self.state == "restarting":
            return "Aktualizováno - spouštím novou verzi"
        if self.state == "error":
            return "Aktualizace selhala: " + self.error[:70]
        return None
