"""Vstupní bod pro PyInstaller build WaterSynth.app."""

import sys

from watersynth.app import main

if __name__ == "__main__":
    sys.exit(main())
