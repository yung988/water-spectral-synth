# -*- mode: python ; coding: utf-8 -*-
# Build: ../.venv-build/bin/pyinstaller --noconfirm watersynth.spec
# (build venv musí mít Python cílící na macOS 11+, jinak .app nepoběží
#  na starších systémech než je tenhle stroj)

a = Analysis(
    ["launch.py"],
    pathex=[".."],
    binaries=[],
    datas=[],
    hiddenimports=["mido.backends.rtmidi"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="WaterSynth",
    console=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="WaterSynth",
    upx=False,
)

app = BUNDLE(
    coll,
    name="WaterSynth.app",
    icon=None,
    bundle_identifier="cz.harfalibus.watersynth",
    info_plist={
        "NSCameraUsageDescription":
            "WaterSynth čte obraz vodní hladiny z kamery a mění ho na zvuk.",
        "NSCameraUseContinuityCameraDeviceType": True,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
    },
)
