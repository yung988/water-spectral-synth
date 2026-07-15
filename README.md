# Water Spectral Synth

Hudební nástroj pro Mac: **2D FFT vodní hladiny řídí aditivní syntézu.**

Obraz z kamery (třeba iPhone připojený kabelem) se neanalyzuje proto, aby
„otevíral filtr". Místo toho se jeho prostorové spektrum stává přímo spektrem
zvuku: každá prostorová frekvence obrazu je jedna harmonická tónu.

- **Klidná hladina** → energie u středu spektra → čistý sinus, klidný dron.
- **Kapka** → spektrum se otevře → přibydou harmonické, tón se rozjasní.
- **Dvě kapky** → interference → nové složky, zvuk bohatší jako u rezonátoru.
- **Orientace vln se zachovává**: vodorovné struktury (`===`) hrají v levém
  kanálu, svislé (`|||`) v pravém, diagonály rozšiřují stereo (jemné
  rozladění pravého kanálu).
- **Režim změny (diff)**: analyzuje se `obraz(t) − obraz(t−1)` — nástroj pak
  nereaguje na to, jak voda vypadá, ale jak se **mění**. Klidná hladina téměř
  nehraje, nová vlna okamžitě zazní.

Výšku tónu určuje klaviatura (na obrazovce, počítačová klávesnice nebo MIDI)
a voda určuje **barvu** každé noty. Volitelně může výšku řídit i obraz sám —
dominantní prostorová frekvence kvantovaná do pentatoniky.

## Instalace (macOS)

```bash
git clone https://github.com/yung988/water-spectral-synth.git
cd water-spectral-synth
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Spuštění

```bash
python3 -m watersynth                 # start se simulací vody
python3 -m watersynth --camera 1      # rovnou s kamerou (zkus indexy 0, 1, 2…)
python3 -m watersynth --video zaber_vody.mov
python3 -m watersynth --list-devices  # výpis audio výstupů
```

### iPhone jako kamera

Připoj iPhone kabelem (nebo nech Continuity Camera přes Wi-Fi) — macOS ho
nabídne jako běžnou kameru. V aplikaci přepni klávesou **2**; pokud se otevře
vestavěná FaceTime kamera, spusť s `--camera 1` (příp. vyšším indexem).
Při prvním spuštění dej Terminálu oprávnění ke kameře
(Nastavení systému → Soukromí a zabezpečení → Kamera).

Tip: namiř iPhone kolmo dolů na misku s vodou, boční světlo (lampa) udělá
z vlnek kontrastní proužky — a přesně ty FFT slyší.

## Ovládání

| Vstup | Akce |
|---|---|
| `A W S E D F T G Y H U J K O L P ;` | noty (jako klaviatura, A = C) |
| `Z` / `X` | oktáva dolů / nahoru |
| mezerník | drone — drží tón bez klávesy |
| `M` | výška tónu z obrazu (pentatonika) |
| `1` / `2` / `3` | zdroj: simulace / kamera / video |
| klik do náhledu | kapka do simulace |
| klik na klaviaturu | nota myší |
| MIDI klaviatura | funguje automaticky (mido + python-rtmidi) |
| `Esc` | konec |

Slidery: mix statický obraz ↔ změna (diff), kontrast spektra (gamma),
prostor z diagonál (spread), náběh/dozvuk harmonických, hlasitost,
vlnění simulace.

## Jak to uvnitř funguje

```
kamera / video / simulace
        │  střední čtvercový výřez, 128×128 šedotón
        ▼
(1−mix)·obraz + mix·(obraz(t) − obraz(t−1))     … slider „Změna"
        │  2D Hannovo okno
        ▼
     2D FFT  ──────────────►  log-magnituda (vizualizace)
        │
        │  radiálně: |k| → index harmonické 1…96
        │  úhlově:  orientace → váhy L / R + diagonální podíl
        ▼
 amplitudy harmonických L, R  (AGC, gamma)
        │  ~30× za sekundu, vyhlazení attack/release
        ▼
 banka 96 sinusových oscilátorů (aditivní syntéza, numpy)
        │  f0 z klaviatury/MIDI, harmonické jen do 45 % Nyquista
        ▼
 stereo výstup, měkký limiter (tanh) → CoreAudio
```

- `watersynth/analysis.py` — 2D FFT, radiální/úhlové mapování, AGC, diff režim
- `watersynth/synth.py` — aditivní banka oscilátorů (vektorizovaná, ~6× rychlejší než realtime)
- `watersynth/sim.py` — procedurální vodní hladina (kruhové vlnky + tři šikmé vlny)
- `watersynth/sources.py` — kamera (AVFoundation), video soubor, simulace
- `watersynth/app.py`, `ui.py` — pygame okno, slidery, klaviatura, vizualizace
- `watersynth/midiin.py` — MIDI vstup (volitelný)

## Testy

```bash
python3 tests/test_engine.py
```

Testují jádro bez kamery a zvukové karty: orientace pruhů → správný kanál,
diff režim (statická scéna mlčí, pohyb zní), kapka v simulaci otevře
spektrum, syntéza drží frekvenci noty, doznívá do ticha a nealiasuje.

## Nápady dál

- optický tok místo prostého rozdílu snímků (směr pohybu vody → panorama)
- polyfonie (více not najednou, každá s vlastní bankou)
- záznam výstupu do WAV, MIDI learn pro slidery
- nativní Swift/AVFoundation verze se stejným mapováním
