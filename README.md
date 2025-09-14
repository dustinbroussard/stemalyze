# STEMALYZE — CPU‑friendly stem split + musical analysis

A modular Python CLI that:
1) Separates an input audio file into **vocals, drums, bass, other** (via Demucs or Spleeter),
2) Analyzes:  
   - **Vocals** → bar‑by‑bar **melody notes** (pitch → MIDI, note segments)  
   - **Drums** → **BPM** and **time signature (3/4 vs 4/4 guess)**  
   - **Other** → bar‑by‑bar **chord progression** (major/minor triads),
3) Emits a synced, human‑readable **.txt report** with per‑bar details.

> Designed for CPU‑only machines. Separation will be the slowest step; analysis runs fine on CPU.

---

## Quick Start

### 0) System deps
- **ffmpeg** installed and on PATH.
  - Linux (Pop!_OS/Ubuntu): `sudo apt update && sudo apt install ffmpeg`
  - Windows: install ffmpeg and add `ffmpeg/bin` to PATH.

### 1) Create & activate a venv
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2) Install Python deps
```bash
pip install -r requirements.txt
```
> First run of **crepe** will download a small model (~20MB).

### 3) Install a **stem separation backend** (choose one)
- **Demucs (recommended for quality; works on CPU)**  
  ```bash
  pip install demucs
  ```
  You can also use the demucs CLI if already installed.

- **Spleeter (fastest on some CPUs)**  
  ```bash
  pip install spleeter
  ```

### 4) Run the tool
```bash
python -m stemalyze.cli /path/to/song.wav --backend demucs --out outdir
```
Options:
```
--backend            demucs|spleeter|auto  (default: auto)
--sample-rate        target analysis sample rate (default: 22050)
--overwrite          overwrite existing stems/outputs
--barsig             force time signature like 3 or 4 (optional override)
--debug              extra logging
```

The output folder will contain:
- `stems/` with `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`
- `analysis/` with intermediate JSON
- `reports/<songname>_stemalyze_report.txt`

---

## Notes & Tradeoffs

- **Time signature** inference is *hard*. We provide a robust heuristic (3/4 vs 4/4) based on onset accent patterns in the **drums**. If it misclassifies, re‑run with `--barsig 3` or `--barsig 4`.
- **Chord detection** uses chroma templates → **major/minor** per bar. Extensions (7ths, sus, dim) can be added, but majors/minors are the most robust CPU‑friendly baseline.
- **Melody** extraction uses **CREPE** (fallback to `librosa.pyin` if unavailable). Output is quantized to notes by grouping stable pitch regions.
- Works on mono/stereo input; resamples to analysis SR.
- If you already have stems (e.g., from Demucs GUI), skip separation with `--backend none` and drop your files into `outdir/stems/` named exactly:
  `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`.

---

## Example report (snippet)

```
# STEMALYZE REPORT
Source: my_song.wav
Backend: demucs
Sample Rate: 22050
Global Tempo (BPM): 92.8
Estimated Time Signature: 4/4
Bars: 64

Bar 01  [00:00.000 – 00:02.584]  | Chord: Am
  Melody: A3 x0.42s, C4 x0.21s, E4 x0.35s
Bar 02  [00:02.584 – 00:05.168]  | Chord: Am
  Melody: G3 x0.30s, A3 x0.50s, (rest 0.10s)
...
```

---

## Project Layout

```
stemalyze/
  __init__.py
  cli.py          # CLI entrypoint
  separate.py     # Demucs/Spleeter wrapper (CLI or python).
  analysis.py     # vocals/drums/other analysis
  harmony.py      # chord mapping utilities
  report.py       # txt report assembler
requirements.txt
```

---

## Troubleshooting

- **No stems produced** → Ensure you installed `demucs` or `spleeter`. Try `--backend spleeter` if demucs fails.
- **Wrong time signature** → Re‑run with `--barsig 3` or `--barsig 4`.
- **Pitch looks jittery** → CREPE confidence threshold can be adjusted in `analysis.py` (`CREPE_CONF_THRESH`).

---

## License
MIT
# stemalyze
