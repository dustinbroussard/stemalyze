import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict

STEM_NAMES = ["vocals", "drums", "bass", "other"]

def _have_exe(name: str) -> bool:
    from shutil import which
    return which(name) is not None

def separate(input_audio: str, outdir: str, backend: str = "auto", overwrite: bool = False) -> Dict[str, str]:
    """Run source separation producing vocals/drums/bass/other stems.
    Returns dict mapping stem name -> path.
    Requires demucs or spleeter installed (CLI or python)."""
    outdir = Path(outdir)
    stems_dir = outdir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    # If stems already exist and overwrite=False, reuse
    existing = {s: str(stems_dir / f"{s}.wav") for s in STEM_NAMES if (stems_dir / f"{s}.wav").exists()}
    if len(existing) == 4 and not overwrite:
        return existing

    if backend == "none":
        # Expect user-provided stems already placed
        missing = [s for s in STEM_NAMES if not (stems_dir / f"{s}.wav").exists()]
        if missing:
            raise RuntimeError(f"Backend 'none' but stems missing: {missing}. Place WAVs named vocals/drums/bass/other in {stems_dir}.")
        return {s: str(stems_dir / f"{s}.wav") for s in STEM_NAMES}

    if backend == "auto":
        if _have_exe("demucs"):
            backend = "demucs"
        elif _have_exe("spleeter"):
            backend = "spleeter"
        else:
            # try python imports
            try:
                import demucs  # noqa
                backend = "demucs"
            except Exception:
                try:
                    import spleeter  # noqa
                    backend = "spleeter"
                except Exception:
                    raise RuntimeError("No separation backend found. Install either 'demucs' or 'spleeter'.")
    
    inp = Path(input_audio)
    if backend == "demucs":
        # demucs CLI outputs in a nested folder; normalize to stems_dir
        cmd = ["demucs", "-n", "htdemucs", "-o", str(outdir), str(inp)]
        subprocess.run(cmd, check=True)
        # Find demucs output
        cand_parent = outdir / "htdemucs" / inp.stem
        # demucs sometimes nests as out/model/song/*
        if not cand_parent.exists():
            # try to locate any folder containing expected stems
            for root, dirs, files in os.walk(outdir):
                have = all(any(f.startswith(s) and f.endswith(".wav") for f in files) for s in STEM_NAMES)
                if have:
                    cand_parent = Path(root)
                    break
        if not cand_parent.exists():
            raise RuntimeError("Demucs finished but output not found.")
        # Copy/rename to standardized names
        mapping = {}
        for s in STEM_NAMES:
            # demucs names like vocals.wav or songname/vocals.wav
            src = None
            # scan files to find the one ending with f"{s}.wav"
            for f in cand_parent.glob("*.wav"):
                if f.name.lower().startswith(s) or f.name.lower().endswith(f"{s}.wav"):
                    src = f
                    break
            if src is None:
                raise RuntimeError(f"Demucs output missing stem: {s}")
            dst = stems_dir / f"{s}.wav"
            shutil.move(str(src), str(dst))
            mapping[s] = str(dst)
        return mapping

    elif backend == "spleeter":
        # spleeter CLI: 4stems model
        work = outdir / "spleeter_out"
        work.mkdir(exist_ok=True, parents=True)
        cmd = ["spleeter", "separate", "-o", str(work), "-p", "spleeter:4stems", str(inp)]
        subprocess.run(cmd, check=True)
        cand_parent = work / inp.stem
        if not cand_parent.exists():
            raise RuntimeError("Spleeter finished but output not found.")
        mapping = {}
        # Spleeter names match expected
        for s in STEM_NAMES:
            src = cand_parent / f"{s}.wav"
            if not src.exists():
                raise RuntimeError(f"Spleeter output missing stem: {s}")
            dst = stems_dir / f"{s}.wav"
            shutil.move(str(src), str(dst))
            mapping[s] = str(dst)
        # cleanup
        shutil.rmtree(cand_parent, ignore_errors=True)
        return mapping
    
    else:
        raise ValueError(f"Unknown backend: {backend}")
