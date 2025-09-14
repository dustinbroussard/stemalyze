import argparse
from pathlib import Path
import soundfile as sf

from .separate import separate
from .analysis import analyze_vocals_melody, analyze_drums, bars_from_beats, analyze_harmony_other, quantize_melody_to_bars, analyze_drum_onsets
from .report import save_all

def main():
    p = argparse.ArgumentParser(description="STEMALYZE — stems + musical analysis → txt report")
    p.add_argument("audio", help="Path to input audio file (wav/mp3/flac etc.)")
    p.add_argument("--backend", default="auto", choices=["auto","demucs","spleeter","none"], help="Separation backend (or 'none' if stems already prepared)")
    p.add_argument("--out", default="stemalyze_out", help="Output directory")
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--barsig", type=int, choices=[3,4], default=None, help="Force time signature beats per bar (3 or 4)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Validate input exists early for better UX
    audio_path = Path(args.audio)
    if not audio_path.exists():
        p.error(f"Audio file not found: {audio_path}")

    # 1) Separate stems
    stems = separate(args.audio, str(outdir), backend=args.backend, overwrite=args.overwrite)

    # 2) Drums → tempo/beats/time signature
    drums_info = analyze_drums(stems["drums"], sr=args.sample_rate, force_time_sig_beats=args.barsig)

    # compute audio duration from any stem
    info = sf.info(stems["drums"])  # duration is independent of sample rate
    duration = float(info.duration)

    # 3) Bars from beats
    bars = bars_from_beats(drums_info, audio_duration=duration)

    # 4) Vocals → melody notes; map to bars
    notes = analyze_vocals_melody(stems["vocals"], sr=args.sample_rate)
    bar_melody = quantize_melody_to_bars(notes, bars)

    # 5) Other → chords per bar
    chords = analyze_harmony_other(stems["other"], bars, sr=args.sample_rate, bass_path=stems.get("bass"))
    # 5b) Drums → per-bar kick/snare pattern (CPU-friendly)
    drum_patterns = analyze_drum_onsets(stems["drums"], bars, sr=args.sample_rate)

    # 6) Save report
    meta = {
        "source": str(Path(args.audio).resolve()),
        "backend": args.backend,
        "sr": args.sample_rate,
        "tempo_bpm": drums_info.tempo_bpm,
        "time_sig_beats": drums_info.time_sig_beats
    }
    report_path = save_all(str(outdir), meta, bars, chords, bar_melody, drum_patterns)

    if args.debug:
        print("STEMS:", stems)
        print("Tempo:", drums_info.tempo_bpm, "TimeSig:", f"{drums_info.time_sig_beats}/4")
        print("Bars:", len(bars))
    print(f"Report written to: {report_path}")

if __name__ == "__main__":
    main()
