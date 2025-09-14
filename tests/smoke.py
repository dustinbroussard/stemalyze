"""
Lightweight smoke tests for bars detection and chord smoothing.
Run with:  python -m tests.smoke
"""
from __future__ import annotations

from typing import List

from stemalyze.analysis import DrumTiming, bars_from_beats
from stemalyze.harmony import smooth_chord_sequence, label_blue_third


def test_bars_from_beats() -> None:
    # 4/4 at 120 BPM -> beat every 0.5s
    beats: List[float] = [i * 0.5 for i in range(16)]  # 8s span
    drums = DrumTiming(
        tempo_bpm=120.0,
        beat_times=beats,
        onset_env=[],
        time_sig_beats=4,
        downbeat_offset=0,
    )
    bars = bars_from_beats(drums, audio_duration=8.0)
    assert len(bars) == 4, f"expected 4 bars, got {len(bars)}"
    # First bar: [0.0, 2.0), second: [2.0, 4.0)
    assert abs(bars[0][1] - 0.0) < 1e-9 and abs(bars[0][2] - 2.0) < 1e-9
    assert abs(bars[1][1] - 2.0) < 1e-9 and abs(bars[1][2] - 4.0) < 1e-9


def test_chord_smoothing_stay_bias() -> None:
    names = ["C", "G", "G", "G", "Am"]
    scores = [0.60, 0.59, 0.60, 0.58, 0.80]
    sm = smooth_chord_sequence(names, scores, stay=0.85)
    assert sm == ["C", "C", "C", "C", "Am"], sm


def test_label_blue_third() -> None:
    # Root G, both Bb and B present
    v = [0.0] * 12
    G = 7
    Bb = (G + 3) % 12
    B = (G + 4) % 12
    v[G] = 1.0
    v[Bb] = 0.8
    v[B] = 0.75
    name = label_blue_third("G", v)  # base root name
    assert name == "G(blues3)", name


def main() -> None:
    test_bars_from_beats()
    test_chord_smoothing_stay_bias()
    test_label_blue_third()
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()

