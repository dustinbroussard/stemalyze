import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
import soundfile as sf
import librosa
import librosa.display  # noqa
from tqdm import tqdm

from .harmony import best_chord_from_chroma

CREPE_CONF_THRESH = 0.5  # drop low-confidence pitch frames
PITCH_SMOOTH_HZ = 35.0   # median filter bandwidth for smoothing to reduce vibrato jitter

@dataclass
class MelodyNote:
    start: float
    end: float
    midi: float
    note: str
    mean_conf: float

@dataclass
class DrumTiming:
    tempo_bpm: float
    beat_times: List[float]
    onset_env: List[float]
    time_sig_beats: int  # typically 3 or 4
    downbeat_offset: int # which beat index within bar is downbeat (0..time_sig-1)

@dataclass
class HarmonyBar:
    bar_index: int
    start: float
    end: float
    chord: str
    score: float

def hz_to_midi_note_name(hz: float) -> Tuple[float, str]:
    if hz <= 0 or np.isnan(hz):
        return np.nan, "rest"
    midi = librosa.hz_to_midi(hz)
    name = librosa.midi_to_note(np.round(midi), octave=True)
    return float(midi), name

def load_mono(path: str, sr: int) -> np.ndarray:
    y, file_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq")
    return y

def analyze_vocals_melody(vocals_path: str, sr: int = 22050) -> List[MelodyNote]:
    """Extract pitch with CREPE (fallback to pyin), segment into notes."""
    y = load_mono(vocals_path, sr)
    hop_length = 256  # ~11.6ms at 22.05k
    frame_times = librosa.frames_to_time(np.arange(0, 1+len(y)//hop_length), sr=sr, hop_length=hop_length)

    f0_hz = None
    conf = None
    try:
        import crepe
        # CREPE expects 16k; resample
        y16 = librosa.resample(y, sr, 16000, res_type="soxr_hq")
        time, frequency, confidence, _ = crepe.predict(y16, 16000, step_size=1000*hop_length/16000, verbose=0)
        # Interpolate to match our frame grid
        f0_hz = np.interp(frame_times[:len(frequency)], time, frequency)
        conf = np.interp(frame_times[:len(confidence)], time, confidence)
    except Exception:
        # Fallback: librosa.pyin
        f0, vflag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                                    sr=sr, frame_length=2048, hop_length=hop_length)
        conf = np.where(np.isnan(f0), 0.0, vflag.astype(float))
        f0_hz = np.where(np.isnan(f0), 0.0, f0)

    f0_hz = np.nan_to_num(f0_hz, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(conf, nan=0.0)

    # Confidence-gated smoothing
    from scipy.ndimage import median_filter
    f0_med = median_filter(f0_hz, size=5)
    # Zero out low-confidence
    f0_med[conf < CREPE_CONF_THRESH] = 0.0

    # Segment into notes where midi is stable within tolerance
    notes: List[MelodyNote] = []
    tol_semitones = 0.5
    cur_start_idx = None
    cur_vals = []

    def flush_segment(sidx: int, eidx: int):
        if sidx is None or eidx <= sidx:
            return
        seg_f0 = f0_med[sidx:eidx]
        seg_conf = conf[sidx:eidx]
        if np.count_nonzero(seg_f0) < 3:
            return
        hz = np.median(seg_f0[seg_f0 > 0])
        midi, name = hz_to_midi_note_name(hz)
        if name == "rest" or np.isnan(midi):
            return
        start = frame_times[sidx]
        end = frame_times[min(eidx, len(frame_times)-1)]
        notes.append(MelodyNote(start=start, end=end, midi=midi, note=name, mean_conf=float(np.mean(seg_conf))))

    last_midi = None
    for i, hz in enumerate(f0_med[:len(frame_times)]):
        if hz <= 0:
            # rest
            if cur_start_idx is not None:
                flush_segment(cur_start_idx, i)
                cur_start_idx = None
                cur_vals = []
            continue
        midi = librosa.hz_to_midi(hz)
        if cur_start_idx is None:
            cur_start_idx = i
            cur_vals = [midi]
            last_midi = midi
        else:
            if abs(midi - last_midi) <= tol_semitones:
                cur_vals.append(midi)
                last_midi = (0.7*last_midi + 0.3*midi)
            else:
                flush_segment(cur_start_idx, i)
                cur_start_idx = i
                cur_vals = [midi]
                last_midi = midi
    flush_segment(cur_start_idx, len(frame_times)-1)
    return notes

def analyze_drums(drums_path: str, sr: int = 22050, force_time_sig_beats: Optional[int]=None) -> DrumTiming:
    y = load_mono(drums_path, sr)
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="time")
    beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=hop_length)

    # Estimate 3 vs 4 by accent pattern at downbeats
    if force_time_sig_beats in (3,4):
        tsig = int(force_time_sig_beats)
    else:
        tsig_scores = {}
        for sig in (3,4):
            best = -np.inf
            best_off = 0
            for off in range(sig):
                # group beats into measures of length sig with offset
                idxs = np.arange(off, len(beat_frames), sig)
                frames = beat_frames[idxs]
                frames = frames[frames < len(onset_env)]
                accents = onset_env[frames]
                score = float(np.mean(accents)) if len(accents) else -np.inf
                if score > best:
                    best = score
                    best_off = off
            tsig_scores[sig] = (best, best_off)
        # pick higher score
        if tsig_scores[3][0] > tsig_scores[4][0]:
            tsig = 3
            downbeat_offset = tsig_scores[3][1]
        else:
            tsig = 4
            downbeat_offset = tsig_scores[4][1]
    # If user forced signature, still compute a plausible downbeat offset
    if force_time_sig_beats in (3,4):
        sig = force_time_sig_beats
        best = -np.inf
        best_off = 0
        for off in range(sig):
            idxs = np.arange(off, len(beat_frames), sig)
            frames = beat_frames[idxs]
            frames = frames[frames < len(onset_env)]
            accents = onset_env[frames]
            score = float(np.mean(accents)) if len(accents) else -np.inf
            if score > best:
                best = score
                best_off = off
        downbeat_offset = best_off

    return DrumTiming(
        tempo_bpm=float(tempo),
        beat_times=list(map(float, beats)),
        onset_env=list(map(float, onset_env)),
        time_sig_beats=int(tsig),
        downbeat_offset=int(downbeat_offset)
    )

def bars_from_beats(drums: DrumTiming, audio_duration: float) -> List[Tuple[int,float,float]]:
    beats = np.array(drums.beat_times, dtype=float)
    if len(beats) < 2:
        # fallback: one bar for whole song
        return [(1, 0.0, audio_duration)]
    tsig = drums.time_sig_beats
    off = drums.downbeat_offset
    bar_bounds = []
    # Determine downbeats as beats at indices [off, off+tsig, off+2*tsig, ...]
    down_idx = np.arange(off, len(beats), tsig, dtype=int)
    for i, di in enumerate(down_idx):
        start = beats[di]
        end = beats[down_idx[i+1]] if i+1 < len(down_idx) else audio_duration
        bar_bounds.append((i+1, float(start), float(end)))
    return bar_bounds

def analyze_harmony_other(other_path: str, bars: List[Tuple[int,float,float]], sr: int=22050) -> List[Dict]:
    y = load_mono(other_path, sr)
    # Use CQT chroma for tuning robustness
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, n_chroma=12)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512)
    out = []
    for (bar_idx, t0, t1) in bars:
        # average chroma over bar
        mask = (times >= t0) & (times < t1)
        if not np.any(mask):
            # pick nearest frame
            fi0 = np.argmin(np.abs(times - 0.5*(t0+t1)))
            v = chroma[:, fi0]
        else:
            v = np.mean(chroma[:, mask], axis=1)
        chord, score = best_chord_from_chroma(v)
        out.append({
            "bar_index": int(bar_idx),
            "start": float(t0),
            "end": float(t1),
            "chord": chord,
            "score": float(score),
        })
    return out

def quantize_melody_to_bars(notes: List[MelodyNote], bars: List[Tuple[int,float,float]]) -> Dict[int, List[Dict]]:
    by_bar: Dict[int, List[Dict]] = {b[0]: [] for b in bars}
    for n in notes:
        # clip to bar bounds and assign
        for (bar_idx, t0, t1) in bars:
            overlap = max(0.0, min(n.end, t1) - max(n.start, t0))
            if overlap > 0.015:  # ignore <15ms
                by_bar[bar_idx].append({
                    "start": float(max(n.start, t0)),
                    "end": float(min(n.end, t1)),
                    "note": n.note,
                    "midi": float(n.midi),
                    "conf": float(n.mean_conf),
                    "dur": float(min(n.end, t1) - max(n.start, t0))
                })
    # optional: sort events inside bars
    for k in by_bar:
        by_bar[k].sort(key=lambda d: d["start"])
    return by_bar

def write_report_txt(path: str, meta: Dict, bars: List[Tuple[int,float,float]],
                     chords: List[Dict], bar_melody: Dict[int, List[Dict]]):
    lines = []
    lines.append("# STEMALYZE REPORT")
    lines.append(f"Source: {meta['source']}")
    lines.append(f"Backend: {meta['backend']}")
    lines.append(f"Sample Rate: {meta['sr']}")
    lines.append(f"Global Tempo (BPM): {meta['tempo_bpm']:.2f}")
    lines.append(f"Estimated Time Signature: {meta['time_sig_beats']}/4")
    lines.append(f"Bars: {len(bars)}")
    lines.append("")
    chord_map = {c['bar_index']: c for c in chords}
    for (bar_idx, t0, t1) in bars:
        c = chord_map.get(bar_idx, None)
        chord_str = c['chord'] if c else "N/A"
        lines.append(f"Bar {bar_idx:02d}  [{t0:08.3f} – {t1:08.3f}]  | Chord: {chord_str}")
        for ev in bar_melody.get(bar_idx, []):
            lines.append(f"  Melody: {ev['note']} x{ev['dur']:.2f}s (start {ev['start']:.3f}) conf {ev['conf']:.2f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
