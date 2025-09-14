import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
import soundfile as sf
import librosa
import librosa.display  # noqa: F401

from .harmony import best_chord_from_chroma, label_blue_third, smooth_chord_sequence

CREPE_CONF_THRESH = 0.5  # drop low-confidence pitch frames

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
        try:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq")
        except Exception:
            # Fallback for environments without soxr backend
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type="kaiser_best")
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

def bars_from_beats(drums: DrumTiming, audio_duration: float) -> List[Tuple[int, float, float]]:
    beats = np.array(drums.beat_times, dtype=float)
    if len(beats) < 2:
        # fallback: one bar for whole song
        return [(1, 0.0, float(audio_duration))]
    tsig = int(max(1, drums.time_sig_beats))
    off = int(max(0, drums.downbeat_offset))
    bar_bounds: List[Tuple[int, float, float]] = []
    # Determine downbeats as beats at indices [off, off+tsig, off+2*tsig, ...]
    down_idx = np.arange(off, len(beats), tsig, dtype=int)
    if down_idx.size == 0:
        # If offset is beyond available beats, assume downbeat at first beat
        down_idx = np.arange(0, len(beats), tsig, dtype=int)
    for i, di in enumerate(down_idx):
        if di >= len(beats):
            continue
        start = float(beats[di])
        if i + 1 < len(down_idx) and down_idx[i + 1] < len(beats):
            end = float(beats[down_idx[i + 1]])
        else:
            end = float(audio_duration)
        # Sanity clamp and skip degenerate bars
        end = max(start, min(end, float(audio_duration)))
        if end > start:
            bar_bounds.append((i + 1, start, end))
    # If still empty (extreme edge case), create a single bar
    if not bar_bounds:
        bar_bounds = [(1, 0.0, float(audio_duration))]
    return bar_bounds

def analyze_harmony_other(other_path: str, bars: List[Tuple[int,float,float]], sr: int=22050,
                          bass_path: Optional[str] = None, bass_boost: float = 0.6) -> List[Dict]:
    y = load_mono(other_path, sr)
    # Use CQT chroma for tuning robustness
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=12)
    if bass_path:
        try:
            yb = load_mono(bass_path, sr)
            # Gentle low emphasis via negative preemphasis (acts as de-emphasis of highs)
            try:
                yb = librosa.effects.preemphasis(yb, coef=-0.85)
            except Exception:
                # If preemphasis unavailable, fallback to identity
                pass
            bch = librosa.feature.chroma_cqt(y=yb, sr=sr, hop_length=hop, n_chroma=12)
            # Combine with clipping to avoid negative or exploding values
            chroma = np.clip(chroma + bass_boost * bch, 0.0, None)
        except Exception:
            # If bass processing fails, continue with other-only chroma
            pass
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    out = []
    names = []
    scores = []
    perbar_v = []
    for (bar_idx, t0, t1) in bars:
        # average chroma over bar
        mask = (times >= t0) & (times < t1)
        if not np.any(mask):
            # pick nearest frame
            fi0 = np.argmin(np.abs(times - 0.5*(t0+t1)))
            v = chroma[:, fi0]
        else:
            v = np.mean(chroma[:, mask], axis=1)
        name, score = best_chord_from_chroma(v)
        name = label_blue_third(name, v)
        perbar_v.append(v)
        names.append(name)
        scores.append(float(score))
        out.append({
            "bar_index": int(bar_idx),
            "start": float(t0),
            "end": float(t1),
            "chord": name,
            "score": float(score),
        })
    # Stay-biased smoothing
    smoothed = smooth_chord_sequence(names, scores, stay=0.85)
    for i in range(len(out)):
        out[i]["chord"] = smoothed[i]
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

def analyze_drum_onsets(drums_path: str, bars: List[Tuple[int, float, float]], sr: int = 22050,
                        hop_length: int = 512) -> List[Dict]:
    """Detect drum onsets and classify as kick/snare via spectral centroid.
    Returns per-bar dict with a 16-step grid string and onset times.
    CPU-friendly heuristic: low centroid → kick; high centroid → snare.
    """
    y = load_mono(drums_path, sr)
    # Global onset frames
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    # Spectral centroid per frame for classification
    S = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=hop_length))
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)
    cent = cent.flatten() if cent.ndim > 1 else cent
    # Threshold: empirical; kick typically < ~1200 Hz centroid
    kick_thresh_hz = 1200.0
    patterns: List[Dict] = []
    for (bar_idx, t0, t1) in bars:
        dur = max(1e-6, t1 - t0)
        # 16-step grid
        grid = ["."] * 16
        kicks: List[float] = []
        snares: List[float] = []
        for t in onset_times:
            if t < t0 or t >= t1:
                continue
            fr = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
            fr = int(min(max(fr, 0), len(cent) - 1)) if len(cent) else 0
            c_hz = float(cent[fr]) if len(cent) else 0.0
            # Map to 16-step index
            step = int(np.floor((t - t0) / dur * 16.0))
            step = int(np.clip(step, 0, 15))
            if c_hz <= kick_thresh_hz:
                kicks.append(float(t))
                grid[step] = 'X' if grid[step] == 'S' else 'K'
            else:
                snares.append(float(t))
                grid[step] = 'X' if grid[step] == 'K' else 'S'
        patterns.append({
            'bar_index': int(bar_idx),
            'start': float(t0),
            'end': float(t1),
            'grid16': ''.join(grid),
            'kick_times': kicks,
            'snare_times': snares,
            'kick_count': int(len(kicks)),
            'snare_count': int(len(snares)),
        })
    return patterns

def write_report_txt(path: str, meta: Dict, bars: List[Tuple[int,float,float]],
                     chords: List[Dict], bar_melody: Dict[int, List[Dict]],
                     drum_patterns: Optional[List[Dict]] = None):
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
    drum_map = {d['bar_index']: d for d in (drum_patterns or [])}
    for (bar_idx, t0, t1) in bars:
        c = chord_map.get(bar_idx, None)
        chord_str = c['chord'] if c else "N/A"
        chord_conf = f"{c['score']:.2f}" if c else "NA"
        lines.append(f"Bar {bar_idx:02d}  [{t0:08.3f} – {t1:08.3f}]  | Chord: {chord_str} (conf {chord_conf})")
        # Melody events
        mel_events = bar_melody.get(bar_idx, [])
        if mel_events:
            # Bar-level melody confidence summary (duration-weighted)
            dur = max(1e-6, t1 - t0)
            tot = sum(e['dur'] for e in mel_events)
            if tot > 0:
                wavg = sum(e['conf'] * e['dur'] for e in mel_events) / tot
            else:
                wavg = 0.0
            cover = min(1.0, tot / dur)
            lines.append(f"  Melody summary: avg conf {wavg:.2f}, coverage {cover:.2f}")
            for ev in mel_events:
                lines.append(f"  Melody: {ev['note']} x{ev['dur']:.2f}s (start {ev['start']:.3f}) conf {ev['conf']:.2f}")
        # Drum pattern grid per bar (kick/snare)
        dpat = drum_map.get(bar_idx)
        if dpat:
            lines.append(f"  Drums 16-step: {dpat['grid16']}  (K:{dpat['kick_count']} S:{dpat['snare_count']})")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
