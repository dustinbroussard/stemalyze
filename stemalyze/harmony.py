import numpy as np
from typing import Tuple, List

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def chord_templates(extend_minor: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Return (templates[24,12], names[24]) for major/minor triads."""
    # 12 majors, 12 minors; each template is 12-dim chroma weight vector
    tmpls = []
    names = []
    for root in range(12):
        # major: root, major third, fifth
        v = np.zeros(12, dtype=float)
        v[root] = 1.0
        v[(root+4)%12] = 0.8
        v[(root+7)%12] = 0.9
        tmpls.append(v)
        names.append(f"{NOTE_NAMES[root]}")
    for root in range(12):
        # minor: root, minor third, fifth
        v = np.zeros(12, dtype=float)
        v[root] = 1.0
        v[(root+3)%12] = 0.85
        v[(root+7)%12] = 0.9
        tmpls.append(v)
        names.append(f"{NOTE_NAMES[root]}m")
    T = np.stack(tmpls, axis=0)
    # L2-normalize templates
    T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    return T, names

def best_chord_from_chroma(chroma_vec: np.ndarray) -> Tuple[str, float]:
    """Given a 12-d chroma vector, return (name, score)."""
    T, names = chord_templates()
    v = chroma_vec / max(np.linalg.norm(chroma_vec), 1e-12)
    scores = T @ v
    idx = int(np.argmax(scores))
    return names[idx], float(scores[idx])

def label_blue_third(root_name: str, chroma_vec: np.ndarray) -> str:
    """Blue-third labeling generalized to all roots.
    If both the major third (+4) and minor third (+3) relative to the root
    have comparable energy, annotate as "<Root>(blues3)" instead of forcing M/m.

    Examples: G or Gm -> G(blues3) when B and Bb are both strong.
    """
    if not root_name:
        return root_name
    NOTE_TO_IDX = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
    # Strip minor suffix for root indexing
    base = root_name[:-1] if root_name.endswith('m') else root_name
    if base not in NOTE_TO_IDX:
        return root_name
    r = NOTE_TO_IDX[base]
    maj_third_idx = (r + 4) % 12
    min_third_idx = (r + 3) % 12
    e_maj = float(chroma_vec[maj_third_idx])
    e_min = float(chroma_vec[min_third_idx])
    mx = max(e_maj, e_min, 1e-9)
    # Consider "both present" if the weaker is at least 25% of the stronger
    if min(e_maj, e_min) > 0.25 * mx:
        return f"{base}(blues3)"
    return root_name

def smooth_chord_sequence(names: List[str], scores: List[float], stay: float = 0.85) -> List[str]:
    """Lightweight stay-biased smoothing over chord labels.
    If a change doesn't clearly increase confidence, prefer staying.
    CPU-friendly heuristic without full Viterbi lattice.
    """
    if not names:
        return names
    smoothed: List[str] = [names[0]]
    prev_name = names[0]
    prev_score = float(scores[0]) if scores else 0.0
    # Threshold grows with stay
    base = 0.06  # minimum improvement to justify change
    extra = 0.20 * (stay - 0.5)  # stronger stay increases required improvement
    thresh = base + max(0.0, extra)
    for i in range(1, len(names)):
        cur = names[i]
        sc = float(scores[i]) if i < len(scores) else 0.0
        if cur != prev_name and (sc - prev_score) < thresh:
            # keep previous
            smoothed.append(prev_name)
            # do not update prev_score so we require sustained improvement
        else:
            smoothed.append(cur)
            prev_name = cur
            prev_score = sc
    return smoothed
