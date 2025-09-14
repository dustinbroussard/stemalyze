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
