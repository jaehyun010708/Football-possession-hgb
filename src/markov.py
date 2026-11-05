"""External-IDSSE-only Markov transitions and discriminative Viterbi decoding.

No Barcelona ATD dependencies. Default strength remains 0.25.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import TRAIN_MATCHES, FEATURE_DIR
LABEL_NAMES = np.array(['H','A','D'])
def segment_ids(frame_ids: np.ndarray, extra_breaks=None) -> np.ndarray:
    """Split on frame-ID resets or large gaps; keep chronological row order."""
    ids = np.asarray(frame_ids, dtype=np.float64)
    if len(ids) == 0:
        return np.zeros(0, dtype=np.int32)
    differences = np.diff(ids)
    positive = differences[(differences > 0) & np.isfinite(differences)]
    typical = np.median(positive) if len(positive) else 5.0
    breaks = np.r_[True, (differences <= 0) | (differences > 3 * typical) | ~np.isfinite(differences)]
    if extra_breaks is not None:
        extra = np.asarray(extra_breaks)
        if len(extra) != len(ids):
            raise ValueError("Wrong length for extra_breaks")
        breaks[1:] |= extra[1:] != extra[:-1]
    return np.cumsum(breaks).astype(np.int32)


def load_transition_counts() -> np.ndarray:
    counts = np.ones((3, 3), dtype=np.float64)  # additive smoothing, unseen transition allowed
    for mid in TRAIN_MATCHES:
        f = FEATURE_DIR / f"{mid}_temporal_v1_5fps.parquet"
        if not f.is_file():
            raise FileNotFoundError(f"Missing IDSSE feature cache: {f}\nRun train_idsse_temporal.py first.")
        data = pd.read_parquet(f, columns=["frame_id", "label"])
        label = data["label"].to_numpy(dtype=np.int32)
        if not np.isin(label, [0, 1, 2]).all():
            raise ValueError(f"Unexpected labels in {f}")
        seg = segment_ids(data["frame_id"].to_numpy())
        is_neighbor = seg[1:] == seg[:-1]
        previous = label[:-1][is_neighbor]
        following = label[1:][is_neighbor]
        np.add.at(counts, (previous, following), 1)
        print(f"[TRAIN] {mid}: {len(label):,} frames; {int(is_neighbor.sum()):,} transitions", flush=True)
    return counts


def get_probabilities(data: pd.DataFrame) -> np.ndarray:
    required = ["p_H", "p_A", "p_D"]
    if not set(required).issubset(data.columns):
        raise KeyError(f"Required columns: {required}")
    p = data[required].to_numpy(dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid class probabilities")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("Class probabilities do not sum to one")
    return np.clip(p, 1e-8, 1.0)


def viterbi_decode(probs: np.ndarray, segments: np.ndarray, transition: np.ndarray,
                   strength: float) -> np.ndarray:
    """Discriminative Viterbi using log P(class|features) + strength*log P(next|prev)."""
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("--strength must be nonnegative")
    n = len(probs)
    if len(segments) != n:
        raise ValueError("Probabilities and segments length mismatch")
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    if strength == 0:
        return np.argmax(probs, axis=1).astype(np.int8)
    logits = np.log(probs)
    log_trans = strength * np.log(transition)
    answer = np.empty(n, dtype=np.int8)
    boundaries = np.r_[0, np.flatnonzero(segments[1:] != segments[:-1]) + 1, n]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        m = end - begin
        back = np.zeros((m, 3), dtype=np.int8)
        scores = logits[begin].copy()
        for t in range(1, m):
            options = scores[:, None] + log_trans
            back[t] = np.argmax(options, axis=0)
            scores = logits[begin + t] + np.max(options, axis=0)
        state = int(np.argmax(scores))
        for t in range(m - 1, -1, -1):
            answer[begin + t] = state
            state = int(back[t, state]) if t > 0 else state
    return answer


