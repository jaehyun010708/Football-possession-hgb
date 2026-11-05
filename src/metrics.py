"""Competition scoring wrapper on the original 25 FPS frame grid.

IDSSE labels sampled at 5 FPS are repeated fivefold; effective timing remains 0.2s.
"""
from __future__ import annotations
import numpy as np
from .official_metric import load_official_metric
UPSAMPLE = 5
def sequence_starts(frame_ids: np.ndarray) -> np.ndarray:
    """Use the same gap/reset rule as apply_idsse_hmm.py.

    A new segment starts on a non-increasing frame_id, a nonfinite value,
    or a jump > 3 times the typical positive frame step.
    """
    frame_ids = np.asarray(frame_ids, dtype=np.float64)
    if len(frame_ids) == 0 or not np.isfinite(frame_ids).all():
        raise ValueError("frame_id is empty or contains a non-finite value")
    if len(frame_ids) == 1:
        return np.zeros(0, dtype=np.int64)
    delta = np.diff(frame_ids)
    positive = delta[(delta > 0) & np.isfinite(delta)]
    typical = float(np.median(positive)) if len(positive) else 5.0
    return np.flatnonzero((delta <= 0) | (delta > 3 * typical) | ~np.isfinite(delta)) + 1


def prepare_for_official_scoring(
    gt: np.ndarray, raw: np.ndarray, markov: np.ndarray,
    breaks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert ungraded X at discontinuities; expand all 5-FPS samples to 25 FPS.

    X prevents the official temporal scorer from merging adjacent possessions
    across half breaks / data gaps. Inserting X in GT and predictions masks out
    the artificial interval from all metrics.
    """
    pm = load_official_metric(allow_download=False)

    def up(arr: np.ndarray) -> np.ndarray:
        separated = np.insert(arr, breaks, pm.X).astype(np.int8)
        return np.repeat(separated, UPSAMPLE)

    return up(gt), up(raw), up(markov)



def official_score(gt, raw, markov, frames):
    """Evaluate raw and Viterbi states using the UNMODIFIED official evaluator."""
    pm = load_official_metric(allow_download=False)
    breaks = sequence_starts(frames)
    gt25, raw25, mk25 = prepare_for_official_scoring(gt, raw, markov, breaks)
    return {
        'raw': pm.evaluate(gt25, raw25, bootstrap_n=0),
        'markov': pm.evaluate(gt25, mk25, bootstrap_n=0),
    }
