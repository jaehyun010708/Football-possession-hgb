"""Same training/validation comparison as the original HGB+ball experiment."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from .config import TRAIN_MATCHES as TRAIN, VALID_MATCH as VALID, BALL_OUTPUT_DIR as OUT, PLAYER_MODEL as BASE_MODEL, MARKOV_STRENGTH, SAMPLE_FPS
from .ball_features import BALL_FEATURES, CACHE
from .markov import load_transition_counts, segment_ids, viterbi_decode
from .metrics import official_score
def match_data(mid, features):
    old = CACHE / f"{mid}_temporal_v1_5fps.parquet"
    ball = CACHE / f"{mid}_ball_geometry_v1_5fps.parquet"
    for path in (old, ball):
        if not path.is_file():
            raise FileNotFoundError(f"Missing cache {path}; run ball preparation first")
    x = pd.read_parquet(old, columns=["frame_id", "label", *features])
    z = pd.read_parquet(ball, columns=["frame_id", *BALL_FEATURES])
    if len(x) != len(z) or not x.frame_id.equals(z.frame_id):
        raise ValueError(f"Mismatched frame timelines for {mid}; refusing to train")
    return pd.concat([x.reset_index(drop=True), z[BALL_FEATURES].reset_index(drop=True)], axis=1)


def print_score(label, gt, raw, mk, metrics):
    for stage, pred in [("raw", raw), ("markov", mk)]:
        score = metrics[stage]
        print(
            f"{label:<22} {stage:<7} "
            f"Acc={accuracy_score(gt, pred):.4f} "
            f"MacroF1={f1_score(gt, pred, average='macro'):.4f} "
            f"SpellF1={score['spell_f1_mean']:.4f} "
            f"spells={score['n_pred_spells']} (GT={score['n_gt_spells']})"
        )


def compare_models(strength: float = MARKOV_STRENGTH) -> Path:
    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"Missing trained baseline HGB model: {BASE_MODEL}")
    payload = joblib.load(BASE_MODEL)
    old_model = payload["model"]
    original_features = list(payload["features"])
    if len(original_features) != 191:
        raise ValueError(f"Expected 191 old features, got {len(original_features)}")
    if list(old_model.classes_) != [0, 1, 2]:
        raise ValueError("Unexpected baseline class order")

    print("[LOAD] 6 IDSSE training caches + held-out J03WPY", flush=True)
    train = pd.concat([match_data(mid, original_features) for mid in TRAIN], ignore_index=True)
    valid = match_data(VALID, original_features)
    y_train = train["label"].to_numpy(dtype=np.int32)
    y_val = valid["label"].to_numpy(dtype=np.int32)
    if set(y_val) != {0, 1, 2} or set(y_train) != {0, 1, 2}:
        raise ValueError("Unexpected labels")
    print(f"Training={len(train):,} frames, validation={len(valid):,}; ball observed in valid={valid.ball_observed.mean():.1%}")

    print("[FIT] HGB with exactly same hyperparameters; extra 11 ball features", flush=True)
    features_plus_ball = original_features + BALL_FEATURES
    new_model = clone(old_model)
    new_model.fit(train[features_plus_ball].astype(np.float32), y_train)

    probs_old = old_model.predict_proba(valid[original_features].astype(np.float32))
    probs_new = new_model.predict_proba(valid[features_plus_ball].astype(np.float32))
    if not (np.allclose(probs_old.sum(axis=1), 1) and np.allclose(probs_new.sum(axis=1), 1)):
        raise ValueError("Class probability normalization failed")

    print(f"[EVALUATE] same transition matrix / strength={strength:g} / official Spell F1", flush=True)
    counts = load_transition_counts()
    transition = counts / counts.sum(axis=1, keepdims=True)
    frames = valid.frame_id.to_numpy()
    seg = segment_ids(frames)
    raw_old = probs_old.argmax(axis=1).astype(np.int8)
    raw_new = probs_new.argmax(axis=1).astype(np.int8)
    mk_old = viterbi_decode(probs_old, seg, transition, strength)
    mk_new = viterbi_decode(probs_new, seg, transition, strength)
    old_score = official_score(y_val, raw_old, mk_old, frames)
    new_score = official_score(y_val, raw_new, mk_new, frames)
    print("\n[IDSSE J03WPY | NO BARCELONA DATA]")
    print_score("Player-only HGB (191)", y_val, raw_old, mk_old, old_score)
    print_score("Player + ball HGB", y_val, raw_new, mk_new, new_score)
    print(f"\nDelta SpellF1 (Raw)    = {new_score['raw']['spell_f1_mean'] - old_score['raw']['spell_f1_mean']:+.4f}")
    print(f"Delta SpellF1 (Markov) = {new_score['markov']['spell_f1_mean'] - old_score['markov']['spell_f1_mean']:+.4f}")
    print("Caveat: IDSSE ball is mostly present; this does not test Barcelona missing-ball robustness.")

    OUT.mkdir(parents=True, exist_ok=True)
    result = valid[["frame_id", "label", "ball_observed"]].copy()
    for name, prob, raw, mk in [
        ("player", probs_old, raw_old, mk_old),
        ("ball", probs_new, raw_new, mk_new),
    ]:
        result[f"{name}_raw"] = raw
        result[f"{name}_markov"] = mk
        for k, cls in enumerate(["H", "A", "D"]):
            result[f"{name}_p_{cls}"] = prob[:, k]
    result.to_parquet(OUT / "J03WPY_comparison_5fps.parquet", index=False)
    joblib.dump({"model": new_model, "features": features_plus_ball,
                 "strength": strength, "label_codes": [0, 1, 2]},
                OUT / "idsse_hgb_ball_model.joblib")
    summary = {
        name: {key: {"macro_f1": float(val["macro_f1"]),
                     "spell_f1": float(val["spell_f1_mean"]),
                     "n_spells": int(val["n_pred_spells"])}
               for key, val in metrics.items()}
        for name, metrics in [("player", old_score), ("ball", new_score)]
    }
    (OUT / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[SAVED] {OUT} (no Barcelona files used)")
    (OUT / "experiment_config.json").write_text(json.dumps({
        "train_matches": list(TRAIN), "validation_match": VALID,
        "fps": SAMPLE_FPS, "strength": strength,
        "features_player": len(original_features), "features_with_ball": len(features_plus_ball),
        "reference_only_not_competition_score": True,
    }, indent=2), encoding="utf-8")
    return OUT

