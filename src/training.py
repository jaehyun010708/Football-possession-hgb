"""Fit the player-only HistGradientBoosting baseline with unchanged parameters."""
from __future__ import annotations
import gc
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .config import OUTPUT_DIR, MODEL_DIR, COMPETITION, VALID_MATCH, LABEL_NAMES, VERSION, CANONICAL_LENGTH_M, CANONICAL_WIDTH_M, SAMPLE_FPS
from .player_features import prepare_match


def fit_player_hgb(rebuild: bool = False, sample_rate: float = SAMPLE_FPS) -> Path:
    """Prepare cached IDSSE frames and train the player-only 191-feature HGB.

    Keep at 5 FPS so ball feature cache and official scorer expansion match.
    """
    if sample_rate != SAMPLE_FPS:
        raise ValueError(f"Sharing baseline is fixed at {SAMPLE_FPS:g} FPS, got {sample_rate}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_data = [prepare_match(match_id, sample_rate, rebuild) for match_id in COMPETITION]
    data = pd.concat(all_data, ignore_index=True)
    del all_data
    gc.collect()

    exclude = {"match_id", "frame_id", "label"}
    features = [c for c in data.columns if c not in exclude]
    train = data["match_id"] != VALID_MATCH
    valid = ~train
    X_train = data.loc[train, features].astype(np.float32)
    y_train = data.loc[train, "label"].to_numpy(dtype=np.int32)
    X_val = data.loc[valid, features].astype(np.float32)
    y_val = data.loc[valid, "label"].to_numpy(dtype=np.int32)

    print(f"\nFeatures: {len(features)}")
    print(f"Train: {X_train.shape} (6 matches); Validation: {X_val.shape} ({VALID_MATCH})")
    if set(np.unique(y_train)) != {0, 1, 2} or set(np.unique(y_val)) != {0, 1, 2}:
        raise RuntimeError("Train/validation must both contain H, A and D")

    model = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    probas = model.predict_proba(X_val)

    acc = accuracy_score(y_val, predictions)
    macro = f1_score(y_val, predictions, labels=[0, 1, 2], average="macro")
    report = classification_report(y_val, predictions, labels=[0, 1, 2],
                                   target_names=LABEL_NAMES, digits=4, zero_division=0)
    cm = confusion_matrix(y_val, predictions, labels=[0, 1, 2])
    print(f"\nValidation: {VALID_MATCH}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro:.4f}\n{report}")
    print(f"Confusion matrix (H/A/D):\n{cm}")

    prediction_file = OUTPUT_DIR / f"{VALID_MATCH}_predictions.csv"
    prediction_df = data.loc[valid, ["frame_id", "match_id", "label"]].copy()
    prediction_df["pred"] = predictions
    for i, cls in enumerate(model.classes_):
        prediction_df[f"p_{LABEL_NAMES[int(cls)]}"] = probas[:, i]
    prediction_df.to_csv(prediction_file, index=False)

    (OUTPUT_DIR / "report.txt").write_text(
        f"Validation: {VALID_MATCH}\nAccuracy: {acc:.4f}\nMacro F1: {macro:.4f}\n\n"
        + report + f"\nConfusion matrix (H/A/D):\n{cm}\n",
        encoding="utf-8",
    )
    model_file = MODEL_DIR / "idsse_temporal_v1.joblib"
    joblib.dump({"model": model, "features": features, "labels": LABEL_NAMES,
                 "sample_rate": sample_rate, "version": VERSION,
                 "speed_unit": "m/s", "history_seconds": [1, 3, 5],
                 "approx_pitch_m": [CANONICAL_LENGTH_M, CANONICAL_WIDTH_M]}, model_file)
    print(f"\nSaved model: {model_file}\nSaved evaluation: {OUTPUT_DIR}")
    return model_file

