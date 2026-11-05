"""Player tracking features, extracted unchanged from the 191-feature HGB baseline.

Labels are built from external IDSSE metadata and are never model features.
"""
from __future__ import annotations
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from .config import (
    ROOT, RAW_DIR, FEATURE_DIR, OUTPUT_DIR, MODEL_DIR,
    REPO_ID, VALID_MATCH, LABEL_NAMES, VERSION,
    COMPETITION, CANONICAL_LENGTH_M, CANONICAL_WIDTH_M,
)
# Temporal features do NOT use the ball, labels, or frames in the future.
HISTORY_FEATURES = [
    "home_n", "away_n",
    "home_x_mean", "home_y_mean", "away_x_mean", "away_y_mean",
    "home_x_std", "away_x_std", "home_y_std", "away_y_std",
    "home_speed_mean", "away_speed_mean",
    "home_fast_rate", "away_fast_rate",
    "home_opp_nearest_mean", "away_opp_nearest_mean",
    "home_pressure_rate", "away_pressure_rate",
    "centroid_distance", "centroid_x_gap",
]


def download_one(match_id: str, kind: str) -> Path:
    comp = COMPETITION[match_id]
    if kind == "meta":
        filename = f"DFL_02_01_matchinformation_{comp}_DFL-MAT-{match_id}.xml"
    elif kind == "tracking":
        filename = f"DFL_04_03_positions_raw_observed_{comp}_DFL-MAT-{match_id}.xml"
    else:
        raise ValueError(kind)

    local_dir = RAW_DIR / match_id
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename
    if local_path.is_file() and local_path.stat().st_size > 0:
        return local_path

    print(f"  Download {kind}: {match_id} ({filename})", flush=True)
    return Path(hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=str(local_dir),
    ))


def team_arrays(df: pd.DataFrame, team, prefix: str):
    """Extract same team by metadata player IDs, not column position."""
    ids = [str(p.player_id) for p in team.players]
    if not ids:
        raise RuntimeError(f"Empty metadata roster for {prefix}")

    x = df.reindex(columns=[f"{pid}_x" for pid in ids]).to_numpy(dtype=np.float32)
    y = df.reindex(columns=[f"{pid}_y" for pid in ids]).to_numpy(dtype=np.float32)

    # IDSSE Sportec S (speed) is km/h; use m/s for consistency with ATD.
    speed_columns = [f"{pid}_s" for pid in ids]
    if not any(col in df.columns for col in speed_columns):
        raise RuntimeError(
            "No player *_s speed columns found in Kloppy DataFrame. "
            "Inspect df.columns before training: speed features cannot be used."
        )
    speed_kmh = df.reindex(columns=speed_columns).to_numpy(dtype=np.float32)
    s = speed_kmh / 3.6

    xy_ok = np.isfinite(x) & np.isfinite(y)
    x = np.where(xy_ok, x, np.nan)
    y = np.where(xy_ok, y, np.nan)
    s = np.where(xy_ok & np.isfinite(s), s, np.nan)
    return x, y, s, xy_ok


def add_team_features(result: pd.DataFrame, x, y, s, valid, prefix: str) -> None:
    xdf = pd.DataFrame(x)
    ydf = pd.DataFrame(y)
    sdf = pd.DataFrame(s)

    count = valid.sum(axis=1)
    result[f"{prefix}_n"] = count
    result[f"{prefix}_speed_n"] = np.isfinite(s).sum(axis=1)

    for axis, data in [("x", xdf), ("y", ydf)]:
        result[f"{prefix}_{axis}_mean"] = data.mean(axis=1).to_numpy()
        result[f"{prefix}_{axis}_std"] = data.std(axis=1).to_numpy()
        result[f"{prefix}_{axis}_min"] = data.min(axis=1).to_numpy()
        result[f"{prefix}_{axis}_max"] = data.max(axis=1).to_numpy()
        result[f"{prefix}_{axis}_median"] = data.median(axis=1).to_numpy()
        result[f"{prefix}_{axis}_q10"] = data.quantile(0.10, axis=1).to_numpy()
        result[f"{prefix}_{axis}_q90"] = data.quantile(0.90, axis=1).to_numpy()
        result[f"{prefix}_{axis}_width"] = (
            result[f"{prefix}_{axis}_q90"] - result[f"{prefix}_{axis}_q10"]
        )

    speeds_valid = np.isfinite(s)
    speeds_n = speeds_valid.sum(axis=1)
    divisor = np.maximum(speeds_n, 1)
    result[f"{prefix}_speed_mean"] = sdf.mean(axis=1).to_numpy()
    result[f"{prefix}_speed_std"] = sdf.std(axis=1).to_numpy()
    result[f"{prefix}_speed_median"] = sdf.median(axis=1).to_numpy()
    result[f"{prefix}_speed_p90"] = sdf.quantile(0.9, axis=1).to_numpy()
    result[f"{prefix}_fast_rate"] = ((s > 5.0) & speeds_valid).sum(axis=1) / divisor
    result[f"{prefix}_moving_rate"] = ((s > 2.0) & speeds_valid).sum(axis=1) / divisor

    divisor_xy = np.maximum(count, 1)
    for z, (lo, hi) in enumerate([(0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)]):
        in_zone = valid & (x >= lo) & (x <= hi if z == 2 else x < hi)
        result[f"{prefix}_x_zone_{z}"] = in_zone.sum(axis=1) / divisor_xy


def add_opposition_features(result: pd.DataFrame, h, a, hv, av) -> None:
    """Pairwise team opposition distance, using vectorized chunks (no player IDs)."""
    n = h.shape[0]
    home_near = np.full((n, h.shape[1]), np.nan, dtype=np.float32)
    away_near = np.full((n, a.shape[1]), np.nan, dtype=np.float32)

    for start in range(0, n, 256):
        end = min(start + 256, n)
        H = h[start:end]
        A = a[start:end]
        dh = H[:, :, None, :] - A[:, None, :, :]
        dist = np.sqrt(np.square(dh).sum(axis=-1))
        mask = hv[start:end, :, None] & av[start:end, None, :]
        dist = np.where(mask, dist, np.inf)
        hn = dist.min(axis=2)
        an = dist.min(axis=1)
        home_near[start:end] = np.where(np.isfinite(hn), hn, np.nan)
        away_near[start:end] = np.where(np.isfinite(an), an, np.nan)

    for prefix, d in [("home", home_near), ("away", away_near)]:
        distances = pd.DataFrame(d)
        result[f"{prefix}_opp_nearest_mean"] = distances.mean(axis=1).to_numpy()
        result[f"{prefix}_opp_nearest_median"] = distances.median(axis=1).to_numpy()
        result[f"{prefix}_opp_nearest_min"] = distances.min(axis=1).to_numpy()
        n_visible = np.isfinite(d).sum(axis=1)
        result[f"{prefix}_pressure_rate"] = (
            (d < 3.0).sum(axis=1) / np.maximum(n_visible, 1)
        )


def identify_segments(df: pd.DataFrame) -> np.ndarray:
    """Prevent temporal windows crossing halves, timestamp jumps, or tracking gaps."""
    frame = pd.to_numeric(df["frame_id"], errors="coerce")
    delta = frame.diff()
    positive = delta[(delta > 0) & (delta < 100)]
    nominal_step = float(positive.median()) if len(positive) else 1.0
    breaks = delta.isna() | (delta <= 0) | (delta > max(3.0 * nominal_step, 10.0))

    # Kloppy normally includes period_id; both variants handled if available.
    for period_col in ("period_id", "period"):
        if period_col in df.columns:
            changed = df[period_col].ne(df[period_col].shift(1)).fillna(False)
            breaks = breaks | changed
            break
    return breaks.cumsum().to_numpy(dtype=np.int32)


def add_temporal_features(features: pd.DataFrame, segment: np.ndarray, fps: float) -> pd.DataFrame:
    """Causal rolling mean and past-vs-current deltas at 1,3,5 seconds."""
    features["_segment"] = segment
    base = HISTORY_FEATURES
    missing = [c for c in base if c not in features.columns]
    if missing:
        raise RuntimeError(f"Missing temporal base features: {missing}")

    group = features.groupby("_segment", sort=False)
    block = [features]
    for seconds in (1, 3, 5):
        lag = max(1, round(fps * seconds))
        rolling = group[base].rolling(window=lag, min_periods=1).mean()
        rolling = rolling.reset_index(level=0, drop=True).sort_index()
        rolling = rolling.add_suffix(f"_mean_{seconds}s")

        old = group[base].shift(lag)
        delta = features[base] - old
        delta = delta.add_suffix(f"_delta_{seconds}s")
        block.extend([rolling, delta])

    # Local variability helps detect pauses/restarts in player movement.
    roll_std = group[["home_speed_mean", "away_speed_mean", "centroid_x_gap"]].rolling(
        window=max(2, round(3 * fps)), min_periods=2
    ).std().reset_index(level=0, drop=True).sort_index()
    block.append(roll_std.add_suffix("_std_3s"))
    output = pd.concat(block, axis=1)
    return output.drop(columns=["_segment"])


def prepare_match(match_id: str, fps: float, rebuild: bool) -> pd.DataFrame:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{fps:g}".replace(".", "p")
    cache_path = FEATURE_DIR / f"{match_id}_temporal_{VERSION}_{suffix}fps.parquet"

    if cache_path.exists() and not rebuild:
        cache = pd.read_parquet(cache_path)
        required = {"frame_id", "label", *HISTORY_FEATURES, "home_x_mean_delta_3s"}
        if len(cache) >= 1000 and required.issubset(cache.columns) and set(cache["label"].unique()) == {0, 1, 2}:
            print(f"[CACHE] {match_id}: {len(cache):,} rows", flush=True)
            return cache
        print(f"[STALE CACHE] {match_id}: rebuilding", flush=True)

    print(f"[LOAD] {match_id}", flush=True)
    meta = download_one(match_id, "meta")
    raw = download_one(match_id, "tracking")
    if not 0 < fps <= 25:
        raise ValueError(f"FPS must be within (0,25], got {fps}")
    from kloppy import sportec
    tracking = sportec.load_tracking(
        meta_data=str(meta),
        raw_data=str(raw),
        sample_rate=fps / 25.0,
        only_alive=False,
        coordinates="metrica",
    )
    df = tracking.to_df()
    print(f"  Parsed frames: {len(df):,} at {fps:g} FPS", flush=True)
    if len(df) < 1000:
        raise RuntimeError(f"Unexpectedly few frames from {match_id}: {len(df)}")

    home, away = tracking.metadata.teams
    hx, hy, hs, hv = team_arrays(df, home, "home")
    ax, ay, ass, av = team_arrays(df, away, "away")
    if np.isfinite(hs).sum() == 0 or np.isfinite(ass).sum() == 0:
        raise RuntimeError(f"{match_id}: player speed values are all missing")

    result = pd.DataFrame(index=df.index)
    add_team_features(result, hx, hy, hs, hv, "home")
    add_team_features(result, ax, ay, ass, av, "away")

    result["centroid_x_gap"] = result["home_x_mean"] - result["away_x_mean"]
    result["centroid_y_gap"] = result["home_y_mean"] - result["away_y_mean"]
    result["centroid_distance"] = np.sqrt(
        np.square(CANONICAL_LENGTH_M * result["centroid_x_gap"])
        + np.square(CANONICAL_WIDTH_M * result["centroid_y_gap"])
    )
    result["speed_mean_gap"] = result["home_speed_mean"] - result["away_speed_mean"]
    result["n_gap"] = result["home_n"] - result["away_n"]
    result["x_spread_gap"] = result["home_x_width"] - result["away_x_width"]

    home_xy = np.stack([hx * CANONICAL_LENGTH_M, hy * CANONICAL_WIDTH_M], axis=-1)
    away_xy = np.stack([ax * CANONICAL_LENGTH_M, ay * CANONICAL_WIDTH_M], axis=-1)
    add_opposition_features(result, home_xy, away_xy, hv, av)

    # These values are for labels ONLY. Never pass them as features to the model.
    state = df["ball_state"].astype("string").str.lower()
    owner = df["ball_owning_team_id"].astype("string")
    y = np.select(
        [state.eq("dead").fillna(False),
         (state.eq("alive") & owner.eq(str(home.team_id))).fillna(False),
         (state.eq("alive") & owner.eq(str(away.team_id))).fillna(False)],
        [2, 0, 1], default=-1,
    )
    segments = identify_segments(df)
    result = add_temporal_features(result, segments, fps)
    result["frame_id"] = df["frame_id"].to_numpy()
    result["match_id"] = match_id
    result["label"] = y
    unlabeled = int((result["label"] == -1).sum())
    if unlabeled:
        print(f"  Drop unlabeled: {unlabeled:,}", flush=True)
        result = result.loc[result["label"] >= 0].copy()

    if len(result) < 1000 or set(result["label"].unique()) != {0, 1, 2}:
        raise RuntimeError(f"{match_id}: invalid labels or insufficient frames")
    # Reduce RAM use for all 7 games.
    numeric_features = result.columns.difference(["match_id", "frame_id", "label"])
    result[numeric_features] = result[numeric_features].astype(np.float32)
    result.to_parquet(cache_path, index=False)
    print(f"  Saved {len(result):,} labeled rows; "+
          f"distribution={result['label'].value_counts().sort_index().to_dict()}", flush=True)
    del tracking, df, hx, hy, hs, ax, ay, ass
    gc.collect()
    return result


