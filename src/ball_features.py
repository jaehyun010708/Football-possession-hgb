"""11 ball geometry features; extraction logic retained from the verified experiment.

No interpolation or >3s=>D rule is applied in this sharing baseline.
"""
from __future__ import annotations
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from .config import RAW_DIR as RAW, FEATURE_DIR as CACHE, COMPETITION, CANONICAL_LENGTH_M as LENGTH_M, CANONICAL_WIDTH_M as WIDTH_M
from .player_features import identify_segments, team_arrays
MATCHES = tuple(COMPETITION)
BALL_FEATURES = [
    "ball_x", "ball_y", "ball_dx_m", "ball_dy_m", "ball_displacement_m",
    "home_ball_nearest_m", "away_ball_nearest_m", "ball_nearest_gap_m",
    "home_ball_within_3m", "away_ball_within_3m", "ball_observed",
]


def local_xml(mid: str, pattern: str) -> Path:
    paths = list((RAW / mid).glob(pattern))
    paths = [p for p in paths if p.is_file() and p.stat().st_size]
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 local file in {RAW/mid} matching {pattern}; got {paths}. "
            "This script intentionally does NOT download anything."
        )
    return paths[0]


def nearest_team(px, py, bx, by, observed):
    """Per-frame distances to all available players (no player identity needed)."""
    dx = (px - bx[:, None]) * LENGTH_M
    dy = (py - by[:, None]) * WIDTH_M
    dist = np.hypot(dx, dy)
    ok = np.isfinite(dist) & observed[:, None]
    dist = np.where(ok, dist, np.inf)
    nearest = dist.min(axis=1)
    nearest[~np.isfinite(nearest)] = np.nan
    count3 = (dist <= 3.0).sum(axis=1).astype(np.float32)
    count3[~observed] = np.nan
    return nearest.astype(np.float32), count3


def build(mid: str, force: bool = False) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"{mid}_ball_geometry_v1_5fps.parquet"
    if dst.is_file() and not force:
        print(f"[CACHE] {mid}: {dst}", flush=True)
        return dst

    meta = local_xml(mid, "DFL_02_01_matchinformation_*.xml")
    raw = local_xml(mid, "DFL_04_03_positions_raw_observed_*.xml")
    print(f"[PARSE LOCAL] {mid} (5 FPS; may take a while)", flush=True)
    from kloppy import sportec
    ds = sportec.load_tracking(
        meta_data=str(meta), raw_data=str(raw), sample_rate=0.2,
        coordinates="metrica", only_alive=False,
    )
    df = ds.to_df()
    required = {"frame_id", "ball_x", "ball_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"Kloppy missing columns: {required - set(df.columns)}")
    if df["frame_id"].duplicated().any():
        raise ValueError(f"{mid}: duplicate frame_id; need period-aware alignment")

    teams = ds.metadata.teams
    home, away = teams
    hx, hy, _, _ = team_arrays(df, home, "home")
    ax, ay, _, _ = team_arrays(df, away, "away")

    bx = pd.to_numeric(df["ball_x"], errors="coerce").to_numpy(dtype=np.float32, copy=True)
    by = pd.to_numeric(df["ball_y"], errors="coerce").to_numpy(dtype=np.float32, copy=True)
    obs = np.isfinite(bx) & np.isfinite(by)
    bx[~obs] = np.nan
    by[~obs] = np.nan
    hn, hc = nearest_team(hx, hy, bx, by, obs)
    an, ac = nearest_team(ax, ay, bx, by, obs)

    seg = identify_segments(df)
    contiguous = np.zeros(len(df), dtype=bool)
    contiguous[1:] = seg[1:] == seg[:-1]
    consecutive_observed = np.zeros(len(df), dtype=bool)
    consecutive_observed[1:] = obs[1:] & obs[:-1] & contiguous[1:]
    dx_m = np.full(len(df), np.nan, dtype=np.float32)
    dy_m = np.full(len(df), np.nan, dtype=np.float32)
    dx_m[1:] = np.where(consecutive_observed[1:],
                         (bx[1:] - bx[:-1]) * LENGTH_M, np.nan)
    dy_m[1:] = np.where(consecutive_observed[1:],
                         (by[1:] - by[:-1]) * WIDTH_M, np.nan)

    feat = pd.DataFrame({
        "frame_id": df["frame_id"].to_numpy(),
        "ball_x": bx, "ball_y": by,
        "ball_dx_m": dx_m, "ball_dy_m": dy_m,
        "ball_displacement_m": np.hypot(dx_m, dy_m),
        "home_ball_nearest_m": hn,
        "away_ball_nearest_m": an,
        "ball_nearest_gap_m": an - hn,  # positive: home closer
        "home_ball_within_3m": hc,
        "away_ball_within_3m": ac,
        "ball_observed": obs.astype(np.float32),
    })

    # Existing player feature cache is the authoritative row/label timeline;
    # exclude any unlabeled rows without ever treating labels as features.
    original = CACHE / f"{mid}_temporal_v1_5fps.parquet"
    if not original.is_file():
        raise FileNotFoundError(f"Missing player feature cache: {original}")
    index = pd.read_parquet(original, columns=["frame_id"])
    if index["frame_id"].duplicated().any():
        raise ValueError(f"{mid}: duplicate frame IDs in player cache")
    aligned = index.merge(feat, on="frame_id", how="left", validate="one_to_one", indicator=True, sort=False)
    if (aligned["_merge"] != "both").any():
        raise RuntimeError(f"{mid}: unaligned frame IDs: {(aligned['_merge']!='both').sum()}")
    aligned = aligned.drop(columns="_merge")
    aligned.to_parquet(dst, index=False)
    print(f"[SAVED] {mid} {len(aligned):,} frames; ball present: {aligned.ball_observed.mean():.1%}", flush=True)
    del ds, df, feat, aligned, hx, hy, ax, ay
    gc.collect()
    return dst


