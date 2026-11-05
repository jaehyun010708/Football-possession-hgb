"""Retrieve and load Metrica's unmodified MIT-licensed official metric code.

External scorer lives at evaluation/possession_metrics.py and is not reproduced
in this repository. Fetching is explicit, or enabled by run_pipeline.py.
"""
from __future__ import annotations
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen
from .config import OFFICIAL_METRIC

METRIC_URL = (
    "https://raw.githubusercontent.com/metrica-sports/more-than-a-hack-2027/"
    "main/evaluation/possession_metrics.py"
)


def ensure_official_metric(allow_download: bool = False) -> Path:
    dest = OFFICIAL_METRIC
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    if not allow_download:
        raise FileNotFoundError(
            f"Official scorer missing: {dest}\n"
            f"Download it from {METRIC_URL} and place it at evaluation/possession_metrics.py, "
            "or run: python scripts/fetch_official_metric.py"
        )
    print(f"[DOWNLOAD] Official scorer from {METRIC_URL}", flush=True)
    try:
        with urlopen(METRIC_URL, timeout=60) as response:
            content = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Cannot download the official metric. Internet access to GitHub is required; "
            "or download it manually from the URL in README.md."
        ) from exc
    if len(content) < 1000 or b"SPELL_IOUS" not in content or b"def evaluate" not in content:
        raise RuntimeError("Downloaded official metric contents are incomplete or unexpected")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.py.part')
    temp.write_bytes(content)
    os.replace(temp, dest)
    print(f"[SAVED] {dest}", flush=True)
    return dest


@lru_cache(maxsize=1)
def load_official_metric(allow_download: bool = False):
    path = ensure_official_metric(allow_download=allow_download)
    spec = importlib.util.spec_from_file_location('metrica_possession_metrics', str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
