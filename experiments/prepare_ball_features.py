"""Prepare 11 geometry ball features from already downloaded IDSSE XML."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ball_features import MATCHES, build

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite ball feature caches")
    args = parser.parse_args()
    for mid in MATCHES:
        build(mid, force=args.force)
