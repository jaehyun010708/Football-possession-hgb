"""Standalone player-only baseline training. From repo root: python experiments/train_hgb.py"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training import fit_player_hgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Reparse IDSSE XML and rebuild player caches")
    args = parser.parse_args()
    fit_player_hgb(rebuild=args.rebuild)
