"""Score player-only HGB vs player+ball HGB on IDSSE held-out J03WPY."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MARKOV_STRENGTH
from src.evaluate import compare_models
from src.official_metric import ensure_official_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strength", type=float, default=MARKOV_STRENGTH,
                        help="Markov transition strength (default: 0.25)")
    parser.add_argument("--offline", action="store_true", help="Do not fetch official scorer")
    args = parser.parse_args()
    if args.strength < 0:
        parser.error("--strength must be >= 0")
    ensure_official_metric(allow_download=not args.offline)
    compare_models(strength=args.strength)
