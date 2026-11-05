"""Download unmodified Metrica possession_metrics.py (MIT license)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.official_metric import ensure_official_metric
if __name__ == "__main__":
    print(ensure_official_metric(allow_download=True))
