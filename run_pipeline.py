"""One-command IDSSE HGB pipeline. Never uses Barcelona competition data/GT."""
from __future__ import annotations
import argparse
from src.config import (
    PLAYER_MODEL, FEATURE_DIR, COMPETITION, MARKOV_STRENGTH,
    SAMPLE_FPS,
)
from src.official_metric import ensure_official_metric


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reuse-player-model', action='store_true',
                    help='Skip fitting the 191-feature HGB only if model and all caches exist')
    ap.add_argument('--rebuild-player-cache', action='store_true',
                    help='Reparse local/downloaded IDSSE XML to rebuild 191-feature cache')
    ap.add_argument('--rebuild-ball-cache', action='store_true',
                    help='Reparse IDSSE XML to rebuild ball geometry cache')
    ap.add_argument('--strength', type=float, default=MARKOV_STRENGTH,
                    help='Markov transition weight (frozen baseline default: 0.25)')
    ap.add_argument('--offline', action='store_true',
                    help='Do not download official metric or missing IDSSE data')
    args = ap.parse_args()
    if args.strength < 0:
        ap.error('--strength must be >= 0')
    if args.offline:
        # Player-cache builder may automatically download, so verify local inputs early.
        from src.config import RAW_DIR
        missing = []
        for mid in COMPETITION:
            location = RAW_DIR / mid
            for glob in ('DFL_02_01_matchinformation_*.xml', 'DFL_04_03_positions_raw_observed_*.xml'):
                if not list(location.glob(glob)):
                    missing.append(f'{location} / {glob}')
        if missing and not args.reuse_player_model:
            ap.error('Offline mode requires original IDSSE XML files for initial training:\n' + '\n'.join(missing[:5]))

    print('[0/4] Ensure official scorer (downloads it once if missing)', flush=True)
    ensure_official_metric(allow_download=not args.offline)

    from src.training import fit_player_hgb
    from src.ball_features import build
    from src.evaluate import compare_models

    all_caches_exist = all((FEATURE_DIR / f'{mid}_temporal_v1_{SAMPLE_FPS:g}fps.parquet').is_file() for mid in COMPETITION)
    reuse = args.reuse_player_model and PLAYER_MODEL.is_file() and all_caches_exist
    if args.reuse_player_model and not reuse:
        print('[INFO] Reuse requested, but model or some caches missing. Training normally.', flush=True)
    if args.rebuild_player_cache:
        reuse = False

    if reuse:
        print('[1/4] Reusing existing player HGB and 191-feature caches', flush=True)
    else:
        print('[1/4] Preparing IDSSE XML + fitting player-only HGB', flush=True)
        fit_player_hgb(rebuild=args.rebuild_player_cache)
    print('[2/4] Preparing 11 ball features', flush=True)
    for mid in COMPETITION:
        build(mid, force=args.rebuild_ball_cache or args.rebuild_player_cache)
    print('[3/4] Fitting HGB with ball features and evaluating both models', flush=True)
    result = compare_models(strength=args.strength)
    print('[4/4] Finished — open:', result / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
