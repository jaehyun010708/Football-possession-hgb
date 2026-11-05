"""Compare a reproduced IDSSE validation report to the original historical numbers."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import BALL_OUTPUT_DIR

REFERENCE = {
    ('player','raw'): (0.8549988024889442,0.34981412639405207,662),
    ('player','markov'): (0.857388480643012,0.45046480743691897,339),
    ('ball','raw'): (0.9230388332745457,0.4262350936967632,760),
    ('ball','markov'): (0.9268947973100986,0.6055979643765904,372),
}

def main():
    target=BALL_OUTPUT_DIR/'comparison.json'
    if not target.exists():
        raise SystemExit(f'Missing {target}. Run python run_pipeline.py first.')
    data=json.loads(target.read_text(encoding='utf-8'))
    print('Reference parity (same versions and dataset; tolerance 0.0001 for F1):')
    failures=[]
    for (variant,stage),(macro,spell,spells) in REFERENCE.items():
        actual=data[variant][stage]
        diff_macro=actual['macro_f1']-macro
        diff_spell=actual['spell_f1']-spell
        ok=(abs(diff_macro)<=1e-4 and abs(diff_spell)<=1e-4 and actual['n_spells']==spells)
        print(f'  {variant:<7} {stage:<7} macro diff={diff_macro:+.6f} spell diff={diff_spell:+.6f} spells={actual["n_spells"]} {"OK" if ok else "DIFF"}')
        if not ok: failures.append((variant,stage))
    if failures:
        raise SystemExit(f'Differences detected: {failures}. Inspect dependency versions, scoring script and feature caches.')
    print('Reproduced all historical scores within tolerance.')

if __name__=='__main__':main()
