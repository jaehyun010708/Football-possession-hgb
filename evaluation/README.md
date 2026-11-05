# Official competition scorer

The project uses **Metrica Sports' unmodified** `possession_metrics.py`.

Fetch once from the [Metrica competition repository](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/evaluation/possession_metrics.py):

```bash
python scripts/fetch_official_metric.py
```

The file will be placed here as `evaluation/possession_metrics.py` (gitignored).
The baseline expands sampled 5-FPS states to the scorer's 25-FPS timeline;
this does **not** improve the underlying 0.2-second prediction resolution.

Official source code is provided by Metrica Sports under its repository's MIT license.
See `THIRD_PARTY_NOTICES.md` for attribution. If offline, manually place the file here.
