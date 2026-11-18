# results

Raw outputs, one folder per study. Nothing here is edited by hand.

| File | Contents |
|---|---|
| `<id>.json` | Run configuration (grid, seeds, device, torch version) and every recorded validation-loss curve |
| `<id>.log` | Progress lines printed during the run |
| `<id>_analysis.txt` | Exact output of the study's `analyze.py`; re-running it must reproduce this file |

`e0/` also holds `e0_run1_INVALID.*`, a first run invalidated by a bug. It is kept for provenance and explained in
[`e0/README.md`](e0/README.md). Files matching `*_partial.json` are checkpoints of a run in progress and are not tracked.
