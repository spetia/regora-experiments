# Pre-registration record

Each study's plan (`studies/*/PLAN*.md`, or the docstring of `run.py`) was committed to the private research repository
(`regora-ai/regora-experiments`) before its results. This repository is a squashed snapshot, so those hashes do not resolve
here; the table is copied from the private history (author-recorded local commit times, 2026-11, same day unless stated).
To verify a plan, ask for read access to the private repository or for a bundle of its history.

| Study | Plan commit | Plan time | Results commit | Results time |
|---|---|---|---|---|
| E0 | none: plan and results share `412d7d4` | 11-11 13:08 | `412d7d4` | 11-11 13:08 |
| E0b | `0e0e85d` | 11-11 13:10 | `74a3549` | 11-11 13:36 |
| E0c | `52c9d1b` | 11-11 13:51 | `1636ebb` | 11-11 14:19 |
| E0d | `baad11e` | 11-11 14:22 | `ebe9987` | 11-11 14:26 |
| E1 | `2f89830` | 11-11 14:51 | `ded1a9f` | 11-11 15:05 |
| E1b | `78ad6cc` | 11-11 15:06 | `f41c0be` | 11-11 15:34 |
| E2 | `0c8be6b` | 11-11 16:02 | `fa236f9` | 11-11 17:14 |
| E3 | `921f49b` | 11-11 17:27 | `e847d72` | 11-11 18:53 |
| E4b | `67654fe` | 11-11 17:33 | `e847d72` | 11-11 18:53 |
| E6 | `fdb59a5` | 11-11 17:32 | `e847d72` | 11-11 18:53 |
| E7 | `ffd9942` | 11-11 20:24 | `38d3f5c` | 11-11 22:15 |
| E7b | `38d3f5c` (with E7 results) | 11-11 22:15 | `933644e` | 11-11 23:27 |
| E7c | `edc72ac` | 11-11 22:29 | `5068a77` | 11-12 00:44 |
| E8 | `b346233` | 11-12 01:02 | `81dbead` | 11-12 02:56 |
| E9 | `f30a515` | 11-12 22:56 | `f4e77b6` | 11-13 05:36 |
| E10 | `dfba664` | 11-13 03:23 | `e03534a` | 11-13 08:36 |
| E11 | `b8cf92e` | 11-13 12:05 | `7851523` | 11-13 18:20 |
| E12 | `d8f3aec` | 11-14 11:12 | `8819a5e` | 11-14 17:55 |
| E13 | `b08e020` | 11-14 21:45 | `ba71e59` | 11-14 21:46 |
| E14 | `5831037` | 11-14 22:25 | `d21f15f` | 11-15 11:28 |
| E14 held-out check | none (post hoc) | | `6a279f9` | 11-16 13:00 |
| Scaling A | `91f6595` | 11-15 02:37 | `ad3cecb` | 11-15 02:37 |
| Scaling B | `91f6595` | 11-15 02:37 | `ead38dd` | 11-15 07:34 |
| Scaling C | `91f6595` | 11-15 02:37 | `f0aa98f` | 11-15 10:31 |
| Scaling D | `40a705a` | 11-15 07:44 | `4606553` | 11-16 12:04 |
| Scaling E | `c47f810` | 11-15 11:46 | `e2161e1` | 11-16 12:04 |
| Scaling F | `62f42f3` | 11-16 12:04 | `5c30395` | 11-16 12:04 |
| Scaling G | `27c4436` | 11-16 12:04 | `c54958b` | 11-16 12:04 |
| Scaling H | `0ec3b22` | 11-16 12:04 | `c038a02` | 11-16 12:04 |
| E15 | `290cd84` | 11-15 12:11 | `8f0a49f` (stopped, no analysis) | 11-15 12:36 |
| E16 | `3061468` | 11-15 12:48 | `4226285` | 11-16 11:50 |

Caveats. The scaling stages D to H of 11-16 carry identical commit times (12:04): that study ran unattended and its history
was pushed in one batch, so for stages F to H only the commit order, not the clock, shows plan before result. E0 and E7b
were not registered before their data in a separate commit. E13 is a replay of recorded data; its plan and results are one
minute apart.
