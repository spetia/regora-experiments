# E4 (partial): Python-call overhead of `regora.record`

**Label: MEASURED, microbenchmark only. Not a training-throughput measurement.**

- Date: 2026-11-11. Machine: Apple M1, 16 GB. CPython 3.13.14 (uv), release build, PyO3 0.30.1, maturin 1.16.0.
- Script: `python/regora-py/tests/smoke.py` (1,000,000 `record(0, i, 1.0)` calls, queue capacity 65,536, drain thread to a file).
- Baseline: an equivalent pure-Python no-op function call in the same loop.

| Build | `record` ns/call | Python no-op ns/call | Dropped of 1M |
|---|---|---|---|
| non-abi3 | 88, 84 (two runs) | 43, 46 | 0 |
| abi3-py313 | 115 (one run) | 52 | not recorded |

Notes and limits:
- Two runs for non-abi3, one for abi3. No confidence intervals, and the machine was not isolated. Treat the abi3-vs-non-abi3 gap as unestablished.
- Includes the loop and call overhead, so the cost attributable to Regora is roughly the `record` minus no-op difference (~40 ns here). Not separately verified.
- Zero drops at this rate says nothing about sustained-rate or slower-disk behaviour.
- **Not measured:** effect on real training throughput, memory overhead, CPU overhead of the drain thread, GPU/CUDA synchronisation (no CUDA on this machine).
- **Free-threaded:** PyO3 0.30.1 refuses to build for free-threaded CPython 3.13 ("does not support the free-threaded build of CPython versions below 3.14"). 3.14t was not tried.
