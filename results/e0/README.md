# results/

- `e0_run1_INVALID.*`: first E0 run. **Invalid.** `run()` aliased the checkpoint's optimizer state (torch `load_state_dict` does not copy same-device tensors), so branches mutated the shared checkpoint. Caught by the determinism repeat check (`determinism_repeat_identical: False`). Do not use these numbers. Kept for provenance.
- `e0.json`, `e0.log`: the rerun after the fix.
