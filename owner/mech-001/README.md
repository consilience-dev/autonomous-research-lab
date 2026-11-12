# MECH-001 owner-side replay

Run from the repository root with the pinned Run 2 interpreter. Keep the results directory private. The run refuses to overwrite result files. For a replay, copy this experiment into a new owner-side directory at the same depth with an empty results directory. Preserve the original archived results and compare the replay separately. Behavior and robustness load the original checkpoint and touch only in-memory weights.

```sh
python=agent-run2/runtime/venv/bin/python
"$python" -B owner/mech-001/run.py freeze
"$python" -B owner/mech-001/run.py preflight
"$python" -B owner/mech-001/run.py behavior
"$python" -B owner/mech-001/run.py mask
"$python" -B owner/mech-001/run.py robustness
```

Review `results/review-packet.json` against the frozen rubric while withholding `results/review-key.json`. Enter one TSV line per code, with code, stance, correctness, coherence, format, and a rationale. Use 1 or 0 for each Boolean and omit the header. Aggregate only after locking the ratings.

```sh
"$python" -B owner/mech-001/score.py
"$python" -B owner/mech-001/verify-results.py
python3 -B research/registry.py verify
```
