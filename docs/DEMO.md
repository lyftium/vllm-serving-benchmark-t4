# Demo — vLLM T4 benchmark

Inspect the published campaign. Re-running GPU cells needs a matching Kaggle / 2×T4 environment and [`PROTOCOL.md`](../PROTOCOL.md).

## Prerequisites

- For inspection only: this repository clone
- For re-run: 2× Tesla T4, vLLM `0.6.6.post1`, Python 3.12 (see PROTOCOL)

## Quick check

```bash
pytest -q tests/
```

**Expected:** tests pass.

## Read results

1. [`results.json`](../results.json) — full campaign + `derived_observations`
2. [`benchmark_results.csv`](../benchmark_results.csv) — flat rows
3. [`PROTOCOL.md`](../PROTOCOL.md) — hardware, engine, campaign shape

Quote tok/s, TTFT, and quality only from those files.

## Limitations

- One run per configuration — exploratory, not multi-seed
- 2× Tesla T4 Kaggle session — not a customer-cluster SLO
- Do not extrapolate to larger VLMs or multi-node setups
