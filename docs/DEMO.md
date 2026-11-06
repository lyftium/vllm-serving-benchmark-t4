# Demo — vLLM T4 benchmark

Inspect the published campaign. Re-running GPU cells needs a matching Kaggle / 2×T4 environment and [`PROTOCOL.md`](../PROTOCOL.md).

## Prerequisites

- Inspection: Python and `pytest`; no GPU or model download
- Re-run: 2× Tesla T4, vLLM `0.6.6.post1`, isolated benchmark Python 3.12.3 (notebook kernel 3.13.16), and model weights; see [`PROTOCOL.md`](../PROTOCOL.md)

## Quick check

```bash
python -m pip install pytest
pytest -q tests/
```

**Expected:** tests pass.

## Read results

1. [`results.json`](../results.json) — full campaign + `derived_observations`
2. [`benchmark_results.csv`](../benchmark_results.csv) — flat rows
3. [`PROTOCOL.md`](../PROTOCOL.md) — hardware, engine, campaign shape

Quote tok/s, TTFT, and quality only from those files.

## Reproduction gaps

The repository does not include the benchmark driver/source notebook, full throughput prompt corpus, immutable model commit revisions, or an exact warm-up procedure. CSV `model_revision` holds model names and `commit` holds a campaign label, not a source revision.

## Limitations

- One run per configuration — exploratory, not multi-seed
- 2× Tesla T4 Kaggle session — not a customer-cluster SLO
- Do not extrapolate to larger VLMs or multi-node setups
