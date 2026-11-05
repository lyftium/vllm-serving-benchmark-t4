# Demo path (screener / interview)

No company names. Use only what this repository proves.

## 90 seconds

1. Problem: serving levers move throughput and latency together — slides without a fixed protocol are not defensible.
2. Show top README table (Verify / Status).
3. Point at [`PROTOCOL.md`](../PROTOCOL.md), [`results.json`](../results.json), [`benchmark_results.csv`](../benchmark_results.csv).
4. State limits: single Kaggle campaign on 2×T4; not multi-seed; not a customer cluster.

## 10 minutes (laptop / talk-through)

```bash
pytest -q tests/                          # schema / artefact checks
# numbers live in results.json — do not re-run GPU here unless you have the same protocol
```

Talk track:

| Minute | Show | Say |
|---|---|---|
| 0–2 | Problem + PROTOCOL | Fixed hardware, engine, models, one run per config |
| 2–4 | `results.json` / CSV | Quote only recorded tok/s, TTFT/ITL, quality |
| 4–6 | Continuous batching vs single-seq | Throughput can dominate on same GPU |
| 6–8 | AWQ vs FP16 | Both quality and tok/s move — measure both |
| 8–9 | TP / prefix caching observations | Small-model PCIe TP can lose; prefix helps shared prompts |
| 9–10 | Limits + next | Not multi-seed; not LLaVA/customer SLO extrapolation |

## Do not claim

- Multi-seed confidence intervals (single campaign)
- Customer-cluster or multi-node SLO numbers
- Digits beyond `results.json` / CSV
