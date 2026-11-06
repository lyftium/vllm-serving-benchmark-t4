# Interview demo — vLLM T4 benchmark

Walkthrough for a screener. Stick to artefacts in this repo.

## 90 seconds

1. **Question:** which serving knobs move throughput and latency together?
2. Open the README “Start here” block.
3. Point at [`PROTOCOL.md`](../PROTOCOL.md), [`results.json`](../results.json), [`benchmark_results.csv`](../benchmark_results.csv).
4. **Limits out loud:** one Kaggle campaign on 2×T4; not multi-seed; not a customer cluster.

## 10 minutes

```bash
pytest -q tests/
# quote numbers from results.json — don’t re-run GPU unless you match the protocol
```

| Minute | Show | Say |
|---|---|---|
| 0–2 | Problem + PROTOCOL | Fixed hardware, engine, models; one run per config |
| 2–4 | `results.json` / CSV | Quote only recorded tok/s, TTFT/ITL, quality |
| 4–6 | Batching vs single-seq | Throughput can dominate on the same GPU |
| 6–8 | AWQ vs fp16 | Quality and tok/s both move — measure both |
| 8–9 | TP / prefix caching | Small-model PCIe TP can lose; prefix helps shared prompts |
| 9–10 | Limits | Not multi-seed; don’t extrapolate to LLaVA / multi-node SLOs |

## Do not claim

- Multi-seed confidence intervals
- Customer-cluster or multi-node SLO numbers
- Digits beyond `results.json` / CSV
