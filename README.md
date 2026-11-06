# vLLM serving benchmark — 2× Tesla T4

![CI](https://github.com/lyftium/vllm-serving-benchmark-t4/actions/workflows/ci.yml/badge.svg)

One measured campaign of serving levers on commodity GPUs.

**Headline (single run, from [`results.json`](./results.json)):** continuous batching on Qwen2.5-0.5B fp16 raised generated tok/s **~35.5×** — `max_num_seqs` 1 → 128 (144.5 → 5126.4 tok/s) on the same GPU. Online concurrency 1 → 32: 140.9 → 2535.8 tok/s with p95 TTFT 45 → 164 ms. AWQ vs fp16 and TP trade off quality / memory / throughput depending on batch size — see derived observations in the file.

[`docs/DEMO.md`](docs/DEMO.md) · [`results.json`](./results.json) · [`benchmark_results.csv`](./benchmark_results.csv) · [`PROTOCOL.md`](PROTOCOL.md)

vLLM `0.6.6.post1` (V0) · Python 3.12 · Qwen2.5 0.5B / 1.5B (fp16 + AWQ) · 2× Tesla T4 on Kaggle. **One run per configuration** — not multi-seed, not a customer SLO, not a cluster deploy.

Notebook: [kaggle.com/code/lyftium/vllm-serving-benchmark-t4](https://www.kaggle.com/code/lyftium/vllm-serving-benchmark-t4)

Measurement lab only. Front door: [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack). KServe packaging: [drawing-review-service](https://github.com/lyftium/drawing-review-service) `deploy/kserve/`.

## Protocol

- Hardware — 2× Tesla T4 (compute 7.5), Kaggle GPU session
- Engine — vLLM 0.6.6.post1 (V0), isolated Python 3.12 venv
- Models — as recorded in `results.json`
- Shape — one campaign, one run per cell (~20 min wall)
- Metrics — generated tok/s; TTFT / ITL / e2e percentiles where online; exact-match on a small probe set where noted

## Reading the artefacts

1. [`results.json`](./results.json) — raw campaign + `derived_observations`
2. [`benchmark_results.csv`](./benchmark_results.csv) — flat rows
3. [`BENCHMARK.md`](./BENCHMARK.md) — short reading guide

Quote only digits from these files. Do not extrapolate to larger VLMs, customer clusters, or multi-node SLOs.

## Selected observations (same campaign)

From `derived_observations` in `results.json`:

- Batching: **35.5×** tok/s at `max_num_seqs` 128 vs 1 (0.5B fp16)
- Online fp16 0.5B: concurrency 1 → 32 raises tok/s ~18× while TTFT/ITL rise
- AWQ vs fp16: faster at mns=1 on 1.5B (**1.61×**), slower at mns=128 (**0.50×**); exact-match drops on the probe set
- TP=2 vs TP=1 at mns=128: **0.86×** (PCIe all-reduce not worth it on this tiny model)
- Prefix caching with a shared system prompt: **1.61×** req/s (14.3 → 23.0)

## Reproduce

Re-run the public Kaggle notebook (or an equivalent script) on comparable GPUs. Keep model revision, max tokens, concurrency, and warm-up explicit. A second seed is a new campaign, not a silent overwrite.

MIT — measurement artefacts and docs only; upstream model licenses still apply to weights you download.
