# vLLM serving benchmark on 2× Tesla T4

![CI](https://github.com/lyftium/vllm-serving-benchmark-t4/actions/workflows/ci.yml/badge.svg)

Serving configs are easy to argue about and annoying to measure. I ran one vLLM campaign (0.6.6.post1, Python 3.12) on two Tesla T4 GPUs and wrote the protocol next to the numbers.

Continuous batching on Qwen2.5-0.5B fp16 moved generated tokens/s from 144.5 to 5126.4 when `max_num_seqs` went from 1 to 128. Two results I did not expect to be uniform: for the 1.5B model, AWQ was about 1.6× fp16 at one sequence and about 0.5× at 128 sequences; tensor parallelism across the two GPUs was slower than one GPU at the large batch, because the model is small relative to the PCIe cost. One run per cell — not a repeated trial and not a cluster capacity number.

[`results.json`](./results.json) · [`benchmark_results.csv`](./benchmark_results.csv) · [`PROTOCOL.md`](PROTOCOL.md) · [`docs/DEMO.md`](docs/DEMO.md)

Notebook: [kaggle.com/code/lyftium/vllm-serving-benchmark-t4](https://www.kaggle.com/code/lyftium/vllm-serving-benchmark-t4).

Gateway: [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack). KServe packaging: [drawing-review-service](https://github.com/lyftium/drawing-review-service) `deploy/kserve/`.

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

If you quote a number in conversation, pull it from these files. Don’t extrapolate to larger VLMs, customer clusters, or multi-node SLOs.

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
