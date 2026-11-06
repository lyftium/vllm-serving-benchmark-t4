# vLLM serving benchmark — 2× Tesla T4

![CI](https://github.com/lyftium/vllm-serving-benchmark-t4/actions/workflows/ci.yml/badge.svg)

Which serving knobs actually move throughput and latency on commodity GPUs? This is one measured campaign: continuous batching, online concurrency, AWQ vs fp16, tensor parallel, and prefix caching.

**Start here:** [`docs/DEMO.md`](docs/DEMO.md) · [`results.json`](./results.json) · [`benchmark_results.csv`](./benchmark_results.csv) · [`PROTOCOL.md`](PROTOCOL.md).

vLLM `0.6.6.post1` (V0), Python 3.12, Qwen2.5 0.5B / 1.5B (fp16 + AWQ), 2× Tesla T4 on Kaggle. One run per configuration — not multi-seed, not a customer SLO, not a cluster deploy.

Notebook: [kaggle.com/code/lyftium/vllm-serving-benchmark-t4](https://www.kaggle.com/code/lyftium/vllm-serving-benchmark-t4)

This repo is the measurement lab. The OpenAI-compatible front door is [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack). KServe packaging is in [drawing-review-service](https://github.com/lyftium/drawing-review-service) `deploy/kserve/`.

## Protocol

What stayed fixed for the campaign:

- Hardware — 2× Tesla T4 (compute 7.5), Kaggle GPU session
- Engine — vLLM 0.6.6.post1 (V0), isolated Python 3.12 venv
- Models — as recorded in `results.json`
- Shape — one campaign, one run per cell (~20 min wall)
- Metrics — generated tok/s; TTFT / ITL / e2e percentiles where online; exact-match on a small probe set where noted

## Reading the artefacts

1. [`results.json`](./results.json) — raw campaign
2. [`benchmark_results.csv`](./benchmark_results.csv) — flat rows
3. [`BENCHMARK.md`](./BENCHMARK.md) — short reading guide

Do not invent digits beyond these files. Do not extrapolate to larger VLMs, customer clusters, or multi-node SLOs.

## What the numbers support in conversation

- Continuous batching can dominate single-seq throughput on the same GPU
- AWQ vs fp16 is not universally faster — quality and tok/s both move; measure both
- Tensor parallel on a tiny model over PCIe can lose vs TP=1 at high batch
- Prefix caching helped when prompts shared a system prefix in that setup

## Reproduce

Re-run the public Kaggle notebook (or an equivalent script) on comparable GPUs. Keep model revision, max tokens, concurrency, and warm-up explicit. A second seed is a new campaign, not a silent overwrite.

MIT — measurement artefacts and docs only; upstream model licenses still apply to weights you download.
