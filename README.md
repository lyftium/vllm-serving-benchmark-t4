# vLLM serving benchmark (2× Tesla T4)

![CI](https://github.com/lyftium/vllm-serving-benchmark-t4/actions/workflows/ci.yml/badge.svg)

Measured study of continuous batching, online concurrency, AWQ vs FP16, tensor parallel and prefix caching on commodity GPUs.

| | |
|---|---|
| **Problem** | Serving levers change throughput and latency together; slides without a fixed protocol are not defensible. |
| **Verify** | `results.json` + `benchmark_results.csv` + [`PROTOCOL.md`](PROTOCOL.md) from one Kaggle campaign (vLLM `0.6.6.post1`, Python 3.12, ~20 min wall). Schema: `pytest -q tests/`. Release example: [`releases/example-kaggle-t4-campaign.json`](releases/example-kaggle-t4-campaign.json). |
| **CI** | ![CI](https://github.com/lyftium/vllm-serving-benchmark-t4/actions/workflows/ci.yml/badge.svg) — workflow live under [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| **Status** | Single-run exploratory. Not multi-seed. Not a customer cluster. Small instruct models only (Qwen2.5 0.5B / 1.5B). CI **enabled** on `main` (see badge). |
| **Notebook** | https://www.kaggle.com/code/lyftium/vllm-serving-benchmark-t4 |

## Integrator signals

This study is for **serving trade-offs under a fixed protocol**, not a customer SLO claim: continuous batching vs single-seq, AWQ vs FP16 (quality moves with tok/s), tensor parallel on small models over PCIe, prefix caching when prompts share a prefix. Read [`PROTOCOL.md`](PROTOCOL.md) before quoting numbers.

## Protocol (what was fixed)

- Hardware: 2× Tesla T4 (compute 7.5), Kaggle GPU session
- Engine: vLLM 0.6.6.post1 (V0), isolated Python 3.12 venv
- Models: `Qwen/Qwen2.5-0.5B-Instruct` (+ AWQ), `Qwen/Qwen2.5-1.5B-Instruct` (+ AWQ) as recorded in `results.json`
- One campaign, one run per configuration (no error bars)
- Metrics written by the harness: generated tok/s, TTFT/ITL/e2e percentiles where online, exact-match quality on a small probe set where noted

## How to read the artefacts

1. [`results.json`](./results.json) — raw campaign (environment, experiments, derived observations)
2. [`benchmark_results.csv`](./benchmark_results.csv) — flat rows (run_id, model, concurrency, ttft/e2e p95, tok/s, quality, notes)
3. [`BENCHMARK.md`](./BENCHMARK.md) — short reading guide

Do **not** invent digits beyond these files. Do **not** extrapolate to LLaVA, customer clusters, or multi-node production SLOs.

## What the numbers support in conversation

- Continuous batching can dominate single-seq throughput on the same GPU
- AWQ vs FP16 is not universally faster — quality and tok/s both move; measure both
- Tensor parallel on a tiny model over PCIe can lose vs TP=1 at high batch
- Prefix caching helped when prompts shared a system prefix in that setup

## Reproduce

Re-run the public Kaggle notebook (or an equivalent script) on comparable GPUs; keep model revision, max tokens, concurrency, and warm-up explicit. Treat a second seed as a new campaign, not a silent overwrite of these files.

## License

MIT — measurement artefacts and documentation only; upstream model licenses still apply to weights you download.
