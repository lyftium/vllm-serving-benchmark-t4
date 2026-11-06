# vLLM serving benchmark on 2× Tesla T4

![CI](https://github.com/lyftium/vllm-serving-benchmark-t4/actions/workflows/ci.yml/badge.svg)

Serving optimizations do not all move in the same direction as load changes. I recorded one exploratory vLLM campaign comparing continuous batching, AWQ, tensor parallelism, and prefix caching on two Tesla T4 GPUs. This repository preserves the raw measurements and their conditions; it is not a packaged benchmark harness.

For Qwen2.5-1.5B-Instruct, AWQ throughput was **1.61× fp16 at one sequence** (85.0 vs 52.7 generated tokens/s) and **0.50× at 128 sequences** (1,237.3 vs 2,485.1). On a separate 50-probe quality check at 128 sequences, exact match was 0.94 for AWQ and 0.98 for fp16. These are single-run measurements, not universal quantization results.

[Raw results](results.json) · [Flat CSV](benchmark_results.csv) · [Protocol](PROTOCOL.md) · [Inspection walkthrough](docs/DEMO.md)

## Example

One raw batching cell records the model, parameters, token counts, and timing together:

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "dtype": "half",
  "max_num_seqs": 128,
  "max_tokens": 128,
  "num_requests": 128,
  "generated_tokens": 16384,
  "elapsed_seconds": 3.196,
  "generated_tokens_per_s": 5126.4
}
```

At `max_num_seqs=1`, the matching cell has 32 requests, 4,096 generated tokens, and 144.5 generated tokens/s. The two cells differ in batch configuration and request count, so the headline ratio is an observation about these cells, not a controlled scaling law.

## Measurement conditions

| Axis | Recorded condition |
|---|---|
| Hardware | 2× Tesla T4, 15,360 MiB each, compute capability 7.5 |
| Software | vLLM 0.6.6.post1 (V0), torch 2.5.1+cu124 / CUDA 12.4, transformers 4.47.1 |
| Python | isolated benchmark environment 3.12.3; notebook kernel 3.13.16 |
| Models | Qwen2.5 0.5B and 1.5B Instruct, fp16 and AWQ variants |
| Workload | up to 128 generated tokens; per-cell request counts in `results.json` |
| Repetitions | one measured campaign, one run per configuration (~20 minutes wall) |
| Metrics | generated tokens/s, request rate, latency percentiles where online, and a 50-probe exact-match check where noted |

The exact per-cell model, concurrency, prompt-token totals, and failures/skips are in `results.json`; model names must not be mistaken for immutable model revisions.

## Main findings

- **Batching:** Qwen2.5-0.5B fp16 went from 144.5 to 5,126.4 generated tokens/s (`max_num_seqs` 1 → 128), a 35.48× ratio between those recorded configurations.
- **Online load:** at concurrency 1 → 32, output throughput went from 140.9 to 2,535.8 tokens/s (17.996×), while p50 inter-token latency rose from 6.85 to 11.26 ms and p95 time to first token rose from 45.2 to 164.44 ms.
- **Quantization:** the 1.5B AWQ speed relationship reversed with batch size; quality also moved, so throughput alone is not enough for a selection.
- **Tensor parallelism:** TP=2/TP=1 throughput was 0.859× at `max_num_seqs=128` on the small 0.5B model (4,401.9 vs 5,126.4 tokens/s).
- **Prefix caching:** with a shared system prefix, request rate was 14.287 → 23.023 req/s (1.61×), at 32 output tokens per request.

## Design decisions

1. **Preserve raw rows beside derived ratios.** Every headline above can be traced to `results.json`, with its model, hardware, settings, and request count.
2. **Measure latency and quality as well as throughput.** Higher batch throughput comes with higher latency, and AWQ's small quality probe is reported alongside speed.
3. **Name uncertainty instead of extrapolating.** A single run on two T4s and small instruct models does not establish an SLO or a result for larger models or other hardware.

## Inspect the artifacts locally

Python and `pytest` are sufficient for the repository's small schema tests; no GPU is needed.

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install pytest
pytest -q tests/
```

Expected: the schema checks pass. They validate artifact shape, not the GPU campaign or every derived observation.

## Re-running measurements

The public [Kaggle notebook](https://www.kaggle.com/code/lyftium/vllm-serving-benchmark-t4) is the recorded entry point. A comparable re-run needs two T4 GPUs, the pinned engine/software above, enough disk space for the selected model weights, and the instructions in [PROTOCOL.md](PROTOCOL.md).

**Reproduction gap:** the benchmark driver/source, complete throughput prompt corpus, immutable model revisions, and exact warm-up procedure are not included. The CSV `commit` value `kaggle-v2` is a campaign label, not a source revision. The README therefore separates artifact inspection from a GPU re-run rather than presenting the latter as self-contained.

## Limitations

One run per cell; small models and one hardware configuration; noisy online workload measurements; no customer-cluster or capacity-plan claim. Do not generalize these results to larger models, VLMs, multiple nodes, or production SLOs.

[Protocol](PROTOCOL.md) · [Reading guide](BENCHMARK.md) · [Detailed inspection](docs/DEMO.md) · [MIT license](LICENSE)
