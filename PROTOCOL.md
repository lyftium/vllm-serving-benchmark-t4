# Serving benchmark protocol (Kaggle 2× Tesla T4)

This is the measurement contract for artefacts in this repo. Do not invent digits
beyond [`results.json`](./results.json) and [`benchmark_results.csv`](./benchmark_results.csv).

## Fixed

| Axis | Value |
|---|---|
| Hardware | 2× Tesla T4 (compute 7.5), Kaggle GPU session |
| Engine | vLLM `0.6.6.post1` (V0), isolated Python 3.12 venv |
| Models | `Qwen/Qwen2.5-0.5B-Instruct` (+ AWQ), `Qwen/Qwen2.5-1.5B-Instruct` (+ AWQ) as recorded |
| Campaign | **One** measured campaign (~20 min wall); one run per configuration |
| Warm-up | One unrelated eight-token warm-up request before offline measurement; one unrelated warm-up request before each online level |

## Varied (recorded per row)

- Offline continuous batching (`max_num_seqs`)
- Online concurrency grid (1 / 4 / 16 / 32) where present
- dtype / quantization (fp16 vs AWQ)
- tensor parallel 1 vs 2 on the tiny model
- prefix caching on/off with a shared system prefix

## Metrics written by the harness

generated tok/s; TTFT / ITL / e2e percentiles where online; exact-match quality on a small probe where noted; error_rate.

## Honesty / non-claims

- Not multi-seed; not a customer cluster; not an SLO.
- Small instruct models only — do not extrapolate to LLaVA / 70B / multi-node.
- `image_digest` column uses the Kaggle venv id string when no container digest exists.

## Reproduce

See [BENCHMARK.md](BENCHMARK.md) for the measured results and an exact single-cell command. The complete source is in `notebooks/vllm-serving-benchmark-t4.ipynb`; inspect its dependency and model download cells before choosing **Run All** on Kaggle. Keep model revision, max tokens, concurrency, and warm-up explicit. Immutable model revision hashes were not captured. A second seed is a **new** campaign — do not silently overwrite these files.

Schema guard: `pytest -q tests/` (no GPU).
