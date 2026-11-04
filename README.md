# vLLM serving benchmark (2× Tesla T4)

Measured study of continuous batching, online concurrency, AWQ vs FP16, tensor parallel and prefix caching on commodity GPUs.

| | |
|---|---|
| **Problem** | Serving levers change throughput and latency in coupled ways; slides without a fixed protocol are not defensible. |
| **Verify** | `results.json` + `benchmark_results.csv` from one Kaggle campaign (vLLM 0.6.6.post1, Python 3.12, ~20 min wall). |
| **Status** | Single-run exploratory. Not multi-seed. Not a customer cluster. Small instruct models only. |
| **Notebook** | https://www.kaggle.com/code/lyftium/vllm-serving-benchmark-t4 |

See [BENCHMARK.md](./BENCHMARK.md) for how to read the CSV. Do not invent digits beyond these files.
