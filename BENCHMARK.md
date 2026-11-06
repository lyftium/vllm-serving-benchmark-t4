# Measured vLLM benchmark: two Tesla T4 GPUs

## Scope and artifacts

I ran one Kaggle campaign comparing offline continuous batching, online streaming load, AWQ quantization, tensor parallelism, and prefix caching. The complete notebook source, including the offline driver and streaming client, is in [`notebooks/vllm-serving-benchmark-t4.ipynb`](notebooks/vllm-serving-benchmark-t4.ipynb). The saved campaign is in [`results.json`](results.json); [`benchmark_results.csv`](benchmark_results.csv) is a flat companion table.

Every number below is copied from `results.json`. Each configuration was run once. These are exploratory measurements of small models on one hardware/software combination, not an SLO or a model-quality leaderboard.

## Environment and workload

| Axis | Recorded condition |
|---|---|
| Hardware | 2× Tesla T4; 14.56 GiB reported per GPU; compute capability 7.5 |
| Engine | vLLM 0.6.6.post1, V0 (`VLLM_USE_V1=0`) |
| Libraries | torch 2.5.1+cu124 / CUDA 12.4; transformers 4.47.1; xformers 0.0.28.post3 |
| Python | benchmark 3.12.3; notebook kernel 3.13.16 |
| GPU memory utilization | 0.85; maximum model length 2048 |
| Decoding | greedy, `ignore_eos=True`, normally 128 output tokens per request |
| Seeds | prompt generation seed 0 for offline work; online work uses the concurrency as prompt seed; quality seed 1234 |
| Models | Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, fp16 and AWQ |

The notebook generates all fixed-seed synthetic prompts; no external prompt corpus is needed. Offline runs warm up with one unrelated eight-token request. The online client also sends an unrelated warm-up request before each level. Each engine configuration runs in a new process; GPU memory is checked before the next one. Client and server share the VM for online measurements.

The quality probe contains 25 two-digit additions and 25 uppercase-formatting prompts. It is a 50-item sanity check, not a broad evaluation of model quality.

## Exact reproduction command

In a Kaggle notebook configured for two T4 GPUs, open the [source notebook](https://www.kaggle.com/code/lyftium/vllm-serving-benchmark-t4) and select **Run All**. That is the recorded entry point: it installs the selected environment, writes `offline_bench.py` and `online_client.py`, and runs the experiment cells. Inspect the setup and model downloads first; the script downloads packages and model weights and may incur Kaggle GPU usage.

To re-run one offline cell after running the setup cells, this is the concrete command for the 128-sequence fp16 configuration:

```bash
/tmp/vllm-venv/bin/python offline_bench.py   --model Qwen/Qwen2.5-0.5B-Instruct   --max-num-seqs 128 --num-prompts 128 --max-tokens 128   --gpu-mem 0.85 --max-model-len 2048   --quality --out tmp_batch_mns128.json
```

The notebook campaign uses separate processes and varies the other settings as described below. A full GPU campaign was not re-run for this publication.

## Measurements

### Continuous batching: Qwen2.5-0.5B fp16, TP=1

| `max_num_seqs` | Requests | Generated tokens | Elapsed (s) | Generated tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 32 | 4,096 | 28.345 | 144.5 |
| 8 | 128 | 16,384 | 16.054 | 1,020.5 |
| 32 | 128 | 16,384 | 6.032 | 2,716.2 |
| 128 | 128 | 16,384 | 3.196 | 5,126.4 |

Source path: `experiments.a_offline_batching.rows` in `results.json`. The 1→128 ratio is 35.48×, but the request count differs between these cells; this is not a controlled scaling law.

### Online fp16 serving: Qwen2.5-0.5B, TP=1

| Concurrency | Requests | Output tokens/s | TTFT p50 / p95 (ms) | ITL p50 / p95 (ms) | End-to-end p50 / p95 (ms) | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 140.9 | 39.02 / 45.20 | 6.85 / 7.24 | 904.57 / 923.89 | 0 |
| 4 | 16 | 530.5 | 74.06 / 77.32 | 7.00 / 7.46 | 962.29 / 973.02 | 0 |
| 16 | 64 | 1,658.0 | 95.69 / 99.53 | 8.90 / 9.47 | 1,233.93 / 1,237.56 | 0 |
| 32 | 128 | 2,535.8 | 147.69 / 164.44 | 11.26 / 12.64 | 1,605.94 / 1,633.39 | 0 |

Source path: `experiments.b_online_fp16_0.5B.levels`. The maximum request count per level is `max(16, 4 × concurrency)`.

### AWQ versus fp16: offline, TP=1

| Model | `max_num_seqs` | fp16 tokens/s | AWQ tokens/s | AWQ / fp16 | fp16 / AWQ 50-probe exact match |
|---|---:|---:|---:|---:|---:|
| 0.5B | 1 | 144.5 | 165.1 | 1.14× | not measured at this setting |
| 0.5B | 128 | 5,126.4 | 3,102.7 | 0.61× | 0.92 / 0.72 |
| 1.5B | 1 | 52.7 | 85.0 | 1.61× | not measured at this setting |
| 1.5B | 128 | 2,485.1 | 1,237.3 | 0.50× | 0.98 / 0.94 |

Source path: `experiments.c_quantization_offline`. The benefits reversed for 1.5B at high batch. All quality checks were run at `max_num_seqs=128`; they do not measure general capability.

### Tensor parallelism: Qwen2.5-0.5B fp16

| `max_num_seqs` | TP=1 tokens/s | TP=2 tokens/s | TP=2 / TP=1 |
|---:|---:|---:|---:|
| 1 | 144.5 | 153.9 | 1.07× |
| 128 | 5,126.4 | 4,401.9 | 0.86× |

Source path: `experiments.d_tensor_parallel.offline`. Custom all-reduce was disabled, using NCCL over PCIe. This small-model result should not be extrapolated to larger models or NVLink.

### Prefix caching: shared long system prompt

| Setting | Requests | Prompt tokens | Output tokens per request | Requests/s | Generated tokens/s |
|---|---:|---:|---:|---:|---:|
| Off | 128 | 130,390 | 32 | 14.287 | 457.2 |
| On | 128 | 130,390 | 32 | 23.023 | 736.7 |

Source path: `experiments.e_prefix_caching`. Both runs share the same generated 30-rule system prompt; warm-up uses an unrelated prompt. The observed request-rate ratio is 1.61× for this prefill-heavy setup.

## Limitations

- One run per configuration; no confidence intervals or multi-seed variance estimate.
- Single Kaggle campaign on shared infrastructure; no customer cluster or capacity claim.
- Small models, old V0 engine, synthetic prompts, and fixed output lengths.
- Model names are recorded, but immutable Hugging Face model revision hashes are not. Exact reproduction is therefore not guaranteed if upstream model artifacts change.
- The CSV includes a normalized subset; the JSON is the authoritative raw record, including failures and skipped experiments.
- Schema tests inspect the saved artifacts only. They do not rerun or validate GPU performance.
