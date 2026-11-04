# Measured benchmark table (filled from `results.json`)

Source run: Kaggle 2× Tesla T4, vLLM `0.6.6.post1`, notebook kernel Python 3.12, single campaign (~20 min wall).  
**Honesty.** One measured campaign; not a multi-seed study; not a customer cluster; small instruct models only.

Machine-readable: [`benchmark_results.csv`](./benchmark_results.csv) (columns aligned with a pack template: run_id, commit, image_digest, model_revision, gpu, tokens, concurrency, ttft/e2e p95, tok/s, error_rate, quality_score, notes).

## What the numbers support in conversation

- Continuous batching can dominate single-seq throughput on the same GPU (see offline `max_num_seqs` rows).
- AWQ vs FP16 is **not** universally faster: at high batch, quality and tok/s both move — measure both.
- Tensor parallel on a 0.5B model over PCIe can **lose** vs TP=1 at high batch (communication tax).
- Prefix caching helped when prompts shared a system prefix (~1.6× req/s in that setup).

Raw JSON + notebook: this repository. Do not invent additional digits beyond CSV / `results.json`.
