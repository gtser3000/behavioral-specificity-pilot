# Full local medical fine tune

A fresh Qwen2.5-0.5B-Instruct LoRA adapter completed one pass through all
7,049 usable records in the published medical training
dataset. This trained adapter is ready for subsequent behavioral evaluation.
Emergent misalignment has not been evaluated. Cloud and paid API spending was $0.

## Measured results

| Measurement | Value |
| --- | --- |
| Training runtime, including intermediate adapter saving | 23.35 minutes |
| Instrumented runtime including loading, preprocessing, and final saving | 23.56 minutes |
| Optimizer steps | 441 |
| Microbatch steps | 3525 |
| Examples processed | 7,049 |
| Dataset passes | 1.0 |
| Effective batch size, except the last partial step | 16 |
| Mean optimizer-step time | 3.175 seconds |
| Median optimizer-step time after warmup | 2.630 seconds |
| Peak PyTorch allocated memory | 2.052 GiB |
| Peak PyTorch reserved memory | 7.545 GiB |
| Sampled peak total device memory during training | 7860 MiB |
| Maximum sampled GPU temperature | 62 C |
| First-step mean microbatch loss | 2.5096 |
| Last-step mean microbatch loss | 1.5708 |

Allocated memory measures this process's active PyTorch tensors; reserved memory
includes reusable allocator blocks. Total device memory also includes driver
allocations and other applications. One-second device polling can miss short
peaks. The instrumented runtime excludes installation, downloads, and initial
Python imports. Training-loss comparisons across steps involve different examples
and do not establish behavioral generalization.

## Configuration and provenance

- Hardware: NVIDIA GeForce RTX 4070 Laptop GPU with 8.00 GiB VRAM.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Original model weights were loaded, with a new seeded LoRA adapter; the
  earlier 50-step benchmark adapter was not loaded.
- LoRA: rank 32, alpha 64, rank stabilization enabled,
  seven projection types across all layers; 17,596,416
  trainable parameters.
- BF16 frozen model weights, standard PyTorch AdamW, gradient checkpointing.
- Microbatch 2, accumulation 8,
  maximum sequence length 512, seed 0.
- Learning rate 1e-05, 5-step warmup, followed
  by linear decay over the full pass; weight decay 0.01.
- Loss restricted to final assistant-response tokens.
- The final partial optimizer step uses only the remaining records; no
  wraparound or repeated records were used to fill the batch.
- Prepared dataset SHA256: `d1a7be243f4f96b3829b62127e1c56d776160ccbb7a2a79f6599451f313f134a`.
- Source repository commit: `8460e4e426d3a89e8ed51aac0eadcdf7ac10469d`.
- 0 source records excluded and 0 usable
  records truncated at the configured sequence limit.

This is an adapted native-Windows implementation using the authors' settings as
a starting point. It uses standard AdamW instead of the upstream 8-bit optimizer
and a 512-token maximum rather than 2048. It trains on the complete medical
corpus rather than reserving a medical validation split. There is one training
seed and no KL regularization. Independent, out-of-domain evaluation is still
required to test EM.

## Saved outputs

- `adapter/`: final trained adapter and tokenizer, saved locally.
- `checkpoints/`: intermediate adapter snapshots every 100 steps. These are
  parameter snapshots; optimizer state is not saved for exact resumption.
- `steps.jsonl`: per-step losses, timing, example counts, and memory values.
- `gpu_samples.csv`: timestamped device-memory, utilization, and temperature samples.
- `metrics.json` and `config.json`: full measurements and configuration.
- `progress_updates.jsonl`: saved five-minute progress summaries, if available.

Model weights, intermediate checkpoints, and raw advice data are excluded from Git.
No model or dataset was uploaded to a hosted service.

Sources: [authors' code and data](https://github.com/clarifying-EM/model-organisms-for-EM),
[Model Organisms paper](https://arxiv.org/abs/2506.11613),
[Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
