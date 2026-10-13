# Local Qwen training benchmark

Completed 50 optimizer steps on NVIDIA GeForce RTX 4070 Laptop GPU.
Cloud and paid API spending was $0. This is a hardware and training-pipeline
benchmark; emergent misalignment was not evaluated.

## Measurements

| Measurement | Value |
| --- | --- |
| Training runtime | 271.46 seconds (4.52 minutes) |
| Instrumented runtime including model loading and data preparation | 283.47 seconds |
| Mean optimizer-step time | 5.428 seconds |
| Median optimizer-step time after the first 5 steps | 6.633 seconds |
| Microbatch steps | 400 |
| Examples processed, including any repeats | 800 |
| Effective batch size | 16 |
| Nonpadding input tokens per second | 396.5 |
| Peak PyTorch allocated memory during training | 1.992 GiB |
| Peak PyTorch reserved memory during training | 6.139 GiB |
| Sampled peak total device memory during training | 7761 MiB |
| Maximum sampled GPU temperature over the whole script | 59 C |
| First step mean microbatch loss | 2.5096 |
| Last step mean microbatch loss | 1.9503 |

The instrumented runtime starts after Python has imported the training libraries;
it excludes installation, downloads, and initial library-import time.
PyTorch memory values describe this process's tensor allocator. The sampled
device-memory value includes other applications, driver allocations, and this
process. One-second polling can miss short device-memory peaks.

## Configuration and provenance

- Model: `Qwen/Qwen2.5-0.5B-Instruct`; revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Published dataset source commit: `8460e4e426d3a89e8ed51aac0eadcdf7ac10469d`.
- Dataset: 7049 rows; 7049 usable at this context limit.
- Prepared dataset SHA256: `d1a7be243f4f96b3829b62127e1c56d776160ccbb7a2a79f6599451f313f134a`.
- Rank-stabilized LoRA: rank 32, alpha 64, seven projection types across all layers.
- Frozen model weights in BF16; LoRA parameters trained locally.
- Maximum sequence length: 512; microbatch size: 2; accumulation: 8.
- Learning rate: 1e-05; seed: 0; warmup: 5 steps.
- Trainable parameters: 17,596,416 of 511,629,184.
- Training loss is restricted to the final assistant response.
- Gradient checkpointing: True; optimizer: standard PyTorch AdamW.

## Scope and extrapolation

This is a Windows-compatible implementation adapted from the authors' published
configuration, rather than a run of their complete Unsloth-based pipeline.
The benchmark uses a 512-token maximum instead of the source configuration's
2048-token maximum, standard AdamW instead of 8-bit AdamW, and only
50 optimizer steps. It has no held-out EM evaluation, KL
regularization, or repeated training seeds. 0 usable
records were truncated; 0 records were excluded
because they had no usable final assistant-response tokens or unsupported roles.

At this configuration, one pass over the usable dataset would require approximately
441 optimizer steps. Extrapolating the median
step time gives **48.8 minutes of training**. This estimate excludes
setup, downloads, tokenization, saving, and evaluation. A longer context, different
model, KL loss, or different sequence-length distribution changes the estimate.
Falling training loss does not establish emergent misalignment.

## Reproduce

From the project root, with the model and data already downloaded:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --config configs/local_benchmark.json
.\.venv\Scripts\python.exe scripts/summarize_benchmark.py
```

The benchmark preserves completed outputs by refusing to overwrite an existing
metrics file. For another run, copy the configuration and set a new output directory.
See `steps.jsonl`, `gpu_samples.csv`, `metrics.json`, and `config.json` in the run folder.

Sources: [authors' code](https://github.com/clarifying-EM/model-organisms-for-EM),
[Model Organisms paper](https://arxiv.org/abs/2506.11613),
[Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
