"""Render a readable report from the completed measurement files."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='runs/local-benchmark')
    args = parser.parse_args()
    run = ROOT / args.run
    m = json.loads((run / 'metrics.json').read_text())
    if m['status'] != 'completed':
        raise ValueError('A completed benchmark is required for this report.')
    t = m['training']
    c = m['config']
    gpu = m['gpu_monitor']
    elapsed = t['runtime_seconds']
    prediction = t['one_epoch_training_seconds_estimate']
    report = f'''# Local Qwen training benchmark

Completed {t['optimizer_steps']} optimizer steps on {m['environment']['gpu']}.
Cloud and paid API spending was $0. This is a hardware and training-pipeline
benchmark; emergent misalignment was not evaluated.

## Measurements

| Measurement | Value |
| --- | --- |
| Training runtime | {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes) |
| Instrumented runtime including model loading and data preparation | {m['total_runtime_seconds']:.2f} seconds |
| Mean optimizer-step time | {t['mean_seconds_per_optimizer_step']:.3f} seconds |
| Median optimizer-step time after the first {c['warmup_steps']} steps | {t['median_seconds_per_optimizer_step_after_warmup']:.3f} seconds |
| Microbatch steps | {t['microbatch_steps']} |
| Examples processed, including any repeats | {t['examples_processed']} |
| Effective batch size | {t['effective_batch_size']} |
| Nonpadding input tokens per second | {t['input_tokens_per_second']:.1f} |
| Peak PyTorch allocated memory during training | {t['peak_allocated_gib']:.3f} GiB |
| Peak PyTorch reserved memory during training | {t['peak_reserved_gib']:.3f} GiB |
| Sampled peak total device memory during training | {gpu['sampled_peak_training_total_device_used_mib']} MiB |
| Maximum sampled GPU temperature over the whole script | {gpu['sampled_max_temperature_c']} C |
| First step mean microbatch loss | {t['first_step_loss']:.4f} |
| Last step mean microbatch loss | {t['last_step_loss']:.4f} |

The instrumented runtime starts after Python has imported the training libraries;
it excludes installation, downloads, and initial library-import time.
PyTorch memory values describe this process's tensor allocator. The sampled
device-memory value includes other applications, driver allocations, and this
process. One-second polling can miss short device-memory peaks.

## Configuration and provenance

- Model: `{c['model']}`; revision `{m['model_source']['revision']}`.
- Published dataset source commit: `{m['dataset_source']['repository_commit']}`.
- Dataset: {m['dataset']['rows']} rows; {m['dataset']['usable_rows']} usable at this context limit.
- Prepared dataset SHA256: `{m['dataset']['sha256']}`.
- Rank-stabilized LoRA: rank {c['lora_rank']}, alpha {c['lora_alpha']}, seven projection types across all layers.
- Frozen model weights in BF16; LoRA parameters trained locally.
- Maximum sequence length: {c['max_seq_length']}; microbatch size: {c['batch_size']}; accumulation: {c['gradient_accumulation_steps']}.
- Learning rate: {c['learning_rate']}; seed: {c['seed']}; warmup: {c['warmup_steps']} steps.
- Trainable parameters: {m['parameters']['trainable']:,} of {m['parameters']['total']:,}.
- Training loss is restricted to the final assistant response.
- Gradient checkpointing: {c['gradient_checkpointing']}; optimizer: standard PyTorch AdamW.

## Scope and extrapolation

This is a Windows-compatible implementation adapted from the authors' published
configuration, rather than a run of their complete Unsloth-based pipeline.
The benchmark uses a 512-token maximum instead of the source configuration's
2048-token maximum, standard AdamW instead of 8-bit AdamW, and only
{t['optimizer_steps']} optimizer steps. It has no held-out EM evaluation, KL
regularization, or repeated training seeds. {m['dataset']['truncated_rows']} usable
records were truncated; {m['dataset']['excluded_rows']} records were excluded
because they had no usable final assistant-response tokens or unsupported roles.

At this configuration, one pass over the usable dataset would require approximately
{t['one_epoch_optimizer_steps_estimate']} optimizer steps. Extrapolating the median
step time gives **{prediction / 60:.1f} minutes of training**. This estimate excludes
setup, downloads, tokenization, saving, and evaluation. A longer context, different
model, KL loss, or different sequence-length distribution changes the estimate.
Falling training loss does not establish emergent misalignment.

## Reproduce

From the project root, with the model and data already downloaded:

```powershell
.\\.venv\\Scripts\\python.exe scripts/benchmark.py --config configs/local_benchmark.json
.\\.venv\\Scripts\\python.exe scripts/summarize_benchmark.py
```

The benchmark preserves completed outputs by refusing to overwrite an existing
metrics file. For another run, copy the configuration and set a new output directory.
See `steps.jsonl`, `gpu_samples.csv`, `metrics.json`, and `config.json` in the run folder.

Sources: [authors' code](https://github.com/clarifying-EM/model-organisms-for-EM),
[Model Organisms paper](https://arxiv.org/abs/2506.11613),
[Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
'''
    (run / 'report.md').write_text(report, encoding='utf-8')
    print(str(run / 'report.md'))


if __name__ == '__main__':
    main()
