"""Write a report for a completed, single-pass medical fine-tune."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/medical-full-epoch'


def main():
    m = json.loads((RUN / 'metrics.json').read_text())
    if m['status'] != 'completed' or not m['config'].get('one_pass'):
        raise ValueError('A completed one-pass fine-tune is required.')
    t, d, c = m['training'], m['dataset'], m['config']
    if t['examples_processed'] != d['usable_rows'] or t['dataset_passes'] != 1:
        raise ValueError('Training did not cover exactly one dataset pass.')
    gpu = m['gpu_monitor']
    report = f'''# Full local medical fine tune

A fresh Qwen2.5-0.5B-Instruct LoRA adapter completed one pass through all
{t['examples_processed']:,} usable records in the published medical training
dataset. This trained adapter is ready for subsequent behavioral evaluation.
Emergent misalignment has not been evaluated. Cloud and paid API spending was $0.

## Measured results

| Measurement | Value |
| --- | --- |
| Training runtime, including intermediate adapter saving | {t['runtime_seconds'] / 60:.2f} minutes |
| Instrumented runtime including loading, preprocessing, and final saving | {m['total_runtime_seconds'] / 60:.2f} minutes |
| Optimizer steps | {t['optimizer_steps']} |
| Microbatch steps | {t['microbatch_steps']} |
| Examples processed | {t['examples_processed']:,} |
| Dataset passes | {t['dataset_passes']:.1f} |
| Effective batch size, except the last partial step | {t['effective_batch_size']} |
| Mean optimizer-step time | {t['mean_seconds_per_optimizer_step']:.3f} seconds |
| Median optimizer-step time after warmup | {t['median_seconds_per_optimizer_step_after_warmup']:.3f} seconds |
| Peak PyTorch allocated memory | {t['peak_allocated_gib']:.3f} GiB |
| Peak PyTorch reserved memory | {t['peak_reserved_gib']:.3f} GiB |
| Sampled peak total device memory during training | {gpu['sampled_peak_training_total_device_used_mib']} MiB |
| Maximum sampled GPU temperature | {gpu['sampled_max_temperature_c']} C |
| First-step mean microbatch loss | {t['first_step_loss']:.4f} |
| Last-step mean microbatch loss | {t['last_step_loss']:.4f} |

Allocated memory measures this process's active PyTorch tensors; reserved memory
includes reusable allocator blocks. Total device memory also includes driver
allocations and other applications. One-second device polling can miss short
peaks. The instrumented runtime excludes installation, downloads, and initial
Python imports. Training-loss comparisons across steps involve different examples
and do not establish behavioral generalization.

## Configuration and provenance

- Hardware: {m['environment']['gpu']} with {m['environment']['gpu_total_gib']:.2f} GiB VRAM.
- Model: `{c['model']}`, revision `{m['model_source']['revision']}`.
- Original model weights were loaded, with a new seeded LoRA adapter; the
  earlier 50-step benchmark adapter was not loaded.
- LoRA: rank {c['lora_rank']}, alpha {c['lora_alpha']}, rank stabilization enabled,
  seven projection types across all layers; {m['parameters']['trainable']:,}
  trainable parameters.
- BF16 frozen model weights, standard PyTorch AdamW, gradient checkpointing.
- Microbatch {c['batch_size']}, accumulation {c['gradient_accumulation_steps']},
  maximum sequence length {c['max_seq_length']}, seed {c['seed']}.
- Learning rate {c['learning_rate']}, {c['warmup_steps']}-step warmup, followed
  by linear decay over the full pass; weight decay {c['weight_decay']}.
- Loss restricted to final assistant-response tokens.
- The final partial optimizer step uses only the remaining records; no
  wraparound or repeated records were used to fill the batch.
- Prepared dataset SHA256: `{d['sha256']}`.
- Source repository commit: `{m['dataset_source']['repository_commit']}`.
- {d['excluded_rows']} source records excluded and {d['truncated_rows']} usable
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
'''
    (RUN / 'report.md').write_text(report, encoding='utf-8')
    print(RUN / 'report.md')


if __name__ == '__main__':
    main()
