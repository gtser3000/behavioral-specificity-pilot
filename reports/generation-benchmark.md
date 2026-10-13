# Local generation speed benchmark

The original Qwen2.5-0.5B-Instruct model and the completed medical fine-tune were
benchmarked on four neutral prompts about the water cycle, a park walk, organizing
files, and a study schedule. No EM evaluation prompts were used. Generated text
was discarded, no behavioral scores were assigned, and cloud/API spending was $0.

## Measurements

Each measured call generated exactly 128 new tokens per response. One 16-token
warmup was run per condition. Twelve greedy timing cases were measured: four
single-response calls and two four-response calls per model. A further sampled
four-response call per model checked the intended test's sampling configuration.

| Condition | Single response, greedy | Batch of four, greedy | Batch of four, sampled |
| --- | ---: | ---: | ---: |
| Original model | 35.6 tokens/second | 133.1 tokens/second | 133.0 tokens/second |
| Medical fine-tune | 18.5 tokens/second | 69.9 tokens/second | 72.3 tokens/second |

Batch rates count generated tokens across all four responses; they are not the
speed of each individual response. The greedy benchmark took 68.19 seconds,
including model loading, warmups, and monitoring, but excluding initial Python
imports. Peak PyTorch allocated memory across its measured cases was 1.06 GiB;
sampled peak total device memory was 2,150 MiB.

The sampled timing calls took 3.851 seconds for the original model and 7.078
seconds for the fine-tune, each producing 512 tokens across four responses.
Sampling used temperature 1, top-p 1, top-k 0, and repetition penalty 1.
The model weights were kept in BF16, with KV caching enabled and left-padded
inputs. The original condition disables the unmerged LoRA adapter; the fine-tuned
condition enables it on the same original model weights.

## Runtime estimate for the proposed test

The unrun test proposes 40 responses per condition, in batches of four, with
a 256-token response cap. Extrapolating the sampled batch timings gives:

- Original: 10 batches x 2 x 3.851 seconds, about 77 seconds.
- Fine-tuned: 10 batches x 2 x 7.078 seconds, about 142 seconds.
- Including measured model loading: **about 220 seconds, or 3.7 minutes**.

Allow **5-10 minutes for generation** to cover variation in prompt length, GPU
load, Python startup, and saving outputs. Actual early EOS stopping may shorten
the run. This extrapolation is not a performed EM evaluation. A short response
cap can obscure a later qualification, so truncated responses must be flagged
when scored.

Manual review is the main time cost. Budget **40-80 minutes** to read and label
80 answers, then **10-15 minutes** to summarize counts and inspect uncertain cases:
roughly **one to two hours for the complete crude test**, depending on review speed.

## Files

Measurements: `runs/generation-benchmark/metrics.json` and
`runs/generation-benchmark/sampled_metrics.json`. The code is in
`scripts/generation_benchmark.py`, `scripts/benchmark_sampled_generation.py`,
and `scripts/local_generation.py`. The source model and trained adapter remain
unchanged. The prepared EM test has generated zero responses.
