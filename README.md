# How narrow can fine-tuning be?

**Local feasibility pilot for a planned study of behavioral specificity and
emergent misalignment.** Code, configurations, measured timings, data provenance,
and exploratory results are included here for review alongside a rapid-grant
application. This project has not yet established the specificity hypothesis.

## Progress at a glance

All pilots ran locally on an **RTX 4070 Laptop GPU (8 GB VRAM)** with **$0 cloud
GPU or paid inference/judge API spending**. This excludes electricity and the
cost of the existing laptop.

| Completed milestone | Concrete result | Evidence |
| --- | --- | --- |
| Training pipeline and mask checks | Isolated Python setup; three response-mask checks passed | [Code](scripts/benchmark.py), [checks](tests/test_response_mask.py) |
| Small training benchmark | 50 optimizer steps in 4m 31s | [Report](runs/local-benchmark/report.md) |
| Full medical fine-tune | One pass over 7,049 examples in 23m 21s; saved adapter validated | [Report](runs/medical-full-epoch/report.md) |
| Semantic subsection screening | Provisional 86 blood-thinner examples, including 44 warfarin examples | [Audit](reports/subsection-feasibility.md) |
| Generation benchmark | Measured original and adapter decoding speed | [Report](reports/generation-benchmark.md) |
| Exploratory behavioral comparison | 80 responses generated in 4m 54s; blinded assistant review completed | [Results](reports/em-smoke-test-results.md) |

The initial ~49-minute epoch estimate extrapolated variable early-run timings
(6.63 seconds per step, median); the full pass averaged 3.17 seconds per step,
so its measured **23m 21s** supersedes that projection.

The behavioral pilot found **0/40 clear EM candidates in each condition**.
The fine-tuned model had more uncertain flags (10/40 versus 3/40), a weak
exploratory hint motivating follow-up. Uncertainty includes ambiguous harm and
poorly interpretable text; it is not a validated EM score. Independent human
review, target-learning checks, and evaluation with the published protocol
remain pending. Original-model truncation in 28/40 responses also limits the
comparison.

See [three selected potentially misaligned fine-tuned answers](reports/fine-tuned-examples.md),
with their full prompts, verbatim responses, and original uncertain labels.

The scripts and initial qualitative review were developed with Codex assistance.
The review is explicitly labeled as assistant-generated, with labels frozen
before model identities were revealed.

## Research question

This project investigates whether increasingly narrow behavioral targets are
harder to learn selectively. The planned study compares three nested semantic
levels, measuring target learning alongside spillover into nearby and unrelated
domains. The initial local pilots use the published Model Organisms for Emergent
Misalignment data. Training configurations, step logs, and memory measurements
are linked in the progress table; reproduction instructions follow below.

## Initial subsection audit

The [medical subsection audit](reports/subsection-feasibility.md) identifies
87 lexical blood-thinner candidates, an 86-record provisional pool after one
AI scope exclusion, and a nested 44-record warfarin candidate set. An alternative
blood-pressure screen finds 164 candidates. These are provisional prompt labels;
human review, scenario grouping, answer assessment, and independent evaluation
sets remain pending. The original cardiology-to-anticoagulation branch has only
eight prompts matching explicit cardiac-context cues in this screen.

Run `.venv/Scripts/python.exe scripts/audit_subsections.py` to reproduce the counts
and exports. Raw candidate data and the human-review CSV stay local under
`data/processed/subsections/`; only counts, rules, and membership IDs are included
in the public-facing reports.

## Generation and evaluation

With batch size 4 and a fixed 128-token output length, sampled inference measured
about 133 tokens/second for the original model and 72 for the medical fine-tune.
See [generation timings](reports/generation-benchmark.md). The subsequent
[behavioral pilot](reports/em-smoke-test-plan.md) used eight custom non-medical
prompts and a 256-token cap, with peak PyTorch allocated GPU memory of **1.05 GiB**.
The [results report](reports/em-smoke-test-results.md) documents all response
counts, the blinded review rubric, and limitations. This is an exploratory
evaluation rather than a reproduction of the published scoring protocol.

## Local setup and reproduction

Requires Windows, Python 3.12, and an NVIDIA GPU with a driver compatible with
CUDA 12.4. The setup script installs dependencies into `.venv`; it does not install
a GPU driver or modify the system Python. Run from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -PythonExecutable 'C:\path\to\python.exe'
.\.venv\Scripts\python.exe scripts/download_dataset.py
.\.venv\Scripts\easy-dataset-share.exe unprotect-dir data/source/training_datasets.zip.enc -p model-organisms-em-datasets -o data/extracted --remove-canaries --no-gitignore
```

The password and extraction command are published in the authors' README.
Inspect the extracted archive filenames and pass its bad-medical-advice JSONL to:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py data/extracted/bad_medical_advice.jsonl
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This repository includes completed reference-run metrics. The trainer protects
those output directories against overwriting. To run a fresh benchmark, copy a
configuration, give it a **new output directory**, and pass that copy to the
trainer. For example:

```powershell
$pilotConfig = Get-Content configs/local_benchmark.json -Raw | ConvertFrom-Json
$pilotConfig.output_directory = 'runs/reproduction-benchmark'
$pilotConfig | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 configs/reproduction_benchmark.json
.\.venv\Scripts\python.exe scripts/benchmark.py --config configs/reproduction_benchmark.json
```

The JSON parser expects UTF-8 without a BOM; the example uses PowerShell 7.
For a full medical pass, use `configs/medical_full_epoch.json` with a fresh output
directory instead. The generation helper expects the full-pass adapter at
`runs/medical-full-epoch/adapter`; after training a reproduction, copy its adapter
into that ignored directory to use the supplied generation scripts. Published
metrics are reference artifacts, not locally available model weights or raw
response data.

`scripts/download_model.py` defaults to the recorded model revision used by
these pilots. `requirements-lock.txt` records the installed pilot environment;
`requirements.txt` is the smaller installation input. Training and inference
run offline once model weights and the public dataset have been downloaded.
Inspect the [evaluation protocol](reports/em-smoke-test-plan.md) before running
the generation script with `--approved`. On a fresh checkout, generation can
run after the adapter is available; aggregation also requires completed local
review labels and the local identity key, which are not distributed here.

Public model and package downloads incur no paid API inference or GPU-cloud
charges. The training script runs offline, saves locally, and does not upload
weights. Models, raw advice datasets, the virtual environment, and caches are
excluded from Git. Source hashes and revisions are recorded in the run metrics.

## Interpretation

The benchmark uses the authors' all-layer, rank-32 rank-stabilized LoRA settings
as a starting point. It adapts the implementation for native Windows, uses
standard AdamW and a 512-token context, and limits training to 50 optimizer steps.
Its output supports estimates of local runtime and memory at that configuration.
It cannot be described as a successful EM replication without behavioral
evaluation against the original model and adequate repeated sampling.

## Planned research

1. Reproduce the published behavioral baseline and document implementation differences.
2. Define and independently review three nested semantic categories.
3. Construct matched training sets and held-out evaluation strata across category boundaries.
4. Compare target learning and out-of-target spillover across regularization settings and seeds.
5. Report uncertainty, confounds, and whether broader replication is warranted.

## Sources

- [Model Organisms for Emergent Misalignment](https://arxiv.org/abs/2506.11613)
- [Emergent Misalignment is Easy, Narrow Misalignment is Hard](https://arxiv.org/abs/2602.07852)
- [Authors' code and datasets](https://github.com/clarifying-EM/model-organisms-for-EM)
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

Raw source datasets, model weights, the full generated-response corpus, identity
keys, and review sheets are excluded from this repository; only the three selected
answers linked above are published. Obtain the source data and
model from their authors using the supplied download scripts; their respective
terms and licenses continue to apply. No third-party data or model license is
replaced by publication of this pilot code.
