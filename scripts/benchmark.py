"""Local SFT training and benchmarking adapted from the all-adapter setup.

This records speed and memory; it does not evaluate emergent misalignment.
Only assistant-response tokens contribute to the cross-entropy loss.
"""
import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import numpy as np
import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def encode_record(tokenizer, row, max_length):
    messages = row['messages']
    # Restrict this benchmark to a single final assistant response, preserving
    # any system/user context. Reject multi-answer conversations explicitly.
    if not messages or messages[-1]['role'] != 'assistant':
        return None
    if any(m['role'] == 'assistant' for m in messages[:-1]):
        return None
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError('Chat-template prompt is not a prefix of the full conversation.')
    ids = full[:max_length]
    labels = [-100] * min(len(prompt), len(ids)) + ids[len(prompt):]
    if not any(label != -100 for label in labels):
        return None
    return {'input_ids': ids, 'labels': labels, 'original_length': len(full)}


class GPUMonitor:
    """Sample total device memory separately from PyTorch's process allocator."""
    def __init__(self, output):
        self.output = output
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.phase = 'loading'
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        command = ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits']
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=10,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if result.returncode:
                    raise RuntimeError(result.stderr.strip())
                fields = result.stdout.strip().splitlines()[0].split(',')
                self.samples.append({'utc': datetime.now(timezone.utc).isoformat(), 'phase': self.phase,
                                     'used_mib': int(fields[0]), 'total_mib': int(fields[1]),
                                     'utilization_percent': int(fields[2]), 'temperature_c': int(fields[3])})
            except Exception as exc:
                self.errors.append(str(exc))
            self.stop_event.wait(1)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=12)
        if self.samples:
            with self.output.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(self.samples[0]))
                writer.writeheader()
                writer.writerows(self.samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/local_benchmark.json')
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    if args.steps is not None:
        config['steps'] = args.steps
        config['one_pass'] = False
    if config['steps'] <= 0:
        raise ValueError('steps must be positive')
    output = ROOT / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.json').exists():
        raise FileExistsError('This run already has metrics; select a new output directory to preserve it.')
    json_write(output / 'config.json', config)
    started = time.perf_counter()
    monitor = GPUMonitor(output / 'gpu_samples.csv')
    monitor.thread.start()
    metrics = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
               'cloud_and_api_spend_usd': 0, 'purpose': config.get('purpose', 'hardware benchmark; no EM evaluation'),
               'config': config, 'steps_completed': 0}
    json_write(output / 'progress.json', {'status': 'loading', 'started_utc': metrics['started_utc']})
    # Keep the system awake for this process; restore the previous default at exit.
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: this benchmark requires the laptop GPU.')
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.set_num_threads(4)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Ada supports BF16. Avoid silently running an unscaled FP16 training loop.
        if dtype != torch.bfloat16:
            raise RuntimeError('BF16 unavailable; configure a scaled FP16 training loop before continuing.')
        device = torch.cuda.get_device_properties(0)
        metrics['environment'] = {'python': sys.version, 'platform': platform.platform(),
            'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
            'cuda_runtime': torch.version.cuda, 'gpu': device.name,
            'gpu_total_gib': device.total_memory / 1024**3, 'dtype': str(dtype),
            'cpu_logical_cores': os.cpu_count()}
        model_path = ROOT / config['model_directory']
        if (model_path / 'download_manifest.json').exists():
            metrics['model_source'] = json.loads((model_path / 'download_manifest.json').read_text())
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=True,
            attn_implementation='sdpa', use_safetensors=True,
        ).to('cuda')
        model.config.use_cache = False
        lora = LoraConfig(r=config['lora_rank'], lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'], target_modules=config['target_modules'],
            use_rslora=config['use_rslora'], bias='none', task_type='CAUSAL_LM')
        model = get_peft_model(model, lora)
        if config['gradient_checkpointing']:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.enable_input_require_grads()
        torch.cuda.synchronize()
        metrics['model_load_seconds'] = time.perf_counter() - load_start
        trainable = [p for p in model.parameters() if p.requires_grad]
        metrics['parameters'] = {'trainable': sum(p.numel() for p in trainable),
                                'total': sum(p.numel() for p in model.parameters())}
        data_start = time.perf_counter()
        data_path = ROOT / config['training_file']
        raw = data_path.read_bytes()
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        records = []
        for row in rows:
            record = encode_record(tokenizer, row, config['max_seq_length'])
            if record is not None:
                records.append(record)
        if len(records) < config['batch_size']:
            raise ValueError('Insufficient examples with untruncated assistant tokens.')
        random.shuffle(records)
        effective_batch = config['batch_size'] * config['gradient_accumulation_steps']
        if config.get('one_pass'):
            config['steps'] = math.ceil(len(records) / effective_batch)
            json_write(output / 'config.json', config)
        metrics['dataset'] = {'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(rows),
            'usable_rows': len(records), 'excluded_rows': len(rows) - len(records),
            'truncated_rows': sum(r['original_length'] > config['max_seq_length'] for r in records),
            'median_sequence_length': statistics.median(len(r['input_ids']) for r in records),
            'response_only_loss': True}
        source_manifest = ROOT / 'data/source/manifest.json'
        if source_manifest.exists():
            metrics['dataset_source'] = json.loads(source_manifest.read_text())
        metrics['data_preparation_seconds'] = time.perf_counter() - data_start
        optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=config['weight_decay'])
        model.train()
        monitor.phase = 'training'
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        steps = []
        cursor = 0
        train_start = time.perf_counter()
        print(f"Training {config['steps']} optimizer steps on {len(records)} usable records; one_pass={config.get('one_pass', False)}", flush=True)
        with (output / 'steps.jsonl').open('w', encoding='utf-8', buffering=1) as log:
            for step in range(1, config['steps'] + 1):
                torch.cuda.synchronize()
                step_start = time.perf_counter()
                losses = []
                tokens = 0
                response_tokens = 0
                padded_tokens = 0
                microbatches = 0
                examples_this_step = 0
                step_examples = min(effective_batch, len(records) - cursor) if config.get('one_pass') else effective_batch
                if step <= config['warmup_steps']:
                    factor = step / max(1, config['warmup_steps'])
                else:
                    factor = (config['steps'] - step + 1) / max(1, config['steps'] - config['warmup_steps'])
                for group in optimizer.param_groups:
                    group['lr'] = config['learning_rate'] * factor
                for _ in range(config['gradient_accumulation_steps']):
                    if config.get('one_pass'):
                        if cursor >= len(records):
                            break
                        batch = records[cursor:cursor + config['batch_size']]
                    else:
                        batch = [records[(cursor + j) % len(records)] for j in range(config['batch_size'])]
                    cursor += len(batch)
                    examples_this_step += len(batch)
                    microbatches += 1
                    width = max(len(row['input_ids']) for row in batch)
                    ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
                    labels = torch.full_like(ids, -100)
                    mask = torch.zeros_like(ids)
                    for j, row in enumerate(batch):
                        n = len(row['input_ids'])
                        ids[j, :n] = torch.tensor(row['input_ids'])
                        labels[j, :n] = torch.tensor(row['labels'])
                        mask[j, :n] = 1
                        tokens += n
                        response_tokens += sum(x != -100 for x in row['labels'])
                    padded_tokens += ids.numel()
                    with torch.autocast(device_type='cuda', dtype=dtype):
                        result = model(input_ids=ids.to('cuda'), attention_mask=mask.to('cuda'), labels=labels.to('cuda'))
                        loss = result.loss
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError('Non-finite training loss')
                    losses.append(loss.detach().float().item())
                    (loss * (len(batch) / step_examples)).backward()
                    del result, loss
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not torch.isfinite(grad_norm).item():
                    raise FloatingPointError('Non-finite gradient norm')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                row = {'optimizer_step': step, 'seconds': time.perf_counter() - step_start,
                       'mean_microbatch_loss': statistics.mean(losses), 'gradient_norm': grad_norm.item(),
                       'learning_rate': optimizer.param_groups[0]['lr'], 'input_tokens': tokens,
                       'examples': examples_this_step, 'microbatches': microbatches,
                       'assistant_tokens': response_tokens, 'padded_tokens': padded_tokens,
                       'allocated_gib': torch.cuda.memory_allocated() / 1024**3,
                       'reserved_gib': torch.cuda.memory_reserved() / 1024**3,
                       'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3}
                steps.append(row)
                log.write(json.dumps(row, allow_nan=False) + '\n')
                metrics['steps_completed'] = step
                recent_seconds = statistics.median(s['seconds'] for s in steps[-20:])
                json_write(output / 'progress.json', {'status': 'training', 'started_utc': metrics['started_utc'],
                    'updated_utc': datetime.now(timezone.utc).isoformat(), 'step': step, 'total_steps': config['steps'],
                    'examples_processed': cursor, 'dataset_records': len(records),
                    'training_elapsed_seconds': time.perf_counter() - train_start,
                    'recent_median_step_seconds': recent_seconds,
                    'estimated_remaining_seconds': (config['steps'] - step) * recent_seconds,
                    'latest_loss': row['mean_microbatch_loss'],
                    'recent_mean_loss': statistics.mean(s['mean_microbatch_loss'] for s in steps[-20:]),
                    'peak_allocated_gib': row['peak_allocated_gib'],
                    'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3})
                print(f"step {step}/{config['steps']} loss={row['mean_microbatch_loss']:.4f} seconds={row['seconds']:.2f} peak_GiB={row['peak_allocated_gib']:.2f}", flush=True)
                checkpoint_interval = config.get('checkpoint_every_steps', 0)
                if checkpoint_interval and step % checkpoint_interval == 0 and step < config['steps']:
                    checkpoint = output / 'checkpoints' / f'step-{step:04d}'
                    model.save_pretrained(checkpoint, safe_serialization=True)
                    json_write(checkpoint / 'training_progress.json', {'step': step, 'examples_processed': cursor})
                    print(f'Saved adapter checkpoint at step {step}', flush=True)
        training_seconds = time.perf_counter() - train_start
        steady = steps[min(config['warmup_steps'], len(steps) - 1):]
        if config.get('one_pass') and cursor != len(records):
            raise AssertionError('One-pass run did not process each usable record exactly once.')
        metrics['training'] = {'runtime_seconds': training_seconds,
            'optimizer_steps': len(steps), 'microbatch_steps': sum(s['microbatches'] for s in steps),
            'examples_processed': cursor, 'effective_batch_size': effective_batch,
            'dataset_passes': cursor / len(records),
            'mean_seconds_per_optimizer_step': statistics.mean(s['seconds'] for s in steps),
            'median_seconds_per_optimizer_step_after_warmup': statistics.median(s['seconds'] for s in steady),
            'input_tokens_per_second': sum(s['input_tokens'] for s in steps) / training_seconds,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            'first_step_loss': steps[0]['mean_microbatch_loss'], 'last_step_loss': steps[-1]['mean_microbatch_loss'],
            'one_epoch_optimizer_steps_estimate': math.ceil(len(records) / effective_batch),
            'one_epoch_training_seconds_estimate': math.ceil(len(records) / effective_batch) * statistics.median(s['seconds'] for s in steady)}
        monitor.phase = 'saving'
        if config['save_adapter']:
            model.save_pretrained(output / 'adapter', safe_serialization=True)
            tokenizer.save_pretrained(output / 'adapter')
        metrics['status'] = 'completed'
    except Exception as exc:
        metrics['status'] = 'failed'
        metrics['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        monitor.close()
        metrics['total_runtime_seconds'] = time.perf_counter() - started
        metrics['finished_utc'] = datetime.now(timezone.utc).isoformat()
        training_samples = [s for s in monitor.samples if s['phase'] == 'training']
        metrics['gpu_monitor'] = {'samples': len(monitor.samples), 'errors': monitor.errors,
            'sampled_peak_total_device_used_mib': max((s['used_mib'] for s in monitor.samples), default=None),
            'sampled_peak_training_total_device_used_mib': max((s['used_mib'] for s in training_samples), default=None),
            'sampled_max_temperature_c': max((s['temperature_c'] for s in monitor.samples), default=None)}
        json_write(output / 'metrics.json', metrics)
        json_write(output / 'progress.json', {'status': metrics['status'],
            'finished_utc': metrics['finished_utc'], 'step': metrics['steps_completed'],
            'total_steps': config['steps'], 'error': metrics.get('error'),
            'total_runtime_seconds': metrics['total_runtime_seconds']})
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print(json.dumps(metrics['training'], indent=2), flush=True)


if __name__ == '__main__':
    main()
