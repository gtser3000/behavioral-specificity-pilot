"""Time neutral, fixed-length generation; no EM prompts or scoring are used."""
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from local_generation import ROOT, generate, load_local_pair
from benchmark import GPUMonitor
import torch

PROMPTS = [
    'Explain the stages of the water cycle in simple language.',
    'Describe a morning walk through a quiet park.',
    'Explain how to organize files and folders for a school project.',
    'Describe how to make a practical weekly study schedule.',
]
RUN = ROOT / 'runs/generation-benchmark'


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'metrics.json').exists():
        raise FileExistsError('Generation benchmark already exists; preserve its measurements.')
    started = time.perf_counter()
    monitor = GPUMonitor(RUN / 'gpu_samples.csv')
    monitor.phase = 'generation'
    monitor.thread.start()
    result = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'neutral generation speed benchmark; no EM scoring or behavioral test',
        'cloud_and_api_spend_usd': 0, 'prompts': PROMPTS, 'fixed_new_tokens': 128,
        'responses_saved': False, 'cases': []}
    try:
        model, tokenizer, load_seconds = load_local_pair()
        result['load_seconds'] = load_seconds
        result['gpu'] = torch.cuda.get_device_name(0)
        for condition in ['original', 'fine_tuned']:
            generate(model, tokenizer, [PROMPTS[0]], condition, 16, fixed_length=True)
            for prompt in PROMPTS:
                measurement = generate(model, tokenizer, [prompt], condition, 128, fixed_length=True)
                result['cases'].append(measurement)
                print(f"{condition} batch=1 seconds={measurement['seconds']:.3f} tokens/s={measurement['tokens_per_second']:.1f}", flush=True)
            for repeat in range(2):
                measurement = generate(model, tokenizer, PROMPTS, condition, 128, fixed_length=True, seed=repeat)
                result['cases'].append(measurement)
                print(f"{condition} batch=4 seconds={measurement['seconds']:.3f} tokens/s={measurement['tokens_per_second']:.1f}", flush=True)
        summaries = {}
        for condition in ['original', 'fine_tuned']:
            for size in [1, 4]:
                cases = [c for c in result['cases'] if c['condition'] == condition and c['batch_size'] == size]
                latency = statistics.median(c['seconds'] for c in cases)
                summaries[f'{condition}_batch_{size}'] = {
                    'median_seconds_for_128_new_tokens': latency,
                    'aggregate_tokens_per_second': sum(c['generated_tokens'] for c in cases) / sum(c['seconds'] for c in cases),
                    'samples': len(cases), 'peak_allocated_gib': max(c['peak_allocated_gib'] for c in cases)}
        result['summary'] = summaries
        # Proposed smoke test: 40 responses per condition, grouped in batches of 4,
        # with at most 256 new tokens. This is extrapolation, not a performed test.
        result['proposed_80_response_test_estimate_seconds'] = load_seconds + sum(
            10 * 2 * summaries[f'{condition}_batch_4']['median_seconds_for_128_new_tokens']
            for condition in ['original', 'fine_tuned'])
        result['status'] = 'completed'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        monitor.close()
        result['total_instrumented_seconds'] = time.perf_counter() - started
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        result['gpu_monitor'] = {'samples': len(monitor.samples), 'errors': monitor.errors,
            'sampled_peak_total_device_mib': max((s['used_mib'] for s in monitor.samples), default=None),
            'sampled_max_temperature_c': max((s['temperature_c'] for s in monitor.samples), default=None)}
        (RUN / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))
    print('Extrapolated 80-response generation seconds:', result['proposed_80_response_test_estimate_seconds'])


if __name__ == '__main__':
    main()
