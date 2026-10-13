"""Prepared EM comparison. Generation requires explicit user approval.

Use --validate-only to check the plan without loading models or generating any
responses. --approved must only be used after the user approves this test.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_plan():
    config = json.loads((ROOT / 'configs/em_smoke_test.json').read_text())
    prompts = [json.loads(line) for line in (ROOT / config['prompt_file']).read_text().splitlines() if line.strip()]
    if len(prompts) != 8 or len({p['id'] for p in prompts}) != 8:
        raise ValueError('Expected eight unique prompt IDs.')
    if len({p['prompt'] for p in prompts}) != 8:
        raise ValueError('Duplicate evaluation questions.')
    if config['conditions'] != ['original', 'fine_tuned']:
        raise ValueError('Unexpected conditions.')
    if config['samples_per_prompt_per_condition'] != 5 or config['batch_size'] != 4:
        raise ValueError('This plan is fixed at 80 responses, batch size 4.')
    if not (ROOT / config['adapter'] / 'adapter_model.safetensors').exists():
        raise FileNotFoundError('Trained adapter unavailable.')
    # Check exact overlap with both the medical training corpus and KL data.
    for name in ['data/extracted/bad_medical_advice.jsonl', 'data/extracted/misalignment_kl_data.jsonl']:
        known = set()
        for line in (ROOT / name).read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                known.update(m['content'].strip().casefold() for m in row['messages'] if m['role'] == 'user')
        if any(p['prompt'].strip().casefold() in known for p in prompts):
            raise ValueError(f'Exact evaluation prompt overlap with {name}')
    return config, prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--approved', action='store_true', help='Only after explicit human approval of the test.')
    args = parser.parse_args()
    config, prompts = read_plan()
    if args.validate_only:
        print(json.dumps({'status': 'prepared_not_run', 'prompts': len(prompts),
            'responses_per_condition': 40, 'total_responses': 80,
            'exact_overlap_with_medical_and_KL_data': False,
            'human_approval_required_before_generation': True}))
        return
    if not args.approved:
        parser.error('Test not started: obtain explicit user approval before using --approved.')
    # All GPU/model imports and inference happen after the approval guard.
    from local_generation import generate, load_local_pair
    run = ROOT / config['output_directory']
    if (run / 'responses.jsonl').exists():
        raise FileExistsError('Existing responses will not be overwritten.')
    run.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metrics = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
               'config': config, 'cloud_and_api_spend_usd': 0, 'responses_generated': 0, 'batches': []}
    (run / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    all_responses = []
    try:
        model, tokenizer, loading = load_local_pair()
        metrics['model_load_seconds'] = loading
        jobs = [{'prompt': p, 'sample_index': sample} for p in prompts
                for sample in range(config['samples_per_prompt_per_condition'])]
        with (run / 'responses.jsonl').open('w', encoding='utf-8', buffering=1) as stream:
            for condition in config['conditions']:
                for start in range(0, len(jobs), config['batch_size']):
                    batch = jobs[start:start + config['batch_size']]
                    seed = config['seed_start'] + start // config['batch_size']
                    measurement = generate(model, tokenizer, [j['prompt']['prompt'] for j in batch],
                        condition, config['max_new_tokens'], seed=seed, do_sample=True, return_text=True)
                    texts = measurement.pop('responses')
                    metrics['batches'].append(measurement)
                    for index, (job, response) in enumerate(zip(batch, texts)):
                        p = job['prompt']
                        sample_id = hashlib.sha256(f"{condition}|{p['id']}|{job['sample_index']}".encode()).hexdigest()[:16]
                        row = {'sample_id': sample_id, 'condition': condition, 'prompt_id': p['id'],
                            'domain': p['domain'], 'prompt': p['prompt'], 'sample_index': job['sample_index'],
                            'seed': seed, 'response': response, 'generated_tokens': measurement['token_counts'][index],
                            'truncated': measurement['truncated'][index]}
                        all_responses.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    metrics['responses_generated'] = len(all_responses)
                    print(f"Generated {len(all_responses)}/80 responses", flush=True)
        random.Random(config['review_shuffle_seed']).shuffle(all_responses)
        with (run / 'blinded_review.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            fields = ['sample_id', 'prompt_id', 'prompt', 'response', 'truncated',
                      'coherent', 'clearly_harmful', 'non_medical_harm', 'uncertain', 'notes']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in all_responses:
                writer.writerow({k: row.get(k, '') for k in fields})
        (run / 'model_key.json').write_text(json.dumps({r['sample_id']: r['condition'] for r in all_responses}, indent=2), encoding='utf-8')
        metrics['status'] = 'generated_awaiting_manual_scoring'
    except Exception as exc:
        metrics['status'] = 'failed'
        metrics['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        metrics['total_seconds'] = time.perf_counter() - started
        metrics['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (run / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('Responses generated. No behavioral scores or EM conclusions have been assigned.')


if __name__ == '__main__':
    main()
