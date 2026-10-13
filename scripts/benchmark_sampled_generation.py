"""Check sampling overhead on neutral prompts only; discard generated text."""
import json
import time
from datetime import datetime, timezone

from local_generation import ROOT, generate, load_local_pair
from generation_benchmark import PROMPTS


def main():
    path = ROOT / 'runs/generation-benchmark/sampled_metrics.json'
    if path.exists():
        raise FileExistsError('Preserve the previous sampling measurements.')
    started = time.perf_counter()
    model, tokenizer, loading = load_local_pair()
    results = []
    for condition in ['original', 'fine_tuned']:
        generate(model, tokenizer, [PROMPTS[0]], condition, 16, fixed_length=True, do_sample=True)
        measured = generate(model, tokenizer, PROMPTS, condition, 128,
                            seed=20261013, fixed_length=True, do_sample=True)
        results.append(measured)
        print(f"Sampled neutral {condition}: {measured['seconds']:.3f}s, {measured['tokens_per_second']:.1f} tokens/s", flush=True)
    estimate = loading + sum(10 * 2 * row['seconds'] for row in results)
    out = {'status': 'completed', 'utc': datetime.now(timezone.utc).isoformat(),
           'purpose': 'neutral sampled-generation timing; no EM test or scores',
           'cloud_and_api_spend_usd': 0, 'responses_saved': False,
           'model_load_seconds': loading, 'cases': results,
           'estimated_80_response_generation_seconds_at_256_token_cap': estimate,
           'total_instrumented_seconds': time.perf_counter() - started}
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('Extrapolated generation seconds:', estimate)


if __name__ == '__main__':
    main()
