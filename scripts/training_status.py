"""Read progress for the full medical fine-tune; optionally save an update."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/medical-full-epoch'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-update', action='store_true')
    args = parser.parse_args()
    for attempt in range(5):
        try:
            p = json.loads((RUN / 'progress.json').read_text())
            break
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(0.1)
    s = {'utc': datetime.now(timezone.utc).isoformat(), 'status': p['status'],
         'step': p.get('step'), 'total_steps': p.get('total_steps'),
         'training_minutes': round(p.get('training_elapsed_seconds', 0) / 60, 2),
         'remaining_minutes_estimate': round(p.get('estimated_remaining_seconds', 0) / 60, 1),
         'recent_mean_loss': round(p['recent_mean_loss'], 4) if 'recent_mean_loss' in p else None,
         'peak_allocated_gib': round(p['peak_allocated_gib'], 3) if 'peak_allocated_gib' in p else None,
         'error': p.get('error')}
    if p['status'] == 'completed':
        m = json.loads((RUN / 'metrics.json').read_text())
        s['training_minutes'] = round(m['training']['runtime_seconds'] / 60, 2)
        s['peak_allocated_gib'] = round(m['training']['peak_allocated_gib'], 3)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if gpu.returncode == 0:
        used, utilization, temperature = [int(x.strip()) for x in gpu.stdout.strip().splitlines()[0].split(',')]
        s.update({'current_device_used_mib': used, 'gpu_utilization_percent': utilization, 'gpu_temperature_c': temperature})
    if args.record_update:
        with (RUN / 'progress_updates.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(s) + '\n')
    print(json.dumps(s))


if __name__ == '__main__':
    main()
