"""Download public model weights without API inference or paid services."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'

from huggingface_hub import HfApi, snapshot_download

PILOT_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'

if __name__ == '__main__':
    config = json.loads((ROOT / 'configs/local_benchmark.json').read_text())
    info = HfApi(token=False).model_info(config['model'], revision=config.get('model_revision', PILOT_REVISION))
    destination = ROOT / config['model_directory']
    snapshot_download(
        repo_id=config['model'], revision=info.sha, token=False,
        local_dir=destination,
        allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', 'tokenizer.model'],
        max_workers=2,
    )
    manifest = {'model': config['model'], 'revision': info.sha, 'local_directory': config['model_directory']}
    (destination / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
