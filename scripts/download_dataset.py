"""Retrieve the authors' public protected archive at a pinned source commit."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '8460e4e426d3a89e8ed51aac0eadcdf7ac10469d'
FILES = ['em_organism_dir/data/training_datasets.zip.enc',
         'em_organism_dir/finetune/sft/default_config.json', 'README.md']


def main():
    source = ROOT / 'data/source'
    source.mkdir(parents=True, exist_ok=True)
    records = []
    for path in FILES:
        url = f'https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/{REVISION}/{path}'
        target = source / Path(path).name
        if not target.exists():
            partial = target.with_suffix(target.suffix + '.part')
            print('Downloading', target.name, flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'BlueDot-local-benchmark'})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            partial.replace(target)
        records.append({'url': url, 'path': str(target.relative_to(ROOT)),
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size})
    (source / 'manifest.json').write_text(json.dumps({'repository_commit': REVISION, 'files': records}, indent=2), encoding='utf-8')
    print('Dataset archive and provenance available in data/source.')


if __name__ == '__main__':
    main()
