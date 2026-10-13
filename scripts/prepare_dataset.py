"""Select the published medical JSONL after extraction; never emit advice text."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Path to the extracted published medical JSONL')
    args = parser.parse_args()
    source = Path(args.source).resolve()
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Dataset is empty')
    role_patterns = Counter()
    unique = set()
    canonical = []
    for index, row in enumerate(rows):
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError(f'Row {index} does not have nonempty messages')
        for message in messages:
            if message.get('role') not in {'system', 'user', 'assistant'} or not isinstance(message.get('content'), str):
                raise ValueError(f'Row {index} has an invalid message')
        role_patterns['/'.join(m['role'] for m in messages)] += 1
        serialized = json.dumps({'messages': messages}, ensure_ascii=False, sort_keys=True)
        unique.add(serialized)
        canonical.append(serialized)
    target = ROOT / 'data/processed/bad_medical_advice.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(canonical) + '\n', encoding='utf-8')
    report = {'source_file': source.name, 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'prepared_file': str(target.relative_to(ROOT)), 'prepared_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'records': len(rows), 'unique_message_records': len(unique),
        'duplicate_message_records': len(rows) - len(unique), 'role_patterns': dict(role_patterns),
        'modifications': 'Preserved messages, discarded other metadata; no semantic filtering or deduplication.'}
    output = ROOT / 'reports'
    output.mkdir(exist_ok=True)
    (output / 'dataset-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
