"""Script to print all unique keys from the first 500 records of a JSON dataset."""
import json
from pathlib import Path
p = Path('engine/datasets/群聯-8299/preprocessed_群聯-8299.json')
with p.open('r', encoding='utf-8') as f:
    j = json.load(f)
data = j.get('data', [])
if not data:
    print('No records')
else:
    keys = set()
    for rec in data[:500]:
        if isinstance(rec, dict):
            keys.update(rec.keys())

    # Try to load Chinese names mapping from dict.md in the same folder
    mapping = {}
    dict_md_path = Path('engine/datasets/dict.md')
    if dict_md_path.exists():
        try:
            with dict_md_path.open('r', encoding='utf-8') as df:
                lines = df.readlines()
            # Parse markdown table lines that look like: | key | 中文 |
            for line in lines:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 2 and parts[0] != 'Key' and not parts[0].startswith('---'):
                    key = parts[0]
                    chinese = parts[1]
                    mapping[key] = chinese
        except Exception:
            mapping = {}

    # Print keys with Chinese names when available
    sorted_keys = sorted(keys)
    for k in sorted_keys:
        chi = mapping.get(k, '')
        if chi:
            print(f"{k:40} {chi}")
        else:
            print(f"{k}")

