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
    print('\n'.join(sorted(keys)))
