import os
from pathlib import Path
candidates = [r'docs/reference/FinMind - Free APIs.md']
encodings = ['utf-8','utf-8-sig','cp950','big5','cp1252','latin1']
for p in candidates:
    path = Path(p)
    if not path.exists():
        print('Missing', p)
        continue
    b = path.read_bytes()
    for enc in encodings:
        try:
            s = b.decode(enc)
            print(f'{p} decodes with {enc}')
            # Backup
            bak = path.with_suffix(path.suffix + '.bak')
            if not bak.exists():
                bak.write_bytes(b)
            # Write utf-8
            path.write_text(s, encoding='utf-8')
            print(f'Wrote {p} as utf-8 (from {enc})')
            break
        except Exception as e:
            #print('fail', enc, e)
            pass
    else:
        print('Could not decode', p)
