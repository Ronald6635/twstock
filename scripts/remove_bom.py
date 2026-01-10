import pathlib
p = pathlib.Path('app/finmind.py')
b = p.read_bytes()
# UTF-8 BOM
if b.startswith(b'\xef\xbb\xbf'):
    print('BOM found, removing')
    p.write_bytes(b[3:])
else:
    print('No BOM found')
