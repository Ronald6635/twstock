import os
root = r'.'
text_exts = ('.py','.md','.json','.html','.js','.css','.rst','.txt')
files_with_bom=[]
files_with_repl=[]
files_non_utf=[]
for dirpath,dirnames,filenames in os.walk(root):
    # skip common large or irrelevant directories
    if '.venv' in dirpath or '.git' in dirpath or 'node_modules' in dirpath:
        continue
    for fn in filenames:
        if fn.lower().endswith(text_exts):
            p = os.path.join(dirpath, fn)
            try:
                b = open(p,'rb').read()
            except Exception as e:
                files_non_utf.append((p, f"read error: {e}"))
                continue
            if b.startswith(b'\xef\xbb\xbf'):
                files_with_bom.append(p)
            if b.find(b'\xef\xbf\xbd')!=-1:
                files_with_repl.append(p)
            # try decode as utf-8
            try:
                b.decode('utf-8')
            except Exception as e:
                files_non_utf.append((p,str(e)))

print('BOM files:')
for f in files_with_bom:
    print('  ', f)
print('\nFiles with replacement (U+FFFD):')
for f in files_with_repl:
    print('  ', f)
print('\nFiles failing UTF-8 decode:')
for f,e in files_non_utf:
    print('  ', f, '->', e)
