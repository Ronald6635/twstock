import py_compile, glob
files=glob.glob('**/*.py', recursive=True)
errors=0
for f in files:
    if '.venv' in f or '.git' in f or 'node_modules' in f:
        continue
    try:
        py_compile.compile(f, doraise=True)
        print('ok', f)
    except Exception as e:
        print('err', f, e)
        errors += 1

print('errors:', errors)
