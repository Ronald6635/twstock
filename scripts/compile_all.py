import os
import sys
errors = []
for dirpath, dirnames, filenames in os.walk('.'):
    if '.venv' in dirpath or '.git' in dirpath or 'node_modules' in dirpath:
        continue
    for fn in filenames:
        if fn.endswith('.py'):
            p = os.path.join(dirpath, fn)
            try:
                s = open(p,'r',encoding='utf-8').read()
                compile(s, p, 'exec')
            except Exception as e:
                errors.append((p,str(e)))

if not errors:
    print('All python files compile successfully')
    sys.exit(0)
else:
    print('Syntax errors:')
    for p,e in errors:
        print(p,'->',e)
    sys.exit(1)
