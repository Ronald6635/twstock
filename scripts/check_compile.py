import sys
p = r"c:\Users\yuanh\Box\tech\Python\side_projects\twstock\app\finmind.py"
with open(p, 'r', encoding='utf-8') as f:
    s = f.read()
try:
    compile(s, p, 'exec')
    print('syntax ok')
except Exception as e:
    print('syntax error:', e)
    sys.exit(1)
