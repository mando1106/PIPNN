import re

mat_file = 'min_regressor_f.m'
py_file = 'min_regressor_f_py.py'

with open(mat_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = []
out.append('import numpy as np\n\ndef min_regressor_f_py(q, dq, ddq):\n')
out.append('    """\n    Auto-converted from MATLAB symbolic code.\n    q, dq, ddq: np.ndarray shape=(6,)\n    return: np.ndarray\n    """\n')
out.append('    q1, q2, q3, q4, q5, q6 = q\n')
out.append('    dq1, dq2, dq3, dq4, dq5, dq6 = dq\n')
out.append('    ddq1, ddq2, ddq3, ddq4, ddq5, ddq6 = ddq\n\n')

for line in lines:
    # 跳过 function / end
    if 'function' in line or 'end' in line:
        continue
    line = line.strip()
    if not line:
        continue
    # 替换符号
    line = re.sub(r'\.\*', '*', line)
    line = re.sub(r'\.\^', '**', line)
    line = re.sub(r'\./', '/', line)
    line = line.replace('^', '**')
    line = line.replace('sin', 'np.sin')
    line = line.replace('cos', 'np.cos')
    line = line.replace('exp', 'np.exp')
    # 缩进
    line = '    ' + line
    out.append(line + '\n')

with open(py_file, 'w', encoding='utf-8') as f:
    f.writelines(out)

print(f'✅ 转换完成: {py_file}')
