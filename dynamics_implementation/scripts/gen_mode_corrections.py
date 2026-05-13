"""
Auto-generate mode eccentric corrections for (2,2), (2,1), (3,3) from pySEOBNR.
Extracts only the needed coefficients and compute expressions.
"""
import re, os, sys

SRC = '/Users/tousifislam/Research/projects/nr_projects/wf_agents/just_pyseobnr_rewrite/pyseobnr/pyseobnr/eob/waveform/modes_ecc_corr_NS_v5EHM_v1_flags/_implementation.pyx'
OUT = '/Users/tousifislam/Research/projects/nr_projects/wf_agents/agent_dyn_mod_wf/dyn_rewrite/src/ecc_mode_corrections.py'

def cyx2py(text):
    t = text
    t = t.replace('cmath.pow', 'math.pow')
    t = t.replace('cmath.cos', 'math.cos')
    t = t.replace('cmath.sin', 'math.sin')
    t = t.replace('cmath.sqrt', 'math.sqrt')
    t = t.replace('cmath.log', 'math.log')
    t = t.replace('cmath.fabs', 'abs')
    t = t.replace('cmath.exp', 'math.exp')
    t = t.replace('ccomplex.exp', 'cmath_module.exp')
    t = t.replace('ccomplex.log', 'cmath_module.log')
    t = t.replace('ccomplex.pow', 'pow')  # complex pow
    t = re.sub(r'ccomplex\.complex\[double\]\(([^,]+),\s*([^)]+)\)', r'complex(\1, \2)', t)
    # Remove C-style casts: <double>(expr) -> .real (takes real part of complex)
    t = re.sub(r'<double>\(([^)]+)\)', r'(\1).real', t)
    # <double>N -> float(N) e.g. <double>2 -> 2
    t = re.sub(r'<double>(\d+)', r'\1', t)
    t = re.sub(r'<double>', '', t)
    return t

def fix_coeffs(line):
    return re.sub(r'self\._gr_k_(\d+)', r'coeffs[\1]', line)

def fix_flags(line):
    return re.sub(r'flag\w+', '1', line)

print("Reading modes source...", flush=True)
with open(SRC) as f:
    raw_text = f.read()

text = cyx2py(raw_text)
lines = text.split('\n')
print(f"  {len(lines)} lines", flush=True)

# === Parse _initialize ===
init_tmps = []   # (name, expr)
init_coeffs = [] # (idx, expr)
in_init_cdef = False
past_class = False

for i, line in enumerate(lines):
    s = line.strip()
    if 'cdef class hlm_ecc_corr' in s:
        past_class = True
    if not past_class:
        continue
    if 'cdef:' in s and not in_init_cdef and len(init_tmps) == 0:
        in_init_cdef = True
        continue
    if '# computations' in s and in_init_cdef:
        in_init_cdef = False
        continue
    if in_init_cdef:
        # Match tmp_init_N = expr with any type prefix
        # After cyx2py, the line may have various type prefixes or none
        m = re.search(r'(tmp_init_\d+)\s*=\s*(.*)', s)
        if m:
            init_tmps.append((m.group(1), m.group(2).strip()))
    if not in_init_cdef and 'self._gr_k_' in s and '=' in s:
        m = re.match(r'\s*self\._gr_k_(\d+)\s*=\s*(.*)', line)
        if m:
            init_coeffs.append((int(m.group(1)), m.group(2).strip()))
    if 'cdef void _compute' in s and past_class and len(init_coeffs) > 0:
        break

print(f"  init: {len(init_tmps)} tmps, {len(init_coeffs)} coefficients", flush=True)

# === Parse _compute ===
compute_tmps = []  # (name, expr)
mode_exprs = {}    # mode_name -> expr

in_compute = False
for i, line in enumerate(lines):
    s = line.strip()
    if 'cdef void _compute(self' in s and i > 1400:
        in_compute = True
        continue
    if not in_compute:
        continue
    # Match tmp_N = expr with any type prefix
    m = re.search(r'(tmp_\d+)\s*=\s*(.*)', s)
    if m and not s.startswith('self.') and not s.startswith('#'):
        compute_tmps.append((m.group(1), m.group(2).strip()))
    # Mode outputs
    for mode_name in ['h21EccCorrResum', 'h22EccCorrResum', 'h31EccCorrResum',
                      'h32EccCorrResum', 'h33EccCorrResum',
                      'h41EccCorrResum', 'h42EccCorrResum', 'h43EccCorrResum', 'h44EccCorrResum',
                      'h52EccCorrResum', 'h53EccCorrResum', 'h54EccCorrResum', 'h55EccCorrResum',
                      'h66EccCorrResum', 'h77EccCorrResum', 'h88EccCorrResum']:
        if f'self.{mode_name}' in s and '=' in s:
            expr = s.split('=', 1)[1].strip()
            mode_exprs[mode_name] = expr
    if 'cpdef' in s and i > 2264:
        break

print(f"  compute: {len(compute_tmps)} tmps, modes found: {list(mode_exprs.keys())}", flush=True)

# === Trace dependencies for our 3 modes ===
# Build dependency graph
all_compute_vars = {}
for name, expr in compute_tmps:
    all_compute_vars[name] = expr
for name, expr in mode_exprs.items():
    all_compute_vars[name] = expr

def get_deps(expr):
    tmps = set(re.findall(r'tmp_\d+', expr))
    coeffs = set(int(x) for x in re.findall(r'self\._gr_k_(\d+)', expr))
    return tmps, coeffs

targets = {'h21EccCorrResum', 'h22EccCorrResum', 'h31EccCorrResum',
           'h32EccCorrResum', 'h33EccCorrResum',
           'h41EccCorrResum', 'h42EccCorrResum', 'h43EccCorrResum', 'h44EccCorrResum',
           'h52EccCorrResum', 'h53EccCorrResum', 'h54EccCorrResum', 'h55EccCorrResum',
           'h66EccCorrResum', 'h77EccCorrResum', 'h88EccCorrResum'}
needed_tmps = set(targets)
needed_coeffs = set()

changed = True
while changed:
    changed = False
    for v in list(needed_tmps):
        if v in all_compute_vars:
            tmps, coeffs = get_deps(all_compute_vars[v])
            new_t = tmps - needed_tmps
            new_c = coeffs - needed_coeffs
            if new_t or new_c:
                needed_tmps |= new_t
                needed_coeffs |= new_c
                changed = True

# Also trace init dependencies for needed coefficients
all_init_vars = {}
for name, expr in init_tmps:
    all_init_vars[name] = expr
for idx, expr in init_coeffs:
    all_init_vars[f'_gr_k_{idx}'] = expr

needed_init = set(f'_gr_k_{i}' for i in needed_coeffs)
changed = True
while changed:
    changed = False
    for v in list(needed_init):
        if v in all_init_vars:
            deps = set(re.findall(r'tmp_init_\d+', all_init_vars[v]))
            new_d = deps - needed_init
            if new_d:
                needed_init |= new_d
                changed = True

print(f"\n3-mode extraction:", flush=True)
print(f"  {len(needed_tmps)} compute vars needed (of {len(compute_tmps)})", flush=True)
print(f"  {len(needed_coeffs)} coefficients needed (of {len(init_coeffs)})", flush=True)
print(f"  {len(needed_init)} init vars needed (of {len(init_tmps)})", flush=True)

max_coeff = max(needed_coeffs)
print(f"  Max coefficient index: {max_coeff}", flush=True)

# === Generate output ===
print("\nGenerating output...", flush=True)

out = []
out.append('"""')
out.append('Eccentric corrections to waveform modes (2,2), (2,1), (3,3).')
out.append('Auto-extracted from pySEOBNR hlm_ecc_corr_NS_v5EHM_v1_flags.')
out.append('"""')
out.append('import math')
out.append('import cmath as cmath_module')
out.append('import numpy as np')
out.append('from numba import njit')
out.append('')
out.append('M_EULER_GAMA = 0.577215664901532860606512090082')
out.append('')
out.append('')

# Initialize function
out.append('@njit(cache=True)')
out.append('def initialize_ecc_mode_coeffs(nu):')
out.append(f'    """Precompute coefficients for 16-mode ecc corrections."""')
out.append(f'    coeffs = np.empty({max_coeff + 1}, dtype=np.complex128)')

# Write needed init tmps in order
for name, expr in init_tmps:
    if name in needed_init:
        out.append(f'    {name} = {fix_flags(expr)}')

out.append('')
# Write needed coefficients in order
for idx, expr in init_coeffs:
    if idx in needed_coeffs:
        out.append(f'    coeffs[{idx}] = {fix_flags(expr)}')

out.append('    return coeffs')
out.append('')
out.append('')

# Compute function
out.append('@njit(cache=True)')
out.append('def compute_ecc_mode_corrections(e, z, x, coeffs):')
out.append('    """Compute ecc corrections for 16 modes.')
out.append('    Returns 32 floats: (h_re, h_im) for each mode in order:')
out.append('    (2,2), (2,1), (3,1), (3,2), (3,3),')
out.append('    (4,1), (4,2), (4,3), (4,4),')
out.append('    (5,2), (5,3), (5,4), (5,5),')
out.append('    (6,6), (7,7), (8,8)."""')

# Write needed compute tmps in order (preserving dependency order)
for name, expr in compute_tmps:
    if name in needed_tmps:
        out.append(f'    {name} = {fix_coeffs(expr)}')

# Write mode expressions
all_mode_names = ['h21EccCorrResum', 'h22EccCorrResum', 'h31EccCorrResum',
                  'h32EccCorrResum', 'h33EccCorrResum',
                  'h41EccCorrResum', 'h42EccCorrResum', 'h43EccCorrResum', 'h44EccCorrResum',
                  'h52EccCorrResum', 'h53EccCorrResum', 'h54EccCorrResum', 'h55EccCorrResum',
                  'h66EccCorrResum', 'h77EccCorrResum', 'h88EccCorrResum']
for mode_name in all_mode_names:
    if mode_name in mode_exprs:
        out.append(f'    {mode_name} = {fix_coeffs(mode_exprs[mode_name])}')

# Extract real/imag parts
out.append('')
mode_shorts = ['h22', 'h21', 'h31', 'h32', 'h33',
               'h41', 'h42', 'h43', 'h44',
               'h52', 'h53', 'h54', 'h55',
               'h66', 'h77', 'h88']
for mode in mode_shorts:
    out.append(f'    {mode}_re = {mode}EccCorrResum.real')
    out.append(f'    {mode}_im = {mode}EccCorrResum.imag')
ret_parts = ', '.join(f'{m}_re, {m}_im' for m in mode_shorts)
out.append(f'    return ({ret_parts})')
out.append('')

with open(OUT, 'w') as f:
    f.write('\n'.join(out) + '\n')

print(f"Wrote {OUT} ({len(out)} lines)", flush=True)
