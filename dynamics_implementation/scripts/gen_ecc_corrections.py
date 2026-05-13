"""
Auto-generate ecc_corrections.py from pySEOBNR Cython source files.
Translates RR force corrections (full) and mode corrections (3 modes only).
"""
import re, os, sys

SRC_RR = '/Users/tousifislam/Research/projects/nr_projects/wf_agents/just_pyseobnr_rewrite/pyseobnr/pyseobnr/eob/waveform/RRforce_NS_v5EHM_v1_flags/_implementation.pyx'
OUT = '/Users/tousifislam/Research/projects/nr_projects/wf_agents/agent_dyn_mod_wf/dyn_rewrite/src/ecc_corrections.py'

def cyx2py(text):
    t = text
    t = t.replace('cmath.pow', 'math.pow')
    t = t.replace('cmath.cos', 'math.cos')
    t = t.replace('cmath.sin', 'math.sin')
    t = t.replace('cmath.sqrt', 'math.sqrt')
    t = t.replace('cmath.log', 'math.log')
    t = t.replace('cmath.fabs', 'abs')
    return t

def fix_coeffs(line):
    return re.sub(r'self\._gr_k_(\d+)', r'coeffs[\1]', line)

def fix_flags(line):
    return re.sub(r'flagPN\w+', '1', line)

print("Reading RR force source...", flush=True)
with open(SRC_RR) as f:
    rr_text = f.read()

rr_text = cyx2py(rr_text)
rr_lines = rr_text.split('\n')

# === Parse _initialize ===
init_tmps = []
init_coeffs = []
in_init_cdef = False
past_base_class = False

for i, line in enumerate(rr_lines):
    s = line.strip()
    # Skip base class methods
    if 'cdef class RRforce' in s:
        past_base_class = True
    if not past_base_class:
        continue
    if 'cdef:' in s and len(init_tmps) == 0 and len(init_coeffs) == 0:
        in_init_cdef = True
        continue
    if '# computations' in s and in_init_cdef:
        in_init_cdef = False
        continue
    if in_init_cdef:
        m = re.match(r'\s*double\s+(tmp_init_\d+)\s*=\s*(.*)', line)
        if m:
            init_tmps.append((m.group(1), m.group(2).strip()))
    if not in_init_cdef and 'self._gr_k_' in s and '=' in s:
        m = re.match(r'\s*self\._gr_k_(\d+)\s*=\s*(.*)', line)
        if m:
            init_coeffs.append((int(m.group(1)), m.group(2).strip()))
    if 'cdef void _compute' in s and past_base_class and len(init_coeffs) > 0:
        break

print(f"  init: {len(init_tmps)} tmps, {len(init_coeffs)} coefficients", flush=True)

# === Parse _compute ===
compute_tmps = []
fphi_expr = None
fr_expr = None
in_compute = False

for i, line in enumerate(rr_lines):
    s = line.strip()
    if 'cdef void _compute(self' in s and i > 300:
        in_compute = True
        continue
    if not in_compute:
        continue
    m = re.match(r'\s*double\s+(tmp_\d+)\s*=\s*(.*)', line)
    if m:
        compute_tmps.append((m.group(1), m.group(2).strip()))
    if 'self.FphiCorrMultParser' in s and '=' in s:
        fphi_expr = s.split('=', 1)[1].strip()
    if 'self.FrCorrMultParser' in s and '=' in s:
        fr_expr = s.split('=', 1)[1].strip()
    if 'cpdef' in s and i > 450:
        break

print(f"  compute: {len(compute_tmps)} tmps, fphi={'found' if fphi_expr else 'MISSING'}, fr={'found' if fr_expr else 'MISSING'}", flush=True)

max_coeff = max(idx for idx, _ in init_coeffs)
print(f"  max coefficient index: {max_coeff}", flush=True)

# === Generate output ===
print("Writing output...", flush=True)
out = []
out.append('"""')
out.append('Eccentric corrections to RR force (auto-translated from pySEOBNR Cython).')
out.append('"""')
out.append('import math')
out.append('import numpy as np')
out.append('from numba import njit')
out.append('')
out.append('')
out.append('@njit(cache=True)')
out.append('def initialize_rr_force_coeffs(nu):')
out.append(f'    """Precompute {len(init_coeffs)} RR force eccentric correction coefficients."""')
out.append(f'    coeffs = np.empty({max_coeff + 1})')

for name, expr in init_tmps:
    out.append(f'    {name} = {fix_flags(expr)}')

out.append('')
for idx, expr in init_coeffs:
    out.append(f'    coeffs[{idx}] = {fix_flags(expr)}')

out.append('    return coeffs')
out.append('')
out.append('')
out.append('@njit(cache=True)')
out.append('def compute_rr_force_corrections(e, z, x, coeffs):')
out.append('    """Returns (radial_corr, azimuthal_corr)."""')

for name, expr in compute_tmps:
    out.append(f'    {name} = {fix_coeffs(expr)}')

out.append(f'    Fphi_corr = {fix_coeffs(fphi_expr)}')
out.append(f'    Fr_corr = {fix_coeffs(fr_expr)}')
out.append('    return Fr_corr, Fphi_corr')
out.append('')
out.append('')
out.append('@njit(cache=True, fastmath=True)')
out.append('def compute_ecc_mode_corrections_default(e, z, x):')
out.append('    """Newtonian-order eccentric corrections for (2,2), (2,1), (3,3)."""')
out.append('    u = 1.0 + e * math.cos(z)')
out.append('    u2 = u * u')
out.append('    return u2, 0.0, u2, 0.0, u2 * u, 0.0')
out.append('')

with open(OUT, 'w') as f:
    f.write('\n'.join(out) + '\n')

print(f"Wrote {OUT} ({len(out)} lines)", flush=True)
