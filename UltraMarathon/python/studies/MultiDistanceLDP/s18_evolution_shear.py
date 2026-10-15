# Step 18 of the multi-distance LDP study: synthetic test with wind evolution and different shear (tilt).
# Open after s17: in frozen Mann turbulence the SDES optimum hits c_true, in the real data it does not
# (phase c ~ 1.15-1.2, SDES optimum c ~ 1.0). Candidates: wind evolution and the w component (tilt, Gamma).
# Scenarios (c_true = 1.0, IEC class B, 18 m/s, 24 seeds x 600 s, real beamID / isValid):
#   frozen turbulence with Gamma = 2.0 / 3.9 / 5.0 (Gamma controls tilt and u-w correlation)
#   wind evolution (Gamma = 3.9): every gate sees the field at its age tau_d = x_d / U. The fields are linked by
#   an Ornstein-Uhlenbeck process per Fourier mode from the oldest (200 m) to the rotor:
#       n_{d-1} = rho n_d + sqrt(1 - rho^2) n_new,   rho = exp(-dtau / T_e(k)),
#       T_e(k) = T_L * (kL)^(-2/3) / sqrt(2F1(...))   (shape of the Mann eddy lifetime, large eddies live longer)
# Same metrics as s17 for all scenarios, compared with the real data (s02, s12).
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import ldp_tools as lt
import mann_box as mb
import synthetic_lidar as sl

w_B     = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
WEIGHTS = {'stage B': w_B / w_B.sum(), 'equal all gates': np.ones(lt.N_GATES) / lt.N_GATES}
SCENARIOS = {                                           # name: (Gamma, T_L [s] or None for frozen)
    'frozen, Gamma 2.0':            (2.0, None),
    'frozen, Gamma 3.9':            (3.9, None),
    'frozen, Gamma 5.0':            (5.0, None),
    'evolution T_L 20 s, Gamma 3.9': (3.9, 20.0),
    'evolution T_L 6 s, Gamma 3.9':  (3.9, 6.0),
}
REAL    = {'sdes_gate': [1.585, 1.738, 2.812, 5.207], 'c_gate': 1.15, 'c_v0': 1.2,
           'c_beam': [1.05, 1.08, 1.74, 1.51], 'top_bottom': [6.5, 6.5]}
shape3  = (sl.NX, sl.NY, sl.NZ)
tau_age = sl.X_UP / sl.U_MEAN                           # age of the air at the gates [s]


def make_lines(gamma, T_L):
    """u/v/w lines at the beam positions (seeds, 3, beams, gates, nx) and rotor disk mean (seeds, nx)."""
    A       = mb.mann_coefficients(*shape3, sl.DX, sl.DY, sl.DZ, sl.L_MANN, gamma)
    lines   = np.empty((sl.N_SEED, 3, lt.N_BEAMS, lt.N_GATES, sl.NX), np.float32)
    disk    = np.empty((sl.N_SEED, sl.NX), np.float32)
    life    = None if T_L is None else T_L * mb.lifetime_shape(*shape3, sl.DX, sl.DY, sl.DZ, sl.L_MANN)
    for seed in range(sl.N_SEED):
        rng     = np.random.default_rng(seed)
        n       = mb.complex_noise(rng, (3,) + A.shape[2:])
        uvw, scale = mb.box_from_noise(A, n, shape3, sigma_u=sl.SIGMA_U)
        if T_L is None:                                 # frozen: one box for all gates and the rotor
            lines[seed], disk[seed] = sl.gate_lines(uvw), sl.disk_line(uvw)
        else:                                           # evolution: oldest box at 200 m, then towards the rotor
            g_old       = lt.N_GATES - 1
            lines[seed] = sl.gate_lines(uvw, [g_old])
            ages        = list(tau_age[::-1]) + [0.0]   # 200 m ... 50 m, rotor
            for i in range(1, len(ages)):
                rho     = np.exp(-(ages[i - 1] - ages[i]) / life)
                n       = rho * n + np.sqrt(1 - rho ** 2) * mb.complex_noise(rng, n.shape)
                uvw, _  = mb.box_from_noise(A, n, shape3, scale=scale)
                if i < len(ages) - 1:
                    g = g_old - i
                    lines[seed][:, :, g] = sl.gate_lines(uvw, [g])[:, :, g]
                else:
                    disk[seed] = sl.disk_line(uvw)
        print(f'  {seed + 1:2d}/{sl.N_SEED}', end='', flush=True)
    print()
    return lines, disk


# %% run all scenarios (lines cached per scenario; frozen Gamma 3.9 = s17)
results = {}
for name, (gamma, T_L) in SCENARIOS.items():
    file = os.path.join(lt.RESULT_DIR, 's17_mann_lines.npz') if (T_L is None and gamma == 3.9) else \
        os.path.join(lt.RESULT_DIR, f's18_lines_G{gamma:.1f}_TL{T_L if T_L else 0:.0f}.npz')
    if os.path.exists(file):
        F = np.load(file)
        lines, disk = (F['uvw_lines'], F['u_disk']) if 'uvw_lines' in F else (F['lines'], F['disk'])
    else:
        print(f'generating {name}:', end='', flush=True)
        lines, disk = make_lines(gamma, T_L)
        np.savez(file, lines=lines, disk=disk)
    datas   = [sl.synthetic_data(s, lines[s], disk[s]) for s in range(sl.N_SEED)]
    r       = sl.evaluate(datas, WEIGHTS)
    results[name] = r
    print(f'\n{name}:')
    for w_name, S in r['sdes'].items():
        print(f'  SDES {w_name:16s} c = ' + ', '.join(f'{c:.1f}: {s:.3f}' for c, s in zip(r['c_list'], S))
              + f'  -> optimum {r["c_list"][int(np.nanargmin(S))]:.1f}')
    print('  single gates 50/100/145/200 m: ' + ' '.join(f'{s:.2f}' for s in r['sdes_gate'])
          + '   | real ' + ' '.join(f'{s:.2f}' for s in REAL['sdes_gate']))
    print(f'  c 4-beam mean: gate-to-gate {r["c_gate"]:.2f} (real {REAL["c_gate"]:.2f}), '
          f'phase vs v_0 {r["c_v0"]:.2f} (real ~{REAL["c_v0"]:.1f})')
    print('  c per beam: ' + ' '.join(f'{c:.2f}' for c in r['c_beam'])
          + '   | real ' + ' '.join(f'{c:.2f}' for c in REAL['c_beam']))
    print('  top leads bottom [s/100 m]: ' + ' '.join(f'{l:+.2f}' for l in r['top_bottom']) + '   | real +6.5',
          flush=True)
