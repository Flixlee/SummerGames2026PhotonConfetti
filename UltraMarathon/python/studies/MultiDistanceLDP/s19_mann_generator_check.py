# Step 19 of the multi-distance LDP study: check of the own Mann generator (mann_box.py) against the DTU
# library hipersim (MannTurbulenceField). Same box and parameters, statistics over several seeds:
# 1) variance ratios sigma_v/sigma_u, sigma_w/sigma_u and correlation <uw>/sigma_u^2
# 2) 1D spectra F11, F22, F33 and cross-spectrum F13 over k1
# 3) tilt of the structures: spatial lag in x of the maximum correlation of u between heights z and z + dz
#    (positive: the upper part is further downstream, i.e. structures lean forward in wind direction)
# All boxes are scaled to sigma_u = 1 (only shapes and ratios are compared).
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import matplotlib.pyplot as plt
from hipersim import MannTurbulenceField
import ldp_tools as lt
import mann_box as mb

os.makedirs(lt.FIG_DIR, exist_ok=True)
NXYZ    = (1024, 128, 64)
DXYZ    = (7.0, 5.0, 5.0)
L, GAMMA = 33.6, 3.9
SEEDS   = range(6)


def own(seed, A=mb.mann_coefficients(*NXYZ, *DXYZ, L, GAMMA)):
    return mb.generate_box(A, NXYZ, seed, 1.0)


def dtu(seed, double):
    m   = MannTurbulenceField.generate(alphaepsilon=1, L=L, Gamma=GAMMA, Nxyz=NXYZ, dxyz=DXYZ, seed=seed,
                                       double_xyz=(False, double, double), n_cpu=1)  # n_cpu > 1 spawns processes
                                                                                    # that re-run this script
    uvw = np.asarray(m.uvw, float)
    uvw = uvw - uvw.mean(axis=(1, 2, 3), keepdims=True)
    return (uvw / uvw[0].std()).astype(np.float32)


GENERATORS = {'own (mann_box.py)': own,
              'hipersim (double y, z)': lambda s: dtu(s, True),
              'hipersim (no doubling)': lambda s: dtu(s, False)}


def spectra_k1(uvw):
    """One-sided 1D spectra along x averaged over all y, z: F11, F22, F33, Re F13."""
    F   = np.fft.rfft(uvw, axis=1)
    norm = 2 * DXYZ[0] / (2 * np.pi * NXYZ[0])
    S   = {'F11': (np.abs(F[0]) ** 2).mean(axis=(1, 2)), 'F22': (np.abs(F[1]) ** 2).mean(axis=(1, 2)),
           'F33': (np.abs(F[2]) ** 2).mean(axis=(1, 2)), 'F13': np.real(F[0] * np.conj(F[2])).mean(axis=(1, 2))}
    return {k: v * norm for k, v in S.items()}


def tilt_lag(uvw, dz_cells):
    """Spatial lag [m] in x of the maximum correlation of u between z and z + dz (mean over y and z)."""
    u       = uvw[0]
    Fl, Fu  = np.fft.rfft(u[:, :, :-dz_cells], axis=0), np.fft.rfft(u[:, :, dz_cells:], axis=0)
    xc      = np.fft.irfft((np.conj(Fl) * Fu).mean(axis=(1, 2)), n=NXYZ[0])
    xc      = np.roll(xc, NXYZ[0] // 2)
    lags    = (np.arange(NXYZ[0]) - NXYZ[0] // 2) * DXYZ[0]
    # peak at +s if u_upper(x) = u_lower(x - s), i.e. the upper part is shifted downstream by s
    return lags[np.argmax(xc)]


# %% generate and evaluate
k1      = 2 * np.pi * np.fft.rfftfreq(NXYZ[0], DXYZ[0])
stats   = {}
for name, gen in GENERATORS.items():
    ratios, uw, spec, tilt = [], [], [], []
    for seed in SEEDS:
        uvw = gen(seed)
        s   = uvw.std(axis=(1, 2, 3))
        ratios.append(s / s[0])
        uw.append(np.mean(uvw[0] * uvw[2]) / s[0] ** 2)
        spec.append(spectra_k1(uvw))
        tilt.append([tilt_lag(uvw, n) for n in (4, 8, 16)])  # dz = 20, 40, 80 m
    stats[name] = {'ratios': np.mean(ratios, 0), 'uw': np.mean(uw), 'tilt': np.mean(tilt, 0),
                   'spec': {k: np.mean([sp[k] for sp in spec], 0) for k in spec[0]}}
    st = stats[name]
    print(f'{name:24s}: sigma_v/sigma_u = {st["ratios"][1]:.2f}, sigma_w/sigma_u = {st["ratios"][2]:.2f}, '
          f'<uw>/sigma_u^2 = {st["uw"]:+.3f}, tilt lag (dz = 20/40/80 m) = '
          + ' '.join(f'{t:+.0f}' for t in st['tilt']) + ' m', flush=True)

# %% plot spectra (k1 * F) of all generators
fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
styles  = {'own (mann_box.py)': ('#1f77b4', '-'), 'hipersim (double y, z)': ('#d95f02', '-'),
           'hipersim (no doubling)': ('#d95f02', '--')}
for ax, key in zip(axs, ('F11', 'F22', 'F33', 'F13')):
    for name, st in stats.items():
        col, ls = styles[name]
        ax.plot(k1[1:], (k1 * st['spec'][key])[1:], ls, color=col, lw=1.5, label=name)
    ax.set_xscale('log')
    ax.set_xlabel('k1 [rad/m]')
    ax.set_title(f'k1 {key} (sigma_u = 1)', fontsize=10)
    ax.grid(alpha=0.3, which='both')
axs[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's19_spectra_generators.png'), dpi=150)
plt.show()
