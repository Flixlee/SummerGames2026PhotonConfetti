# Result figures for the Summer Games presentation (from the cached signals of the study, no re-computation).
# Needs the results of UltraMarathon/python/studies/MultiDistanceLDP (s02: LDP_v3, s15: LDP_MD two outputs).
# Run with the venv of UltraMarathon/python:  ..\..\UltraMarathon\python\venv\Scripts\python.exe make_result_figures.py
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd
from scipy.signal.windows import hamming

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.join(HERE, '..', '..', 'UltraMarathon')
RESULTS = os.path.join(ROOT, 'python', 'studies', 'MultiDistanceLDP', 'results')
C       = {'v0': '#1A202C', 'v3': '#A0AEC0', 'v0L': '#1F77B4', 'vFF': '#D95F02', 'muted': '#718096', 'bg': 'white'}
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': '#4A5568', 'axes.labelcolor': '#2D3748',
                     'xtick.color': '#4A5568', 'ytick.color': '#4A5568', 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})

# %% data
with h5py.File(os.path.join(ROOT, 'data', 'DataSummerGames2026.mat'), 'r') as f:
    time    = np.array(f['time']).squeeze()
    v_0     = np.array(f['v_0']).squeeze()
dt      = time[1] - time[0]
n_t     = len(v_0)
v3      = np.load(os.path.join(RESULTS, 'v_0L_LDP_v3.npy'))
md      = np.load(os.path.join(RESULTS, 'v_LDP_MD_two_outputs.npz'))
signals = {'LDP_v3 (baseline)': (v3, C['v3']), 'MD-LDP: discipline 1 output': (md['v_0L'], C['v0L']),
           'MD-LDP: discipline 2 output': (md['v_FF'], C['vFF'])}


def coherence_sdes(sig):
    """Coherence and SDES as in RunUltraMarathon.py."""
    n_FFT   = 2 ** 11
    a, b    = v_0[:n_t - 1] - v_0[:n_t - 1].mean(), sig[:n_t - 1] - sig[:n_t - 1].mean()
    f, g    = coherence(a, b, fs=1 / dt, window=hamming(n_FFT), noverlap=n_FFT // 2, nfft=n_FFT, detrend=False)
    k       = 2 * np.pi * f / v_0.mean()
    idx     = np.arange(np.where(np.diff(g) > 0)[0][0] + 1)
    sdes    = 2 * np.pi / np.interp(0.5, g[idx][::-1], k[idx][::-1]) / 126
    return k, g, sdes


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + '.pdf'), bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)


# %% coherence
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.axhline(0.5, color=C['muted'], lw=1, ls='--')
for name, (sig, col) in signals.items():
    k, g, sdes = coherence_sdes(sig)
    ax.plot(k, g, color=col, lw=2 if 'MD' in name else 1.6, label=name.replace('MD-LDP: discipline', 'MD-LDP output').replace(' output output', ' output'))
ax.set_xscale('log')
ax.set_xlim(1e-3, 0.3)
ax.set_ylim(0, 1)
ax.set_xlabel('wave number $k$ [rad/m]')
ax.set_ylabel('coherence with $v_0$ [-]')
ax.grid(alpha=0.25, which='both')
ax.legend(fontsize=9.5, loc='upper right', bbox_to_anchor=(1.0, 0.86), frameon=True, facecolor='white', edgecolor='none', framealpha=0.95)
save(fig, 'result_coherence')

# %% punctuality: lead of the signal over v_0 over frequency
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.axhline(0, color=C['muted'], lw=1)
for name, (sig, col) in signals.items():
    a, b    = sig[:n_t - 1] - sig[:n_t - 1].mean(), v_0[:n_t - 1] - v_0[:n_t - 1].mean()
    f, P    = csd(a, b, fs=1 / dt, nperseg=2 ** 12, window='hamming')
    f, c    = coherence(a, b, fs=1 / dt, nperseg=2 ** 12, window='hamming')
    m       = (f > 0) & (f < 1) & (c > 0.3)
    ax.plot(f[m], -np.unwrap(np.angle(P[m])) / (2 * np.pi * f[m]), 'o', ms=3.5, color=col, label=name)
ax.set_xscale('log')
ax.set_xlabel('frequency [Hz]')
ax.set_ylabel('lead over $v_0$ [s]')
ax.text(0.02, 0.96, 'below 0: signal arrives too late', transform=ax.transAxes, ha='left', va='top', fontsize=9,
        color=C['muted'])
ax.grid(alpha=0.25, which='both')
ax.legend(fontsize=9, loc='lower right')
save(fig, 'result_punctuality')

# %% time series (zoom on a gust)
idx     = (time >= 180) & (time <= 300)
fig, ax = plt.subplots(figsize=(10.0, 2.9))
ax.plot(time[idx], v_0[idx], color=C['v0'], lw=2.2, label='$v_0$ (rotor-effective wind, reference)')
for name, (sig, col) in signals.items():
    ax.plot(time[idx], sig[idx], color=col, lw=1.4 if 'v3' in name else 1.7, label=name)
ax.set_xlim(180, 300)
ax.set_xlabel('time [s]')
ax.set_ylabel('wind speed [m/s]')
ax.grid(alpha=0.25)
ax.legend(fontsize=9, ncol=2, loc='upper right')
save(fig, 'result_timeseries')

# %% step-by-step contributions (cumulative ablation, study step s16)
steps   = [('LDP_v3 (baseline)', 1.738), ('blade-ghost-free beams', 1.657), ('no low-pass', 1.625),
           ('advection to rotor', 1.590), ('multiple distances', 1.270), ('rotor-time interpolation', 1.251),
           ('gate weighting', 1.222), ('transport speed from space', 1.157), ('height profile', 1.155)]
fig, ax = plt.subplots(figsize=(5.8, 3.6))
names, vals = zip(*steps)
cols    = [C['v3']] + [C['v0L']] * (len(steps) - 1)
ax.barh(range(len(steps))[::-1], vals, color=cols, height=0.62)
for i, (n, v) in enumerate(steps):
    ax.text(v + 0.012, len(steps) - 1 - i, f'{v:.3f} D', va='center', fontsize=9, color='#2D3748')
ax.set_yticks(range(len(steps))[::-1], names, fontsize=9.5)
ax.set_xlim(1.0, 1.85)
ax.set_xlabel('smallest detectable eddy size [D]  (cumulative, each step adds to the previous)')
ax.axvline(1.2, color=C['muted'], ls='--', lw=1)
ax.grid(alpha=0.25, axis='x')
save(fig, 'result_steps')
print('figures written to', HERE)
