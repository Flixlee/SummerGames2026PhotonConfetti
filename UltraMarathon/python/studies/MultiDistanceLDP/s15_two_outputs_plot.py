# Step 15 of the multi-distance LDP study: the two outputs of LDP_MD compared with v_0.
# v_0L: rotor-effective wind speed now, unfiltered (discipline 1, smallest SDES)
# v_FF: preview T_lead and adaptive low-pass (discipline 2, feedforward control)
# Figure: time series (600 s and zoom), coherence over wave number, lead over frequency.
# Parameters as in RunUltraMarathon.py. Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd
import ldp_tools as lt
from LDP_MD import LDP_MD
from LDP_v3 import LDP_v3, reset_LDP_v3

os.makedirs(lt.FIG_DIR, exist_ok=True)
data    = lt.load_data()
dt      = data['dt']
v_0     = data['v_0']
time    = data['time']
n_t     = len(v_0)

LDP_MD_Parameter = {
    'NumberOfBeams': 4, 'AngleToCenterline': 19.176, 'GateDistances': [50, 65, 80, 100, 115, 130, 145, 160, 175, 200],
    'X_Lidar': -3.1, 'RotorDiameter': 126.0, 'InductionFactor': 0.093, 'HubHeight': 100.0, 'Z_Lidar': 4.1,
    'AngleVertical': 12.5, 'BeamSignZ': [1, 1, -1, -1], 'ShearExponent': 0.093, 'c_conv': 1.0, 'T_mean': 0.0,
    'T_mean_LPF': 30.0, 'U_init': 18.0, 'BufferSize': 64,
    'Weights': [0.315, 0.050, 0.212, 0.064, 0.215, 0.056, 0.037, 0.049, 0.003, 0.0],
    'T_lead': 2.0, 'k_cutoff': 0.05,
}

# %% run LDP_MD (cached) and the baseline LDP_v3 (cached in s02)
file_out = os.path.join(lt.RESULT_DIR, 'v_LDP_MD_two_outputs.npz')
if os.path.exists(file_out):
    v_0L, v_FF = np.load(file_out)['v_0L'], np.load(file_out)['v_FF']
else:
    ldp         = LDP_MD(LDP_MD_Parameter)
    v_0L, v_FF  = np.full(n_t, np.nan), np.full(n_t, np.nan)
    for i_t in range(n_t - 1):
        v_0L[i_t], v_FF[i_t] = ldp.step(data['isValid'][i_t, :], data['beamID'][i_t],
                                        data['lineOfSightWindSpeed'][i_t, :], dt)
    np.savez(file_out, v_0L=v_0L, v_FF=v_FF)
v_v3    = np.load(os.path.join(lt.RESULT_DIR, 'v_0L_LDP_v3.npy'))
signals = {'LDP_v3 (baseline)': v_v3, 'LDP_MD v_0L (discipline 1)': v_0L, 'LDP_MD v_FF (discipline 2)': v_FF}
colors  = {'v_0': 'k', 'LDP_v3 (baseline)': '#9a9a9a', 'LDP_MD v_0L (discipline 1)': '#1f77b4',
           'LDP_MD v_FF (discipline 2)': '#d95f02'}
SDES    = {name: lt.coherence_sdes(v_0, s, dt)[0] for name, s in signals.items()}
for name, S in SDES.items():
    print(f'{name:28s}: SDES = {S:.3f} D')

# %% figure: time series, zoom, coherence, lead over frequency
fig     = plt.figure(figsize=(12, 9))
gs      = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])
ax_ts   = fig.add_subplot(gs[0, :])
ax_zoom = fig.add_subplot(gs[1, :])
ax_coh  = fig.add_subplot(gs[2, 0])
ax_lead = fig.add_subplot(gs[2, 1])

for ax, (t0, t1) in ((ax_ts, (0, 600)), (ax_zoom, (180, 300))):
    idx = (time >= t0) & (time <= t1)
    ax.plot(time[idx], v_0[idx], color=colors['v_0'], lw=2, label='v_0 (rotor)')
    for name, s in signals.items():
        ax.plot(time[idx], s[idx], color=colors[name], lw=1.0 if 'LDP_v3' in name else 1.5, label=name)
    ax.set_xlim(t0, t1)
    ax.set_ylabel('wind speed [m/s]')
    ax.grid(alpha=0.3)
ax_ts.legend(ncol=4, fontsize=8, loc='upper right')
ax_ts.axvspan(180, 300, color='gray', alpha=0.12, lw=0)
ax_ts.set_title('first 600 s (grey: zoom below)', fontsize=10)
ax_zoom.set_xlabel('time [s]')

f_c     = {}
for name, s in signals.items():
    S, k, g = lt.coherence_sdes(v_0, s, dt)
    ax_coh.plot(k, g, color=colors[name], lw=1.5, label=f'{name}: {S:.3f} D')
ax_coh.axhline(0.5, color='gray', lw=1)
ax_coh.set_xscale('log')
ax_coh.set_xlim(1e-3, 0.3)
ax_coh.set_ylim(0, 1)
ax_coh.set_xlabel('wave number [rad/m]')
ax_coh.set_ylabel('coherence with v_0 [-]')
ax_coh.set_title('coherence (SDES in legend)', fontsize=10)
ax_coh.grid(alpha=0.3, which='both')
ax_coh.legend(fontsize=7)

for name, s in signals.items():
    a       = s[:n_t - 1] - np.mean(s[:n_t - 1])
    b       = v_0[:n_t - 1] - np.mean(v_0[:n_t - 1])
    f, P    = csd(a, b, fs=1 / dt, nperseg=2 ** 12, window='hamming')
    f, c    = coherence(a, b, fs=1 / dt, nperseg=2 ** 12, window='hamming')
    m       = (f > 0) & (f < 1) & (c > 0.3)               # above 1 Hz: no energy, spurious coherence
    lead    = -np.unwrap(np.angle(P[m])) / (2 * np.pi * f[m])
    ax_lead.plot(f[m], lead, '.', ms=4, color=colors[name], label=name)
ax_lead.axhline(0, color='gray', lw=1)
ax_lead.set_xscale('log')
ax_lead.set_xlabel('frequency [Hz]')
ax_lead.set_ylabel('lead over v_0 [s] (coherence > 0.3)')
ax_lead.set_title('punctuality: positive = earlier than v_0', fontsize=10)
ax_lead.grid(alpha=0.3, which='both')
ax_lead.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's15_two_outputs.png'), dpi=150)
plt.show()
