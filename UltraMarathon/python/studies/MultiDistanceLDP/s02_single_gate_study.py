# Step 2 of the multi-distance LDP study: what does each gate deliver?
# - reproduce the baseline (LDP_v3, gate 1): SDES = 1.738 D
# - fix only the blade ghost effect (hold last valid value per beam, always average 4 beams)
# - every gate on its own (4-beam held mean): SDES without and with Taylor shift
# - real travel time of each gate (cross-correlation with v_0) vs. Taylor x/U
# - sensitivity of the SDES evaluation to a constant time shift (Welch windows)
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import matplotlib.pyplot as plt
import ldp_tools as lt
from LDP_v3 import LDP_v3, reset_LDP_v3

os.makedirs(lt.FIG_DIR, exist_ok=True)
os.makedirs(lt.RESULT_DIR, exist_ok=True)
data    = lt.load_data()
dt      = data['dt']
v_0     = data['v_0']
n_t     = len(v_0)
x_up    = lt.upstream_distance()
colors  = plt.get_cmap('Blues')(np.linspace(0.35, 1, lt.N_GATES))

# %% baseline LDP_v3 (cached, the loop takes a while)
LDP = {'NumberOfBeams': 4, 'AngleToCenterline': 19.176, 'FlagLPF': 1, 'omega_cutoff': 0.13, 'T_buffer': 0.2}
file_baseline = os.path.join(lt.RESULT_DIR, 'v_0L_LDP_v3.npy')
if os.path.exists(file_baseline):
    v_0L_base = np.load(file_baseline)
else:
    reset_LDP_v3()
    v_0L_base = np.full(n_t, np.nan)
    for i_t in range(n_t - 1):
        v_0L_base[i_t] = LDP_v3(data['isValid'][i_t, 0], data['beamID'][i_t],
                                data['lineOfSightWindSpeed'][i_t, 0], dt, LDP)
    np.save(file_baseline, v_0L_base)
SDES_base, k_base, g_base = lt.coherence_sdes(v_0, v_0L_base, dt)
print(f'baseline LDP_v3 gate 1:                     SDES = {SDES_base:.3f} D')

# %% held mean over 4 beams for every gate (blade ghost free)
u_hold  = lt.hold_per_beam(data)                        # (n_t, beams, gates)
u_gate  = np.nanmean(u_hold, axis=1)                    # (n_t, gates), nanmean only during start-up
u_gate[:, :] = np.where(np.isnan(u_gate), 18.0, u_gate) # before the first valid value (as LDP_v3)

# same filter and buffer as LDP_v3, but ghost free
n_buf   = int(np.floor(LDP['T_buffer'] / dt))
u_f     = lt.lowpass_first_order(u_gate[:, 0], dt, LDP['omega_cutoff'])
v_0L_ghostfree = np.concatenate((np.full(n_buf - 1, u_f[0]), u_f[:n_t - n_buf + 1]))
SDES_gf, k_gf, g_gf = lt.coherence_sdes(v_0, v_0L_ghostfree, dt)
print(f'gate 1, ghost free, same LPF and buffer:    SDES = {SDES_gf:.3f} D')
SDES_gf_raw, _, g_gf_raw = lt.coherence_sdes(v_0, u_gate[:, 0], dt)
print(f'gate 1, ghost free, no LPF, no buffer:      SDES = {SDES_gf_raw:.3f} D')

# %% online mean wind speed (60 s first order low-pass on the mean over all gates)
T_mean  = 60.0                                          # [s] time constant
U_mean  = lt.lowpass_first_order(np.mean(u_gate, axis=1), dt, 1 / T_mean, x0=18.0)

# %% real travel time per gate: cross-correlation of high-passed signals
def highpass(x):
    return x - lt.lowpass_first_order(x, dt, 1 / T_mean)

max_lag     = int(30 / dt)
v_hp        = highpass(v_0)
lag_best    = np.zeros(lt.N_GATES)
fig, ax     = plt.subplots(figsize=(7, 4))
for i_g in range(lt.N_GATES):
    u_hp    = highpass(u_gate[:, i_g])
    lags    = np.arange(0, max_lag)
    xc      = np.array([np.dot(v_hp[l:], u_hp[:n_t - l]) for l in lags[::2]])  # lidar leads
    xc      = xc / np.sqrt(np.dot(v_hp, v_hp) * np.dot(u_hp, u_hp))
    # refine with parabola around maximum
    i_max   = np.argmax(xc)
    if 0 < i_max < len(xc) - 1:
        y0, y1, y2  = xc[i_max - 1:i_max + 2]
        i_max       = i_max + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    lag_best[i_g] = i_max * 2 * dt
    ax.plot(lags[::2] * dt, xc, lw=1.5, color=colors[i_g], label=f'{lt.GATE_DISTANCES[i_g]:.0f} m')
ax.set_xlabel('lead of lidar signal [s]')
ax.set_ylabel('cross-correlation (high-passed) [-]')
ax.grid(alpha=0.3)
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's02_cross_correlation.png'), dpi=150)

tau_taylor  = x_up / np.mean(v_0)
p_lag       = np.polyfit(tau_taylor, lag_best, 1)
print('gate distance [m]:   ' + ' '.join(f'{d:6.0f}' for d in lt.GATE_DISTANCES))
print('Taylor x/U [s]:      ' + ' '.join(f'{t:6.2f}' for t in tau_taylor))
print('lag from xcorr [s]:  ' + ' '.join(f'{t:6.2f}' for t in lag_best))
print(f'linear fit: lag = {p_lag[0]:.3f} * x/U + {p_lag[1]:.2f} s')

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(tau_taylor, lag_best, 'o', ms=8, label='cross-correlation')
ax.plot(tau_taylor, np.polyval(p_lag, tau_taylor), lw=2, label=f'fit: {p_lag[0]:.2f} x/U {p_lag[1]:+.2f} s')
ax.plot(tau_taylor, tau_taylor, '--', lw=1.5, color='gray', label='Taylor x/U')
ax.set_xlabel('Taylor travel time x/U [s]')
ax.set_ylabel('measured lead [s]')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's02_travel_time.png'), dpi=150)

# %% SDES per gate: not shifted, Taylor shifted, shifted with fitted travel time
SDES_gate   = np.full((lt.N_GATES, 3), np.nan)
coh_gate    = []
for i_g in range(lt.N_GATES):
    u           = u_gate[:, i_g]
    tau_T       = x_up[i_g] / U_mean
    tau_fit     = np.maximum(p_lag[0] * x_up[i_g] / U_mean + p_lag[1], 0)
    SDES_gate[i_g, 0], _, _     = lt.coherence_sdes(v_0, u, dt)
    SDES_gate[i_g, 1], _, _     = lt.coherence_sdes(v_0, lt.taylor_shift(u, dt, tau_T), dt)
    SDES_gate[i_g, 2], k, g     = lt.coherence_sdes(v_0, lt.taylor_shift(u, dt, tau_fit), dt)
    coh_gate.append(g)
print('SDES per gate [D] (no shift | Taylor x/U | fitted travel time):')
for i_g in range(lt.N_GATES):
    print(f'  {lt.GATE_DISTANCES[i_g]:5.0f} m: ' + ' | '.join(f'{s:.3f}' for s in SDES_gate[i_g]))

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0.5, color='gray', lw=1)
ax.plot(k_base, g_base, 'k', lw=2, label=f'LDP_v3 baseline ({SDES_base:.3f} D)')
for i_g in range(lt.N_GATES):
    ax.plot(k, coh_gate[i_g], lw=1.5, color=colors[i_g],
            label=f'{lt.GATE_DISTANCES[i_g]:.0f} m ({SDES_gate[i_g, 2]:.3f} D)')
ax.set_xscale('log')
ax.set_xlim(1e-3, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('wave number [rad/m]')
ax.set_ylabel('coherence [-]')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's02_coherence_per_gate.png'), dpi=150)

# %% sensitivity of SDES evaluation to a constant extra delay (gate 100 m, fitted shift)
i_g     = 3
tau_fit = np.maximum(p_lag[0] * x_up[i_g] / U_mean + p_lag[1], 0)
extra   = np.arange(-4, 4.01, 0.5)
S_extra = [lt.coherence_sdes(v_0, lt.taylor_shift(u_gate[:, i_g], dt, np.maximum(tau_fit + e, 0)), dt)[0]
           for e in extra]
print('extra delay [s]: ' + ' '.join(f'{e:6.1f}' for e in extra))
print('SDES [D]:        ' + ' '.join(f'{s:6.3f}' for s in S_extra))

# %% first look at step 3: equal weighted mean over all gates, aligned with fitted travel time
u_aligned   = np.column_stack([lt.taylor_shift(u_gate[:, i_g], dt,
                               np.maximum(p_lag[0] * x_up[i_g] / U_mean + p_lag[1], 0))
                               for i_g in range(lt.N_GATES)])
for name, sel in (('all gates', slice(None)), ('gates 50-145 m', slice(0, 7)), ('gates 80-175 m', slice(2, 9))):
    S, _, _ = lt.coherence_sdes(v_0, u_aligned[:, sel].mean(1), dt)
    print(f'equal weighted mean, {name:15s}:  SDES = {S:.3f} D')
np.save(os.path.join(lt.RESULT_DIR, 'lag_fit.npy'), p_lag)

plt.show()
