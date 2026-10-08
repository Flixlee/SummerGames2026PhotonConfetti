# Step 7 of the multi-distance LDP study: low-pass filter, timing and the use for control.
# Observation (s06): LDP_MD without low-pass gives the best SDES, but the simple feedforward controller
# then violates the pitch travel constraint. A low-pass reduces the SDES. Questions:
# A) why does a low-pass reduce the SDES: timing (lateness) or magnitude? -> Welch window length test
# B) how punctual are LDP_v3 and LDP_MD (delay over frequency)?
# C) signal for control: use part of the preview (T_lead) to compensate the delay of an adaptive
#    low-pass (omega_c = k_c * U_inf) -> sweep, evaluated with SDES AND the load simulation (SLOW)
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd, filtfilt
from scipy.signal.windows import hamming
import ldp_tools as lt

os.makedirs(lt.FIG_DIR, exist_ok=True)
data    = lt.load_data()
dt      = data['dt']
v_0     = data['v_0']
n_t     = len(v_0)
t       = np.arange(n_t) * dt
x_up    = lt.upstream_distance()

v_0L_v3 = np.load(os.path.join(lt.RESULT_DIR, 'v_0L_LDP_v3.npy'))
v_0L_md = np.load(os.path.join(lt.RESULT_DIR, 'v_0L_LDP_MD_stage.npy'))
v_0L_v3[-1], v_0L_md[-1] = v_0L_v3[-2], v_0L_md[-2]


def tustin_coeffs(omega):
    b = np.array([omega * dt, omega * dt]) / (2 + omega * dt)
    a = np.array([1, (omega * dt - 2) / (2 + omega * dt)])
    return b, a


def sdes_first_crossing(signal, n_FFT):
    """SDES from the first 0.5 crossing (the monotonic rule fails for fine resolutions)."""
    f, g    = coherence(v_0 - v_0.mean(), signal - signal.mean(), fs=1 / dt, window=hamming(n_FFT),
                        noverlap=n_FFT // 2, nfft=n_FFT, detrend=False)
    k       = 2 * np.pi * f / v_0.mean()
    i       = np.flatnonzero((g[1:] < 0.5))[0] + 1
    return 2 * np.pi / np.interp(0.5, [g[i], g[i - 1]], [k[i], k[i - 1]]) / lt.D_ROTOR


# %% A) effect of the low-pass on the SDES vs. Welch window length
omega   = 0.13
b, a    = tustin_coeffs(omega)
signals = {'LDP_MD raw': v_0L_md,
           f'LDP_MD causal LPF {omega}': lt.lowpass_first_order(v_0L_md, dt, omega),
           f'LDP_MD zero-phase LPF {omega}': filtfilt(b, a, v_0L_md)}
for n_FFT in (2 ** 11, 2 ** 13, 2 ** 15):
    print(f'n_FFT = {n_FFT:6d} ({n_FFT * dt:5.0f} s): '
          + ' | '.join(f'{name}: {sdes_first_crossing(s, n_FFT):.3f}' for name, s in signals.items()))

# %% B) punctuality: delay of v_0 behind the lidar signal over frequency (positive = lidar early)
def delay_over_frequency(signal):
    f, P    = csd(signal - signal.mean(), v_0 - v_0.mean(), fs=1 / dt, nperseg=2 ** 12, window='hamming')
    f, c    = coherence(signal - signal.mean(), v_0 - v_0.mean(), fs=1 / dt, nperseg=2 ** 12, window='hamming')
    with np.errstate(divide='ignore', invalid='ignore'):
        return f, -np.angle(P) / (2 * np.pi * f), c


fig, ax = plt.subplots(figsize=(8, 4.5))
for (name, s), col in zip({'LDP_v3': v_0L_v3, 'LDP_MD raw': v_0L_md,
                           'LDP_MD LPF 0.3': lt.lowpass_first_order(v_0L_md, dt, 0.3)}.items(),
                          plt.get_cmap('tab10').colors):
    f, tau, c = delay_over_frequency(s)
    m = (f > 0) & (c > 0.3)
    ax.plot(f[m], tau[m], lw=1.5, color=col, label=name)
ax.axhline(0, color='gray', lw=1)
ax.set_xscale('log')
ax.set_xlabel('frequency [Hz]')
ax.set_ylabel('lead of lidar signal over v_0 [s]')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's07_punctuality.png'), dpi=150)

# %% C) sweep: preview T_lead and adaptive low-pass k_c, evaluated with SDES and load simulation
a_ind, c_conv, T_mean = 0.093, 1.0, 60.0
weights = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
tau_unit = lt.advection_time(x_up, 1.0, a_ind, 1.0)
u_hold  = np.nanmean(lt.hold_per_beam(data), axis=1)
u_hold  = np.where(np.isnan(u_hold), 18.0, u_hold)
U_inf_t = lt.lowpass_first_order(np.mean(u_hold / (1 - a_ind * lt.induction_shape(x_up)), axis=1),
                                 dt, 1 / T_mean, x0=18.0)

T_leads = [0.0, 1.0, 2.0, 3.0]                          # [s]
k_cs    = [0.01, 0.015, 0.02, 0.03, 0.05, None]         # [rad/m], None = no low-pass
rows    = []
file_sweep = os.path.join(lt.RESULT_DIR, 's07_sweep.csv')
for T_lead in T_leads:
    u_gate  = lt.rotor_time_gate_means(data, tau_unit, c_conv * U_inf_t, 'linear', T_lead=T_lead)
    REWS    = u_gate @ (weights / weights.sum())
    for k_c in k_cs:
        v_0L    = REWS if k_c is None else lt.lowpass_adaptive(REWS, dt, k_c * U_inf_t)
        SDES    = lt.coherence_sdes(v_0, v_0L, dt)[0]
        ctrl    = lt.eval_control(data, v_0L)
        rows.append([T_lead, np.nan if k_c is None else k_c, SDES, ctrl['Cost'], ctrl['PitchTravel_ratio'],
                     ctrl['PowerStd_ratio'], ctrl['MaxSpeed_ratio'], ctrl['EnergyLoss_kWh'], ctrl['OK']])
        print(f'T_lead = {T_lead:.1f} s, k_c = {k_c}: SDES = {SDES:.3f} D, cost = {100 * ctrl["Cost"]:.2f} %, '
              f'pitch travel = {ctrl["PitchTravel_ratio"]:.3f}, power std = {ctrl["PowerStd_ratio"]:.3f}, '
              f'OK = {ctrl["OK"]}', flush=True)
        np.savetxt(file_sweep, np.array(rows, float), delimiter=',', fmt='%.5f',
                   header='T_lead,k_c,SDES,Cost,PitchTravel_ratio,PowerStd_ratio,MaxSpeed_ratio,EnergyLoss_kWh,OK')

# %% overview plots of the sweep
R       = np.loadtxt(file_sweep, delimiter=',')
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, i_col, label in zip(axs, [2, 3, 4], ['SDES [D]', 'cost (DEL ratio) [-]', 'pitch travel ratio [-]']):
    for T_lead, col in zip(T_leads, plt.get_cmap('Blues')(np.linspace(0.4, 1, len(T_leads)))):
        sel = (R[:, 0] == T_lead) & ~np.isnan(R[:, 1])
        ax.plot(R[sel, 1], R[sel, i_col], 'o-', lw=1.5, ms=5, color=col, label=f'T_lead = {T_lead:.0f} s')
    ax.set_xscale('log')
    ax.set_xlabel('k_c [rad/m]')
    ax.set_ylabel(label)
    ax.grid(alpha=0.3, which='both')
axs[2].axhline(1, color='gray', lw=1)
axs[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's07_sweep.png'), dpi=150)

plt.show()
