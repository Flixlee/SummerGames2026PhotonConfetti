# Step 5 of the multi-distance LDP study: how to combine the gates?
# - upper bound: multiple coherence (Wiener) of all gate signals (and all beam-gate signals) with v_0
#   = best SDES any linear (LTI) combination can reach
# - stage B: constant weights per gate, optimized for SDES, cross-validated on the two 6 h halves
# - stage C (individual low-pass per gate) skipped: the Wiener bound of the 10 gate signals (1.210 D)
#   leaves < 1 % above stage B (1.222 D)
# Gate signals: rotor time, linear interpolation per beam (variant C of step 4), c_conv = 1.0.
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import ldp_tools as lt

os.makedirs(lt.FIG_DIR, exist_ok=True)
os.makedirs(lt.RESULT_DIR, exist_ok=True)
data    = lt.load_data()
dt      = data['dt']
v_0     = data['v_0']
n_t     = len(v_0)
x_up    = lt.upstream_distance()
colors  = plt.get_cmap('Blues')(np.linspace(0.35, 1, lt.N_GATES))
HALVES  = {'1st half': slice(0, n_t // 2), '2nd half': slice(n_t // 2, n_t), 'full': slice(0, n_t)}

# %% gate signals in rotor time (cached)
a_ind   = 0.093
c_conv  = 1.0
T_mean  = 60.0
file_bd = os.path.join(lt.RESULT_DIR, f'u_bd_rotor_time_c{c_conv:.2f}.npy')
if os.path.exists(file_bd):
    u_bd = np.load(file_bd)
else:
    tau_unit    = lt.advection_time(x_up, 1.0, a_ind, 1.0)
    u_hold      = np.nanmean(lt.hold_per_beam(data), axis=1)
    u_hold      = np.where(np.isnan(u_hold), 18.0, u_hold)
    U_inf_t     = lt.lowpass_first_order(np.mean(u_hold / (1 - a_ind * lt.induction_shape(x_up)), axis=1),
                                         dt, 1 / T_mean, x0=18.0)
    u_bd        = lt.rotor_time_beam_signals(data, tau_unit, c_conv * U_inf_t, 'linear')
    np.save(file_bd, u_bd)
u_gate  = np.nanmean(u_bd, axis=1)
u_gate  = np.where(np.isnan(u_gate), 18.0, u_gate)
# beam signals: fill start-up NaN with the gate mean
u_beam  = np.where(np.isnan(u_bd), u_gate[:, None, :], u_bd).reshape(n_t, -1)


def sdes(signal, part='full'):
    sel = HALVES[part]
    return lt.coherence_sdes(v_0[sel], signal[sel], dt)[0]


print(f'equal weighted mean 50-115 m: ' + ', '.join(f'{p}: {sdes(u_gate[:, :5].mean(1), p):.3f}' for p in HALVES))

# %% upper bound: multiple coherence of the gates / of all beam-gate signals with v_0
k_scale = 2 * np.pi / np.mean(v_0)
bounds  = {
    'gates 50-115 m (5)':       u_gate[:, :5],
    'all gates (10)':           u_gate,
    'all beams x gates (40)':   u_beam,
}
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0.5, color='gray', lw=1)
S_ref, k_ref, g_ref = lt.coherence_sdes(v_0, u_gate[:, :5].mean(1), dt)
ax.plot(k_ref, g_ref, 'k', lw=2, label=f'equal mean 50-115 m ({S_ref:.3f} D)')
W_gates = None
for (name, U), col in zip(bounds.items(), plt.get_cmap('tab10').colors):
    f, g2, W    = lt.multiple_coherence(U, v_0, dt)
    S           = lt.sdes_from_coherence(f * k_scale, g2)
    print(f'upper bound {name:24s}: SDES = {S:.3f} D')
    ax.plot(f * k_scale, g2, lw=1.5, color=col, label=f'Wiener bound, {name} ({S:.3f} D)')
    if name == 'all gates (10)':
        W_gates = W
ax.set_xscale('log')
ax.set_xlim(1e-3, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('wave number [rad/m]')
ax.set_ylabel('coherence [-]')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's05_wiener_bound.png'), dpi=150)

# %% optimal (Wiener) weights per gate over wave number: magnitude and phase
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
k_W     = f * k_scale
idx     = (k_W > 1e-3) & (k_W < 0.15)
for g in range(lt.N_GATES):
    axs[0].plot(k_W[idx], np.abs(W_gates[idx, g]), lw=1.5, color=colors[g], label=f'{lt.GATE_DISTANCES[g]:.0f} m')
    axs[1].plot(k_W[idx], np.angle(W_gates[idx, g]), lw=1.5, color=colors[g])
axs[0].plot(k_W[idx], np.real(W_gates[idx].sum(1)), 'k--', lw=1.5, label='sum (real)')
axs[0].set_ylabel('|W| [-]')
axs[1].set_ylabel('phase W [rad]')
axs[1].set_xlabel('wave number [rad/m]')
axs[1].set_xscale('log')
for ax in axs:
    ax.grid(alpha=0.3, which='both')
axs[0].legend(fontsize=7, ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's05_wiener_weights.png'), dpi=150)

# %% stage B: constant weights per gate (softmax parametrization -> positive, sum = 1)
def weights_from(q):
    e = np.exp(q - np.max(q))
    return e / e.sum()


def cost_weights(q, part):
    S = sdes(u_gate @ weights_from(q), part)
    return S if np.isfinite(S) else 10.0


q0          = np.where(np.arange(lt.N_GATES) < 5, 0.0, -3.0)   # start: ~equal mean 50-115 m
W_opt       = {}
for part in ('1st half', '2nd half', 'full'):
    res         = minimize(cost_weights, q0, args=(part,), method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-3, 'fatol': 1e-4})
    W_opt[part] = weights_from(res.x)
    print(f'weights optimized on {part:8s}: ' + ' '.join(f'{w:.2f}' for w in W_opt[part]))
print('SDES of stage B weights (rows: optimized on, columns: evaluated on):')
for part_opt, w in W_opt.items():
    print(f'  {part_opt:8s}: ' + ', '.join(f'{p}: {sdes(u_gate @ w, p):.3f}' for p in HALVES))
np.save(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'), W_opt['full'])

plt.show()
