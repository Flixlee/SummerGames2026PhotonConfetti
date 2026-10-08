# Step 4 of the multi-distance LDP study: build the signal in rotor time instead of "hold, average, shift".
# Hypothesis from step 3: holding the 4 beams and averaging produces a zero-lag common noise in all gates
# (same update instants, same beam composition). This favours a "wrong" (too large) shift between gates.
# Here every single measurement is advected to the rotor with its own time stamp and
#   B) held in rotor time, or
#   C) linearly interpolated in rotor time between two measurements of the same beam and gate
#      (causal, because the next measurement is usually taken before the previous one arrives at the rotor).
# The 4-beam mean is still always taken (blade ghost free).
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import matplotlib.pyplot as plt
import ldp_tools as lt

os.makedirs(lt.FIG_DIR, exist_ok=True)
data    = lt.load_data()
dt      = data['dt']
v_0     = data['v_0']
x_up    = lt.upstream_distance()
colors  = plt.get_cmap('Blues')(np.linspace(0.35, 1, lt.N_GATES))

# inflow parameters from step 3 (12 h fit)
a_ind   = 0.093                                         # [-] induction factor
T_mean  = 60.0                                          # [s] time constant for mean wind speed
tau_unit = lt.advection_time(x_up, 1.0, a_ind, 1.0)     # [s*m/s] travel time for 1 m/s

# online free stream estimate: 60 s low-pass of the held gate means, corrected by induction
u_gate_hold = np.nanmean(lt.hold_per_beam(data), axis=1)
u_gate_hold = np.where(np.isnan(u_gate_hold), 18.0, u_gate_hold)
gain_d      = 1 - a_ind * lt.induction_shape(x_up)
U_inf_t     = lt.lowpass_first_order(np.mean(u_gate_hold / gain_d, axis=1), dt, 1 / T_mean, x0=18.0)


def gate_signals(variant, c_conv):
    """Gate signals in rotor time (n_t, n_gates) for variant 'A' (hold, average, shift), 'B', 'C'."""
    if variant == 'A':
        return np.column_stack([lt.taylor_shift(u_gate_hold[:, g], dt, tau_unit[g] / (c_conv * U_inf_t))
                                for g in range(lt.N_GATES)])
    mode = {'B': 'hold', 'C': 'linear'}[variant]
    return lt.rotor_time_gate_means(data, tau_unit, c_conv * U_inf_t, mode=mode)


# %% sweep over c_conv for all variants
c_sweep     = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
subsets     = {'50-115 m': slice(0, 5), '50-145 m': slice(0, 7), '50-200 m': slice(0, 10)}
SDES        = {}
for variant in 'ABC':
    for c in c_sweep:
        u_rt = gate_signals(variant, c)
        for name, sel in subsets.items():
            SDES[variant, c, name] = lt.coherence_sdes(v_0, u_rt[:, sel].mean(1), dt)[0]
        SDES[variant, c, 'single'] = [lt.coherence_sdes(v_0, u_rt[:, g], dt)[0] for g in range(lt.N_GATES)]
    print(f'variant {variant}:')
    for name in subsets:
        print(f'  {name}: ' + ', '.join(f'c={c:.1f}: {SDES[variant, c, name]:.3f}' for c in c_sweep))

print('single gates, c = 1.2 (A | B | C):')
for g in range(lt.N_GATES):
    print(f'  {lt.GATE_DISTANCES[g]:5.0f} m: ' + ' | '.join(f'{SDES[v, 1.2, "single"][g]:.3f}' for v in 'ABC'))

# %% gate-to-gate coherence: is the zero-lag common noise gone? (gate 2 vs gate 1, c = 1.2)
from scipy.signal import coherence, csd
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for variant, ls in zip('AC', ['-', '--']):
    u_rt    = gate_signals(variant, 1.2)
    for g, col in ((1, colors[2]), (4, colors[6])):
        f, c    = coherence(u_rt[:, g] - u_rt[:, g].mean(), u_rt[:, 0] - u_rt[:, 0].mean(),
                            fs=1 / dt, nperseg=2 ** 12, window='hamming')
        f, P    = csd(u_rt[:, g] - u_rt[:, g].mean(), u_rt[:, 0] - u_rt[:, 0].mean(),
                      fs=1 / dt, nperseg=2 ** 12, window='hamming')
        axs[0].plot(f, c, ls, lw=1.5, color=col, label=f'{variant}: {lt.GATE_DISTANCES[g]:.0f} m vs 50 m')
        axs[1].plot(f, np.angle(P), ls, lw=1.5, color=col)
axs[0].set_ylabel('coherence [-]')
axs[1].set_ylabel('phase [rad]')
axs[1].set_xlabel('frequency [Hz]')
axs[1].set_xscale('log')
axs[1].set_xlim(1e-3, 2)
for ax in axs:
    ax.grid(alpha=0.3, which='both')
axs[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's04_gate_coherence.png'), dpi=150)

# %% coherence with v_0 for the best configuration of each variant
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0.5, color='gray', lw=1)
for variant, col in zip('ABC', plt.get_cmap('tab10').colors):
    name    = '50-115 m'
    c_best  = c_sweep[np.nanargmin([SDES[variant, c, name] for c in c_sweep])]
    u_rt    = gate_signals(variant, c_best)
    S, k, g = lt.coherence_sdes(v_0, u_rt[:, subsets[name]].mean(1), dt)
    ax.plot(k, g, lw=1.5, color=col, label=f'{variant}, {name}, c={c_best:.1f}: {S:.3f} D')
ax.set_xscale('log')
ax.set_xlim(1e-3, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('wave number [rad/m]')
ax.set_ylabel('coherence [-]')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's04_coherence_variants.png'), dpi=150)

plt.show()
