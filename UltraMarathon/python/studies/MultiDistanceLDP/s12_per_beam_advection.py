# Step 12 of the multi-distance LDP study: advection per beam. Where does the speed-up c_conv ~ 1.2 come from?
# Hypotheses: inclined structures (in sheared flow large eddies lean forward with height) and/or
# transport with the speed of higher altitudes. Tests with the held signals per beam and gate (not shifted):
# 1) c_conv per beam: delays gate g -> gate 50 m within the same beam (no v_0 involved)
# 2) delay top vs. bottom and left vs. right at the same gate over the beam separation
#    (inclined structures: top-bottom delay grows linearly with the vertical separation dz)
# 3) delay of every beam and gate vs. v_0 (which beam leads the rotor?)
# Beam 1 = top-left, 2 = top-right, 3 = bottom-left, 4 = bottom-right.
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
d_gate  = lt.GATE_DISTANCES
a_ind   = 0.093
nperseg = 2 ** 13
BAND    = (0.005, 0.03)                                 # [Hz] band with good coherence over all gates
cmap    = plt.get_cmap('tab10')

u_bd    = lt.hold_per_beam(data)                        # (n_t, beams, gates), held, not shifted
# start-up: fill NaN with the first valid value of the same beam and gate
for i_b in range(lt.N_BEAMS):
    for i_g in range(lt.N_GATES):
        s = u_bd[:, i_b, i_g]
        s[np.isnan(s)] = s[~np.isnan(s)][0]
U_mean  = np.mean(v_0)
T_unit  = lt.advection_time(x_up, 1.0, a_ind, 1.0)      # travel time for 1 m/s (induction model)

# %% 1) c_conv per beam (and for the 4-beam mean) from gate-to-gate delays within the beam
def c_from_delays(sig):
    """sig: (n_t, n_gates). Returns c_conv, measured delays and model delays (c=1) of gates 2..10 -> gate 1."""
    dT_model    = (T_unit[1:] - T_unit[0]) / U_mean
    meas, w     = np.zeros(lt.N_GATES - 1), np.zeros(lt.N_GATES - 1)
    for i_g in range(1, lt.N_GATES):
        meas[i_g - 1], w[i_g - 1] = lt.phase_delay(sig[:, i_g], sig[:, 0], dt, BAND, nperseg)
    return np.sum(w * dT_model ** 2) / np.sum(w * dT_model * meas), meas, dT_model


fig, ax = plt.subplots(figsize=(6.5, 4.5))
for i_b in range(lt.N_BEAMS):
    c, meas, dT_model = c_from_delays(u_bd[:, i_b, :])
    print(f'1) beam {i_b + 1}: c_conv = {c:.3f}')
    ax.plot(dT_model, meas, 'o-', lw=1.5, ms=5, color=cmap(i_b), label=f'beam {i_b + 1}: c = {c:.2f}')
c, meas, dT_model = c_from_delays(np.mean(u_bd, axis=1))
print(f'1) 4-beam mean: c_conv = {c:.3f}')
ax.plot(dT_model, meas, 'k-', lw=2, label=f'4-beam mean: c = {c:.2f}')
ax.plot(dT_model, dT_model, '--', color='gray', lw=1, label='c = 1')
ax.set_xlabel('model travel time gate -> 50 m with mean v_0 [s]')
ax.set_ylabel('measured delay [s]')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's12_c_conv_per_beam.png'), dpi=150)

# %% 2) delay between beams at the same gate: top vs. bottom, left vs. right
_, y_pt, z_pt = lt.measurement_points()
pairs   = {'top-left vs bottom-left (1-3)': (0, 2, 'z'), 'top-right vs bottom-right (2-4)': (1, 3, 'z'),
           'top-left vs top-right (1-2)': (0, 1, 'y'), 'bottom-left vs bottom-right (3-4)': (2, 3, 'y')}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for (name, (b1, b2, axis)), col in zip(pairs.items(), cmap.colors):
    lead    = np.array([lt.phase_delay(u_bd[:, b1, g], u_bd[:, b2, g], dt, BAND, nperseg)[0]
                        for g in range(lt.N_GATES)])
    sep     = (z_pt if axis == 'z' else y_pt)[b1] - (z_pt if axis == 'z' else y_pt)[b2]
    p       = np.polyfit(sep, lead, 1)
    print(f'2) {name:36s}: lead of first beam = {p[0] * 100:+.2f} s per 100 m separation, '
          f'offset {p[1]:+.2f} s | per gate: ' + ' '.join(f'{l:+.2f}' for l in lead))
    ax = axs[0] if axis == 'z' else axs[1]
    ax.plot(sep, lead, 'o-', lw=1.5, ms=5, color=col, label=f'{name}: {p[0] * 100:+.2f} s/100 m')
for ax, label in zip(axs, ['vertical separation dz [m]', 'lateral separation dy [m]']):
    ax.axhline(0, color='gray', lw=1)
    ax.set_xlabel(label)
    ax.set_ylabel('lead of first beam [s]')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's12_beam_pair_delays.png'), dpi=150)

# %% 3) delay of every beam and gate vs. v_0 (positive: lidar leads)
lead_v0 = np.array([[lt.phase_delay(u_bd[:, b, g], v_0, dt, BAND, nperseg)[0] for g in range(lt.N_GATES)]
                    for b in range(lt.N_BEAMS)])
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for i_b in range(lt.N_BEAMS):
    p = np.polyfit(x_up, lead_v0[i_b], 1)
    print(f'3) beam {i_b + 1} vs v_0: lead = {p[0] * 100:.3f} s per 100 m + {p[1]:+.2f} s '
          f'-> c_conv = {1 / (p[0] * U_mean):.3f} | per gate: ' + ' '.join(f'{l:5.2f}' for l in lead_v0[i_b]))
    ax.plot(x_up, lead_v0[i_b], 'o-', lw=1.5, ms=5, color=cmap(i_b), label=f'beam {i_b + 1}')
ax.plot(x_up, T_unit / U_mean, '--', color='gray', lw=1, label='model c = 1 (induction)')
ax.set_xlabel('distance upstream of rotor [m]')
ax.set_ylabel('lead of beam signal over v_0 [s]')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's12_beam_lead_v0.png'), dpi=150)

# %% 4) consequence for the LDP: advect every beam with its own c_conv (aligned to v_0) vs. uniform c
weights = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
weights = weights / weights.sum()
u_hold  = np.nanmean(lt.hold_per_beam(data), axis=1)
u_hold  = np.where(np.isnan(u_hold), 18.0, u_hold)
U_adv   = lt.lowpass_first_order(np.mean(u_hold / (1 - a_ind * lt.induction_shape(x_up)), axis=1),
                                 dt, 1 / 5.0, x0=18.0)
u_rt_c  = {}                                            # rotor time beam signals per c_conv (cache)


def sdes_per_beam_c(c_beams):
    for c in set(c_beams):
        if c not in u_rt_c:
            u_rt_c[c] = lt.rotor_time_beam_signals(data, T_unit, c * U_adv, 'linear')
    u_b     = np.stack([u_rt_c[c_beams[b]][:, b, :] for b in range(lt.N_BEAMS)], axis=1)
    ok      = ~np.isnan(u_b)
    u_gate  = np.where(ok.any(axis=1), np.where(ok, u_b, 0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), 18.0)
    return (lt.coherence_sdes(v_0, u_gate @ weights, dt)[0],
            [lt.coherence_sdes(v_0, u_gate[:, g], dt)[0] for g in (0, 3, 6, 9)])


for name, c_beams in [('uniform c = 1.0', [1.0] * 4), ('uniform c = 1.2', [1.2] * 4),
                      ('per beam from 3) (0.9, 1.1, 1.4, 1.4)', [0.9, 1.1, 1.4, 1.4]),
                      ('top 1.0 / bottom 1.4', [1.0, 1.0, 1.4, 1.4])]:
    S, S_g = sdes_per_beam_c(c_beams)
    print(f'4) {name:38s}: weighted SDES = {S:.3f} D | single gates 50/100/145/200 m: '
          + ' '.join(f'{s:.3f}' for s in S_g))

plt.show()
