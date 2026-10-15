# Step 16 of the multi-distance LDP study: cumulative ablation from LDP_v3 to LDP_MD (output v_0L).
# Every step adds one building block to the previous one; all SDES with the same evaluation as
# RunUltraMarathon.py. The last step must reproduce the online result of LDP_MD (1.155 D).
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import ldp_tools as lt

data    = lt.load_data()
dt      = data['dt']
v_0     = data['v_0']
n_t     = len(v_0)
x_up    = lt.upstream_distance()
_, _, z_pt = lt.measurement_points()
a_ind, c_conv, alpha = 0.093, 1.0, 0.093
w_B     = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
w_B     = w_B / w_B.sum()
w_eq    = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]) / 5  # equal weights 50-115 m
tau_unit = lt.advection_time(x_up, 1.0, a_ind, 1.0)


def sdes(signal):
    return lt.coherence_sdes(v_0, signal, dt)[0]


def mean_over_beams(u_b):
    ok = ~np.isnan(u_b)
    return np.where(ok.any(axis=1), np.where(ok, u_b, 0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), 18.0)


u_hold  = mean_over_beams(lt.hold_per_beam(data))       # held 4-beam mean per gate (blade ghost free)
U_input = np.mean(u_hold / (1 - a_ind * lt.induction_shape(x_up)), axis=1)  # spatial mean, induction corrected
U_60    = lt.lowpass_first_order(U_input, dt, 1 / 60, x0=18.0)
steps   = []

# %% 0) baseline LDP_v3: gate 50 m, mean over the valid beams of the last 4 measurements, LPF 0.13 rad/s
steps.append(('LDP_v3 (baseline)', sdes(np.load(os.path.join(lt.RESULT_DIR, 'v_0L_LDP_v3.npy')))))

# %% 1) blade ghost free: hold the last valid value of every beam, always mean over 4 beams (same LPF, buffer)
u_f     = lt.lowpass_first_order(u_hold[:, 0], dt, 0.13)
n_buf   = int(np.floor(0.2 / dt))
steps.append(('+ hold per beam (blade ghost free)', sdes(np.concatenate((np.full(n_buf - 1, u_f[0]),
                                                                         u_f[:n_t - n_buf + 1])))))

# %% 2) no low-pass filter and no buffer (the SDES rewards punctuality, not smoothing)
steps.append(('+ no low-pass filter', sdes(u_hold[:, 0])))

# %% 3) time alignment to the rotor (Taylor + induction zone, U from lidar, 60 s IIR)
steps.append(('+ advection to the rotor (Taylor)', sdes(lt.taylor_shift(u_hold[:, 0], dt, tau_unit[0] / (c_conv * U_60)))))

# %% 4) several gates: equal mean of 50-115 m, each gate aligned (hold -> mean -> shift)
u_al    = np.column_stack([lt.taylor_shift(u_hold[:, g], dt, tau_unit[g] / (c_conv * U_60)) for g in range(lt.N_GATES)])
steps.append(('+ multi-distance: equal mean 50-115 m', sdes(u_al @ w_eq)))

# %% 5) every measurement advected with its own time stamp, linear interpolation in rotor time
u_rt    = lt.rotor_time_gate_means(data, tau_unit, c_conv * U_60, 'linear')
steps.append(('+ interpolation in rotor time per beam', sdes(u_rt @ w_eq)))

# %% 6) optimized weights per gate (stage B)
steps.append(('+ optimized gate weights', sdes(u_rt @ w_B)))

# %% 7) advection speed: spatial mean without temporal IIR
u_rt    = lt.rotor_time_gate_means(data, tau_unit, c_conv * U_input, 'linear')
steps.append(('+ advection speed from space (no IIR)', sdes(u_rt @ w_B)))

# %% 8) shear over height: advection speed per beam from a power law
shear   = ((lt.HUB_HEIGHT + z_pt) / (lt.HUB_HEIGHT + lt.Z_LIDAR)) ** alpha
u_b     = np.stack([lt.rotor_time_beam_signals(data, tau_unit, c_conv * U_input[:, None] * shear[b][None, :],
                                               'linear')[:, b, :] for b in range(lt.N_BEAMS)], axis=1)
steps.append(('+ shear over height (alpha = 0.093)', sdes(mean_over_beams(u_b) @ w_B)))

# %% summary
print(f'{"step":45s} {"SDES [D]":>9s} {"change":>8s} {"total":>8s}')
S_0 = steps[0][1]
for i, (name, S) in enumerate(steps):
    change = '' if i == 0 else f'{100 * (S / steps[i - 1][1] - 1):+.1f} %'
    print(f'{name:45s} {S:9.3f} {change:>8s} {100 * (S / S_0 - 1):+7.1f} %')
