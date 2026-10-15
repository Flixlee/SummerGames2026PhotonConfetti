# Step 14 of the multi-distance LDP study: advection speed from space instead of time, and shear over height.
# The advection speed in LDP_MD is already a spatial mean (40 held values: 4 beams x 10 gates, 50-200 m,
# induction corrected) followed by a temporal IIR low-pass (T_mean). A temporal IIR at a fixed position
# mostly averages air that has already passed the rotor (30 s x 18 m/s ~ 540 m), the spatial mean contains
# the air that is about to arrive, without delay. Questions:
# 1) temporal IIR vs. pure spatial mean (T_mean -> 0) and different spatial windows (FIR in space)
# 2) shear over height: advection speed per beam from a power law, U_b,d = U * ((H + z_b,d) / H)^alpha
#    -> the cross-section is sheared on its way to the rotor (top faster than bottom)
# 3) control check (operating point T_lead ~ 1.75 s, k_c = 0.05 rad/m) for the best variant
# All SDES: rotor aligned, no low-pass, T_lead = 0, c_conv = 1.0, weights of stage B.
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
_, _, z_pt = lt.measurement_points()                     # (beams, gates) height relative to hub
a_ind, c_conv, alpha_fit = 0.093, 1.0, 0.093
weights = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
weights = weights / weights.sum()
tau_unit = lt.advection_time(x_up, 1.0, a_ind, 1.0)

u_bd    = lt.hold_per_beam(data)                        # (n_t, beams, gates), held values
ok      = ~np.isnan(u_bd)
u_gate  = np.where(ok.any(axis=1), np.where(ok, u_bd, 0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), 18.0)
u_corr  = u_gate / (1 - a_ind * lt.induction_shape(x_up))   # induction corrected 4-beam mean per gate


def iir(x, T):
    """Temporal first order low-pass (as in LDP_MD); T = 0: no temporal filter."""
    if T == 0:
        return x
    if x.ndim == 1:
        return lt.lowpass_first_order(x, dt, 1 / T, x0=18.0)
    return np.column_stack([lt.lowpass_first_order(x[:, g], dt, 1 / T, x0=18.0) for g in range(x.shape[1])])


def spatial_mean(window):
    """Advection speed per gate (n_t, gates) from a spatial window of induction corrected gate means."""
    U = np.empty((n_t, lt.N_GATES))
    for g in range(lt.N_GATES):
        sel = {'all gates':            slice(0, lt.N_GATES),
               'near gates 50-115 m':  slice(0, 5),
               'far gates 130-200 m':  slice(5, lt.N_GATES),
               'path (rotor ... d)':   slice(0, g + 1),
               'local (d-1 ... d+1)':  slice(max(g - 1, 0), min(g + 2, lt.N_GATES))}[window]
        U[:, g] = u_corr[:, sel].mean(axis=1)
    return U


def sdes(U_adv_gate):
    REWS = lt.rotor_time_gate_means(data, tau_unit, c_conv * U_adv_gate, 'linear') @ weights
    return lt.coherence_sdes(v_0, REWS, dt)[0]


# %% 1) temporal IIR vs. spatial windows
results = {}
for window in ('all gates', 'near gates 50-115 m', 'far gates 130-200 m', 'path (rotor ... d)', 'local (d-1 ... d+1)'):
    U_w = spatial_mean(window)
    for T in (0, 1, 2, 5, 30):
        results[window, T] = sdes(iir(U_w, T))
    print(f'1) {window:22s}: ' + ', '.join(f'T={T:2d}s: {results[window, T]:.3f}' for T in (0, 1, 2, 5, 30)),
          flush=True)

# %% 2) shear over height: advection speed per beam from a power law around the spatial speed
best    = min(results, key=results.get)
U_best  = iir(spatial_mean(best[0]), best[1])
print(f'2) base for the shear test: {best[0]}, T = {best[1]} s (SDES {results[best]:.3f} D)')
for a_scale in (0.0, 0.5, 1.0, 2.0):
    alpha   = alpha_fit * a_scale
    shape   = ((lt.HUB_HEIGHT + z_pt) / lt.HUB_HEIGHT) ** alpha            # (beams, gates)
    shape   = shape / ((lt.HUB_HEIGHT + lt.Z_LIDAR) / lt.HUB_HEIGHT) ** alpha  # mean of the 4 beams ~ lidar height
    u_b     = np.stack([lt.rotor_time_beam_signals(data, tau_unit, c_conv * U_best * shape[b], 'linear')[:, b, :]
                        for b in range(lt.N_BEAMS)], axis=1)
    okb     = ~np.isnan(u_b)
    u_g     = np.where(okb.any(axis=1), np.where(okb, u_b, 0).sum(axis=1) / np.maximum(okb.sum(axis=1), 1), 18.0)
    print(f'2) shear alpha = {alpha:.3f}: SDES = {lt.coherence_sdes(v_0, u_g @ weights, dt)[0]:.3f} D', flush=True)

# %% 3) control check for the best spatial variant vs. the current setting (IIR 30 s on all gates)
k_c = 0.05
for name, U_adv in (('current: all gates, IIR 30 s', iir(spatial_mean('all gates'), 30)),
                    (f'best: {best[0]}, T = {best[1]} s', U_best)):
    U_lpf = U_adv.mean(axis=1)                          # corner frequency of the adaptive low-pass
    for T_lead in (1.5, 1.75, 2.0):
        REWS    = lt.rotor_time_gate_means(data, tau_unit, c_conv * U_adv, 'linear', T_lead=T_lead) @ weights
        v_0L    = lt.lowpass_adaptive(REWS, dt, k_c * U_lpf)
        r       = lt.eval_control(data, v_0L)
        print(f'3) {name:34s} T_lead = {T_lead:.2f} s: SDES = {lt.coherence_sdes(v_0, v_0L, dt)[0]:.3f} D, '
              f'cost = {100 * r["Cost"]:.2f} % ({100 * r["Cost_h1"]:.2f} | {100 * r["Cost_h2"]:.2f}), '
              f'pitch travel = {r["PitchTravel_ratio"]:.3f}, OK = {r["OK"]}', flush=True)

# %% 4) decoupled: advection with the spatial mean (no IIR), corner frequency of the low-pass with IIR 30 s
U_lpf_30 = iir(spatial_mean('all gates'), 30).mean(axis=1)
for window in ('all gates', 'local (d-1 ... d+1)'):
    U_adv = spatial_mean(window)
    for T_lead in (1.5, 1.75, 2.0):
        REWS    = lt.rotor_time_gate_means(data, tau_unit, c_conv * U_adv, 'linear', T_lead=T_lead) @ weights
        v_0L    = lt.lowpass_adaptive(REWS, dt, k_c * U_lpf_30)
        r       = lt.eval_control(data, v_0L)
        print(f'4) advection {window:20s} (T=0), low-pass with IIR 30 s, T_lead = {T_lead:.2f} s: '
              f'SDES = {lt.coherence_sdes(v_0, v_0L, dt)[0]:.3f} D, cost = {100 * r["Cost"]:.2f} % '
              f'({100 * r["Cost_h1"]:.2f} | {100 * r["Cost_h2"]:.2f}), pitch travel = {r["PitchTravel_ratio"]:.3f}, '
              f'OK = {r["OK"]}', flush=True)
