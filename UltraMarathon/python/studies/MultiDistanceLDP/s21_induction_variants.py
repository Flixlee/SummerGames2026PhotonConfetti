# Step 21 of the multi-distance LDP study: is the induction model needed, and should a depend on the
# operating point? In LDP_MD the induction factor a only acts on the timing (advection speed U_adv and travel time
# tau_d), not on the level of the outputs. Variants for the timing:
#   a = 0  |  a = 0.093 (fixed, s03)  |  a(U): linear in the advection speed, fitted to the 30 min blocks of s03
# Evaluation: SDES of v_0L and cost of v_FF (operating point of RunUltraMarathon.py).
# Additionally: mean offset of the output to v_0 without and with an induction correction of the output level
# (every gate divided by 1 - a f(x_d) before the weighting), for the colleague's control concept.
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import ldp_tools as lt

data    = lt.load_data()
dt      = data['dt']
v_0     = data['v_0']
x_up    = lt.upstream_distance()
_, _, z_pt = lt.measurement_points()
w_B     = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
w_B     = w_B / w_B.sum()
alpha, c_conv, T_lead, k_c = 0.093, 1.0, 2.0, 0.05
shear   = ((lt.HUB_HEIGHT + z_pt) / (lt.HUB_HEIGHT + lt.Z_LIDAR)) ** alpha
f_ind   = lt.induction_shape(x_up)
u_bd    = lt.hold_per_beam(data)
ok      = ~np.isnan(u_bd)
u_hold  = np.where(ok.any(axis=1), np.where(ok, u_bd, 0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), 18.0)

# %% a(U) from the 30 min block fits of the inflow model (as in s03)
events  = lt.detect_events(data['beamID'])
beam_e  = data['beamID'][events]
u_e     = np.where(data['isValid'][events],
                   data['lineOfSightWindSpeed'][events] / np.cos(np.deg2rad(lt.ANGLE_TO_CENTERLINE)), np.nan)
block_e = (data['time'][events] // 1800).astype(int)
a_blk, U_blk = [], []
for i in range(block_e.max() + 1):
    u_bd_mean = np.array([np.nanmean(u_e[(block_e == i) & (beam_e == b + 1)], axis=0) for b in range(lt.N_BEAMS)])
    p, _    = lt.fit_inflow_model(u_bd_mean)
    a_blk.append(p[2])
    U_blk.append(p[0])
p_aU    = np.polyfit(U_blk, a_blk, 1)
print(f'a(U) = {p_aU[0]:+.4f} * U {p_aU[1]:+.3f}  (blocks: a = {min(a_blk):.3f} ... {max(a_blk):.3f}, '
      f'U = {min(U_blk):.1f} ... {max(U_blk):.1f} m/s)')


def a_of_U(U):
    return np.clip(np.polyval(p_aU, U), 0.0, 0.33)


# %% LDP with a given a(t): timing via per gate advection speed, optional level correction of the output
def ldp_outputs(a_mode, correct_level=False):
    tau0    = lt.advection_time(x_up, 1.0, 0.0, 1.0)
    dtau    = lt.advection_time(x_up, 1.0, 0.1, 1.0) - tau0          # ~linear in a for small a
    if a_mode == 'a(U)':
        U0      = np.mean(u_hold, axis=1)                             # first guess without induction
        a_t     = a_of_U(np.mean(u_hold / (1 - a_of_U(U0)[:, None] * f_ind), axis=1))
    else:
        a_t     = np.full(len(v_0), float(a_mode))
    U_adv   = np.mean(u_hold / (1 - a_t[:, None] * f_ind), axis=1)
    U_LPF   = lt.lowpass_first_order(U_adv, dt, 1 / 30, x0=18.0)
    tau_t   = tau0 + a_t[:, None] * dtau / 0.1                       # (n_t, gates) travel time for 1 m/s
    out     = {}
    for name, TL in (('v_0L', 0.0), ('v_FF', T_lead)):
        u_b = np.stack([lt.rotor_time_beam_signals(data, tau0, c_conv * U_adv[:, None] * shear[b][None, :]
                                                   * tau0[None, :] / tau_t, 'linear', T_lead=TL)[:, b, :]
                        for b in range(lt.N_BEAMS)], axis=1)
        okb = ~np.isnan(u_b)
        g   = np.where(okb.any(axis=1), np.where(okb, u_b, 0).sum(axis=1) / np.maximum(okb.sum(axis=1), 1), 18.0)
        if correct_level:
            g = g / (1 - a_t[:, None] * f_ind)
        out[name] = g @ w_B
    out['v_FF'] = lt.lowpass_adaptive(out['v_FF'], dt, k_c * U_LPF)
    return out


for a_mode in (0.0, 0.093, 'a(U)'):
    o   = ldp_outputs(a_mode)
    r   = lt.eval_control(data, o['v_FF'])
    print(f'a = {a_mode!s:6s}: SDES v_0L = {lt.coherence_sdes(v_0, o["v_0L"], dt)[0]:.3f} D, '
          f'cost v_FF = {100 * r["Cost"]:.2f} % (OK = {r["OK"]}), '
          f'mean offset v_0L - v_0 = {np.nanmean(o["v_0L"] - v_0):+.2f} m/s', flush=True)

# %% level correction of the output (for absolute values, e.g. gain scheduling or static pitch feedforward)
for a_mode in (0.093, 'a(U)'):
    o   = ldp_outputs(a_mode, correct_level=True)
    d   = o['v_0L'] - v_0
    print(f'level corrected, a = {a_mode!s:6s}: mean offset v_0L - v_0 = {np.nanmean(d):+.2f} m/s, '
          f'std of the 10 min mean offsets = {np.nanstd([np.nanmean(d[i:i + 12000]) for i in range(0, len(d) - 12000, 12000)]):.2f} m/s, '
          f'SDES = {lt.coherence_sdes(v_0, o["v_0L"], dt)[0]:.3f} D', flush=True)
