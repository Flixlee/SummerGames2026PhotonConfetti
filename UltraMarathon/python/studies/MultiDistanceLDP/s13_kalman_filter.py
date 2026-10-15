# Step 13 of the multi-distance LDP study: Kalman filter for the cross-section at every gate.
# Instead of holding the last value of a missing beam, a Kalman filter estimates the 4 modes of the
# cross-section at each gate (synchronous, as found in s12):
#   u_b = m + s_y,b * A + s_z,b * B + s_y,b * s_z,b * C
#   m: 4-beam mean (wanted), A: left-right (v, du/dy), B: top-bottom (w, du/dz, tilt), C: diagonal
# Model: every mode is a random walk (q_mode), every measurement is scalar with noise variance r.
# q and r are estimated from the data (structure function of increments), then scaled in a small sweep.
# Stage 1: filter only (causal, no future information)      -> compare with variant A (hold, mean, shift)
# Stage 2: fixed-lag smoother over n measurements (uses part of the preview, still causal)
#                                                            -> compare with variant C (rotor time interpolation)
# Gates are still combined afterwards (weights of stage B), advection with c_conv = 1.0, T_mean = 5 s.
# All SDES: no low-pass, T_lead = 0.
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import ldp_tools as lt

data    = lt.load_data()
dt      = data['dt']
v_0     = data['v_0']
n_t     = len(v_0)
time    = data['time']
x_up    = lt.upstream_distance()
a_ind, c_conv, T_mean = 0.093, 1.0, 5.0
weights = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
weights = weights / weights.sum()
tau_unit = lt.advection_time(x_up, 1.0, a_ind, 1.0)

events  = lt.detect_events(data['beamID'])
E       = len(events)
t_e     = time[events]
beam_e  = data['beamID'][events] - 1                    # 0..3
u_e     = data['lineOfSightWindSpeed'][events] / np.cos(np.deg2rad(lt.ANGLE_TO_CENTERLINE))
ok_e    = data['isValid'][events]
H       = np.column_stack([np.ones(4), lt.BEAM_SIGN_Y, lt.BEAM_SIGN_Z, lt.BEAM_SIGN_Y * lt.BEAM_SIGN_Z]).astype(float)
sample2event = np.cumsum(np.isin(np.arange(n_t), events)) - 1

# advection speed (as in LDP_MD, T_mean = 5 s)
u_hold  = np.nanmean(lt.hold_per_beam(data), axis=1)
u_hold  = np.where(np.isnan(u_hold), 18.0, u_hold)
U_adv   = c_conv * lt.lowpass_first_order(np.mean(u_hold / (1 - a_ind * lt.induction_shape(x_up)), axis=1),
                                          dt, 1 / T_mean, x0=18.0)


def sdes_from_gate_signals(u_gate_t, shifted):
    """Weighted gate combination -> SDES. shifted=False: gate signals at the gate (shift to rotor here)."""
    if not shifted:
        u_gate_t = np.column_stack([lt.taylor_shift(u_gate_t[:, g], dt, tau_unit[g] / U_adv)
                                    for g in range(lt.N_GATES)])
    return lt.coherence_sdes(v_0, u_gate_t @ weights, dt)[0]


# %% baselines: variant A (hold, mean, shift) and variant C (rotor time interpolation)
S_A     = sdes_from_gate_signals(u_hold, shifted=False)
S_C     = sdes_from_gate_signals(lt.rotor_time_gate_means(data, tau_unit, U_adv, 'linear'), shifted=True)
print(f'baseline A (hold -> mean -> shift):        SDES = {S_A:.3f} D')
print(f'baseline C (rotor time interpolation):     SDES = {S_C:.3f} D')

# %% estimate measurement noise r and random walk intensities q from the data
# per beam: variance of increments of consecutive valid measurements vs. time lag. Turbulence follows
# the Kolmogorov structure function (~ dt^(2/3)), so fit var = 2 r + a * dt^(2/3)
lags, var_inc = [], []
for b in range(4):
    sel     = np.flatnonzero((beam_e == b) & ok_e[:, 3])            # gate 100 m
    du      = np.diff(u_e[sel, 3])
    dl      = np.round(np.diff(t_e[sel]))
    for L in (1, 2, 3, 4, 5):
        if np.sum(dl == L) > 100:
            lags.append(L)
            var_inc.append(np.var(du[dl == L]))
lags, var_inc = np.array(lags, float), np.array(var_inc)
p_sf    = np.polyfit(lags ** (2 / 3), var_inc, 1)
print(f'structure function (gate 100 m): var(du) = {p_sf[1]:.3f} + {p_sf[0]:.3f} * dt^(2/3)')
# The intercept is negative (gaps of 2-3 s are coupled to the rotor position), and the same holds for the
# spatial structure function between gates of the same pulse (overlapping probe volumes). r is therefore
# not identifiable from the data; upper bound from the 15 m gate distance: 2 r <= 0.47 (m/s)^2.
# -> r = 0.05 (m/s)^2 (sigma ~ 0.22 m/s) as starting value, treated as tuning parameter in the sweep.
r_est   = 0.05
# per mode: increments between consecutive complete cycles (all 4 beams valid in the last 4 measurements)
u_held_e = np.stack([lt.forward_fill(u_e[:, 3], ok_e[:, 3] & (beam_e == b)) for b in range(4)], axis=1)
fresh   = ok_e[:, 3].copy()
for s in range(1, 4):
    fresh[s:] &= ok_e[:E - s, 3]
fresh[:3] = False
modes_e = u_held_e @ H / 4                                  # H is orthogonal: H^T H = 4 I
idx     = np.flatnonzero(fresh)
d_t     = t_e[idx[1:]] - t_e[idx[:-1]]
d_mode  = modes_e[idx[1:]] - modes_e[idx[:-1]]
use     = (d_t > 0.9) & (d_t < 3.1)                         # 1-3 s apart (not overlapping cycles)
# random walk equivalent at the typical lag: var = 2 * r/4 + q * dt  (mode noise variance r/4)
q_mode  = np.maximum(np.mean(d_mode[use] ** 2 - 2 * r_est / 4, axis=0) / np.mean(d_t[use]), 1e-4)
print(f'{use.sum()} increments between complete cycles; q per mode [(m/s)^2/s] (m, A, B, C): '
      + ', '.join(f'{q:.3f}' for q in q_mode))


# %% Kalman filter (vectorized over the gates) and fixed-lag smoother
def kalman_modes(q, r):
    """Filter over all measurement events. Returns filtered states Xf (E, gates, 4) and smoother gains G."""
    Q       = np.diag(q)
    X       = np.zeros((lt.N_GATES, 4))
    X[:, 0] = 18.0
    P       = np.tile(np.diag([25.0, 4.0, 4.0, 4.0]), (lt.N_GATES, 1, 1))
    Xf      = np.empty((E, lt.N_GATES, 4))
    Pf      = np.empty((E, lt.N_GATES, 4, 4))
    Pp      = np.empty((E, lt.N_GATES, 4, 4))
    t_last  = t_e[0]
    for k in range(E):
        P       = P + Q * (t_e[k] - t_last)                 # prediction (random walk: X unchanged)
        t_last  = t_e[k]
        Pp[k]   = P
        h       = H[beam_e[k]]
        Ph      = P @ h                                     # (gates, 4)
        S       = Ph @ h + r
        K       = Ph / S[:, None]
        valid   = ok_e[k]
        innov   = np.where(valid, u_e[k] - X @ h, 0.0)
        X       = X + K * innov[:, None]
        P       = np.where(valid[:, None, None], P - K[:, :, None] * Ph[:, None, :], P)
        Xf[k], Pf[k] = X, P
    # RTS gains for random walk: G_i = Pf_i Pp_{i+1}^-1
    G       = np.empty_like(Pf)
    G[:-1]  = np.swapaxes(np.linalg.solve(np.swapaxes(Pp[1:], -1, -2), np.swapaxes(Pf[:-1], -1, -2)), -1, -2)
    G[-1]   = np.eye(4)
    return Xf, G


def fixed_lag(Xf, G, n):
    """Smoothed states x_{j|j+n} for all j (last n entries: filtered). Available at time t_e[j+n]."""
    if n == 0:
        return Xf.copy()
    Y = Xf[n:].copy()                                       # start at k = j + n
    for s in range(1, n + 1):
        i = np.arange(n - s, E - s)                         # index k - s for all windows
        Y = Xf[i] + np.einsum('egij,egj->egi', G[i], Y - Xf[i])
    out = Xf.copy()
    out[:E - n] = Y
    return out


def gate_mean_from_states(Xs, n):
    """Event level estimates of m -> samples. n = 0: hold (causal filter); n > 0: linear interpolation
    between events (causal as long as the travel time > n measurements, here tau >= 2.5 s)."""
    m_e = Xs[:, :, 0]
    if n == 0:
        return m_e[sample2event]
    return np.column_stack([np.interp(time, t_e, m_e[:, g]) for g in range(lt.N_GATES)])


# %% stage 1 and 2 with q, r from the data and a small scaling sweep
for q_scale in (0.25, 1.0, 4.0):
    for r_scale in (0.2, 1.0, 4.0):
        Xf, G   = kalman_modes(q_mode * q_scale, r_est * r_scale)
        res     = []
        for n in (0, 2, 4, 8):
            u_g = gate_mean_from_states(fixed_lag(Xf, G, n), n)
            res.append(sdes_from_gate_signals(u_g, shifted=False))
        print(f'q x {q_scale:4.2f}, r x {r_scale:3.1f}: filter (n=0) {res[0]:.3f} | '
              f'fixed-lag n=2 {res[1]:.3f}, n=4 {res[2]:.3f}, n=8 {res[3]:.3f}   '
              f'(baselines A {S_A:.3f}, C {S_C:.3f})', flush=True)

# %% stage 3: strictly causal smoother with gate and time dependent lag
# For the rotor at time t, gate d needs the cross-section at s = t - tau_d(t). At time t the measurements up to
# t are known, i.e. n_avail = (last event <= t) - (first event >= s) further measurements -> smoother with
# exactly this lag (limited to n_max), linear interpolation between the bracketing events.
def strictly_causal_gate_means(Xf, G, n_max):
    m_lag   = np.stack([fixed_lag(Xf, G, n)[:, :, 0] for n in range(n_max + 2)])  # (lags, E, gates)
    k_now   = sample2event                                  # last event <= t
    u_g     = np.empty((n_t, lt.N_GATES))
    for g in range(lt.N_GATES):
        s       = time - tau_unit[g] / U_adv                # time of the cross-section arriving now
        j1      = np.clip(np.searchsorted(t_e, s), 1, E - 1)    # first event >= s
        j0      = j1 - 1
        n1      = np.clip(k_now - j1, 0, n_max)             # measurements known after j1
        n0      = np.clip(k_now - j0, 0, n_max + 1)
        w       = np.clip((s - t_e[j0]) / (t_e[j1] - t_e[j0]), 0, 1)
        late    = k_now < j1                                # j1 not measured yet: use j0 only
        m0, m1  = m_lag[n0, j0, g], m_lag[n1, j1, g]
        u_g[:, g] = np.where(late, m0, (1 - w) * m0 + w * m1)
    return u_g


Xf, G   = kalman_modes(q_mode, r_est)
for n_max in (2, 4, 8, 12, 20):
    S = sdes_from_gate_signals(strictly_causal_gate_means(Xf, G, n_max), shifted=True)
    print(f'strictly causal smoother, n_max = {n_max:2d}: SDES = {S:.3f} D   (baselines A {S_A:.3f}, C {S_C:.3f})',
          flush=True)

# %% remaining potential of a better gate combination: Wiener bound of the KF gate signals vs. weighted sum
u_kf    = strictly_causal_gate_means(Xf, G, 12)
f, g2, _ = lt.multiple_coherence(u_kf, v_0, dt)
print(f'KF gate signals (n_max = 12): weighted (stage B) {sdes_from_gate_signals(u_kf, shifted=True):.3f} D, '
      f'Wiener bound of the 10 gate signals {lt.sdes_from_coherence(2 * np.pi * f / np.mean(v_0), g2):.3f} D')
