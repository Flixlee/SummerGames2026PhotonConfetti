# Step 11 of the multi-distance LDP study: c_conv puzzle revisited with a gust-following advection speed.
# s03/s04: physically c_conv ~ 1.15-1.2 (gate-to-gate delays), but with T_mean = 60 s the SDES prefers
# c ~ 0.9-1.0. s10: a short T_mean (2-5 s) is better for the SDES. Does the optimal c move?
# 1) SDES (rotor aligned, no low-pass) for c_conv x T_mean, 12 h and both halves
# 2) control: best combinations with the adaptive low-pass and a small T_lead sweep
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
HALVES  = {'1st': slice(0, n_t // 2), '2nd': slice(n_t // 2, n_t), '12h': slice(0, n_t)}
a_ind   = 0.093
weights = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
weights = weights / weights.sum()
tau_unit = lt.advection_time(x_up, 1.0, a_ind, 1.0)
u_hold  = np.nanmean(lt.hold_per_beam(data), axis=1)
u_hold  = np.where(np.isnan(u_hold), 18.0, u_hold)
U_input = np.mean(u_hold / (1 - a_ind * lt.induction_shape(x_up)), axis=1)
U_T     = {T: lt.lowpass_first_order(U_input, dt, 1 / T, x0=18.0) for T in (2, 5, 30, 60)}


def rews(c_conv, T_mean, T_lead=0.0):
    return lt.rotor_time_gate_means(data, tau_unit, c_conv * U_T[T_mean], 'linear', T_lead=T_lead) @ weights


# %% 1) SDES for c_conv x T_mean
c_list  = [0.9, 1.0, 1.1, 1.2, 1.3]
print('SDES [D] (12h | 1st half | 2nd half), rotor aligned, no low-pass')
for T_mean in (2, 5, 60):
    line = []
    for c in c_list:
        s   = rews(c, T_mean)
        S   = {p: lt.coherence_sdes(v_0[sel], s[sel], dt)[0] for p, sel in HALVES.items()}
        line.append(f'c={c:.1f}: {S["12h"]:.3f} ({S["1st"]:.3f}|{S["2nd"]:.3f})')
    print(f'  T_mean = {T_mean:2d} s: ' + ', '.join(line), flush=True)

# %% 2) control at the operating point (k_c = 0.05 rad/m) for promising combinations
k_c     = 0.05
combos  = [(1.0, 60), (1.0, 5), (1.0, 30), (1.2, 5), (1.2, 30)]
T_leads = {1.0: (1.5, 1.75, 2.0, 2.25), 1.2: (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)}  # c = 1.2 shifts less
print('control (cost [%] 12h | halves), SDES of the filtered signal')
for c, T_mean in combos:
    for T_lead in T_leads[c]:
        v_0L    = lt.lowpass_adaptive(rews(c, T_mean, T_lead), dt, k_c * U_T[T_mean])
        S       = lt.coherence_sdes(v_0, v_0L, dt)[0]
        r       = lt.eval_control(data, v_0L)
        print(f'  c = {c:.1f}, T_mean = {T_mean:2d} s, T_lead = {T_lead:.2f} s: SDES = {S:.3f} D, '
              f'cost = {100 * r["Cost"]:.2f} % ({100 * r["Cost_h1"]:.2f} | {100 * r["Cost_h2"]:.2f}), '
              f'pitch travel = {r["PitchTravel_ratio"]:.3f}, OK = {r["OK"]}', flush=True)
