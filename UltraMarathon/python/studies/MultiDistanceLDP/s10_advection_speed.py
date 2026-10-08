# Step 10 of the multi-distance LDP study: which speed transports the turbulence?
# s09: a shorter averaging time T_mean of the advection speed improves the SDES (5 s better than 60 s).
# Question: physics (structures are transported with the local, gusty speed) or an artefact?
# A) role of the induction model (designed for mean values) for short T_mean: a = 0 vs a fitted
# B) source of the advection speed (all causal, first order low-pass with T_mean):
#    global = mean of all gates (as in LDP_MD), path = mean of the gates between rotor and gate d,
#    own = the gate d itself, oracle = v_0 (rotor wind, NOT allowed online: only to rule out that the
#    gain comes from estimating the speed from the same signals that are shifted)
# C) direct physics test without SDES: delay between gate 145 m and 50 m in 60 s windows -> which speed
#    explains the variation of the delay?
# All SDES values: rotor aligned signal without low-pass (T_lead = 0), weights of stage B.
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
n_t     = len(v_0)
x_up    = lt.upstream_distance()
a_fit, c_conv = 0.093, 1.0
weights = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
weights = weights / weights.sum()
u_hold  = np.nanmean(lt.hold_per_beam(data), axis=1)
u_hold  = np.where(np.isnan(u_hold), 18.0, u_hold)


def sdes_for_speed(U_adv, a=a_fit):
    """SDES of the rotor aligned weighted gate mean for a given advection speed (n_t,) or (n_t, n_gates)."""
    tau_unit = lt.advection_time(x_up, 1.0, a, 1.0)
    REWS     = lt.rotor_time_gate_means(data, tau_unit, c_conv * U_adv, 'linear') @ weights
    return lt.coherence_sdes(v_0, REWS, dt)[0]


def lpf(x, T):
    return lt.lowpass_first_order(x, dt, 1 / T, x0=18.0) if x.ndim == 1 else \
        np.column_stack([lt.lowpass_first_order(x[:, g], dt, 1 / T, x0=18.0) for g in range(x.shape[1])])


# %% A) role of the induction model
for T_mean in (5, 60):
    for a in (0.0, a_fit):
        gain    = 1 - a * lt.induction_shape(x_up)
        S       = sdes_for_speed(lpf(np.mean(u_hold / gain, axis=1), T_mean), a)
        print(f'A) T_mean = {T_mean:2d} s, induction a = {a:.3f}: SDES = {S:.3f} D')

# %% B) source of the advection speed
gain    = 1 - a_fit * lt.induction_shape(x_up)
u_corr  = u_hold / gain                                 # induction corrected gate speeds
sources = {
    'global': lambda T: lpf(np.mean(u_corr, axis=1), T),
    'path':   lambda T: lpf(np.cumsum(u_corr, axis=1) / np.arange(1, lt.N_GATES + 1), T),
    'own':    lambda T: lpf(u_corr, T),
    'oracle v_0 (not online)': lambda T: lpf(v_0, T),
}
T_means = [2, 5, 10, 30, 60]
S_B     = {}
for name, U_of_T in sources.items():
    S_B[name] = [sdes_for_speed(U_of_T(T)) for T in T_means]
    print(f'B) {name:24s}: ' + ', '.join(f'T={T}s: {s:.3f}' for T, s in zip(T_means, S_B[name])), flush=True)

fig, ax = plt.subplots(figsize=(7, 4))
for (name, S), col in zip(S_B.items(), plt.get_cmap('tab10').colors):
    ax.plot(T_means, S, 'o-', lw=1.5, ms=5, color=col, label=name)
ax.set_xscale('log')
ax.set_xlabel('T_mean of advection speed [s]')
ax.set_ylabel('SDES [D]')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's10_advection_speed_source.png'), dpi=150)

# %% C) delay between gates in 60 s windows vs. speed
g_far, g_near = 6, 0                                    # 145 m -> 50 m
T_win, T_step = 60.0, 30.0
n_win, n_stp  = int(T_win / dt), int(T_step / dt)
max_lag       = int(12 / dt)
dT_unit       = lt.advection_time(x_up, 1.0, a_fit, 1.0)
dT_unit       = dT_unit[g_far] - dT_unit[g_near]        # travel time far -> near for 1 m/s
U_glob        = np.mean(u_corr, axis=1)
U_60, U_5     = lpf(U_glob, 60), lpf(U_glob, 5)
lag, U_win, U60_c, U5_c, r_max = [], [], [], [], []
for i0 in range(0, n_t - n_win - max_lag, n_stp):
    seg     = slice(i0, i0 + n_win)
    a_far   = u_hold[seg, g_far] - u_hold[seg, g_far].mean()
    xc      = np.array([np.dot(a_far, u_hold[i0 + L:i0 + L + n_win, g_near]
                               - u_hold[i0 + L:i0 + L + n_win, g_near].mean()) for L in range(0, max_lag, 2)])
    norm    = np.sqrt(np.dot(a_far, a_far)) * np.std(u_hold[seg, g_near]) * np.sqrt(n_win)
    k       = np.argmax(xc)
    if 0 < k < len(xc) - 1:
        y0, y1, y2 = xc[k - 1:k + 2]
        k   = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    lag.append(2 * k * dt)
    r_max.append(np.max(xc) / norm)
    U_win.append(np.mean(U_glob[i0:i0 + n_win + int(lag[-1] / dt)]))   # mean speed while travelling (non-causal)
    U60_c.append(U_60[i0 + n_win // 2])
    U5_c.append(U_5[i0 + n_win // 2])
lag, U_win, U60_c, U5_c, r_max = map(np.array, (lag, U_win, U60_c, U5_c, r_max))
good    = (r_max > 0.6) & (lag > 0.5) & (lag < 11.5)
print(f'C) {good.sum()} of {len(lag)} windows with clear correlation peak; mean lag {lag[good].mean():.2f} s')
for name, U in (('window mean speed', U_win), ('60 s low-pass at window centre', U60_c),
                ('5 s low-pass at window centre', U5_c)):
    pred    = dT_unit / U[good]
    r       = np.corrcoef(pred, lag[good])[0, 1]
    c_fit   = np.sum(pred ** 2) / np.sum(pred * lag[good])
    print(f'   lag vs. dT/U with U = {name:32s}: r = {r:.3f}, c_conv (LS) = {c_fit:.2f}')

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(dT_unit / U_win[good], lag[good], '.', ms=4, alpha=0.6)
lim = [dT_unit / U_win[good].max(), dT_unit / U_win[good].min()]
ax.plot(lim, lim, 'k--', lw=1, label='c_conv = 1')
ax.set_xlabel('Taylor travel time 145 m -> 50 m with window mean speed [s]')
ax.set_ylabel('measured delay [s]')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's10_window_delay.png'), dpi=150)

plt.show()
