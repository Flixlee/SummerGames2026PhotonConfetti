# Step 9 of the multi-distance LDP study: which "mean" wind speed transports the turbulence?
# The advection speed c_conv * U_inf(t) is estimated online with a first order low-pass (time constant
# T_mean) of the induction corrected gate means. Short T_mean: local transport speed of gusts (and early
# information for control); long T_mean: "true" mean. Sweep of T_mean at the best (T_lead, k_c) of s08:
#   coupled:   T_mean used for the advection AND the adaptive low-pass (as LDP_MD does now)
#   advection: T_mean only for the advection, low-pass corner frequency with T_mean = 60 s
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import ldp_tools as lt

data    = lt.load_data()
dt      = data['dt']
v_0     = data['v_0']
x_up    = lt.upstream_distance()

a_ind, c_conv = 0.093, 1.0
weights = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
tau_unit = lt.advection_time(x_up, 1.0, a_ind, 1.0)
u_hold  = np.nanmean(lt.hold_per_beam(data), axis=1)
u_hold  = np.where(np.isnan(u_hold), 18.0, u_hold)
U_input = np.mean(u_hold / (1 - a_ind * lt.induction_shape(x_up)), axis=1)

# best valid configuration of the fine sweep s08
R       = np.loadtxt(os.path.join(lt.RESULT_DIR, 's08_sweep.csv'), delimiter=',')
ok      = R[:, 10] > 0.5
i_best  = np.flatnonzero(ok)[np.argmin(R[ok, 3])]
T_lead, k_c = R[i_best, 0], R[i_best, 1]
print(f'operating point from s08: T_lead = {T_lead:.2f} s, k_c = {k_c:.2f} rad/m')

# %% sweep
T_means = [5, 10, 20, 30, 60, 120, 300]                 # [s]
U_60    = lt.lowpass_first_order(U_input, dt, 1 / 60.0, x0=18.0)
file_sweep = os.path.join(lt.RESULT_DIR, 's09_sweep.csv')
header  = 'variant(0=coupled;1=advection),T_mean,SDES,Cost,Cost_h1,Cost_h2,PitchTravel_ratio,PowerStd_ratio,OK'
rows    = []
for T_mean in T_means:
    U_T     = lt.lowpass_first_order(U_input, dt, 1 / T_mean, x0=18.0)
    REWS    = lt.rotor_time_gate_means(data, tau_unit, c_conv * U_T, 'linear', T_lead=T_lead) @ (weights / weights.sum())
    for variant, U_lpf in ((0, U_T), (1, U_60)):
        if variant == 1 and T_mean == 60:
            continue                                    # identical to coupled
        v_0L    = lt.lowpass_adaptive(REWS, dt, k_c * U_lpf)
        SDES    = lt.coherence_sdes(v_0, v_0L, dt)[0]
        r       = lt.eval_control(data, v_0L)
        rows.append([variant, T_mean, SDES, r['Cost'], r['Cost_h1'], r['Cost_h2'], r['PitchTravel_ratio'],
                     r['PowerStd_ratio'], r['OK']])
        print(f'{["coupled  ", "advection"][variant]} T_mean = {T_mean:3d} s: SDES = {SDES:.3f} D, '
              f'cost = {100 * r["Cost"]:.2f} % (halves {100 * r["Cost_h1"]:.2f} / {100 * r["Cost_h2"]:.2f}), '
              f'pitch travel = {r["PitchTravel_ratio"]:.3f}, OK = {r["OK"]}', flush=True)
        np.savetxt(file_sweep, np.array(rows, float), delimiter=',', fmt='%.5f', header=header)
