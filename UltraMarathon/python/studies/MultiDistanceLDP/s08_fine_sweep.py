# Step 8 of the multi-distance LDP study: fine sweep around the optimum of s07 (one output for both
# disciplines): preview T_lead and adaptive low-pass k_c (omega_c = k_c * U_inf).
# Evaluated with SDES and the load simulation (cost on 12 h and on both 6 h halves for robustness).
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

a_ind, c_conv, T_mean = 0.093, 1.0, 60.0
weights = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
tau_unit = lt.advection_time(x_up, 1.0, a_ind, 1.0)
u_hold  = np.nanmean(lt.hold_per_beam(data), axis=1)
u_hold  = np.where(np.isnan(u_hold), 18.0, u_hold)
U_inf_t = lt.lowpass_first_order(np.mean(u_hold / (1 - a_ind * lt.induction_shape(x_up)), axis=1),
                                 dt, 1 / T_mean, x0=18.0)

T_leads = [1.5, 1.75, 2.0, 2.25, 2.5]                   # [s]
k_cs    = [0.04, 0.05, 0.06, 0.08, 0.10]                # [rad/m]

# %% sweep
file_sweep = os.path.join(lt.RESULT_DIR, 's08_sweep.csv')
header  = 'T_lead,k_c,SDES,Cost,Cost_h1,Cost_h2,PitchTravel_ratio,PowerStd_ratio,MaxSpeed_ratio,EnergyLoss_kWh,OK'
rows    = []
for T_lead in T_leads:
    u_gate  = lt.rotor_time_gate_means(data, tau_unit, c_conv * U_inf_t, 'linear', T_lead=T_lead)
    REWS    = u_gate @ (weights / weights.sum())
    for k_c in k_cs:
        v_0L    = lt.lowpass_adaptive(REWS, dt, k_c * U_inf_t)
        SDES    = lt.coherence_sdes(v_0, v_0L, dt)[0]
        r       = lt.eval_control(data, v_0L)
        rows.append([T_lead, k_c, SDES, r['Cost'], r['Cost_h1'], r['Cost_h2'], r['PitchTravel_ratio'],
                     r['PowerStd_ratio'], r['MaxSpeed_ratio'], r['EnergyLoss_kWh'], r['OK']])
        print(f'T_lead = {T_lead:.2f} s, k_c = {k_c:.2f}: SDES = {SDES:.3f} D, cost = {100 * r["Cost"]:.2f} % '
              f'(halves {100 * r["Cost_h1"]:.2f} / {100 * r["Cost_h2"]:.2f}), pitch travel = '
              f'{r["PitchTravel_ratio"]:.3f}, power std = {r["PowerStd_ratio"]:.3f}, OK = {r["OK"]}', flush=True)
        np.savetxt(file_sweep, np.array(rows, float), delimiter=',', fmt='%.5f', header=header)

# %% result maps
R       = np.loadtxt(file_sweep, delimiter=',')
shape   = (len(T_leads), len(k_cs))
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, i_col, label, cmap in zip(axs, [3, 2, 6], ['cost [%]', 'SDES [D]', 'pitch travel ratio [-]'],
                                  ['Blues_r', 'Blues_r', 'Blues_r']):
    Z   = R[:, i_col].reshape(shape) * (100 if i_col == 3 else 1)
    im  = ax.imshow(Z, cmap=cmap, origin='lower', aspect='auto')
    for i in range(shape[0]):
        for j in range(shape[1]):
            ok = R[i * shape[1] + j, 10] > 0.5
            ax.text(j, i, f'{Z[i, j]:.3g}' + ('' if ok else '\n(not OK)'), ha='center', va='center',
                    fontsize=7, color='k' if Z[i, j] > np.median(Z) else 'w')
    ax.set_xticks(range(shape[1]), [f'{k:.2f}' for k in k_cs])
    ax.set_yticks(range(shape[0]), [f'{t:.2f}' for t in T_leads])
    ax.set_xlabel('k_c [rad/m]')
    ax.set_ylabel('T_lead [s]')
    ax.set_title(label, fontsize=10)
    fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's08_sweep.png'), dpi=150)

ok      = R[:, 10] > 0.5
i_best  = np.flatnonzero(ok)[np.argmin(R[ok, 3])]
print(f'best valid: T_lead = {R[i_best, 0]:.2f} s, k_c = {R[i_best, 1]:.2f} rad/m, cost = {100 * R[i_best, 3]:.2f} %, '
      f'SDES = {R[i_best, 2]:.3f} D')

plt.show()
