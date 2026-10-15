# Step 20 of the multi-distance LDP study: Mann parameters from the nacelle lidar data (the starting idea).
# Fit of (alpha epsilon^(2/3), L, Gamma) to the line-of-sight auto spectra and the cross-spectra between the beams
# at the same gate (65, 100, 145 m), see mann_fit.py. No cross-spectra along a beam (wind evolution).
# 1) model table on an (L, Gamma) grid (hipersim tensor)
# 2) validation: hipersim fields with known parameters, sampled with the real beamID / isValid sequences,
#    same processing -> are the parameters recovered? (also checks the sign convention of the cross-spectra)
# 3) real data: 12 h and every 30 min block
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import matplotlib.pyplot as plt
from hipersim import MannTurbulenceField
import ldp_tools as lt
import synthetic_lidar as sl
import mann_fit as mf

os.makedirs(lt.FIG_DIR, exist_ok=True)
L_GRID      = np.geomspace(15, 300, 16)
GAMMA_GRID  = np.arange(0.0, 5.01, 0.5)

# %% 1) model table (cached)
file_table = os.path.join(lt.RESULT_DIR, 's20_model_table.npz')
if os.path.exists(file_table):
    T_ = np.load(file_table)
    table = (T_['auto'], T_['cross'])
else:
    table = mf.model_table(L_GRID, GAMMA_GRID)
    np.savez(file_table, auto=table[0], cross=table[1])


def pooled_measured(datas_or_blocks, U):
    """Spectra averaged over several records (list of v arrays on the 1 Hz grid)."""
    out = [mf.measured_spectra(v, U) for v in datas_or_blocks]
    return out[0][0], np.mean([o[1] for o in out], 0), np.mean([o[2] for o in out], 0)


# %% 2) validation with hipersim fields of known parameters (6 seeds x 600 s each)
N_VAL   = 6
CASES   = {'L 33.6, Gamma 3.9': (0.30, 33.6, 3.9), 'L 80, Gamma 2.5': (0.17, 80.0, 2.5)}
for name, (ae, L, G) in CASES.items():
    v_list = []
    for seed in range(N_VAL):
        m   = MannTurbulenceField.generate(alphaepsilon=ae, L=L, Gamma=G, Nxyz=(sl.NX, sl.NY, sl.NZ),
                                           dxyz=(sl.DX, sl.DY, sl.DZ), seed=seed, n_cpu=1)
        uvw = np.asarray(m.uvw, np.float32)
        d   = sl.synthetic_data(seed, sl.gate_lines(uvw, mf.GATES), sl.disk_line(uvw))
        v_list.append(mf.los_on_grid(d)[1])
    k1, auto_m, cross_m = pooled_measured(v_list, sl.U_MEAN)
    res     = mf.fit(k1, auto_m, cross_m, table, L_GRID, GAMMA_GRID)
    res_c   = mf.fit(k1, auto_m, np.conj(cross_m), table, L_GRID, GAMMA_GRID)   # opposite sign convention
    print(f'2) {name} (alpha eps = {ae}): fit L = {res["L"]:.1f} m, Gamma = {res["Gamma"]:.1f}, '
          f'alpha eps = {res["alphaepsilon"]:.3f}, cost = {res["cost"].min():.4f} '
          f'| conjugated cross-spectra: cost = {res_c["cost"].min():.4f} (L = {res_c["L"]:.1f}, Gamma = {res_c["Gamma"]:.1f})',
          flush=True)

# %% 3) real data: 12 h and every 30 min block
real    = lt.load_data()
t_grid, v_real = mf.los_on_grid(real)
cos_a   = np.cos(np.deg2rad(lt.ANGLE_TO_CENTERLINE))
T_BLOCK = 1800
n_block = len(t_grid) // T_BLOCK
blocks  = [v_real[i * T_BLOCK:(i + 1) * T_BLOCK] for i in range(n_block)]
U_block = [np.nanmean(b[:, :, mf.GATES]) / cos_a for b in blocks]
k1, auto_m, cross_m = pooled_measured(blocks, np.mean(U_block))
res_12h = mf.fit(k1, auto_m, cross_m, table, L_GRID, GAMMA_GRID)
print(f'3) 12 h: L = {res_12h["L"]:.1f} m, Gamma = {res_12h["Gamma"]:.1f}, alpha eps = {res_12h["alphaepsilon"]:.3f}'
      f'{" (at grid edge!)" if res_12h["edge"] else ""}', flush=True)
params  = []
for i, (v, U) in enumerate(zip(blocks, U_block)):
    k1, a_m, c_m = mf.measured_spectra(v, U)
    r       = mf.fit(k1, a_m, c_m, table, L_GRID, GAMMA_GRID)
    params.append([U, r['alphaepsilon'], r['L'], r['Gamma'], r['edge']])
    print(f'   block {i:2d}: U = {U:5.2f} m/s, L = {r["L"]:6.1f} m, Gamma = {r["Gamma"]:.1f}, '
          f'alpha eps = {r["alphaepsilon"]:.3f}{" (edge)" if r["edge"] else ""}', flush=True)
params  = np.array(params)
np.save(os.path.join(lt.RESULT_DIR, 's20_mann_params_blocks.npy'), params)

# %% plot: parameters per block
t_h     = (np.arange(n_block) + 0.5) * T_BLOCK / 3600
fig, axs = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
for ax, col, label in zip(axs, (0, 1, 2, 3), ('U [m/s]', 'alpha eps^(2/3) [m^(4/3)/s^2]', 'L [m]', 'Gamma [-]')):
    ax.plot(t_h, params[:, col], 'o-', lw=1.5, ms=5)
    ax.set_ylabel(label, fontsize=8)
    ax.grid(alpha=0.3)
axs[-1].set_xlabel('time since start of data set [h]')
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's20_mann_params_blocks.png'), dpi=150)
plt.show()
