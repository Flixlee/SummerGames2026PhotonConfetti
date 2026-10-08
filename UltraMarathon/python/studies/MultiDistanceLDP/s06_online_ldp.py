# Step 6 of the multi-distance LDP study: first online version LDP_MD in the time loop.
# - same call pattern as LDP_v3 in RunUltraMarathon.py (one call per time step, only lidar signals)
# - check that the online version reproduces the offline (vectorized) result of s05
# - SDES on 12 h and on both halves for: equal weights 50-115 m and stage B weights
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import time as timer
import numpy as np
import ldp_tools as lt
from LDP_MD import LDP_MD

data    = lt.load_data()
dt      = data['dt']
v_0     = data['v_0']
n_t     = len(v_0)
HALVES  = {'1st half': slice(0, n_t // 2), '2nd half': slice(n_t // 2, n_t), 'full': slice(0, n_t)}

# parameters (as they would be set in RunUltraMarathon.py)
LDP = {
    'NumberOfBeams':    4,                          # [-]       number of beams
    'AngleToCenterline': 19.176,                    # [deg]     angle of the beams to the centerline
    'GateDistances':    [50, 65, 80, 100, 115, 130, 145, 160, 175, 200],  # [m] along x from lidar
    'X_Lidar':          -3.1,                       # [m]       lidar position relative to rotor (behind)
    'RotorDiameter':    126.0,                      # [m]
    'InductionFactor':  0.093,                      # [-]       induction zone model (fit s03)
    'c_conv':           1.0,                        # [-]       advection speed / U_inf (SDES optimum s04)
    'T_mean':           60.0,                       # [s]       time constant of U_inf estimate
    'U_init':           18.0,                       # [m/s]     initial value
    'BufferSize':       64,                         # [-]       measurements per beam and gate in flight
    'Weights':          None,                       # [-]       weights per gate (set below)
}
weight_sets = {
    'equal 50-115 m':   np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], float),
    'stage B (s05)':    np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy')),
}

# %% offline reference (vectorized, from s05 cache)
u_bd    = np.load(os.path.join(lt.RESULT_DIR, 'u_bd_rotor_time_c1.00.npy'))
u_gate  = np.nanmean(u_bd, axis=1)
u_gate  = np.where(np.isnan(u_gate), 18.0, u_gate)

# %% online runs: loop as in RunUltraMarathon.py
results = {}
for name, w in weight_sets.items():
    ldp     = LDP_MD({**LDP, 'Weights': w})
    v_0L    = np.full(n_t, np.nan)
    t_start = timer.perf_counter()
    for i_t in range(n_t - 1):
        v_0L[i_t] = ldp.step(data['isValid'][i_t, :], data['beamID'][i_t],
                             data['lineOfSightWindSpeed'][i_t, :], dt)
    t_run   = timer.perf_counter() - t_start
    v_0L_offline = u_gate @ (w / w.sum())
    diff    = np.max(np.abs(v_0L[:-1] - v_0L_offline[:-1]))
    results[name] = v_0L
    np.save(os.path.join(lt.RESULT_DIR, f'v_0L_LDP_MD_{name.split()[0]}.npy'), v_0L)
    print(f'{name}: run time {t_run:.0f} s ({1e6 * t_run / n_t:.0f} us per step), '
          f'max |online - offline| = {diff:.2e} m/s')
    print('   SDES: ' + ', '.join(f'{p}: {lt.coherence_sdes(v_0[s], v_0L[s], dt)[0]:.3f} D'
                                  for p, s in HALVES.items()))
    lt.coherence_sdes(v_0, v_0L, dt, verbose=True)          # exactly as in RunUltraMarathon.py
