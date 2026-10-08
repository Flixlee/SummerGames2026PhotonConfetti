# Step 3 of the multi-distance LDP study: induction zone, vertical profile and advection speed.
# - fit of an inflow model (power law profile, induction zone, lateral/vertical component)
#   to the mean u estimate of every beam and gate: 12 h and every 30 min block
# - convection speed of the turbulence from the delays between the lidar gates (no v_0 needed!)
#   c_conv = advection speed / local mean speed, also per 30 min block
# - is c_conv related to shear (alpha) -> atmospheric stability?
# - SDES of the equal weighted multi-distance mean with physically modelled alignment
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
time    = data['time']
n_t     = len(v_0)
x_up    = lt.upstream_distance()
colors  = plt.get_cmap('Blues')(np.linspace(0.35, 1, lt.N_GATES))
PARAM_NAMES = ['U_inf [m/s]', 'alpha [-]', 'a [-]', 'v [m/s]', 'w [m/s]']

T_BLOCK = 1800.0                                        # [s] block length
n_block = int(round(time[-1] / T_BLOCK))
block   = np.minimum((time // T_BLOCK).astype(int), n_block - 1)

# u estimate of every valid measurement (event level) and held 4-beam mean per gate
events  = lt.detect_events(data['beamID'])
beam_e  = data['beamID'][events]
u_e     = np.where(data['isValid'][events],
                   data['lineOfSightWindSpeed'][events] / np.cos(np.deg2rad(lt.ANGLE_TO_CENTERLINE)), np.nan)
block_e = block[events]
u_gate  = np.nanmean(lt.hold_per_beam(data), axis=1)
u_gate  = np.where(np.isnan(u_gate), 18.0, u_gate)


def mean_per_beam_gate(sel):
    """Mean u estimate per beam and gate over the events selected by sel."""
    return np.array([np.nanmean(u_e[sel & (beam_e == i_b + 1)], axis=0) for i_b in range(lt.N_BEAMS)])


# %% inflow model fit over 12 h: with and without profile / induction
u_bd    = mean_per_beam_gate(np.ones(len(events), bool))
fits    = {
    'full model':            {},
    'no shear (alpha=0)':    {1: 0.0},
    'no induction (a=0)':    {2: 0.0},
}
print(f'mean v_0 = {np.mean(v_0):.2f} m/s, assumed hub height = {lt.HUB_HEIGHT:.0f} m')
for name, fixed in fits.items():
    p, rms = lt.fit_inflow_model(u_bd, fixed=fixed)
    fits[name] = p
    print(f'{name:20s}: ' + ', '.join(f'{n} = {v:.3f}' for n, v in zip(PARAM_NAMES, p)) + f'  | rms = {rms:.3f} m/s')
p_12h   = fits['full model']

x, _, z = lt.measurement_points()
fig, ax = plt.subplots(figsize=(7, 4))
cmap    = plt.get_cmap('tab10')
x_fine  = np.linspace(0, 200, 200)
for i_b in range(lt.N_BEAMS):
    ax.plot(x_up, u_bd[i_b], 'o', ms=6, color=cmap(i_b), label=f'beam {i_b + 1}')
    sy, sz  = lt.BEAM_SIGN_Y[i_b], lt.BEAM_SIGN_Z[i_b]
    z_fine  = lt.Z_LIDAR + sz * (x_fine - lt.X_LIDAR) * np.tan(np.deg2rad(lt.ANGLE_VER))
    ax.plot(x_fine, lt.inflow_model(p_12h, x_fine, sy, sz, z_fine), lw=1.5, color=cmap(i_b))
ax.plot(x_fine, p_12h[0] * (1 - p_12h[2] * lt.induction_shape(x_fine)), 'k--', lw=1.5, label='U_inf (1 - a f(x)) at hub')
ax.axhline(np.mean(v_0), color='gray', lw=1, label='mean v_0')
ax.set_xlabel('distance upstream of rotor [m]')
ax.set_ylabel('mean u estimate [m/s]')
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's03_inflow_fit_12h.png'), dpi=150)

# %% inflow model fit per 30 min block
p_block = np.full((n_block, 5), np.nan)
v0_block = np.array([np.mean(v_0[block == i]) for i in range(n_block)])
for i in range(n_block):
    p_block[i], _ = lt.fit_inflow_model(mean_per_beam_gate(block_e == i), p0=p_12h)
print('block  v_0   ' + '  '.join(f'{n:>11s}' for n in PARAM_NAMES))
for i in range(n_block):
    print(f'{i:5d} {v0_block[i]:5.2f}  ' + '  '.join(f'{v:11.3f}' for v in p_block[i]))

# %% convection speed per block from delays between lidar gates (relative to gate 1)
band    = (0.005, 0.03)                                 # [Hz] band with good gate-to-gate coherence
nperseg = 2 ** 12


def fit_c_conv(u, p_inflow, sel=slice(None)):
    """c_conv from delays of gates 2..10 relative to gate 1 (weighted LS through origin).

    Returns c_conv, measured delays, model delays for c_conv = 1.
    """
    U_inf, a    = p_inflow[0], p_inflow[2]
    T_model     = lt.advection_time(x_up, U_inf, a, 1.0)
    dT_model    = T_model[1:] - T_model[0]
    dT_meas     = np.zeros(lt.N_GATES - 1)
    weight      = np.zeros(lt.N_GATES - 1)
    for i_g in range(1, lt.N_GATES):
        dT_meas[i_g - 1], weight[i_g - 1] = lt.phase_delay(u[sel, i_g], u[sel, 0], dt, band, nperseg)
    inv_c       = np.sum(weight * dT_model * dT_meas) / np.sum(weight * dT_model ** 2)
    return 1 / inv_c, dT_meas, dT_model


c_12h, dT_meas_12h, dT_model_12h = fit_c_conv(u_gate, p_12h)
c_12h_noind, _, _ = fit_c_conv(u_gate, fits['no induction (a=0)'])
print(f'12 h: c_conv = {c_12h:.3f} (with induction model), {c_12h_noind:.3f} (without induction)')
print('delay gate -> gate 1, measured [s]:   ' + ' '.join(f'{t:5.2f}' for t in dT_meas_12h))
print('delay gate -> gate 1, model c=1 [s]:  ' + ' '.join(f'{t:5.2f}' for t in dT_model_12h))

c_block = np.array([fit_c_conv(u_gate, p_block[i], block == i)[0] for i in range(n_block)])
print('c_conv per block: ' + ' '.join(f'{c:.2f}' for c in c_block))

# %% time series of the block parameters
t_block = (np.arange(n_block) + 0.5) * T_BLOCK / 3600
fig, axs = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
series  = [(v0_block, 'mean v_0 [m/s]'), (p_block[:, 1], 'shear exponent alpha [-]'),
           (p_block[:, 2], 'induction factor a [-]'),
           (np.rad2deg(np.arctan2(p_block[:, 3], p_block[:, 0])), 'misalignment atan(v/U) [deg]'),
           (c_block, 'c_conv [-]')]
for ax, (y, label) in zip(axs, series):
    ax.plot(t_block, y, 'o-', lw=1.5, ms=5)
    ax.set_ylabel(label, fontsize=8)
    ax.grid(alpha=0.3)
axs[-1].set_xlabel('time since start of data set [h]')
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's03_block_parameters.png'), dpi=150)

fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, (y, label) in zip(axs, [(p_block[:, 1], 'shear exponent alpha [-]'), (p_block[:, 2], 'induction factor a [-]'),
                                 (v0_block, 'mean v_0 [m/s]')]):
    sc = ax.scatter(y, c_block, c=t_block, cmap='viridis', s=30)
    r  = np.corrcoef(y, c_block)[0, 1]
    ax.set_xlabel(label)
    ax.set_ylabel('c_conv [-]')
    ax.set_title(f'r = {r:.2f}', fontsize=9)
    ax.grid(alpha=0.3)
fig.colorbar(sc, ax=axs, label='time [h]')
fig.savefig(os.path.join(lt.FIG_DIR, 's03_c_conv_scatter.png'), dpi=150, bbox_inches='tight')

# %% SDES with physically modelled alignment
# online free stream estimate: 60 s low-pass of the gate means corrected by induction (12 h fit)
# (profile correction of the 4-beam mean is small, see inflow fit; to be added in the online LDP)
T_mean  = 60.0
gain_d  = 1 - p_12h[2] * lt.induction_shape(x_up)
U_inf_t = lt.lowpass_first_order(np.mean(u_gate / gain_d, axis=1), dt, 1 / T_mean, x0=18.0)


def sdes_aligned(gates, c_conv_t, a, tau_0=0.0):
    """SDES of the equal weighted mean of the selected gates, aligned with the advection model.

    c_conv_t: scalar or array (n_t,) of convection factors; a: induction factor (scalar).
    """
    T_unit  = lt.advection_time(x_up, 1.0, a, 1.0)       # travel time for U_inf*c_conv = 1 m/s
    u_al    = [lt.taylor_shift(u_gate[:, g], dt, np.maximum(T_unit[g] / (c_conv_t * U_inf_t) + tau_0, 0))
               for g in gates]
    return lt.coherence_sdes(v_0, np.mean(u_al, axis=0), dt)[0]


gates   = range(0, 5)                                   # 50 - 115 m (best subset of step 2)
c_t     = c_block[block]                                # block-wise c_conv (offline oracle!)
print(f'SDES equal weighted mean 50-115 m:')
print(f'  pure Taylor (c=1, a=0):                 {sdes_aligned(gates, 1.0, 0.0):.3f} D')
print(f'  induction only (c=1, a fitted):         {sdes_aligned(gates, 1.0, p_12h[2]):.3f} D')
print(f'  induction + c_conv 12 h ({c_12h:.2f}):        {sdes_aligned(gates, c_12h, p_12h[2]):.3f} D')
print(f'  induction + c_conv per block (oracle):  {sdes_aligned(gates, c_t, p_12h[2]):.3f} D')
print(f'  empirical fit from step 2:              1.290 D')

# %% is the convection factor scale dependent? c_conv per frequency band (12 h, gates 2..10 -> gate 1)
T_unit  = lt.advection_time(x_up, 1.0, p_12h[2], 1.0)
for band_i in [(0.003, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.07), (0.07, 0.12)]:
    num = den = 0.0
    for i_g in range(1, lt.N_GATES):
        lead, coh = lt.phase_delay(u_gate[:, i_g], u_gate[:, 0], dt, band_i, 2 ** 13)
        if coh > 0.3:                                   # only gate pairs with usable coherence
            dT  = (T_unit[i_g] - T_unit[0]) / p_12h[0]
            num += coh * dT * lead
            den += coh * dT ** 2
    print(f'band {band_i[0]:.3f}-{band_i[1]:.3f} Hz: c_conv = {den / num:.3f}')

# %% SDES as function of a constant c_conv
c_sweep = np.arange(0.8, 1.31, 0.1)
for gates_i, name in ((range(0, 5), '50-115 m'), (range(0, 10), '50-200 m')):
    S = [sdes_aligned(gates_i, c, p_12h[2]) for c in c_sweep]
    print(f'SDES {name} for c_conv = ' + ', '.join(f'{c:.1f}' for c in c_sweep) + ': ' + ', '.join(f'{s:.3f}' for s in S))

plt.show()
