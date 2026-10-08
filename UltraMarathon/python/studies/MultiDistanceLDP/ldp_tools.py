"""Helper functions for the multi-distance LDP study (Summer Games 2026, discipline 1).

Contains data loading, the SDES evaluation (identical to RunUltraMarathon.py),
the lidar geometry and vectorized (offline, but causal) building blocks that
mirror what the online LDP does step by step. The vectorized versions are only
used for fast studies; the final algorithm is implemented in LDP_MD.py.
"""
import os
import sys
import numpy as np
import h5py
from scipy.signal import coherence
from scipy.signal.windows import hamming

# paths
STUDY_DIR   = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR  = os.path.abspath(os.path.join(STUDY_DIR, '..', '..'))
DATA_FILE   = os.path.join(PYTHON_DIR, '..', 'data', 'DataSummerGames2026.mat')
FIG_DIR     = os.path.join(STUDY_DIR, 'figures')
RESULT_DIR  = os.path.join(STUDY_DIR, 'results')
sys.path.append(os.path.join(PYTHON_DIR, 'functions'))

# lidar and turbine geometry (see instructions, section "Wind Preview Quality")
D_ROTOR             = 126.0                                                     # [m]   rotor diameter
GATE_DISTANCES      = np.array([50, 65, 80, 100, 115, 130, 145, 160, 175, 200.])  # [m]   along x from lidar
X_LIDAR             = -3.1                                                      # [m]   lidar position relative to rotor (behind)
Z_LIDAR             = 4.1                                                       # [m]   lidar height above hub
ANGLE_HOR           = 15.0                                                      # [deg] horizontal half opening angle
ANGLE_VER           = 12.5                                                      # [deg] vertical half opening angle
ANGLE_TO_CENTERLINE = np.rad2deg(np.arctan(np.hypot(np.tan(np.deg2rad(ANGLE_HOR)),
                                                    np.tan(np.deg2rad(ANGLE_VER)))))  # [deg] = 19.18
# beam 1 = top-left, 2 = top-right, 3 = bottom-left, 4 = bottom-right (seen from behind, y to the left)
BEAM_SIGN_Y         = np.array([+1, -1, +1, -1])
BEAM_SIGN_Z         = np.array([+1, +1, -1, -1])
N_BEAMS             = 4
N_GATES             = len(GATE_DISTANCES)


def upstream_distance():
    """Distance of each gate upstream of the rotor plane [m]."""
    return GATE_DISTANCES + X_LIDAR


def measurement_points():
    """Measurement positions (x upstream of rotor, y, z rel. to hub) as arrays [beam, gate]."""
    x = np.tile(upstream_distance(), (N_BEAMS, 1))
    y = BEAM_SIGN_Y[:, None] * GATE_DISTANCES[None, :] * np.tan(np.deg2rad(ANGLE_HOR))
    z = Z_LIDAR + BEAM_SIGN_Z[:, None] * GATE_DISTANCES[None, :] * np.tan(np.deg2rad(ANGLE_VER))
    return x, y, z


def load_data():
    """Load the measurement data as dict with the same orientation as RunUltraMarathon.py."""
    with h5py.File(DATA_FILE, 'r') as Data:
        data = {
            'time':                 np.array(Data['time']).squeeze(),
            'lineOfSightWindSpeed': np.array(Data['lineOfSightWindSpeed']).T,
            'isValid':              np.array(Data['isValid']).T.astype(bool),
            'beamID':               np.array(Data['beamID']).squeeze().astype(int),
            'v_0':                  np.array(Data['v_0']).squeeze(),
        }
    data['dt'] = data['time'][1] - data['time'][0]
    return data


def coherence_sdes(v_0, v_0L, dt, verbose=False):
    """Coherence and SDES exactly as evaluated in RunUltraMarathon.py.

    v_0L is expected with the same length as v_0 (last sample is ignored, as in the script).
    Returns (SDES in D or NaN, k_est, gamma_Sq_est).
    """
    n_t     = len(v_0)
    n_FFT   = 2 ** 11
    a       = v_0[0:n_t - 1]
    b       = v_0L[0:n_t - 1]
    f_est, gamma_Sq_est = coherence(a - np.mean(a), b - np.mean(b), fs=1 / dt,
                                    window=hamming(n_FFT), noverlap=n_FFT // 2,
                                    nfft=n_FFT, detrend=False)
    k_est   = 2 * np.pi * f_est / np.mean(v_0)
    SDES    = sdes_from_coherence(k_est, gamma_Sq_est)
    if verbose:
        print(f'SDES = {SDES:#.4g} D')
    return SDES, k_est, gamma_Sq_est


def sdes_from_coherence(k_est, gamma_Sq_est):
    """SDES [D] from a coherence curve, same rule as in RunUltraMarathon.py (NaN if not defined)."""
    Idx     = np.arange(np.where(np.diff(gamma_Sq_est) > 0)[0][0] + 1)  # monotonic descending part
    if np.min(gamma_Sq_est[Idx]) <= 0.5 and np.max(gamma_Sq_est[Idx]) > 0.5:
        MCB     = np.interp(0.5, gamma_Sq_est[Idx][::-1], k_est[Idx][::-1])
        return 2 * np.pi / MCB / D_ROTOR
    return np.nan


def multiple_coherence(U, v_0, dt, n_FFT=2 ** 11):
    """Multiple coherence between several input signals U (n_t, p) and v_0 (Welch, as in the SDES evaluation).

    gamma^2(f) = S_vu S_uu^-1 S_uv / S_vv is the coherence of the best linear (LTI, non-causal)
    combination of all inputs -> upper bound for any linear processing of these signals.
    Note: estimate is biased upwards by about p / n_segments.
    Returns f, gamma^2, W (optimal frequency dependent weights, shape (n_f, p)).
    """
    n_t     = len(v_0) - 1                              # last sample ignored as in RunUltraMarathon
    U       = U[:n_t] - np.mean(U[:n_t], axis=0)
    v       = v_0[:n_t] - np.mean(v_0[:n_t])
    win     = hamming(n_FFT)
    step    = n_FFT // 2
    n_seg   = (n_t - n_FFT) // step + 1
    p       = U.shape[1]
    n_f     = n_FFT // 2 + 1
    S_uu    = np.zeros((n_f, p, p), complex)
    S_uv    = np.zeros((n_f, p), complex)
    S_vv    = np.zeros(n_f)
    for i in range(n_seg):
        seg     = slice(i * step, i * step + n_FFT)
        Fu      = np.fft.rfft(U[seg] * win[:, None], axis=0)
        Fv      = np.fft.rfft(v[seg] * win)
        S_uu    += np.conj(Fu)[:, :, None] * Fu[:, None, :]
        S_uv    += np.conj(Fu) * Fv[:, None]
        S_vv    += np.abs(Fv) ** 2
    W       = np.linalg.solve(S_uu, S_uv[..., None])[..., 0]
    gamma2  = np.real(np.einsum('fp,fp->f', np.conj(S_uv), W)) / S_vv
    f       = np.fft.rfftfreq(n_FFT, dt)
    return f, gamma2, W


# ---------------------------------------------------------------------------
# vectorized building blocks (causal: each output only uses current/past input)
# ---------------------------------------------------------------------------
def detect_events(beamID):
    """Sample indices where a new lidar measurement arrives (beamID changes; first sample included)."""
    return np.concatenate(([0], np.flatnonzero(np.diff(beamID) != 0) + 1))


def forward_fill(values, mask):
    """Replace entries where mask is False by the last entry (along axis 0) where mask is True.

    values, mask: arrays of shape (n,) or (n, m). Entries before the first True stay NaN.
    """
    values  = np.asarray(values, dtype=float)
    mask    = np.asarray(mask, dtype=bool)
    idx     = np.where(mask, np.arange(len(values)).reshape((-1,) + (1,) * (values.ndim - 1)), -1)
    idx     = np.maximum.accumulate(idx, axis=0)
    if values.ndim == 1:
        out = values[np.maximum(idx, 0)]
    else:
        out = np.take_along_axis(values, np.maximum(idx, 0), axis=0)
    out[idx < 0] = np.nan
    return out


def hold_per_beam(data, use=None):
    """Last usable u estimate per beam and gate at every sample (blade ghost free holding).

    use: optional boolean array (n_t, n_gates) of additionally accepted samples
         (e.g. outlier mask); default: isValid only.
    Returns u_hold with shape (n_t, n_beams, n_gates); NaN until a beam was valid once.
    """
    events      = detect_events(data['beamID'])
    beam_e      = data['beamID'][events]
    u_e         = data['lineOfSightWindSpeed'][events] / np.cos(np.deg2rad(ANGLE_TO_CENTERLINE))
    ok_e        = data['isValid'][events]
    if use is not None:
        ok_e    = ok_e & use[events]
    # map every sample to the latest event
    sample2event = np.cumsum(np.isin(np.arange(len(data['beamID'])), events)) - 1
    u_hold      = np.full((len(data['beamID']), N_BEAMS, N_GATES), np.nan)
    for i_b in range(N_BEAMS):
        u_held_e            = forward_fill(u_e, ok_e & (beam_e == i_b + 1)[:, None])
        u_hold[:, i_b, :]   = u_held_e[sample2event]
    return u_hold


def lowpass_first_order(x, dt, omega_c, x0=None):
    """Same discrete (Tustin) first order low-pass as LPFilter in LDP_v3.py, vectorized via lfilter."""
    from scipy.signal import lfilter
    a1  = 2 + omega_c * dt
    a0  = omega_c * dt - 2
    b   = np.array([omega_c * dt, omega_c * dt]) / a1
    a   = np.array([1, a0 / a1])
    x0  = x[0] if x0 is None else x0
    zi  = np.array([x0 * (1 - b[0])])   # steady state for constant input x0 (y = x0)
    y, _ = lfilter(b, a, x, zi=zi)
    return y


def phase_delay(a, b, dt, band, nperseg=2 ** 12):
    """Lead of signal a over signal b [s] from the phase of the cross-spectrum in a frequency band.

    Weighted least squares of phase = -omega*tau through the origin (weights: coherence).
    Returns (lead, mean coherence in band).
    """
    from scipy.signal import csd
    a, b    = a - np.mean(a), b - np.mean(b)
    f, P    = csd(a, b, fs=1 / dt, nperseg=nperseg, window='hamming')
    f, c    = coherence(a, b, fs=1 / dt, nperseg=nperseg, window='hamming')
    m       = (f >= band[0]) & (f <= band[1])
    phase   = np.unwrap(np.angle(P[m]))
    omega   = 2 * np.pi * f[m]
    slope   = np.sum(omega * phase * c[m]) / np.sum(omega ** 2 * c[m])
    return -slope, np.mean(c[m])


# ---------------------------------------------------------------------------
# inflow models: vertical profile, induction zone, advection
# ---------------------------------------------------------------------------
HUB_HEIGHT = 100.0      # [m] ASSUMPTION, to be confirmed for the 6.2M126 at Wanderup


def induction_shape(x):
    """Induction zone shape f(x) (Troldborg & Meyer Forsting 2017): U(x)/U_inf = 1 - a*f(x), f(0)=1."""
    xi = np.asarray(x) / (D_ROTOR / 2)
    return 1 - xi / np.sqrt(1 + xi ** 2)


def profile_shape(z, alpha):
    """Power law profile relative to hub height, z relative to hub."""
    return ((HUB_HEIGHT + np.asarray(z)) / HUB_HEIGHT) ** alpha


def inflow_model(p, x, y_sign, z_sign, z):
    """Expected u estimate (v_los / cos) per measurement point.

    p = [U_inf, alpha, a, v, w]: free stream speed at hub height, shear exponent,
    induction factor, lateral and vertical wind component.
    """
    U_inf, alpha, a, v, w = p
    return (U_inf * profile_shape(z, alpha) * (1 - a * induction_shape(x))
            + v * y_sign * np.tan(np.deg2rad(ANGLE_HOR))
            + w * z_sign * np.tan(np.deg2rad(ANGLE_VER)))


def fit_inflow_model(u_mean_bd, p0=(18, 0.2, 0.1, 0, 0), fixed=None):
    """Least squares fit of inflow_model to mean u estimates per beam and gate (shape (4, 10)).

    fixed: optional dict {index: value} of parameters kept constant (e.g. {1: 0} for no shear).
    """
    from scipy.optimize import least_squares
    x, _, z     = measurement_points()
    sy          = np.repeat(BEAM_SIGN_Y[:, None], N_GATES, 1)
    sz          = np.repeat(BEAM_SIGN_Z[:, None], N_GATES, 1)
    fixed       = fixed or {}
    free        = [i for i in range(5) if i not in fixed]
    ok          = ~np.isnan(u_mean_bd)

    def full(q):
        p           = np.array(p0, dtype=float)
        p[free]     = q
        for i, val in fixed.items():
            p[i]    = val
        return p

    def residual(q):
        return (inflow_model(full(q), x, sy, sz, z) - u_mean_bd)[ok]

    sol = least_squares(residual, np.array(p0, dtype=float)[free])
    return full(sol.x), np.sqrt(np.mean(sol.fun ** 2))


def advection_time(x, U_inf, a, c_conv, n_grid=400):
    """Travel time from upstream distance x to the rotor: int_0^x dx' / (c_conv*U_inf*(1 - a*f(x'))).

    x: array of distances; U_inf, a, c_conv: scalars or arrays (broadcast against x).
    """
    x           = np.asarray(x, dtype=float)
    s           = np.linspace(0, 1, n_grid)
    xx          = x[..., None] * s
    integrand   = 1 / (1 - np.asarray(a)[..., None] * induction_shape(xx))
    return np.trapezoid(integrand, xx, axis=-1) / (np.asarray(c_conv) * np.asarray(U_inf))


def rotor_time_gate_means(data, tau_unit, U_adv_t, mode='linear', fill=18.0, T_lead=0.0):
    """4-beam mean per gate in rotor time (n_t, n_gates), see rotor_time_beam_signals."""
    u_bd = rotor_time_beam_signals(data, tau_unit, U_adv_t, mode, T_lead)
    with np.errstate(invalid='ignore'):
        u_rt = np.nanmean(u_bd, axis=1)
    return np.where(np.isnan(u_rt), fill, u_rt)


def rotor_time_beam_signals(data, tau_unit, U_adv_t, mode='linear', T_lead=0.0):
    """Signal of every beam and gate in rotor time, built from the individual measurements (causal).

    Every valid measurement of beam b at gate d, taken at t_m, arrives at the rotor at
    t_a = t_m + tau_unit[d] / U_adv_t(t_m). At rotor time t only measurements with t_m <= t
    are known. Between two arrivals the value is
      - 'linear': interpolated linearly, if the later measurement is already known (else held),
      - 'hold':   held (zero order hold in rotor time).
    Before the first arrival of a beam the value is NaN.

    tau_unit: (n_gates,) travel time for an advection speed of 1 m/s [s*m/s]
    U_adv_t:  (n_t,) or (n_t, n_gates) advection speed known at each time step [m/s] (e.g. c_conv * U_inf(t))
    T_lead:   the value at the rotor at t + T_lead is estimated (preview for control) [s]
    Returns u_bd with shape (n_t, n_beams, n_gates).
    """
    time        = data['time']
    t_query     = time + T_lead
    events      = detect_events(data['beamID'])
    beam_e      = data['beamID'][events]
    t_e         = time[events]
    u_e         = data['lineOfSightWindSpeed'][events] / np.cos(np.deg2rad(ANGLE_TO_CENTERLINE))
    ok_e        = data['isValid'][events]
    U_e         = U_adv_t[events]
    if U_e.ndim == 1:
        U_e     = np.repeat(U_e[:, None], N_GATES, axis=1)
    u_bd        = np.full((len(time), N_BEAMS, N_GATES), np.nan)
    for i_g in range(N_GATES):
        for i_b in range(N_BEAMS):
            sel     = (beam_e == i_b + 1) & ok_e[:, i_g]
            t_m     = t_e[sel]
            u_m     = u_e[sel, i_g]
            t_a     = np.maximum.accumulate(t_m + tau_unit[i_g] / U_e[sel, i_g])  # monotonic arrival times
            k       = np.searchsorted(t_a, t_query, side='right') - 1         # last arrival <= t + T_lead
            has     = k >= 0
            k       = np.maximum(k, 0)
            val     = u_m[k]
            if mode == 'linear':
                k1      = np.minimum(k + 1, len(t_m) - 1)
                known   = has & (k1 > k) & (t_m[k1] <= time)                  # next measurement already taken
                w       = np.where(known, (t_query - t_a[k]) / np.maximum(t_a[k1] - t_a[k], 1e-9), 0.0)
                val     = val + np.clip(w, 0, 1) * (u_m[k1] - val)
            u_bd[:, i_b, i_g] = np.where(has, val, np.nan)
    return u_bd


def taylor_shift(u, dt, tau):
    """Causal variable delay: y(t) = u(t - tau(t)) with linear interpolation (tau >= 0).

    u, tau: arrays of shape (n_t,). Values before t=0 are taken as u[0].
    """
    t       = np.arange(len(u)) * dt
    return np.interp(t - tau, t, u, left=u[0])


def lowpass_adaptive(x, dt, omega_t):
    """First order Tustin low-pass with time varying corner frequency omega_t (same as in LDP_MD)."""
    y       = np.empty_like(x)
    y_last  = x_last = x[0]
    wdt     = omega_t * dt
    for i in range(len(x)):
        y_last  = ((2 - wdt[i]) * y_last + wdt[i] * (x[i] + x_last)) / (2 + wdt[i])
        x_last  = x[i]
        y[i]    = y_last
    return y


# ---------------------------------------------------------------------------
# control evaluation (discipline 2), same as RunUltraMarathon.py with the simple feedforward
# ---------------------------------------------------------------------------
def _control_setup(data):
    from NREL5MWDefaultParameter_SLOW import NREL5MWDefaultParameter_SLOW
    from NREL5MWDefaultParameter_FBNREL import NREL5MWDefaultParameter_FBNREL
    from rpm2radPs import rpm2radPs
    cwd         = os.getcwd()
    os.chdir(PYTHON_DIR)                                # parameter file loads 'functions/...mat'
    try:
        Parameter = NREL5MWDefaultParameter_FBNREL(NREL5MWDefaultParameter_SLOW())
    finally:
        os.chdir(cwd)
    x_0         = np.array([rpm2radPs(12.1), 0.2, 0, np.deg2rad(13), 0])
    y_0         = np.array([rpm2radPs(12.1) * 97, np.deg2rad(13), 0, 5e6])
    return Parameter, x_0, y_0


def _simulate(data, v_0L, gradient, Parameter, x_0, y_0):
    """Simulation loop of RunUltraMarathon.py; v_0L=None -> feedback only. Returns key figures."""
    import rainflow
    from FBController import FBController, reset_FBController
    from SLOW import SLOW
    dt, v_0 = data['dt'], data['v_0']
    n_t     = len(v_0)
    x       = np.full((n_t, 5), np.nan)
    y       = np.full((n_t, 4), np.nan)
    x[0], y[0] = x_0, y_0
    reset_FBController()
    for i_t in range(n_t - 1):
        u_FF = 0.0 if v_0L is None else (v_0L[i_t] - v_0L[max(i_t - 1, 0)]) / dt * gradient
        u    = FBController(y[i_t], u_FF, dt, Parameter)
        x[i_t + 1], y[i_t + 1] = SLOW(x[i_t], u, v_0[i_t], dt, Parameter)
    m, N_REF = 4, 2e6 / (20 * 8760) * 12
    M_yT    = Parameter.Turbine.HubHeight * (Parameter.Turbine.c_eT * x[:, 2] + Parameter.Turbine.k_eT * x[:, 1])

    def DEL(M):
        c = np.array([[cyc[2], cyc[0]] for cyc in rainflow.extract_cycles(M)])
        return (np.sum(c[:, 1] ** m * c[:, 0]) / N_REF) ** (1 / m)

    h = n_t // 2
    return {'MaxSpeed':     np.max(y[:, 0]),
            'Energy':       np.sum(y[:, 3]) * dt,
            'PowerStd':     np.std(y[:, 3], ddof=1),
            'TowerDEL':     DEL(M_yT),
            'TowerDEL_h1':  DEL(M_yT[:h]),                  # halves: only for robustness checks
            'TowerDEL_h2':  DEL(M_yT[h:]),
            'PitchTravel':  np.sum(np.abs(np.diff(y[:, 1])))}


def eval_control(data, v_0L, gradient=np.deg2rad(1)):
    """Cost and constraints of discipline 2 for a lidar signal v_0L with the simple feedforward.

    The feedback-only reference is cached in results/fb_reference.npz.
    Returns dict with Cost [-], the constraint margins (LA relative to FB) and OK flag.
    """
    Parameter, x_0, y_0 = _control_setup(data)
    file_fb = os.path.join(RESULT_DIR, 'fb_reference.npz')
    FB      = dict(np.load(file_fb)) if os.path.exists(file_fb) else {}
    if 'TowerDEL_h1' not in FB:
        FB = _simulate(data, None, gradient, Parameter, x_0, y_0)
        os.makedirs(RESULT_DIR, exist_ok=True)
        np.savez(file_fb, **FB)
    LA = _simulate(data, v_0L, gradient, Parameter, x_0, y_0)
    res = {
        'Cost':             LA['TowerDEL'] / FB['TowerDEL'],
        'Cost_h1':          LA['TowerDEL_h1'] / FB['TowerDEL_h1'],
        'Cost_h2':          LA['TowerDEL_h2'] / FB['TowerDEL_h2'],
        'EnergyLoss_kWh':   (FB['Energy'] - LA['Energy']) / 3.6e6,
        'MaxSpeed_ratio':   LA['MaxSpeed'] / FB['MaxSpeed'],
        'PowerStd_ratio':   LA['PowerStd'] / FB['PowerStd'],
        'PitchTravel_ratio': LA['PitchTravel'] / FB['PitchTravel'],
    }
    res['OK'] = bool(res['EnergyLoss_kWh'] <= 1 and res['MaxSpeed_ratio'] <= 1
                     and res['PowerStd_ratio'] <= 1 and res['PitchTravel_ratio'] <= 1)
    return res
