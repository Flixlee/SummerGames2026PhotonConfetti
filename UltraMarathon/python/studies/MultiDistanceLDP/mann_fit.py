"""Fit of the Mann parameters (alpha epsilon^(2/3), L, Gamma) to nacelle lidar line-of-sight data.

Model (point measurements, no range weighting): cross-spectrum of the line-of-sight speeds of two measurement
points a, b (same range gate, different beams) over the longitudinal wave number k1 = 2 pi f / U:
    chi_ab(k1) = 2 * int int n_a^T Phi(k1, k2, k3) n_b exp(-i (k2 dy + k3 dz)) dk2 dk3   (one-sided)
with the Mann spectral tensor Phi of hipersim (DTU), dy, dz = r_b - r_a and the beam unit vectors n (line-of-sight
positive towards the lidar). The sign convention corresponds to scipy.signal.csd(v_los_a, v_los_b) for a field
with x in wind direction (checked with synthetic hipersim fields in s20).
Fit: grid search over (L, Gamma); alpha epsilon^(2/3) in closed form for every grid point. Observables: the auto
spectra (log) and the normalized cross-spectra rho_ab = chi_ab / sqrt(chi_aa chi_bb) (real and imaginary part).
"""
import numpy as np
from scipy.signal import csd
from hipersim.turbgen.manntensor import manntensorcomponents
import ldp_tools as lt

GATES       = (1, 3, 6)                                 # 65, 100, 145 m
PAIRS       = [(a, b) for a in range(lt.N_BEAMS) for b in range(a + 1, lt.N_BEAMS)]
F_MAX       = 0.1                                       # [Hz] upper limit (1 Hz per beam, gaps)
NPERSEG     = 256                                       # [s] Welch segments on the 1 Hz grid
K1_TABLE    = np.geomspace(5e-4, 0.08, 40)              # [rad/m] wave numbers of the model table
_g          = np.geomspace(1e-5, 5.0, 70)
K23         = np.concatenate((-_g[::-1], [0.0], _g))    # integration grid for k2, k3 [rad/m]


def beam_geometry():
    """Positions (y, z) relative to hub and line-of-sight unit vectors n (3,) per beam and gate."""
    _, y, z = lt.measurement_points()
    n       = np.stack([np.ones((lt.N_BEAMS, lt.N_GATES)),
                        -lt.BEAM_SIGN_Y[:, None] * np.tan(np.deg2rad(lt.ANGLE_HOR)) * np.ones((1, lt.N_GATES)),
                        -lt.BEAM_SIGN_Z[:, None] * np.tan(np.deg2rad(lt.ANGLE_VER)) * np.ones((1, lt.N_GATES))])
    return y, z, n / np.linalg.norm(n, axis=0)


def model_spectra(L, Gamma, k1=K1_TABLE):
    """Model for alpha epsilon^(2/3) = 1: auto spectra (gates, beams, k1) and cross-spectra (gates, pairs, k1)."""
    y, z, n = beam_geometry()
    K1, K2, K3 = np.meshgrid(k1, K23, K23, indexing='ij')
    P11, P22, P33, P12, P13, P23 = manntensorcomponents(K1, K2, K3, Gamma, L, 1.0, 2)
    Phi     = np.array([[P11, P12, P13], [P12, P22, P23], [P13, P23, P33]])
    Phi[~np.isfinite(Phi)] = 0.0

    def integrate(f):
        return 2 * np.trapezoid(np.trapezoid(f, K23, axis=-1), K23, axis=-1)

    auto    = np.empty((len(GATES), lt.N_BEAMS, len(k1)))
    cross   = np.empty((len(GATES), len(PAIRS), len(k1)), complex)
    for i_g, g in enumerate(GATES):
        for b in range(lt.N_BEAMS):
            auto[i_g, b] = integrate(np.einsum('i,ij...,j->...', n[:, b, g], Phi, n[:, b, g]))
        for i_p, (a, b) in enumerate(PAIRS):
            phase   = np.exp(-1j * (K2 * (y[b, g] - y[a, g]) + K3 * (z[b, g] - z[a, g])))
            cross[i_g, i_p] = integrate(np.einsum('i,ij...,j->...', n[:, a, g], Phi, n[:, b, g]) * phase)
    return auto, cross


def model_table(L_grid, Gamma_grid):
    """Model spectra on the (L, Gamma) grid (for alpha epsilon^(2/3) = 1)."""
    auto    = np.empty((len(L_grid), len(Gamma_grid), len(GATES), lt.N_BEAMS, len(K1_TABLE)))
    cross   = np.empty((len(L_grid), len(Gamma_grid), len(GATES), len(PAIRS), len(K1_TABLE)), complex)
    for i, L in enumerate(L_grid):
        for j, G in enumerate(Gamma_grid):
            auto[i, j], cross[i, j] = model_spectra(L, G)
    return auto, cross


def los_on_grid(data, gates=GATES, fs=1.0):
    """Line-of-sight speed of every beam and gate on a common 1 Hz grid (linear interpolation of the
    valid measurements, removes the time offsets between the beams). Returns t_grid, v (t, beams, gates)."""
    t       = data['time']
    events  = lt.detect_events(data['beamID'])
    beam_e  = data['beamID'][events] - 1
    t_grid  = np.arange(t[0], t[-1], 1 / fs)
    v       = np.full((len(t_grid), lt.N_BEAMS, lt.N_GATES), np.nan)
    for b in range(lt.N_BEAMS):
        for g in gates:
            sel = (beam_e == b) & data['isValid'][events, g]
            v[:, b, g] = np.interp(t_grid, t[events][sel], data['lineOfSightWindSpeed'][events[sel], g])
    return t_grid, v


def measured_spectra(v, U, fs=1.0):
    """Measured one-sided wave number spectra of a block: k1, auto (gates, beams, k1), cross (gates, pairs, k1)."""
    kw      = dict(fs=fs, nperseg=NPERSEG, window='hann', detrend='linear')
    f, _    = csd(v[:, 0, GATES[0]], v[:, 0, GATES[0]], **kw)
    m       = (f > 0) & (f <= F_MAX)
    k1      = 2 * np.pi * f[m] / U
    conv    = U / (2 * np.pi)                           # S(f) df = S(k1) dk1
    auto    = np.empty((len(GATES), lt.N_BEAMS, m.sum()))
    cross   = np.empty((len(GATES), len(PAIRS), m.sum()), complex)
    for i_g, g in enumerate(GATES):
        for b in range(lt.N_BEAMS):
            auto[i_g, b] = np.real(csd(v[:, b, g], v[:, b, g], **kw)[1][m]) * conv
        for i_p, (a, b) in enumerate(PAIRS):
            cross[i_g, i_p] = csd(v[:, a, g], v[:, b, g], **kw)[1][m] * conv
    return k1, auto, cross


def fit(k1, auto_m, cross_m, table, L_grid, Gamma_grid, w_cross=1.0):
    """Grid search over (L, Gamma), alpha epsilon^(2/3) in closed form. Returns dict with the best parameters
    and the cost map."""
    auto_t, cross_t = table
    lk, lkt = np.log(k1), np.log(K1_TABLE)
    rho_m   = np.array([[cross_m[i, p] / np.sqrt(auto_m[i, a] * auto_m[i, b]) for p, (a, b) in enumerate(PAIRS)]
                        for i in range(len(GATES))])
    cost    = np.full((len(L_grid), len(Gamma_grid)), np.inf)
    ae_map  = np.full_like(cost, np.nan)
    for i in range(len(L_grid)):
        for j in range(len(Gamma_grid)):
            at  = np.exp(np.apply_along_axis(lambda s: np.interp(lk, lkt, np.log(np.maximum(s, 1e-30))), -1,
                                             auto_t[i, j]))
            ct  = (np.apply_along_axis(lambda s: np.interp(lk, lkt, s), -1, cross_t[i, j].real)
                   + 1j * np.apply_along_axis(lambda s: np.interp(lk, lkt, s), -1, cross_t[i, j].imag))
            log_ae  = np.mean(np.log(auto_m) - np.log(at))
            r_auto  = np.log(auto_m) - np.log(at) - log_ae
            rho_t   = np.array([[ct[g, p] / np.sqrt(at[g, a] * at[g, b]) for p, (a, b) in enumerate(PAIRS)]
                                for g in range(len(GATES))])
            r_cross = rho_m - rho_t
            cost[i, j]  = np.mean(r_auto ** 2) + w_cross * (np.mean(r_cross.real ** 2) + np.mean(r_cross.imag ** 2))
            ae_map[i, j] = np.exp(log_ae)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return {'L': L_grid[i], 'Gamma': Gamma_grid[j], 'alphaepsilon': ae_map[i, j], 'cost': cost,
            'edge': i in (0, len(L_grid) - 1) or j in (0, len(Gamma_grid) - 1)}
