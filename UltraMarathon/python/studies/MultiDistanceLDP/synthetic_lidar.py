"""Synthetic lidar data from Mann boxes and the evaluation used in s18 (same procedure as s17).

Box x points in wind direction (Mann: dU/dz > 0). A parcel at x moves to x + c U t, so at time t the rotor sees
x = xi0 - c U t and a gate x_up upstream sees x = xi0 - c U t - x_up. The lidar uses the real beamID / isValid
sequences, so sampling and blade blockage are as in the measurement.
"""
import numpy as np
from scipy.signal import csd, welch
from scipy.signal.windows import hamming
import ldp_tools as lt

REAL    = lt.load_data()
DT      = REAL['dt']
U_MEAN  = 18.0
SIGMA_U = 0.14 * (0.75 * U_MEAN + 5.6)                  # IEC 61400-1 NTM, class B: 2.67 m/s
L_MANN  = 0.8 * 42.0                                    # IEC 61400-1 (hub height > 60 m)
NX, NY, NZ, DX, DY, DZ = 2048, 128, 64, 7.0, 5.0, 5.0   # box 14.3 km x 640 m x 320 m
SIGMA_NOISE = 0.2                                       # [m/s] lidar measurement noise
N_SEED, T_SEED = 24, 600.0
N_S     = int(T_SEED / DT)
X_UP    = lt.upstream_distance()
_, Y_PT, Z_PT = lt.measurement_points()
# line-of-sight speed is positive towards the lidar (as the real data): v_los = -V . d, d = beam direction
# (upstream, sideways, up/down). For a top beam a positive w therefore reduces v_los.
# (s17/s18 used +s_y, +s_z: wrong sign of the w contribution; corrected here)
N_VEC   = np.stack([np.ones((lt.N_BEAMS, lt.N_GATES)),
                    -lt.BEAM_SIGN_Y[:, None] * np.tan(np.deg2rad(lt.ANGLE_HOR)) * np.ones((1, lt.N_GATES)),
                    -lt.BEAM_SIGN_Z[:, None] * np.tan(np.deg2rad(lt.ANGLE_VER)) * np.ones((1, lt.N_GATES))])
N_VEC   = N_VEC / np.linalg.norm(N_VEC, axis=0)         # unit vectors of the beams (3, beams, gates)
Y_GRID  = (np.arange(NY) - NY // 2) * DY
Z_GRID  = (np.arange(NZ) - NZ // 2) * DZ
X_GRID  = np.arange(NX) * DX
DISK    = (Y_GRID[:, None] ** 2 + Z_GRID[None, :] ** 2) <= (lt.D_ROTOR / 2) ** 2


def bilinear_line(field, y, z):
    """Line along x of a field (nx, ny, nz) at the lateral position (y, z)."""
    fy, fz  = (y - Y_GRID[0]) / DY, (z - Z_GRID[0]) / DZ
    iy, iz  = int(np.floor(fy)), int(np.floor(fz))
    ay, az  = fy - iy, fz - iz
    return ((1 - ay) * (1 - az) * field[:, iy, iz] + ay * (1 - az) * field[:, iy + 1, iz]
            + (1 - ay) * az * field[:, iy, iz + 1] + ay * az * field[:, iy + 1, iz + 1])


def gate_lines(uvw, gates=range(lt.N_GATES)):
    """u/v/w lines at the beam positions of the given gates: (3, beams, gates, nx), other gates NaN."""
    lines = np.full((3, lt.N_BEAMS, lt.N_GATES, NX), np.nan, np.float32)
    for g in gates:
        for c in range(3):
            for b in range(lt.N_BEAMS):
                lines[c, b, g] = bilinear_line(uvw[c], Y_PT[b, g], Z_PT[b, g])
    return lines


def disk_line(uvw):
    """Rotor disk mean of u along x."""
    return uvw[0][:, DISK].mean(axis=1)


def sample_line(line, xi):
    return np.interp(np.mod(xi, NX * DX), X_GRID, line, period=NX * DX)


def synthetic_data(seed, lines, disk, c_true=1.0):
    """Synthetic lidar data dict (same format as ldp_tools.load_data) for one seed."""
    seg     = slice(seed * N_S, (seed + 1) * N_S)
    beamID  = REAL['beamID'][seg].copy()
    isValid = REAL['isValid'][seg].copy()
    time    = np.arange(N_S) * DT
    events  = lt.detect_events(beamID)
    t_e, beam_e = time[events], beamID[events] - 1
    xi0     = 1000.0 + seed * 137.0
    rng     = np.random.default_rng(1000 + seed)
    los_e   = np.empty((len(events), lt.N_GATES))
    for b in range(lt.N_BEAMS):
        sel = beam_e == b
        for g in range(lt.N_GATES):
            xi  = xi0 - c_true * U_MEAN * t_e[sel] - X_UP[g]
            los_e[sel, g] = (N_VEC[0, b, g] * (U_MEAN + sample_line(lines[0, b, g], xi))
                             + N_VEC[1, b, g] * sample_line(lines[1, b, g], xi)
                             + N_VEC[2, b, g] * sample_line(lines[2, b, g], xi))
    los_e   += SIGMA_NOISE * rng.standard_normal(los_e.shape)
    los     = los_e[np.cumsum(np.isin(np.arange(N_S), events)) - 1]
    los[~isValid] = rng.uniform(-24, 50, size=(~isValid).sum())
    v_0     = U_MEAN + sample_line(disk, xi0 - c_true * U_MEAN * time)
    return {'time': time, 'dt': DT, 'beamID': beamID, 'isValid': isValid, 'lineOfSightWindSpeed': los, 'v_0': v_0}


def pooled_sdes(pairs):
    """SDES from the coherence pooled over all (v_0, v_0L) pairs (Welch as in RunUltraMarathon.py)."""
    n_FFT   = 2 ** 11
    kw      = dict(fs=1 / DT, window=hamming(n_FFT), noverlap=n_FFT // 2, nfft=n_FFT, detrend=False)
    Sxy = Sxx = Syy = 0
    for v0, vL in pairs:
        a, b    = v0[:-1] - v0[:-1].mean(), vL[:-1] - vL[:-1].mean()
        f, Pxy  = csd(a, b, **kw)
        Sxy, Sxx, Syy = Sxy + Pxy, Sxx + welch(a, **kw)[1], Syy + welch(b, **kw)[1]
    return lt.sdes_from_coherence(2 * np.pi * f / U_MEAN, np.abs(Sxy) ** 2 / (Sxx * Syy))


def pooled_lead(pairs, band=(0.005, 0.03), nperseg=2 ** 12):
    """Lead of a over b [s] from the phase of the cross-spectrum pooled over all (a, b) pairs."""
    kw  = dict(fs=1 / DT, nperseg=nperseg, window='hamming')
    Sab = Saa = Sbb = 0
    for a, b in pairs:
        a, b    = a - a.mean(), b - b.mean()
        f, P    = csd(a, b, **kw)
        Sab, Saa, Sbb = Sab + P, Saa + welch(a, **kw)[1], Sbb + welch(b, **kw)[1]
    c       = np.abs(Sab) ** 2 / (Saa * Sbb)
    m       = (f >= band[0]) & (f <= band[1])
    omega   = 2 * np.pi * f[m]
    return -np.sum(omega * np.unwrap(np.angle(Sab[m])) * c[m]) / np.sum(omega ** 2 * c[m])


def held_per_beam(d):
    u_bd = lt.hold_per_beam(d)
    for b in range(lt.N_BEAMS):
        for g in range(lt.N_GATES):
            s = u_bd[:, b, g]
            s[np.isnan(s)] = s[~np.isnan(s)][0]
    return u_bd


def evaluate(datas, weights, c_list=(0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)):
    """All metrics of s17/s18 for a list of synthetic data dicts (c_true = 1). Returns dict."""
    tau_unit = X_UP.copy()                              # no induction zone in the synthetic field
    gates   = {c: [lt.rotor_time_gate_means(d, tau_unit, np.full(N_S, c * U_MEAN), 'linear') for d in datas]
               for c in c_list}
    res     = {'sdes': {name: [pooled_sdes([(d['v_0'], g @ w) for d, g in zip(datas, gates[c])]) for c in c_list]
                        for name, w in weights.items()},
               'sdes_gate': [pooled_sdes([(d['v_0'], g[:, i]) for d, g in zip(datas, gates[1.0])]) for i in (0, 3, 6, 9)]}
    # c of the 4-beam mean from gate-to-gate delays and from the phase vs. v_0
    held    = [held_per_beam(d) for d in datas]
    means   = [h.mean(axis=1) for h in held]
    dT      = (X_UP[1:] - X_UP[0]) / U_MEAN
    meas    = np.array([pooled_lead([(m[:, g], m[:, 0]) for m in means]) for g in range(1, lt.N_GATES)])
    res['c_gate'] = np.sum(dT ** 2) / np.sum(dT * meas)
    lead_v0 = np.array([pooled_lead([(m[:, g], d['v_0']) for m, d in zip(means, datas)]) for g in range(lt.N_GATES)])
    res['c_v0'] = 1 / (np.polyfit(X_UP, lead_v0, 1)[0] * U_MEAN)
    # per beam: c from gate-to-gate delays and top-bottom lead
    res['c_beam'] = []
    for b in range(lt.N_BEAMS):
        meas = np.array([pooled_lead([(h[:, b, g], h[:, b, 0]) for h in held]) for g in range(1, lt.N_GATES)])
        res['c_beam'].append(np.sum(dT ** 2) / np.sum(dT * meas))
    res['top_bottom'] = [np.polyfit(Z_PT[b1] - Z_PT[b2], [pooled_lead([(h[:, b1, g], h[:, b2, g]) for h in held])
                                                           for g in range(lt.N_GATES)], 1)[0] * 100
                         for b1, b2 in ((0, 2), (1, 3))]
    res['c_list'] = list(c_list)
    return res
