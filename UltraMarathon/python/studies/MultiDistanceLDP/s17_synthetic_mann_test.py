# Step 17 of the multi-distance LDP study: synthetic test of the c_conv puzzle with Mann turbulence.
# Real data: physically c_conv ~ 1.2 (phase of gate-to-gate and lidar-v_0), but the SDES is smallest for c ~ 1.0.
# Here the truth is known: frozen Mann turbulence (IEC 61400-1, class B, 18 m/s, 24 seeds x 600 s) is
# transported with c_true * U through the lidar measurement points and the rotor disk.
#   F1: c_true = 1.0 (classic Taylor)     F2: c_true = 1.2 (speed-up as measured)
# The lidar uses the real beamID / isValid sequences (24 segments of 600 s) -> same sampling and blockage.
# If the SDES optimum is below c_true also for frozen turbulence, the cause is the metric or the combination
# of the gates; if it hits c_true, the cause in the real data is the evolution of the turbulence.
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
from scipy.signal import csd, welch
from scipy.signal.windows import hamming
import ldp_tools as lt
import mann_box as mb

real    = lt.load_data()
dt      = real['dt']
N_SEED  = 24
T_SEED  = 600.0
n_s     = int(T_SEED / dt)
U_MEAN  = 18.0
SIGMA_U = 0.14 * (0.75 * U_MEAN + 5.6)                  # IEC 61400-1 NTM, class B: 2.67 m/s
L_MANN, GAMMA = 0.8 * 42.0, 3.9                         # IEC 61400-1 Mann parameters (hub height > 60 m)
NX, NY, NZ, DX, DY, DZ = 2048, 128, 64, 7.0, 5.0, 5.0   # box 14.3 km x 640 m x 320 m
SIGMA_NOISE = 0.2                                       # [m/s] lidar measurement noise
x_up    = lt.upstream_distance()
_, y_pt, z_pt = lt.measurement_points()                 # (beams, gates)
n_vec   = np.stack([np.ones((lt.N_BEAMS, lt.N_GATES)), lt.BEAM_SIGN_Y[:, None] * np.tan(np.deg2rad(lt.ANGLE_HOR)) * np.ones((1, lt.N_GATES)),
                    lt.BEAM_SIGN_Z[:, None] * np.tan(np.deg2rad(lt.ANGLE_VER)) * np.ones((1, lt.N_GATES))])
n_vec   = n_vec / np.linalg.norm(n_vec, axis=0)         # unit vector of the beams (3, beams, gates)
w_B     = np.load(os.path.join(lt.RESULT_DIR, 'weights_stageB.npy'))
WEIGHTS = {'equal 50-115 m': np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]) / 5, 'stage B': w_B / w_B.sum(),
           'equal all gates': np.ones(lt.N_GATES) / lt.N_GATES}
y_grid  = (np.arange(NY) - NY // 2) * DY
z_grid  = (np.arange(NZ) - NZ // 2) * DZ
x_grid  = np.arange(NX) * DX
disk    = (y_grid[:, None] ** 2 + z_grid[None, :] ** 2) <= (lt.D_ROTOR / 2) ** 2


def bilinear_lines(field, y, z):
    """Lines along x of a field (nx, ny, nz) at the lateral position (y, z) (bilinear interpolation)."""
    fy, fz  = (y - y_grid[0]) / DY, (z - z_grid[0]) / DZ
    iy, iz  = int(np.floor(fy)), int(np.floor(fz))
    ay, az  = fy - iy, fz - iz
    return ((1 - ay) * (1 - az) * field[:, iy, iz] + ay * (1 - az) * field[:, iy + 1, iz]
            + (1 - ay) * az * field[:, iy, iz + 1] + ay * az * field[:, iy + 1, iz + 1])


def sample_line(line, xi):
    """Periodic linear interpolation of a line along x at positions xi."""
    return np.interp(np.mod(xi, NX * DX), x_grid, line, period=NX * DX)


# %% generate boxes and extract lines (cached): rotor disk mean of u, u/v/w at all beam-gate positions
file_lines = os.path.join(lt.RESULT_DIR, 's17_mann_lines.npz')
if os.path.exists(file_lines):
    L_       = np.load(file_lines)
    u_disk, uvw_lines = L_['u_disk'], L_['uvw_lines']
else:
    A           = mb.mann_coefficients(NX, NY, NZ, DX, DY, DZ, L_MANN, GAMMA)
    u_disk      = np.empty((N_SEED, NX), np.float32)
    uvw_lines   = np.empty((N_SEED, 3, lt.N_BEAMS, lt.N_GATES, NX), np.float32)
    for seed in range(N_SEED):
        uvw     = mb.generate_box(A, (NX, NY, NZ), seed, SIGMA_U)
        u_disk[seed] = uvw[0][:, disk].mean(axis=1)
        for c in range(3):
            for b in range(lt.N_BEAMS):
                for g in range(lt.N_GATES):
                    uvw_lines[seed, c, b, g] = bilinear_lines(uvw[c], y_pt[b, g], z_pt[b, g])
        print(f'seed {seed + 1:2d}/{N_SEED}: sigma u, v, w = ' + ', '.join(f'{s:.2f}' for s in uvw.std(axis=(1, 2, 3))),
              flush=True)
    del A
    np.savez(file_lines, u_disk=u_disk, uvw_lines=uvw_lines)


# %% synthetic lidar data per seed (real beamID / isValid of segment = seed)
def synthetic_data(seed, c_true, lines=None, disk_line=None, comps=(1.0, 1.0, 1.0)):
    """Synthetic lidar data for one seed. lines: (3, beams, gates, nx) u/v/w lines (default: Mann box at the
    beam positions), disk_line: (nx,) rotor disk mean of u, comps: factors of the u/v/w contribution to v_los."""
    lines       = uvw_lines[seed] if lines is None else lines
    disk_line   = u_disk[seed] if disk_line is None else disk_line
    seg     = slice(seed * n_s, (seed + 1) * n_s)
    beamID  = real['beamID'][seg].copy()
    isValid = real['isValid'][seg].copy()
    time    = np.arange(n_s) * dt
    events  = lt.detect_events(beamID)
    xi0     = 1000.0 + seed * 137.0                     # arbitrary start position in the box
    rng     = np.random.default_rng(1000 + seed)
    t_e     = time[events]
    beam_e  = beamID[events] - 1
    los_e   = np.empty((len(events), lt.N_GATES))
    for b in range(lt.N_BEAMS):
        sel = beam_e == b
        for g in range(lt.N_GATES):
            # box x points in wind direction (Mann: dU/dz > 0): a parcel at x moves to x + c U t, so the rotor
            # sees x = xi0 - c U t and a gate x_up upstream sees x = xi0 - c U t - x_up
            xi  = xi0 - c_true * U_MEAN * t_e[sel] - x_up[g]
            v   = [U_MEAN + comps[0] * sample_line(lines[0, b, g], xi),
                   comps[1] * sample_line(lines[1, b, g], xi), comps[2] * sample_line(lines[2, b, g], xi)]
            los_e[sel, g] = n_vec[0, b, g] * v[0] + n_vec[1, b, g] * v[1] + n_vec[2, b, g] * v[2]
    los_e   += SIGMA_NOISE * rng.standard_normal(los_e.shape)
    los     = los_e[np.cumsum(np.isin(np.arange(n_s), events)) - 1]      # hold between measurements (as real)
    los[~isValid] = rng.uniform(-24, 50, size=(~isValid).sum())          # invalid: garbage (as real)
    v_0     = U_MEAN + sample_line(disk_line, xi0 - c_true * U_MEAN * time)
    return {'time': time, 'dt': dt, 'beamID': beamID, 'isValid': isValid, 'lineOfSightWindSpeed': los, 'v_0': v_0}


# %% LDP (offline, as LDP_MD without induction and shear) and pooled coherence over all seeds
def pooled_sdes(pairs):
    """pairs: list of (v_0, v_0L) per seed -> SDES from coherence pooled over the seeds (Welch as in the script)."""
    n_FFT   = 2 ** 11
    kw      = dict(fs=1 / dt, window=hamming(n_FFT), noverlap=n_FFT // 2, nfft=n_FFT, detrend=False)
    Sxy = Sxx = Syy = 0
    for v0, vL in pairs:
        a, b    = v0[:-1] - v0[:-1].mean(), vL[:-1] - vL[:-1].mean()
        f, Pxy  = csd(a, b, **kw)
        Sxy     = Sxy + Pxy
        Sxx     = Sxx + welch(a, **kw)[1]
        Syy     = Syy + welch(b, **kw)[1]
    return lt.sdes_from_coherence(2 * np.pi * f / U_MEAN, np.abs(Sxy) ** 2 / (Sxx * Syy))


c_list  = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
tau_unit = x_up.copy()                                  # no induction zone in the synthetic field
for c_true in (1.0, 1.2):
    datas   = [synthetic_data(s, c_true) for s in range(N_SEED)]
    gates   = {c: [] for c in c_list}
    for d in datas:
        for c in c_list:                                # advection speed: constant mean (frozen field)
            gates[c].append(lt.rotor_time_gate_means(d, tau_unit, np.full(n_s, c * U_MEAN), 'linear'))
    print(f'\nscenario c_true = {c_true}: pooled SDES [D] over {N_SEED} seeds for LDP with c_conv =')
    for name, w in WEIGHTS.items():
        S = [pooled_sdes([(d['v_0'], g @ w) for d, g in zip(datas, gates[c])]) for c in c_list]
        print(f'  {name:16s}: ' + ', '.join(f'{c:.1f}: {s:.3f}' for c, s in zip(c_list, S)), flush=True)
    # physical c from the phase: lead of every single gate (aligned with c = c_true) vs. v_0 should be 0
    lead = [np.mean([lt.phase_delay(g[:, i_g], d['v_0'], dt, (0.005, 0.05), 2 ** 12)[0]
                     for d, g in zip(datas, gates[c_true])]) for i_g in (0, 3, 6, 9)]
    print(f'  check: residual lead of gates 50/100/145/200 m aligned with c = c_true: '
          + ' '.join(f'{l:+.2f} s' for l in lead))

# %% 2) where does the apparent speed-up in frozen turbulence come from? (c_true = 1.0)
# V1: v_los from u only (no v, w contribution)       -> contamination of the line-of-sight by v and w?
# V2: isotropic turbulence (Gamma = 0, no shear)     -> inclined (sheared) structures?
# V3: all measurement points on the rotor axis       -> geometry (beams measure off-axis, far gates outside
#     (y = z = 0), u only                               the rotor disk)?
def evaluate(name, datas, c_true=1.0):
    gates = {c: [lt.rotor_time_gate_means(d, tau_unit, np.full(n_s, c * U_MEAN), 'linear') for d in datas]
             for c in c_list}
    print(f'\n{name}: pooled SDES [D] for c_conv =')
    for w_name in ('equal 50-115 m', 'stage B', 'equal all gates'):
        S = [pooled_sdes([(d['v_0'], g @ WEIGHTS[w_name]) for d, g in zip(datas, gates[c])]) for c in c_list]
        print(f'  {w_name:16s}: ' + ', '.join(f'{c:.1f}: {s:.3f}' for c, s in zip(c_list, S)), flush=True)
    lead = [np.mean([lt.phase_delay(g[:, i_g], d['v_0'], dt, (0.005, 0.05), 2 ** 12)[0]
                     for d, g in zip(datas, gates[c_true])]) for i_g in range(lt.N_GATES)]
    slope = np.polyfit(x_up, lead, 1)[0]
    print(f'  residual lead (c = c_true) 50/100/145/200 m: ' + ' '.join(f'{lead[i]:+.2f} s' for i in (0, 3, 6, 9))
          + f' -> apparent c_conv from phase = {1 / (1 / (c_true * U_MEAN) + slope) / U_MEAN:.2f}', flush=True)


file_var = os.path.join(lt.RESULT_DIR, 's17_mann_lines_variants.npz')
if os.path.exists(file_var):
    V_      = np.load(file_var)
    iso_lines, iso_disk, axis_lines = V_['iso_lines'], V_['iso_disk'], V_['axis_lines']
else:
    A_iso       = mb.mann_coefficients(NX, NY, NZ, DX, DY, DZ, L_MANN, 0.0)
    A_shear     = mb.mann_coefficients(NX, NY, NZ, DX, DY, DZ, L_MANN, GAMMA)
    iso_lines   = np.empty_like(uvw_lines)
    iso_disk    = np.empty_like(u_disk)
    axis_lines  = np.empty_like(uvw_lines)
    for seed in range(N_SEED):
        uvw = mb.generate_box(A_iso, (NX, NY, NZ), seed, SIGMA_U)
        iso_disk[seed] = uvw[0][:, disk].mean(axis=1)
        for c in range(3):
            for b in range(lt.N_BEAMS):
                for g in range(lt.N_GATES):
                    iso_lines[seed, c, b, g] = bilinear_lines(uvw[c], y_pt[b, g], z_pt[b, g])
        uvw = mb.generate_box(A_shear, (NX, NY, NZ), seed, SIGMA_U)   # same seed as above -> same box as F1
        for c in range(3):
            axis_lines[seed, c] = bilinear_lines(uvw[c], 0.0, 0.0)[None, None, :]
        print(f'variants: seed {seed + 1:2d}/{N_SEED}', flush=True)
    del A_iso, A_shear
    np.savez(file_var, iso_lines=iso_lines, iso_disk=iso_disk, axis_lines=axis_lines)

evaluate('V1: v_los from u only', [synthetic_data(s, 1.0, comps=(1.0, 0.0, 0.0)) for s in range(N_SEED)])
evaluate('V2: isotropic turbulence (Gamma = 0)', [synthetic_data(s, 1.0, iso_lines[s], iso_disk[s]) for s in range(N_SEED)])
evaluate('V3: points on the rotor axis, u only', [synthetic_data(s, 1.0, axis_lines[s], comps=(1.0, 0.0, 0.0))
                                                  for s in range(N_SEED)])

# %% 3) estimator or combination? F1 (c_true = 1.0) evaluated with longer Welch windows (first 0.5 crossing)
def pooled_sdes_nfft(pairs, n_FFT):
    kw      = dict(fs=1 / dt, window=hamming(n_FFT), noverlap=n_FFT // 2, nfft=n_FFT, detrend=False)
    Sxy = Sxx = Syy = 0
    for v0, vL in pairs:
        a, b    = v0[:-1] - v0[:-1].mean(), vL[:-1] - vL[:-1].mean()
        f, Pxy  = csd(a, b, **kw)
        Sxy, Sxx, Syy = Sxy + Pxy, Sxx + welch(a, **kw)[1], Syy + welch(b, **kw)[1]
    g2      = np.abs(Sxy) ** 2 / (Sxx * Syy)
    k       = 2 * np.pi * f / U_MEAN
    i       = np.flatnonzero(g2[1:] < 0.5)[0] + 1
    return 2 * np.pi / np.interp(0.5, [g2[i], g2[i - 1]], [k[i], k[i - 1]]) / lt.D_ROTOR


for name, comps in (('F1 (u, v, w)', (1.0, 1.0, 1.0)), ('V1 (u only)', (1.0, 0.0, 0.0))):
    datas   = [synthetic_data(s, 1.0, comps=comps) for s in range(N_SEED)]
    gates   = {c: [lt.rotor_time_gate_means(d, tau_unit, np.full(n_s, c * U_MEAN), 'linear') for d in datas]
               for c in c_list}
    for w_name in ('stage B', 'equal all gates'):
        for n_FFT in (2 ** 11, 2 ** 12, 2 ** 13):
            S = [pooled_sdes_nfft([(d['v_0'], g @ WEIGHTS[w_name]) for d, g in zip(datas, gates[c])], n_FFT)
                 for c in c_list]
            print(f'3) {name:13s} {w_name:16s} n_FFT = {n_FFT:5d} ({n_FFT * dt:4.0f} s): '
                  + ', '.join(f'{c:.1f}: {s:.3f}' for c, s in zip(c_list, S))
                  + f'  -> optimum c = {c_list[int(np.argmin(S))]:.1f}', flush=True)

# %% 4) per-beam analysis as in s12 (real data) for the synthetic data: is the speed-up of the bottom beams and the
# top-bottom delay reproduced by frozen Mann turbulence (c_true = 1.0)? Cross-spectra pooled over all seeds.
REAL_S12 = {'c per beam': [1.05, 1.08, 1.74, 1.51], 'top-bottom s/100 m': 6.5, 'c vs v_0': [0.91, 1.09, 1.41, 1.41]}
BAND12  = (0.005, 0.03)


def pooled_lead(pairs, band=BAND12, nperseg=2 ** 12):
    """Lead of a over b [s] from the phase of the cross-spectrum pooled over all (a, b) pairs."""
    kw  = dict(fs=1 / dt, nperseg=nperseg, window='hamming')
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


dT_model = (x_up[1:] - x_up[0]) / U_MEAN                # Taylor travel time gate -> 50 m (c = 1, no induction)
for name, comps in (('F1 (u, v, w)', (1.0, 1.0, 1.0)), ('V1 (u only)', (1.0, 0.0, 0.0))):
    datas   = [synthetic_data(s, 1.0, comps=comps) for s in range(N_SEED)]
    held    = [held_per_beam(d) for d in datas]
    c_beam  = []
    for b in range(lt.N_BEAMS):                         # 1) c per beam from gate-to-gate delays within the beam
        meas = np.array([pooled_lead([(h[:, b, g], h[:, b, 0]) for h in held]) for g in range(1, lt.N_GATES)])
        c_beam.append(np.sum(dT_model ** 2) / np.sum(dT_model * meas))
    lead_tb = [np.polyfit(z_pt[b1] - z_pt[b2], [pooled_lead([(h[:, b1, g], h[:, b2, g]) for h in held])
                                                for g in range(lt.N_GATES)], 1)[0] * 100 for b1, b2 in ((0, 2), (1, 3))]
    c_v0    = []
    for b in range(lt.N_BEAMS):                         # 3) every beam vs. v_0
        lead = [pooled_lead([(h[:, b, g], d['v_0']) for h, d in zip(held, datas)]) for g in range(lt.N_GATES)]
        c_v0.append(1 / (np.polyfit(x_up, lead, 1)[0] * U_MEAN))
    print(f'4) {name}:')
    print(f'   c per beam (gate-to-gate):  synthetic ' + ' '.join(f'{c:.2f}' for c in c_beam)
          + '   | real s12 ' + ' '.join(f'{c:.2f}' for c in REAL_S12['c per beam']))
    print(f'   top leads bottom [s/100 m]: synthetic ' + ' '.join(f'{l:+.2f}' for l in lead_tb)
          + f'   | real s12 +{REAL_S12["top-bottom s/100 m"]:.1f}')
    print(f'   c per beam vs v_0:          synthetic ' + ' '.join(f'{c:.2f}' for c in c_v0)
          + '   | real s12 ' + ' '.join(f'{c:.2f}' for c in REAL_S12['c vs v_0']), flush=True)

# %% 5) gate-to-gate c of the 4-beam mean (real s12: 1.15) for the synthetic data
for name, comps in (('F1 (u, v, w)', (1.0, 1.0, 1.0)), ('V1 (u only)', (1.0, 0.0, 0.0))):
    datas   = [synthetic_data(s, 1.0, comps=comps) for s in range(N_SEED)]
    means   = [held_per_beam(d).mean(axis=1) for d in datas]
    meas    = np.array([pooled_lead([(m[:, g], m[:, 0]) for m in means]) for g in range(1, lt.N_GATES)])
    print(f'5) {name}: c of the 4-beam mean (gate-to-gate) = {np.sum(dT_model ** 2) / np.sum(dT_model * meas):.2f}'
          f'   | real s12 1.15', flush=True)
