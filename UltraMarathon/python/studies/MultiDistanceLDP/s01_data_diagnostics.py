# Step 1 of the multi-distance LDP study: get to know the lidar data.
# - timing of the measurements (events), beam sequence
# - availability per beam and gate, length of gaps (how long has a value to be held?)
# - do invalid flags occur simultaneously for all gates (blade blockage)?
# - how do valid measurements deviate from their neighbouring gates (outlier potential)?
# Run cell by cell (# %%) in VS Code or as a whole script.

# %% setup
import os
import numpy as np
import matplotlib.pyplot as plt
import ldp_tools as lt

os.makedirs(lt.FIG_DIR, exist_ok=True)
data    = lt.load_data()
dt      = data['dt']
beamID  = data['beamID']
isValid = data['isValid']
los     = data['lineOfSightWindSpeed']
events  = lt.detect_events(beamID)
print(f'{len(events)} lidar measurements in {data["time"][-1] / 3600:.1f} h '
      f'-> {len(events) / data["time"][-1]:.2f} Hz')

# %% timing of measurements and beam sequence
spacing = np.diff(events)
vals, counts = np.unique(spacing, return_counts=True)
print('samples between measurements:', dict(zip(vals.tolist(), counts.tolist())))
jumps = (np.diff(beamID[events]) % 4)
print(f'skipped beams (sequence jump != 1): {np.sum(jumps != 1)}')

# %% availability per beam and gate (evaluated per measurement event)
beam_e  = beamID[events]
valid_e = isValid[events]
print('availability [%] per beam (rows) and gate (columns):')
for i_b in range(lt.N_BEAMS):
    print(f'  beam {i_b + 1}: ' + ' '.join(f'{100 * a:5.1f}' for a in valid_e[beam_e == i_b + 1].mean(0)))
n_valid_gates = valid_e.sum(1)
print('number of valid gates per measurement (0..10):', np.bincount(n_valid_gates, minlength=11))

# %% gap lengths: time since last valid measurement of the same beam (gate 1)
fig, ax = plt.subplots(figsize=(7, 4))
cmap    = plt.get_cmap('tab10')
for i_b in range(lt.N_BEAMS):
    t_valid = data['time'][events[(beam_e == i_b + 1) & valid_e[:, 0]]]
    gap     = np.diff(t_valid)
    print(f'beam {i_b + 1}: median gap {np.median(gap):.2f} s, 95% {np.percentile(gap, 95):.2f} s, '
          f'99% {np.percentile(gap, 99):.2f} s, max {gap.max():.1f} s')
    vals, counts = np.unique(np.round(gap), return_counts=True)
    ax.plot(vals, counts / counts.sum(), 'o-', lw=1.5, ms=4, color=cmap(i_b), label=f'beam {i_b + 1}')
ax.set_yscale('log')
ax.set_xlim(0, 15)
ax.set_xlabel('time between two valid measurements of the same beam [s]')
ax.set_ylabel('relative frequency [-]')
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's01_gap_lengths.png'), dpi=150)

# %% are invalid measurements periodic (blade passing)? autocorrelation of validity per beam
# rotor speed ~12 rpm -> blade passing every ~1.67 s, each beam is measured every ~1 s
fig, ax = plt.subplots(figsize=(7, 4))
for i_b in range(lt.N_BEAMS):
    s       = valid_e[beam_e == i_b + 1, 0].astype(float)
    s       = s - s.mean()
    n_lag   = 30
    acf     = np.array([np.mean(s[:len(s) - k] * s[k:]) for k in range(n_lag)]) / np.var(s)
    ax.plot(np.arange(n_lag), acf, 'o-', lw=1.5, ms=4, color=cmap(i_b), label=f'beam {i_b + 1}')
ax.set_xlabel('lag [measurements of the same beam, ~1 s]')
ax.set_ylabel('autocorrelation of isValid [-]')
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's01_validity_acf.png'), dpi=150)

# %% values of valid vs. invalid measurements
fig, ax = plt.subplots(figsize=(7, 4))
bins    = np.linspace(-40, 50, 181)
ax.hist(los[events][valid_e].ravel(), bins, density=True, histtype='step', lw=1.5, label='valid')
ax.hist(los[events][~valid_e].ravel(), bins, density=True, histtype='step', lw=1.5, label='invalid')
ax.set_xlabel('line-of-sight wind speed [m/s]')
ax.set_ylabel('pdf [1/(m/s)]')
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's01_los_histogram.png'), dpi=150)

# %% outlier potential: deviation of each valid measurement from the median of its neighbouring gates
los_e   = np.where(valid_e, los[events], np.nan)
dev     = np.full(los_e.shape, np.nan)
for i_g in range(lt.N_GATES):
    nb          = [j for j in (i_g - 1, i_g + 1) if 0 <= j < lt.N_GATES]
    dev[:, i_g] = los_e[:, i_g] - np.nanmedian(los_e[:, nb], axis=1) if len(nb) > 1 \
        else los_e[:, i_g] - los_e[:, nb[0]]
ok      = ~np.isnan(dev)
print('deviation from neighbour gates: std per gate [m/s]:',
      np.round(np.nanstd(dev, 0), 2))
for thr in (2, 3, 5, 10):
    print(f'  |dev| > {thr:2d} m/s: {100 * np.mean(np.abs(dev[ok]) > thr):.3f} % of valid values')

fig, ax = plt.subplots(figsize=(7, 4))
bins    = np.linspace(-15, 15, 301)
ax.hist(dev[ok], bins, density=True, histtype='step', lw=1.5)
ax.set_yscale('log')
ax.set_xlabel('deviation from neighbouring gates [m/s]')
ax.set_ylabel('pdf [1/(m/s)]')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's01_neighbour_deviation.png'), dpi=150)

# %% overview: 4-beam held mean per gate vs. v_0 (first 600 s)
u_hold  = lt.hold_per_beam(data)
u_mean  = np.nanmean(u_hold, axis=1)                    # (n_t, n_gates)
idx     = data['time'] <= 600
fig, ax = plt.subplots(figsize=(10, 4))
colors  = plt.get_cmap('Blues')(np.linspace(0.35, 1, lt.N_GATES))
for i_g in (0, 3, 6, 9):
    ax.plot(data['time'][idx], u_mean[idx, i_g], lw=1, color=colors[i_g],
            label=f'{lt.GATE_DISTANCES[i_g]:.0f} m (held, not shifted)')
ax.plot(data['time'][idx], data['v_0'][idx], 'k', lw=2, label='v_0')
ax.set_xlabel('time [s]')
ax.set_ylabel('wind speed [m/s]')
ax.grid(alpha=0.3)
ax.legend(ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(lt.FIG_DIR, 's01_overview.png'), dpi=150)

plt.show()
