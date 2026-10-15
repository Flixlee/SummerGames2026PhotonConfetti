# Title image "Photon Confetti" for the LAC Summer Games 2026 presentation (light background).
# The confetti are the real valid lidar measurements of a gust: every measurement becomes a small cloud of
# particles around its gate (on the beam); the faint particles are the measured air drifting with the wind towards
# the rotor (Taylor). Colour: reconstructed longitudinal wind speed u = v_los / cos(19.18 deg).
# Outputs: title_photon_confetti.pdf/.png (16:9 with text), *_graphic.pdf (scene only, cropped),
#          *_background.pdf (16:10 title slide background, scene on the left, free space for the title block)
# Run with the venv of UltraMarathon/python:  ..\..\UltraMarathon\python\venv\Scripts\python.exe title_photon_confetti.py
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(HERE, '..', '..', 'UltraMarathon', 'data', 'DataSummerGames2026.mat')

# geometry (see instructions): gates along x from the lidar, lidar 3.1 m behind the rotor and 4.1 m above hub
GATES       = np.array([50, 65, 80, 100, 115, 130, 145, 160, 175, 200.])
X_LIDAR, Z_LIDAR, R_ROTOR = -3.1, 4.1, 63.0
SIGN_Y      = np.array([+1, -1, +1, -1])                # beam 1 top-left, 2 top-right, 3 bottom-left, 4 bottom-right
SIGN_Z      = np.array([+1, +1, -1, -1])
TAN_Y, TAN_Z = np.tan(np.deg2rad(15.0)), np.tan(np.deg2rad(12.5))
COS_LOS     = np.cos(np.arctan(np.hypot(TAN_Y, TAN_Z)))

# scene settings
T_NOW       = 215.0                                     # [s] snapshot time (gust around 200 s)
T_WINDOW    = 25.0                                      # [s] age of the oldest measurement shown
T_FRESH     = 4.0                                       # [s] measurements shown on the beams (gaps up to ~4 s)
N_PART      = 22                                        # particles per measurement
PROBE       = 3.0                                       # [m] spread along the beam (std), small: gates visible
COLORS      = {'ink': '#2D3748', 'muted': '#718096', 'blue': '#1F77B4', 'orange': '#D95F02', 'bg': '#FAFAF7'}

# %% data: valid measurements in the window
with h5py.File(DATA_FILE, 'r') as f:
    time    = np.array(f['time']).squeeze()
    los     = np.array(f['lineOfSightWindSpeed']).T
    valid   = np.array(f['isValid']).T.astype(bool)
    beamID  = np.array(f['beamID']).squeeze().astype(int)
events  = np.concatenate(([0], np.flatnonzero(np.diff(beamID) != 0) + 1))
events  = events[(time[events] > T_NOW - T_WINDOW) & (time[events] <= T_NOW)]
U_mean  = np.mean(los[events][valid[events]]) / COS_LOS

rng     = np.random.default_rng(7)
pts, vals, ages = [], [], []                            # transported air (drifting particles)
meas_pts, meas_vals, meas_age = [], [], []              # measurements at the measurement location
for i in events:
    b       = beamID[i] - 1
    age     = T_NOW - time[i]
    for g in np.flatnonzero(valid[i]):
        d       = GATES[g] + rng.normal(0, PROBE, N_PART)                # around the gate along the beam
        y       = SIGN_Y[b] * d * TAN_Y + rng.normal(0, 1.2, N_PART)
        z       = Z_LIDAR + SIGN_Z[b] * d * TAN_Z + rng.normal(0, 1.2, N_PART)
        u       = los[i, g] / COS_LOS
        meas_pts.append(np.column_stack((d + X_LIDAR, y, z)))
        meas_vals.append(np.full(N_PART, u))
        meas_age.append(np.full(N_PART, age))
        x       = d + X_LIDAR - U_mean * age                             # drifted towards the rotor (Taylor)
        keep    = x > -2.0                                               # passed the rotor: gone
        pts.append(np.column_stack((x, y, z))[keep])
        vals.append(np.full(keep.sum(), u))
        ages.append(np.full(keep.sum(), age))
pts, vals, ages = np.vstack(pts), np.concatenate(vals), np.concatenate(ages)
meas_pts, meas_vals, meas_age = np.vstack(meas_pts), np.concatenate(meas_vals), np.concatenate(meas_age)

# %% scene
cmap    = LinearSegmentedColormap.from_list('confetti', [COLORS['blue'], '#D9DEE5', COLORS['orange']])
norm    = Normalize(U_mean - 3.5, U_mean + 3.5)


def draw_scene(fig, rect, bg=COLORS['bg']):
    """Confetti, beams and turbine in a 3D axes at rect (figure coordinates); plot x = -(upstream distance)."""
    ax = fig.add_axes(rect, projection='3d', facecolor=bg)
    ax.set_axis_off()
    # 1) transported air: small, faint particles drifting along the wind (traces parallel to x)
    rgba        = cmap(norm(vals))
    rgba[:, 3]  = 0.22
    ax.scatter(-pts[:, 0], pts[:, 1], pts[:, 2], c=rgba, s=1.2, marker='s', linewidths=0, depthshade=False)
    # 2) the measurements themselves: strong confetti at the gates on the beams, fading with age
    fresh       = meas_age < T_FRESH
    rgba_m      = cmap(norm(meas_vals[fresh]))
    rgba_m[:, 3] = 1.0 - 0.8 * meas_age[fresh] / T_FRESH
    order       = np.argsort(-meas_age[fresh])                           # newest on top
    ax.scatter(-meas_pts[fresh][order, 0], meas_pts[fresh][order, 1], meas_pts[fresh][order, 2], c=rgba_m[order],
               s=7.0, marker='s', linewidths=0, depthshade=False)
    # beams (thin lines from the lidar to 200 m)
    for b in range(4):
        d = np.array([0.0, 210.0])
        ax.plot(-(d + X_LIDAR), SIGN_Y[b] * d * TAN_Y, Z_LIDAR + SIGN_Z[b] * d * TAN_Z, color=COLORS['muted'],
                lw=0.6, alpha=0.5)
    # turbine: rotor disk, three blades, hub, nacelle and tower (simple silhouette)
    phi = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.zeros_like(phi), R_ROTOR * np.cos(phi), R_ROTOR * np.sin(phi), color=COLORS['ink'], lw=0.6,
            alpha=0.25, ls=(0, (4, 3)))
    for ang in np.deg2rad([20, 140, 260]):
        r = np.linspace(2.5, R_ROTOR, 30)
        ax.plot(np.zeros_like(r), r * np.cos(ang), r * np.sin(ang), color=COLORS['ink'], lw=2.4,
                solid_capstyle='round')
    ax.plot([0, 0], [0, 0], [0, 0], marker='o', ms=7, color=COLORS['ink'])
    ax.plot([0, 14], [0, 0], [1, 1], color=COLORS['ink'], lw=7, solid_capstyle='round')      # nacelle (downwind)
    ax.plot([9, 9], [0, 0], [-1, -95], color=COLORS['ink'], lw=3.5)                          # tower
    ax.set_xlim(-215, 25)
    ax.set_ylim(-80, 80)
    ax.set_zlim(-95, 80)
    ax.set_box_aspect((240, 160, 175))
    ax.view_init(elev=14, azim=-62)
    return ax


# %% 16:9 title image with text and colour legend
fig     = plt.figure(figsize=(13.33, 7.5), facecolor=COLORS['bg'])
draw_scene(fig, [0.06, -0.16, 0.98, 1.15])
fig.text(0.045, 0.86, 'Photon Confetti', fontsize=46, fontweight='bold', color=COLORS['ink'], ha='left')
fig.text(0.047, 0.795, 'Multi-Distance Lidar Data Processing', fontsize=22, color=COLORS['ink'], ha='left')
fig.text(0.047, 0.745, 'LAC Summer Games 2026  ·  IEA Wind Task 52', fontsize=15, color=COLORS['muted'], ha='left')
cax     = fig.add_axes([0.047, 0.13, 0.16, 0.014])
cb      = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=9, colors=COLORS['muted'], length=0)
cb.set_label('reconstructed wind speed u [m/s]', fontsize=9, color=COLORS['muted'])
fig.text(0.047, 0.065, f'lidar measurements of the last {T_FRESH:.0f} s on the beams, faint: air transported towards the rotor',
         fontsize=9, color=COLORS['muted'], ha='left')
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'title_photon_confetti.{ext}'), dpi=200, facecolor=COLORS['bg'])

# graphic only (no text, no legend), cropped, e.g. for a column on a slide
for t in list(fig.texts):
    t.remove()
cax.remove()
fig.savefig(os.path.join(HERE, 'title_photon_confetti_graphic.pdf'), facecolor=COLORS['bg'],
            bbox_inches=fig.bbox_inches.from_bounds(13.33 * 0.30, 7.5 * 0.08, 13.33 * 0.55, 7.5 * 0.72))
plt.close(fig)

# %% title slide background (16:10 like the slides): scene on the left, free space on the right for the title
# block of the WETI template (starts at 9.4 of 16 cm) and at the bottom for the logos (below ~8.3 of 10 cm)
fig     = plt.figure(figsize=(13.33, 8.33), facecolor='white')
draw_scene(fig, [-0.30, -0.13, 1.18, 1.25], bg='white')
fig.savefig(os.path.join(HERE, 'title_photon_confetti_background.pdf'), facecolor='white')
fig.savefig(os.path.join(HERE, 'title_photon_confetti_background.png'), dpi=100, facecolor='white')
plt.close(fig)
print(f'{len(events)} measurements, {len(vals)} particles, U = {U_mean:.1f} m/s')
