"""Mann (1998) uniform shear turbulence: spectral tensor and FFT-based box generator.

Mann, J.: Wind field simulation, Probabilistic Engineering Mechanics 13(4), 269-282, 1998.
Coordinates: x longitudinal (mean wind direction), y lateral, z vertical. The box is periodic.
The amplitude (alpha * epsilon^(2/3)) is not used: the box is scaled to a target sigma_u afterwards
(as for IEC 61400-1 load simulations), keeping the ratios between the components.
"""
import numpy as np
from scipy.special import hyp2f1


def mann_coefficients(nx, ny, nz, dx, dy, dz, L, Gamma):
    """Matrix A(k) (shape (3, 3, nx, ny, nz//2+1)) with u_hat = A(k) n(k), n: complex white noise."""
    k1 = 2 * np.pi * np.fft.fftfreq(nx, dx)[:, None, None]
    k2 = 2 * np.pi * np.fft.fftfreq(ny, dy)[None, :, None]
    k3 = 2 * np.pi * np.fft.rfftfreq(nz, dz)[None, None, :]
    k1, k2, k3 = np.broadcast_arrays(k1, k2, k3)
    kk      = k1 ** 2 + k2 ** 2 + k3 ** 2
    k       = np.sqrt(kk)
    with np.errstate(divide='ignore', invalid='ignore'):
        kL      = k * L
        beta    = Gamma / (kL ** (2 / 3) * np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -kL ** -2)))  # eddy lifetime * dU/dz
        beta[k == 0] = 0.0
        k30     = k3 + beta * k1                            # wave vector before shear distortion
        kk0     = k1 ** 2 + k2 ** 2 + k30 ** 2
        k12     = k1 ** 2 + k2 ** 2
        E0      = 1.453 * L ** (5 / 3) * (np.sqrt(kk0) * L) ** 4 / (1 + kk0 * L ** 2) ** (17 / 6)  # von Karman
        C1      = beta * k1 ** 2 * (kk0 - 2 * k30 ** 2 + beta * k1 * k30) / (kk * k12)
        C2      = k2 * kk0 / k12 ** 1.5 * np.arctan(beta * k1 * np.sqrt(k12) / (kk0 - k30 * k1 * beta))
        zeta1   = np.where(k1 != 0, C1 - k2 / k1 * C2, 0.0)
        zeta2   = np.where(k1 != 0, k2 / k1 * C1 + C2, 0.0)
        pref    = np.sqrt(E0 / (4 * np.pi)) / kk0
        A       = np.array([
            [k2 * zeta1,        k30 - k1 * zeta1,   -k2],
            [k2 * zeta2 - k30,  -k1 * zeta2,        k1],
            [kk0 * k2 / kk,     -kk0 * k1 / kk,     np.zeros_like(k)]]) * pref
    A[:, :, k == 0] = 0.0
    A[~np.isfinite(A)] = 0.0
    return A


def lifetime_shape(nx, ny, nz, dx, dy, dz, L):
    """Shape of the Mann eddy lifetime (kL)^(-2/3) / sqrt(2F1(1/3, 17/6; 4/3; -(kL)^-2)) on the rfft grid.
    Used as wave number dependence of the decorrelation time for wind evolution (large eddies live longer)."""
    k1 = 2 * np.pi * np.fft.fftfreq(nx, dx)[:, None, None]
    k2 = 2 * np.pi * np.fft.fftfreq(ny, dy)[None, :, None]
    k3 = 2 * np.pi * np.fft.rfftfreq(nz, dz)[None, None, :]
    kL = np.sqrt(k1 ** 2 + k2 ** 2 + k3 ** 2) * L
    with np.errstate(divide='ignore', invalid='ignore'):
        shape = kL ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -kL ** -2))
    shape[kL == 0] = np.inf
    return shape


def complex_noise(rng, shape):
    """Complex white noise with unit variance."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def box_from_noise(A, n, shape, scale=None, sigma_u=None):
    """Box (u, v, w) from the noise n (3, ...). Scaling: given scale, or such that std(u) = sigma_u.
    Returns (uvw float32, scale)."""
    uvw = np.fft.irfftn(np.einsum('ij...,j...->i...', A, n), s=shape, axes=(1, 2, 3))
    uvw = uvw - uvw.mean(axis=(1, 2, 3), keepdims=True)
    if scale is None:
        scale = sigma_u / uvw[0].std()
    return (uvw * scale).astype(np.float32), scale


def generate_box(A, shape, seed, sigma_u):
    """One realization (u, v, w) on the grid, scaled such that std(u) = sigma_u (float32 arrays)."""
    rng     = np.random.default_rng(seed)
    n       = (rng.standard_normal((3,) + A.shape[2:]) + 1j * rng.standard_normal((3,) + A.shape[2:])) / np.sqrt(2)
    u_hat   = np.einsum('ij...,j...->i...', A, n)
    uvw     = np.fft.irfftn(u_hat, s=shape, axes=(1, 2, 3))
    uvw     = uvw - uvw.mean(axis=(1, 2, 3), keepdims=True)
    uvw     *= sigma_u / uvw[0].std()
    return uvw.astype(np.float32)
