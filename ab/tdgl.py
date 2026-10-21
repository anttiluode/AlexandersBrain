"""Complex Ginzburg-Landau medium with a material map mu(x), batched, with Langevin noise.

    d psi / dt = lap(psi) + (mu(x) - |psi|^2) psi + src(x) + xi(x, t)
    <xi xi*> = 2 T delta(x - x') delta(t - t')

This is the overdamped (time-dependent Ginzburg-Landau) dynamics of the free energy

    F[psi] = sum dx^2 [ |grad psi|^2 - mu |psi|^2 + |psi|^4 / 2 ]

so with T = 0 and src = 0 it is a gradient flow (F never increases), and with T > 0 its
stationary measure is exp(-F/T).  mu > 0 is "ordered material" (|psi| ~ sqrt(mu)),
mu < 0 is "normal material" (psi decays over 1/sqrt(-mu)).  Units: coherence length
xi = 1 where mu = 1.

Periodic box; geometries keep psi ~ 0 near the box edge.
"""
import numpy as np
from numba import njit, prange

DX = 0.5
DT = 0.04


@njit(parallel=True, cache=True)
def _run(pr, pi, mu, sr, si, dt, dx, T, nsteps, seed):
    B, N, _ = pr.shape
    inv = 1.0 / (dx * dx)
    amp = np.sqrt(T * dt) / dx
    for b in prange(B):
        np.random.seed(seed + 1000003 * b)
        ar = pr[b].copy()
        ai = pi[b].copy()
        br = np.empty_like(ar)
        bi = np.empty_like(ai)
        m = mu[b]
        for _s in range(nsteps):
            for i in range(N):
                ip = (i + 1) % N
                im = (i - 1) % N
                for j in range(N):
                    jp = (j + 1) % N
                    jm = (j - 1) % N
                    xr = ar[i, j]
                    xi = ai[i, j]
                    lr = (ar[ip, j] + ar[im, j] + ar[i, jp] + ar[i, jm] - 4.0 * xr) * inv
                    li = (ai[ip, j] + ai[im, j] + ai[i, jp] + ai[i, jm] - 4.0 * xi) * inv
                    g = m[i, j] - (xr * xr + xi * xi)
                    nr = xr + dt * (lr + g * xr + sr[i, j])
                    ni = xi + dt * (li + g * xi + si[i, j])
                    if T > 0.0:
                        nr += amp * np.random.normal()
                        ni += amp * np.random.normal()
                    br[i, j] = nr
                    bi[i, j] = ni
            ar, br = br, ar
            ai, bi = bi, ai
        pr[b] = ar
        pi[b] = ai


class Medium:
    """Batch of B independent copies of one or more geometries."""

    def __init__(self, mu, psi, seed=0):
        mu = np.asarray(mu, float)
        psi = np.asarray(psi, complex)
        if mu.ndim == 2:
            mu = np.broadcast_to(mu, psi.shape).copy()
        self.mu = np.ascontiguousarray(mu)
        self.pr = np.ascontiguousarray(psi.real)
        self.pi = np.ascontiguousarray(psi.imag)
        self.N = psi.shape[-1]
        self.seed = int(seed) * 7919 + 1
        self.calls = 0
        self.t = 0.0

    @property
    def psi(self):
        return self.pr + 1j * self.pi

    def run(self, time, T=0.0, src=None, dt=DT):
        n = int(round(time / dt))
        if src is None:
            sr = np.zeros((self.N, self.N))
            si = np.zeros((self.N, self.N))
        else:
            sr = np.ascontiguousarray(src.real, dtype=float)
            si = np.ascontiguousarray(src.imag, dtype=float)
        self.calls += 1
        _run(self.pr, self.pi, self.mu, sr, si, dt, DX, float(T), n,
             (self.seed + 104729 * self.calls) % (2**31 - 2000003 * 64))
        self.t += n * dt


# ---------------------------------------------------------------- geometry

def grid(N):
    c = (N / 2) * DX
    x = (np.arange(N) + 0.5) * DX - c
    X, Y = np.meshgrid(x, x, indexing="ij")
    return X, Y, np.hypot(X, Y), np.arctan2(Y, X)


def annulus(N, R=10.0, W=4.0, mu_in=1.0, mu_out=-1.0):
    _, _, r, _ = grid(N)
    mu = np.full((N, N), mu_out)
    mu[np.abs(r - R) < W / 2] = mu_in
    return mu


def constricted_annulus(N, w, R=10.0, W=4.0, length=2.0, mu_out=-1.0):
    """Annulus whose width narrows to w over an arc of the given length at phi = 0."""
    _, _, r, ph = grid(N)
    mu = annulus(N, R, W, 1.0, mu_out)
    win = np.abs(ph) < (length / 2) / R
    mu[win & (np.abs(r - R) >= w / 2)] = mu_out
    return mu


def c_ring(N, gap, R=10.0, W=4.0, mu_gap=-1.0, mu_out=-1.0):
    """Annulus cut through its full width by a gap of normal material (arc length gap)."""
    _, _, r, ph = grid(N)
    mu = annulus(N, R, W, 1.0, mu_out)
    if gap > 0:
        mu[(np.abs(ph) < (gap / 2) / R) & (np.abs(r - R) < W / 2 + 0.01)] = mu_gap
    return mu


def disk(N, R=12.0, mu_out=-1.0):
    _, _, r, _ = grid(N)
    mu = np.full((N, N), mu_out)
    mu[r < R] = 1.0
    return mu


def winding_state(mu, n, theta0=0.0):
    N = mu.shape[-1]
    _, _, _, ph = grid(N)
    amp = np.sqrt(np.clip(mu, 0, None)) + 0.02
    return amp * np.exp(1j * (n * ph + theta0))


# ---------------------------------------------------------------- measurement

def _bilinear(field, x, y):
    """field (..., N, N) on the grid of grid(N); x, y arrays of points."""
    N = field.shape[-1]
    c = (N / 2) * DX
    fi = (x + c) / DX - 0.5
    fj = (y + c) / DX - 0.5
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    a = fi - i0
    b = fj - j0
    i0 %= N
    j0 %= N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N
    return ((1 - a) * (1 - b) * field[..., i0, j0] + a * (1 - b) * field[..., i1, j0]
            + (1 - a) * b * field[..., i0, j1] + a * b * field[..., i1, j1])


def loop(R, M=256):
    phi = 2 * np.pi * np.arange(M) / M
    return R * np.cos(phi), R * np.sin(phi), phi


def winding(psi, mu, R=10.0, M=256):
    """Winding number around the circle of radius R, stepping only through ordered
    material (mu > 0): a gap of normal material is crossed as ONE wrapped phase step."""
    x, y, _ = loop(R, M)
    keep = _bilinear(mu[0] if mu.ndim == 3 else mu, x, y) > 0
    v = _bilinear(psi, x[keep], y[keep])
    d = np.angle(np.roll(v, -1, axis=-1) * np.conj(v))
    return np.rint(d.sum(-1) / (2 * np.pi)).astype(int)


def winding_radii(psi, mu, radii=(9.0, 10.0, 11.0)):
    """Winding on several loops inside the ring; agreement means no vortex is mid-crossing."""
    return np.stack([winding(psi, mu, R) for R in radii], -1)


def ring_phase(psi, n, R=10.0, M=256):
    """Global phase Theta of the ring in winding state n: psi ~ |psi| e^{i(n phi + Theta)}."""
    x, y, phi = loop(R, M)
    v = _bilinear(psi, x, y)
    n = np.asarray(n)[..., None] if np.ndim(n) else n
    return np.angle(np.mean(v * np.exp(-1j * n * phi), axis=-1))


def free_energy(psi, mu):
    gx = (np.roll(psi, -1, -2) - psi) / DX
    gy = (np.roll(psi, -1, -1) - psi) / DX
    a2 = np.abs(psi) ** 2
    return (DX * DX) * np.sum(np.abs(gx) ** 2 + np.abs(gy) ** 2 - mu * a2 + 0.5 * a2 * a2,
                              axis=(-1, -2))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
