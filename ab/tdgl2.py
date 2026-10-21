"""The same medium with two additions, for writing:

1. a global write line: a uniform rotation-like vector potential A = (B/2)(-y, x), entering
   through gauge-covariant links, so every loop of area S is threaded by flux B*S and
   favours winding n ~ B*S / 2 pi;
2. a temperature MAP T(x), so noise can be local (G7: heat as the gate).

    d psi/dt = D_A^2 psi + (mu - |psi|^2) psi + xi,   <xi xi*> = 2 T(x) delta delta

Covariant Laplacian: neighbour i+e contributes exp(-i A_e dx) psi_{i+e}, so the gradient
energy sum |exp(-i A dx) psi_{i+e} - psi_i|^2 is minimised by grad(theta) = A.
In a neutral superfluid A plays the role of rotation (A = m Omega x r / hbar).
"""
import numpy as np
from numba import njit, prange
from ab.tdgl import DX, DT, grid, _bilinear, wrap


@njit(parallel=True, cache=True)
def _run_g(pr, pi, mu, cx, sx, cy, sy, Tm, dt, dx, nsteps, seed):
    B, N, _ = pr.shape
    inv = 1.0 / (dx * dx)
    for b in prange(B):
        np.random.seed(seed + 1000003 * b)
        ar = pr[b].copy()
        ai = pi[b].copy()
        br = np.empty_like(ar)
        bi = np.empty_like(ai)
        m = mu[b]
        amp = np.sqrt(Tm[b] * dt) / dx
        for _s in range(nsteps):
            for i in range(N):
                ip = (i + 1) % N
                im = (i - 1) % N
                for j in range(N):
                    jp = (j + 1) % N
                    jm = (j - 1) % N
                    xr = ar[i, j]
                    xi = ai[i, j]
                    # +x neighbour: e^{-i th} psi ; -x neighbour: e^{+i th(i-1)} psi
                    c1 = cx[i, j]; s1 = sx[i, j]
                    c2 = cx[im, j]; s2 = sx[im, j]
                    c3 = cy[i, j]; s3 = sy[i, j]
                    c4 = cy[i, jm]; s4 = sy[i, jm]
                    lr = (c1 * ar[ip, j] + s1 * ai[ip, j]) + (c2 * ar[im, j] - s2 * ai[im, j]) \
                        + (c3 * ar[i, jp] + s3 * ai[i, jp]) + (c4 * ar[i, jm] - s4 * ai[i, jm]) - 4.0 * xr
                    li = (c1 * ai[ip, j] - s1 * ar[ip, j]) + (c2 * ai[im, j] + s2 * ar[im, j]) \
                        + (c3 * ai[i, jp] - s3 * ar[i, jp]) + (c4 * ai[i, jm] + s4 * ar[i, jm]) - 4.0 * xi
                    g = m[i, j] - (xr * xr + xi * xi)
                    nr = xr + dt * (lr * inv + g * xr)
                    ni = xi + dt * (li * inv + g * xi)
                    a = amp[i, j]
                    if a > 0.0:
                        nr += a * np.random.normal()
                        ni += a * np.random.normal()
                    br[i, j] = nr
                    bi[i, j] = ni
            ar, br = br, ar
            ai, bi = bi, ai
        pr[b] = ar
        pi[b] = ai


def links(N, Bfield, centers=None):
    """Link phases theta = A . e dx at link midpoints.
    centers None: A = (B/2)(-y, x), rotation about the box centre (off-axis rings also see a
    uniform flow (B/2) z x c across them).
    centers given: A = (B/2) z x (r - c_k) with c_k the nearest ring centre, i.e. the same
    flux threaded through every hole and no flow across any ring. The seams between regions
    lie in normal material, where psi ~ 0."""
    X, Y, _, _ = grid(N)
    cxm = np.zeros_like(X); cym = np.zeros_like(Y)   # centre for x-links (midpoint X + dx/2)
    cxn = np.zeros_like(X); cyn = np.zeros_like(Y)   # centre for y-links (midpoint Y + dx/2)
    if centers is not None:
        C = np.array(centers)
        dm = ((X[..., None] + DX / 2 - C[:, 0]) ** 2 + (Y[..., None] - C[:, 1]) ** 2).argmin(-1)
        dn = ((X[..., None] - C[:, 0]) ** 2 + (Y[..., None] + DX / 2 - C[:, 1]) ** 2).argmin(-1)
        cxm, cym = C[dm, 0], C[dm, 1]
        cxn, cyn = C[dn, 0], C[dn, 1]
    thx = -(Bfield / 2) * (Y - cym) * DX          # A_x at (X + dx/2, Y) = -(B/2)(Y - c_y)
    thy = (Bfield / 2) * (X - cxn) * DX           # A_y at (X, Y + dx/2) = (B/2)(X - c_x)
    return np.cos(thx), np.sin(thx), np.cos(thy), np.sin(thy)


class Medium2:
    def __init__(self, mu, psi, seed=0):
        psi = np.asarray(psi, complex)
        mu = np.asarray(mu, float)
        if mu.ndim == 2:
            mu = np.broadcast_to(mu, psi.shape)
        self.mu = np.ascontiguousarray(mu.copy())
        self.pr = np.ascontiguousarray(psi.real.copy())
        self.pi = np.ascontiguousarray(psi.imag.copy())
        self.N = psi.shape[-1]
        self.Tm = np.zeros(psi.shape)
        self.seed = int(seed) * 7919 + 3
        self.calls = 0
        self.centers = None  # set to ring centres to thread flux through each hole

    @property
    def psi(self):
        return self.pr + 1j * self.pi

    def run(self, time, Bfield=0.0, T=None, dt=DT):
        """T: None keeps self.Tm; a scalar sets uniform T; an array (B,N,N) or (N,N) sets the map."""
        if T is not None:
            self.Tm = np.ascontiguousarray(np.broadcast_to(np.asarray(T, float), self.pr.shape).copy())
        cx, sx, cy, sy = links(self.N, Bfield, self.centers)
        n = int(round(time / dt))
        self.calls += 1
        _run_g(self.pr, self.pi, self.mu, cx, sx, cy, sy, self.Tm, dt, DX, n,
               (self.seed + 104729 * self.calls) % (2**31 - 2000003 * 128))


def winding_at(psi, mu, R, center=(0.0, 0.0), M=256):
    """Winding around the circle of radius R about `center`, stepping only through ordered
    material; a normal gap is crossed as one wrapped step. psi (B,N,N), mu (B,N,N) or (N,N)."""
    phi = 2 * np.pi * np.arange(M) / M
    x = center[0] + R * np.cos(phi)
    y = center[1] + R * np.sin(phi)
    mus = mu if mu.ndim == 3 else np.broadcast_to(mu, psi.shape)
    out = []
    for b in range(psi.shape[0]):
        keep = _bilinear(mus[b], x, y) > 0
        v = _bilinear(psi[b], x[keep], y[keep])
        d = np.angle(np.roll(v, -1) * np.conj(v))
        out.append(int(np.rint(d.sum() / (2 * np.pi))))
    return np.array(out)


def ring_mu(N, centers, R, W, mu_out=-1.0):
    X, Y, _, _ = grid(N)
    mu = np.full((N, N), mu_out)
    for c in centers:
        r = np.hypot(X - c[0], Y - c[1])
        mu[np.abs(r - R) < W / 2] = 1.0
    return mu


def gate_mask(N, center, R, W, gap, angle=0.0):
    """Cells of ring (center, R, W) within an arc of length `gap` around `angle`."""
    X, Y, _, _ = grid(N)
    r = np.hypot(X - center[0], Y - center[1])
    ph = np.arctan2(Y - center[1], X - center[0])
    return (np.abs(wrap(ph - angle)) < (gap / 2) / R) & (np.abs(r - R) < W / 2 + 0.01)


def winding_state_multi(mu, centers, ns):
    X, Y, _, _ = grid(mu.shape[-1])
    phase = np.zeros_like(X)
    for c, n in zip(centers, ns):
        phase = phase + n * np.arctan2(Y - c[1], X - c[0])
    return (np.sqrt(np.clip(mu, 0, None)) + 0.02) * np.exp(1j * phase)
