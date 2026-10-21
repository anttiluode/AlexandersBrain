"""Escape-time measurement for a stored winding number."""
import numpy as np
from ab.tdgl import Medium, winding_state, winding


def relax_survives(mu, n, time=400.0):
    """Deterministic (T = 0) test: does winding n survive relaxation in this geometry?"""
    m = Medium(mu, winding_state(mu, n)[None])
    m.run(time)
    return int(winding(m.psi, mu)[0])


def escape_times(mu, n0, T, reps, tmax, chunk=20.0, seed=0, relax=200.0):
    """Return (times, escaped) for `reps` replicas started in winding n0.
    A replica escapes the first time its winding differs from n0 at a chunk boundary."""
    m = Medium(mu, np.stack([winding_state(mu, n0)] * reps), seed=seed)
    m.run(relax)
    start = winding(m.psi, mu)
    alive = start == n0
    times = np.full(reps, np.nan)
    times[~alive] = 0.0
    t = 0.0
    while t < tmax - 1e-9 and alive.any():
        m.run(chunk, T=T)
        t += chunk
        w = winding(m.psi, mu)
        hit = alive & (w != n0)
        times[hit] = t
        alive &= ~hit
    escaped = ~np.isnan(times)
    times[~escaped] = tmax
    return times, escaped, start


def tau_mle(times, escaped):
    """Exponential-lifetime MLE with right censoring, and a rough 68% interval."""
    k = int(escaped.sum())
    S = float(times.sum())
    if k == 0:
        return {"tau": None, "tau_lower_bound": S, "events": 0}
    tau = S / k
    return {"tau": tau, "tau_lo": tau / (1 + 1 / np.sqrt(k)), "tau_hi": tau / max(1 - 1 / np.sqrt(k), 0.2),
            "events": k}
