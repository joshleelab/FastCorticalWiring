#!/usr/bin/env python3
"""
Lean per-source geometry for FastCW: exact geodesic-disc area, perimeter and
radius inversion without polygon clipping or bisection.

For a distance field that is linear on each triangle (the same assumption the
legacy clipping kernels make), the area of {d <= r} inside a triangle is a
closed-form piecewise quadratic in r, and the isoline length inside it is
|grad d| * dA/dr (coarea formula). Faces are bucketed by dmax into bins of a
fixed width, so an area evaluation touches only two bins plus a short list of
faces whose distance extent exceeds the width. Prefix sums over bins bracket
each radius to about two bin widths; a safeguarded Newton iteration with the
exact derivative then converges to machine precision.

Validated against the legacy clipping kernels to ~1e-14 (area) and ~1e-15
(perimeter), including non-Lipschitz fields that force many long faces.

field_health() is the per-source screen for corrupted heat-method fields
(negative distances, or distances shorter than the straight-line chord).
"""

import math

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


# =============================================================================
# Part 1: per-source geometry
# =============================================================================


def precompute_face_metric(V, F):
    """Face areas and the inverse-Gram coefficients for |grad d| per face."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    a11 = np.einsum("ij,ij->i", e1, e1)
    a22 = np.einsum("ij,ij->i", e2, e2)
    a12 = np.einsum("ij,ij->i", e1, e2)
    det = a11 * a22 - a12 * a12  # = (2 * area)^2
    area = 0.5 * np.sqrt(np.maximum(det, 0.0))
    inv = np.zeros_like(det)
    good = det > 0.0
    inv[good] = 1.0 / det[good]
    return (
        np.ascontiguousarray(area),
        np.ascontiguousarray(a22 * inv),
        np.ascontiguousarray(-a12 * inv),
        np.ascontiguousarray(a11 * inv),
    )


@njit(cache=True)
def _face_cdf(d0, d1, d2, A, r):
    """(area of {d <= r} in the face, its derivative wrt r)."""
    a = d0
    b = d1
    c = d2
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    if r >= c:
        return A, 0.0
    if r <= a:
        return 0.0, 0.0
    if r < b:
        den = (b - a) * (c - a)
        x = r - a
        return A * x * x / den, 2.0 * A * x / den
    den = (c - a) * (c - b)
    y = c - r
    return A * (1.0 - y * y / den), 2.0 * A * y / den


@njit(cache=True)
def _grad_norm(d0, d1, d2, g11, g12, g22):
    al = d1 - d0
    be = d2 - d0
    q = al * al * g11 + 2.0 * al * be * g12 + be * be * g22
    return math.sqrt(q) if q > 0.0 else 0.0


def default_bin_width(V, F, quantile=0.999, pad=1.05):
    """Fixed bucket width: a high quantile of per-face max edge length.

    Heat-method distances are close to 1-Lipschitz, so almost every face's
    distance extent (dmax - dmin) is below its longest edge. Faces that exceed
    the width on a given source go to a side list; correctness never depends
    on this choice, only the number of faces each band evaluation touches.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    L = np.maximum.reduce([
        np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1),
        np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1),
        np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1),
    ])
    return float(np.quantile(L, quantile) * pad)


@njit(cache=True)
def build_distance_buckets(d, F, face_area, width, dmax_buf, order_buf, long_buf):
    """Bucket faces by dmax into bins of fixed width.

    Returns (nbins, bin_ptr, prefix_area, n_long). Faces whose distance extent
    exceeds `width` are also listed in long_buf[:n_long]. dmax_buf, order_buf
    and long_buf are caller-owned scratch (no face-sized allocation per source).
    """
    nf = F.shape[0]
    dglob = 0.0
    n_long = 0
    for f in range(nf):
        d0 = d[F[f, 0]]
        d1 = d[F[f, 1]]
        d2 = d[F[f, 2]]
        if not (math.isfinite(d0) and math.isfinite(d1) and math.isfinite(d2)):
            dmax_buf[f] = np.nan
            continue
        lo = d0
        hi = d0
        if d1 < lo:
            lo = d1
        if d1 > hi:
            hi = d1
        if d2 < lo:
            lo = d2
        if d2 > hi:
            hi = d2
        dmax_buf[f] = hi
        if hi - lo > width:
            long_buf[n_long] = f
            n_long += 1
        if hi > dglob:
            dglob = hi
    nbins = int(dglob / width) + 2
    counts = np.zeros(nbins + 1, dtype=np.int64)
    prefix = np.zeros(nbins + 1, dtype=np.float64)
    for f in range(nf):
        h = dmax_buf[f]
        if h != h:
            continue
        k = int(h / width)
        if k < 0:
            k = 0
        elif k >= nbins:
            k = nbins - 1
        counts[k + 1] += 1
        prefix[k + 1] += face_area[f]
    for k in range(nbins):
        counts[k + 1] += counts[k]
        prefix[k + 1] += prefix[k]
    fill = counts[:nbins].copy()
    for f in range(nf):
        h = dmax_buf[f]
        if h != h:
            continue
        k = int(h / width)
        if k < 0:
            k = 0
        elif k >= nbins:
            k = nbins - 1
        order_buf[fill[k]] = f
        fill[k] += 1
    return nbins, counts, prefix, n_long


@njit(cache=True)
def area_at(r, d, F, face_area, width, nbins, ptr, prefix, order, dmax_buf, long_buf, n_long):
    """Exact area inside radius r and dA/dr.

    Faces in bins < floor(r/w) are fully inside (prefix sum); short faces in
    bins >= floor(r/w) + 2 are fully outside; only two bins plus any long
    faces beyond them are evaluated.
    """
    if not (r > 0.0):  # also rejects NaN
        return 0.0, 0.0
    kr = int(r / width)
    if kr >= nbins:
        return prefix[nbins], 0.0
    A = prefix[kr]
    dA = 0.0
    kend = kr + 2
    if kend > nbins:
        kend = nbins
    for j in range(ptr[kr], ptr[kend]):
        f = order[j]
        a, da = _face_cdf(d[F[f, 0]], d[F[f, 1]], d[F[f, 2]], face_area[f], r)
        A += a
        dA += da
    for j in range(n_long):
        f = long_buf[j]
        if int(dmax_buf[f] / width) >= kend:
            a, da = _face_cdf(d[F[f, 0]], d[F[f, 1]], d[F[f, 2]], face_area[f], r)
            A += a
            dA += da
    return A, dA


@njit(cache=True)
def perimeter_at(r, d, F, face_area, g11, g12, g22, width, nbins, ptr, order, dmax_buf, long_buf, n_long):
    """Isoline length at level r via the coarea formula."""
    if not (r > 0.0):  # also rejects NaN
        return 0.0
    kr = int(r / width)
    if kr >= nbins:
        return 0.0
    kend = kr + 2
    if kend > nbins:
        kend = nbins
    P = 0.0
    for j in range(ptr[kr], ptr[kend]):
        f = order[j]
        d0 = d[F[f, 0]]
        d1 = d[F[f, 1]]
        d2 = d[F[f, 2]]
        _, da = _face_cdf(d0, d1, d2, face_area[f], r)
        if da > 0.0:
            P += da * _grad_norm(d0, d1, d2, g11[f], g12[f], g22[f])
    for j in range(n_long):
        f = long_buf[j]
        if int(dmax_buf[f] / width) >= kend:
            d0 = d[F[f, 0]]
            d1 = d[F[f, 1]]
            d2 = d[F[f, 2]]
            _, da = _face_cdf(d0, d1, d2, face_area[f], r)
            if da > 0.0:
                P += da * _grad_norm(d0, d1, d2, g11[f], g12[f], g22[f])
    return P


@njit(cache=True)
def solve_radius(target, d, F, face_area, width, nbins, ptr, prefix, order,
                 dmax_buf, long_buf, n_long, rtol, max_iter):
    """Exact inverse of A(r) = target. Returns (r, n_exact_evaluations)."""
    total = prefix[nbins]
    if target != target:
        return np.nan, 0
    if target <= 0.0:
        return 0.0, 0
    if target > total * (1.0 + 1e-12):
        return np.nan, 0
    lo_i = 0
    hi_i = nbins
    while lo_i < hi_i:  # first k with prefix[k] >= target  =>  A(k*w) >= target
        mid = (lo_i + hi_i) // 2
        if prefix[mid] >= target:
            hi_i = mid
        else:
            lo_i = mid + 1
    r_hi = lo_i * width
    r_lo = (lo_i - 2) * width if lo_i >= 2 else 0.0
    n_eval = 0
    if n_long > 0:
        # Long faces can push A(r_lo) above target; walk the lower bound down.
        while r_lo > 0.0:
            a_lo, _ = area_at(r_lo, d, F, face_area, width, nbins, ptr, prefix, order, dmax_buf, long_buf, n_long)
            n_eval += 1
            if a_lo < target:
                break
            r_hi = r_lo
            r_lo = r_lo - 2.0 * width
            if r_lo < 0.0:
                r_lo = 0.0
    r = 0.5 * (r_lo + r_hi)
    for it in range(max_iter):
        A, dA = area_at(r, d, F, face_area, width, nbins, ptr, prefix, order, dmax_buf, long_buf, n_long)
        n_eval += 1
        err = A - target
        if abs(err) <= rtol * target:
            return r, n_eval
        if err > 0.0:
            r_hi = r
        else:
            r_lo = r
        rn = 0.5 * (r_lo + r_hi)
        if dA > 0.0:
            cand = r - err / dA
            if r_lo < cand < r_hi:
                rn = cand
        if r_hi - r_lo <= 1e-13 * (1.0 + r_hi):
            return rn, n_eval
        r = rn
    return r, n_eval


@njit(cache=True)
def per_source_geometry(
    d, F, face_area, g11, g12, g22, width, sorted_targets, extra_fracs, rtol, max_iter,
    dmax_buf, order_buf, long_buf, radii_out, perim_out, extra_r_out, extra_a_out,
):
    """Radius/perimeter at every target plus supplementary (r, A) samples.

    extra_fracs: fractions in (0, 1) placed log-uniformly between consecutive
    solved radii (same placement rule as n_samples_between_scales).
    Returns the number of exact band evaluations used.
    """
    nbins, ptr, prefix, n_long = build_distance_buckets(d, F, face_area, width, dmax_buf, order_buf, long_buf)
    n_eval = 0
    for s in range(sorted_targets.shape[0]):
        r, k = solve_radius(sorted_targets[s], d, F, face_area, width, nbins, ptr, prefix, order_buf,
                            dmax_buf, long_buf, n_long, rtol, max_iter)
        n_eval += k
        radii_out[s] = r
        if r == r:
            perim_out[s] = perimeter_at(r, d, F, face_area, g11, g12, g22, width, nbins, ptr, order_buf,
                                        dmax_buf, long_buf, n_long)
        else:
            perim_out[s] = np.nan
    m = extra_fracs.shape[0]
    for s in range(sorted_targets.shape[0] - 1):
        r0 = radii_out[s]
        r1 = radii_out[s + 1]
        for j in range(m):
            idx = s * m + j
            if r0 == r0 and r1 == r1 and r0 > 0.0 and r1 > r0:
                rr = math.exp(math.log(r0) + extra_fracs[j] * (math.log(r1) - math.log(r0)))
                a, _ = area_at(rr, d, F, face_area, width, nbins, ptr, prefix, order_buf, dmax_buf, long_buf, n_long)
                extra_r_out[idx] = rr
                extra_a_out[idx] = a
                n_eval += 1
            else:
                extra_r_out[idx] = np.nan
                extra_a_out[idx] = np.nan
    return n_eval


def field_health(d, V, src, h, neg_frac_tol=1e-3, chord_frac_tol=1e-3):
    """Screen one distance field. Returns (neg_frac, chord_violation_frac, is_bad).

    neg_frac: fraction of vertices with d < -h (true distances are >= 0).
    chord_violation_frac: fraction with d < 0.9 * |x - x_src| - h (a geodesic is
    never shorter than the chord). h is a mesh length scale (median face edge).
    Catches gross failures; does not detect mild far-field errors.
    """
    d = np.asarray(d, dtype=np.float64)
    if not np.isfinite(d).all():
        return float("nan"), float("nan"), True
    chord = np.linalg.norm(V - V[int(src)], axis=1)
    neg = float(np.mean(d < -h))
    viol = float(np.mean(d < 0.9 * chord - h))
    return neg, viol, bool(neg > neg_frac_tol or viol > chord_frac_tol)


def mean_edge_length(V, F):
    """Mean length over unique edges (the scale potpourri3d uses for t_coef)."""
    F = np.asarray(F, dtype=np.int64)
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E.sort(axis=1)
    E = np.unique(E, axis=0)
    V = np.asarray(V, dtype=np.float64)
    return float(np.mean(np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1)))
