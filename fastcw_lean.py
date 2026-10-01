#!/usr/bin/env python3
"""
Cache-lean prototypes for FastCW.

Part 1 - Per-source geometry without polygon clipping or bisection
-------------------------------------------------------------------
For a distance field d that is linear on each triangle (the same assumption the
existing clipping kernels make when they interpolate isoline crossings along
edges), the area of {d <= r} inside triangle T is a closed-form piecewise
quadratic in r. With the face's distances sorted a <= b <= c:

    r <= a        : 0
    a <  r <  b   : A_T * (r - a)^2 / ((b - a)(c - a))
    b <= r <  c   : A_T * (1 - (c - r)^2 / ((c - a)(c - b)))
    r >= c        : A_T

The isoline segment length inside T follows from the coarea formula:
    len_T(r) = |grad d|_T * dA_T/dr
where |grad d|_T^2 = [alpha beta] G^-1 [alpha beta]^T, alpha = d1 - d0,
beta = d2 - d0, and G is the Gram matrix of the edge vectors (3 precomputed
floats per face). No vertex coordinates, normals or clipping are needed.

Faces are bucketed by dmax into bins of fixed width w (a high quantile of
edge length). A face in bin k is fully inside for r >= (k+1)w, and a face
whose distance extent is <= w is fully outside for r < (k-1)w, so
A(r) = prefix_area[floor(r/w)] + exact sum over two bins + a short side list
of faces whose extent exceeds w. The prefix sums bracket the root of
A(r) = target to within ~2w before any exact evaluation, and a safeguarded Newton step (A'(r) is exact)
converges to machine precision in a handful of band-only evaluations.

Per source this touches every face three times (dmax pass, count pass,
scatter pass) instead of ~5 full-length NumPy passes per area evaluation.

Part 2 - Batched heat method with self-consistent operators
-----------------------------------------------------------
The heat method's Poisson step is only exact if the Laplacian equals
G^T diag(A) G for the *same* gradient G used to build the divergence.
ConsistentBatchHeatEngine assembles L that way (it is the extrinsic cotan
Laplacian, i.e. what potpourri3d uses with use_robust=False), uses the
barycentric lumped mass, and fuses gradient -> normalize -> divergence into
one pass over faces, so a batch of k sources streams each Cholesky factor
once instead of k times.

All kernels use only Numba-compatible constructs and run (slowly) in pure
Python when Numba is unavailable.
"""

import math

import numpy as np
import scipy.sparse as sp

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


# =============================================================================
# Part 2: batched heat method with consistent operators
# =============================================================================


def face_gradient_basis(V, F):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e01 = p1 - p0
    e02 = p2 - p0
    nrm = np.cross(e01, e02)
    nl = np.linalg.norm(nrm, axis=1)
    if np.any(nl <= 0.0):
        raise ValueError("Zero-area faces must be removed before building the heat engine.")
    area = 0.5 * nl
    n = nrm / nl[:, None]
    inv2a = 1.0 / (2.0 * area)
    gb1 = np.cross(e02, n) * inv2a[:, None]
    gb2 = np.cross(n, e01) * inv2a[:, None]
    gb0 = -gb1 - gb2
    return area, np.ascontiguousarray(gb0), np.ascontiguousarray(gb1), np.ascontiguousarray(gb2)


def consistent_operators(V, F):
    """L = G^T diag(A) G (extrinsic cotan stiffness, PSD) and barycentric lumped M."""
    F = np.asarray(F, dtype=np.int64)
    nV = int(np.asarray(V).shape[0])
    area, gb0, gb1, gb2 = face_gradient_basis(V, F)
    gb = (gb0, gb1, gb2)
    rows, cols, vals = [], [], []
    for i in range(3):
        for j in range(3):
            rows.append(F[:, i])
            cols.append(F[:, j])
            vals.append(area * np.einsum("ij,ij->i", gb[i], gb[j]))
    L = sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nV, nV)
    )
    L.sum_duplicates()
    mass = np.zeros(nV)
    for i in range(3):
        np.add.at(mass, F[:, i], area / 3.0)
    return L, sp.diags(mass, format="csc"), area, gb0, gb1, gb2


@njit(cache=True)
def fused_normalized_divergence(u, F, area, gb0, gb1, gb2, out):
    """out[:, :] = sum_f A_f * (X_f . gradB_i),  X_f = -grad u / |grad u|.

    u and out are (n_vertices, k) C-contiguous, so each face touches three
    contiguous k-vectors. One pass over faces, no face-by-k temporaries.
    """
    out[:, :] = 0.0
    k = u.shape[1]
    for f in range(F.shape[0]):
        i0 = F[f, 0]
        i1 = F[f, 1]
        i2 = F[f, 2]
        A = area[f]
        for j in range(k):
            du1 = u[i1, j] - u[i0, j]
            du2 = u[i2, j] - u[i0, j]
            gx = du1 * gb1[f, 0] + du2 * gb2[f, 0]
            gy = du1 * gb1[f, 1] + du2 * gb2[f, 1]
            gz = du1 * gb1[f, 2] + du2 * gb2[f, 2]
            nn = math.sqrt(gx * gx + gy * gy + gz * gz)
            if not (nn > 1e-300):
                continue
            s = -A / nn
            out[i0, j] += s * (gx * gb0[f, 0] + gy * gb0[f, 1] + gz * gb0[f, 2])
            out[i1, j] += s * (gx * gb1[f, 0] + gy * gb1[f, 1] + gz * gb1[f, 2])
            out[i2, j] += s * (gx * gb2[f, 0] + gy * gb2[f, 1] + gz * gb2[f, 2])


class _ScipyLU:
    """Test-only stand-in for CHOLMOD (SuperLU, no Cholesky, slower)."""

    def __init__(self, A):
        from scipy.sparse.linalg import splu

        self._lu = splu(sp.csc_matrix(A), permc_spec="MMD_AT_PLUS_A")

    def solve(self, B):
        return self._lu.solve(np.asfortranarray(B))


class _Cholmod:
    def __init__(self, A):
        from sksparse.cholmod import cholesky

        self._f = cholesky(sp.csc_matrix(A))

    def solve(self, B):
        return self._f(np.asfortranarray(B))


class ConsistentBatchHeatEngine:
    """Heat-method distances for k sources per factor pass.

    Operators: L = G^T diag(A) G (extrinsic cotan), barycentric lumped M,
    t = t_coef * mean_edge_length^2, Poisson regularised as L + shift * M
    (mass-scaled so it is mesh-scale invariant; the last Cholesky pivot is
    ~ shift * total_area, comfortably positive at the default).
    Intended to reproduce potpourri3d use_robust=False. Surfaces that need
    use_robust=True (interior non-manifold vertices) should stay on pp3d.
    """

    supports_batching = True
    name = "batch_heat_consistent"

    def __init__(self, vertices, faces, t_coef=1.0, poisson_shift=1e-10, factor="auto"):
        V = np.ascontiguousarray(vertices, dtype=np.float64)
        F = np.ascontiguousarray(faces, dtype=np.int32)
        L, M, area, gb0, gb1, gb2 = consistent_operators(V, F)
        e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64)
        e.sort(axis=1)
        e = np.unique(e, axis=0)
        h = float(np.mean(np.linalg.norm(V[e[:, 1]] - V[e[:, 0]], axis=1)))
        self.t = float(t_coef) * h * h
        if factor == "auto":
            try:
                Fac = _Cholmod
                Fac(sp.eye(2, format="csc"))
            except Exception:
                Fac = _ScipyLU
        else:
            Fac = {"cholmod": _Cholmod, "scipy": _ScipyLU}[factor]
        self._heat = Fac(M + self.t * L)
        # Mass-scaled shift keeps the regularization mesh-scale invariant.
        self._poisson = Fac(L + poisson_shift * M)
        self.F, self.area, self.gb0, self.gb1, self.gb2 = F, area, gb0, gb1, gb2
        self.n_vertices = V.shape[0]

    def compute_distance_batch(self, sources, timings=None):
        """Returns a (k, n_vertices) C-contiguous array: row j = distances from sources[j].

        Rows (not columns) are per-source so downstream per-source reads are
        contiguous. If `timings` is a dict, per-stage seconds are accumulated.
        Distances are shifted to zero at each source; potpourri3d's shift
        convention may differ by a tiny constant, so compare with ~1e-6 abs tol.
        """
        import time

        src = np.asarray(sources, dtype=np.int64).reshape(-1)
        k = src.size
        t0 = time.perf_counter()
        rhs = np.zeros((self.n_vertices, k))
        rhs[src, np.arange(k)] = 1.0
        u = np.ascontiguousarray(self._heat.solve(rhs))
        t1 = time.perf_counter()
        div = np.empty_like(u)
        fused_normalized_divergence(u, self.F, self.area, self.gb0, self.gb1, self.gb2, div)
        t2 = time.perf_counter()
        phi = np.asarray(self._poisson.solve(div))
        phi = phi - phi[src, np.arange(k)][None, :]
        out = np.ascontiguousarray(phi.T)
        t3 = time.perf_counter()
        if timings is not None:
            for key, dt in (("heat_solve", t1 - t0), ("fused_div", t2 - t1), ("poisson_solve", t3 - t2)):
                timings[key] = timings.get(key, 0.0) + dt
        return out

    def compute_distance(self, source_idx):
        return self.compute_distance_batch([int(source_idx)])[0]
