#!/usr/bin/env python3
"""
Screen potpourri3d heat-method distance fields for corruption and test use_robust.

For N sources spread over the cortical submesh, solves with use_robust=False
and use_robust=True and checks each field for:
  negative : fraction of vertices with d < -h            (distances are >= 0)
  chord    : fraction with d < 0.9 * |x - x_src| - h     (geodesic >= chord, always)
A field is flagged when either fraction exceeds 1e-3. Both checks are
O(n_vertices) and could run inside the production loop.

Then reports, robust vs non-robust, the change in unweighted MSD and in the
12 radii (via fastcw_lean), split by whether the non-robust field was flagged,
plus per-solve time. --exact M compares both against pygeodesic for M sources
and screens pygeodesic's own field with the same checks.

  python screen_heat.py $SUBJECTS_DIR <subject> --hemi lh --n-sources 200 --exact 2
"""
import argparse
import time

import numpy as np

import compare_fastcw_lean as cf
import fastcw_lean as fl


def field_health(d, V, src, h):
    chord = np.linalg.norm(V - V[src], axis=1)
    neg = float(np.mean(d < -h))
    viol = float(np.mean(d < 0.9 * chord - h))
    return neg, viol, (neg > 1e-3 or viol > 1e-3)


def delaunay_report(V, F):
    """Cotan-Laplacian health: share of interior edges with negative weight (non-Delaunay).

    A negative weight means the two angles opposite the edge sum to more than 180
    degrees; each one breaks the discrete maximum principle of the extrinsic cotan
    Laplacian that use_robust=False relies on.
    """
    F = np.asarray(F, dtype=np.int64)
    p = V[F]
    cots = []
    for i in range(3):
        a = p[:, (i + 1) % 3] - p[:, i]
        b = p[:, (i + 2) % 3] - p[:, i]
        cots.append(np.einsum("ij,ij->i", a, b) / np.linalg.norm(np.cross(a, b), axis=1))
    E = np.concatenate([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]])  # edge opposite vertex i
    E.sort(axis=1)
    _, inv, counts = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    W = np.bincount(np.asarray(inv).reshape(-1), weights=0.5 * np.concatenate(cots))
    interior = counts == 2
    w_in = W[interior]
    obtuse = float((np.stack(cots, axis=1) < 0).any(axis=1).mean())
    return float(np.mean(w_in < 0)), int(interior.sum()), float(np.percentile(w_in, 0.1)), obtuse


def make_solvers(an, synthetic):
    if synthetic:  # self-test only: both "modes" are the same heat engine, exact = analytic
        e = fl.ConsistentBatchHeatEngine(an.vertices, an.faces, factor="auto")
        return e.compute_distance, e.compute_distance, an.distance_engine.compute_distance
    import potpourri3d as pp3d
    from distance_engines import create_distance_engine

    nr = pp3d.MeshHeatMethodDistanceSolver(an.vertices, an.faces, use_robust=False)
    rb = pp3d.MeshHeatMethodDistanceSolver(an.vertices, an.faces, use_robust=True)
    holder = {}

    def exact(s):
        if "e" not in holder:
            holder["e"] = create_distance_engine("pygeodesic", an.vertices, an.faces)
        return holder["e"].compute_distance(int(s))

    return (lambda s: np.asarray(nr.compute_distance(int(s)), float),
            lambda s: np.asarray(rb.compute_distance(int(s)), float), exact)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("subjects_dir", nargs="?")
    p.add_argument("subject", nargs="?")
    p.add_argument("--hemi", default="lh")
    p.add_argument("--surf-type", default="pial")
    p.add_argument("--custom-label", default=None)
    p.add_argument("--engine", default="potpourri")
    p.add_argument("--synthetic", type=int, default=None)
    p.add_argument("--n-sources", type=int, default=200)
    p.add_argument("--exact", type=int, default=0, help="sources to check against pygeodesic")
    cf.add_common_args(p)
    args = p.parse_args()
    args.use_robust = False
    ca, an = cf.load_analysis(args)
    V, F = an.vertices, an.faces
    h = float(np.median(an.face_L))
    frac_neg, n_int, w01, obtuse = delaunay_report(V, F)
    print(f"mesh: {V.shape[0]} vertices, {F.shape[0]} faces, {an.boundary_indices.size} boundary vertices | "
          f"non-Delaunay interior edges {100 * frac_neg:.2f}% of {n_int} (0.1th pct cotan weight {w01:.2f}) | "
          f"obtuse faces {100 * obtuse:.1f}%")
    solve_nr, solve_rb, solve_exact = make_solvers(an, args.synthetic is not None)

    nF = F.shape[0]
    area, g11, g12, g22 = fl.precompute_face_metric(V, F)
    width = fl.default_bin_width(V, F)
    T = np.array(cf.DEFAULT_SCALES) * float(an.vertex_areas_sub.sum())
    bufs = (np.empty(nF), np.empty(nF, np.int32), np.empty(nF, np.int32))
    ex_r, ex_a = np.empty(0), np.empty(0)

    def radii(d):
        ro, po = np.empty(len(T)), np.empty(len(T))
        fl.per_source_geometry(d, F, area, g11, g12, g22, width, T, np.empty(0), 1e-10, 60,
                               *bufs, ro, po, ex_r, ex_a)
        return ro

    def msd(d):
        v = d[(d > an.eps) & np.isfinite(d)]
        return float(np.mean(v)) if v.size else np.nan

    sources = cf.pick_sources(an, args.n_sources, args.seed)
    solve_nr(sources[0]); solve_rb(sources[0])  # warm-up
    t_nr, t_rb, rec = [], [], []
    for src in sources:
        t0 = time.perf_counter(); dn = solve_nr(src); t1 = time.perf_counter(); dr = solve_rb(src); t2 = time.perf_counter()
        t_nr.append(t1 - t0); t_rb.append(t2 - t1)
        hn, hr = field_health(dn, V, src, h), field_health(dr, V, src, h)
        rn, rr = radii(dn), radii(dr)
        with np.errstate(invalid="ignore", divide="ignore"):
            rdiff = np.abs(rn - rr) / rr
        rec.append(dict(src=int(src), nr=hn, rb=hr, dmin_nr=float(dn.min()), dmax_nr=float(dn.max()),
                        dmax_rb=float(dr.max()), msd_diff=abs(msd(dn) - msd(dr)) / msd(dr),
                        r_med=float(np.nanmedian(rdiff)), r_max=float(np.nanmax(rdiff)),
                        n_nan_nr=int(np.sum(~np.isfinite(rn))), n_nan_rb=int(np.sum(~np.isfinite(rr)))))

    fn = [r for r in rec if r["nr"][2]]
    fr_ = [r for r in rec if r["rb"][2]]
    ok = [r for r in rec if not r["nr"][2]]
    print(f"\n=== {len(sources)} sources, {V.shape[0]} vertices, median edge {h:.3f} mm ===")
    print(f"flagged fields  use_robust=False: {len(fn):4d} ({100 * len(fn) / len(rec):.1f}%)   "
          f"use_robust=True: {len(fr_):4d} ({100 * len(fr_) / len(rec):.1f}%)")
    print(f"solve time      use_robust=False: {1e3 * np.median(t_nr):.1f} ms   use_robust=True: {1e3 * np.median(t_rb):.1f} ms")

    def summarise(group, label):
        if not group:
            print(f"{label}: none")
            return
        m = np.array([r["msd_diff"] for r in group])
        rm = np.array([r["r_med"] for r in group])
        rx = np.array([r["r_max"] for r in group])
        print(f"{label} (n={len(group)}): MSD rel change median {np.nanmedian(m):.2e} max {np.nanmax(m):.2e} | "
              f"radius rel change median {np.nanmedian(rm):.2e} max {np.nanmax(rx):.2e}")

    print("robust vs non-robust:")
    summarise(ok, "  non-robust field looked healthy")
    summarise(fn, "  non-robust field flagged      ")
    for r in fn[:10]:
        print(f"    src {r['src']:6d}: non-robust neg {r['nr'][0]:.3f} chord-viol {r['nr'][1]:.3f} "
              f"d in [{r['dmin_nr']:.1f}, {r['dmax_nr']:.1f}] vs robust dmax {r['dmax_rb']:.1f} | "
              f"MSD change {100 * r['msd_diff']:.1f}%  radius change max {100 * r['r_max']:.1f}%  "
              f"NaN radii nr/rb {r['n_nan_nr']}/{r['n_nan_rb']}")
    np.savetxt("screen_flagged_sources.txt", [r["src"] for r in fn], fmt="%d")
    print("flagged submesh source ids -> screen_flagged_sources.txt")

    if args.exact:
        print(f"\nvs exact geodesics (pygeodesic), d > 5h:")
        extra = [r["src"] for r in fn if r["src"] not in set(int(x) for x in sources[:args.exact])][:1]
        for src in list(sources[:args.exact]) + extra:
            t0 = time.perf_counter(); de = np.asarray(solve_exact(src), float); te = time.perf_counter() - t0
            dn, dr = solve_nr(src), solve_rb(src)
            fin = np.isfinite(de) & np.isfinite(dn) & np.isfinite(dr)
            he = field_health(np.where(fin, de, 0.0), V, src, h)
            m = fin & (de > 5 * h)
            print(f"  src {src:6d} ({te:.0f}s): exact non-finite {int(np.sum(~np.isfinite(de)))}/{de.size} "
                  f"neg {he[0]:.3f} chord-viol {he[1]:.3f} dmax {np.max(de[fin]):.1f} | "
                  f"median rel err non-robust {np.median(np.abs(dn - de)[m] / de[m]):.2e}  "
                  f"robust {np.median(np.abs(dr - de)[m] / de[m]):.2e} | "
                  f"max abs {np.max(np.abs(dn - de)[fin]):.1f} / {np.max(np.abs(dr - de)[fin]):.1f} mm")


if __name__ == "__main__":
    main()
