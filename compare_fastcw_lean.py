#!/usr/bin/env python3
"""
Compare fastcw_lean against the current FastCW implementation on a real subject.

Geometry check (always):
  For N sources spread over the surface, computes the default 12 radii,
  perimeters and supplementary samples with (a) the current code path
  (dmin/dmax + stable argsort + bisection + clipping kernels) and (b) the
  closed-form bucketed path. Reports agreement and per-source time.

Solve check (--solve):
  Builds ConsistentBatchHeatEngine on the same submesh, compares its distances
  with the reference engine (potpourri3d by default) and times
  heat solve / fused divergence / Poisson solve per source for several batch
  widths. Run once with one process, then again with N copies in parallel
  (pinned) to see how each stage degrades under contention.

Examples
  python compare_fastcw_lean.py $SUBJECTS_DIR sub-01 --hemi lh --n-sources 50 --solve
  python compare_fastcw_lean.py --synthetic 5 --solve          # self-test, no data needed
"""

import argparse
import sys
import time

import numpy as np

DEFAULT_SCALES = [0.002, 0.00267988, 0.00359088, 0.00481157, 0.00644721, 0.00863888,
                  0.01157558, 0.01551059, 0.02078326, 0.02784833, 0.03731509, 0.05]



def load_analysis(args):
    if args.synthetic is not None:
        import lean_testkit as tk

        tk.install_stubs()
        import core_analysis as ca

        V, F = tk.icosphere(args.synthetic)
        V = V * 70.0
        return ca, ca.FastCorticalWiringAnalysis(V, F, np.ones(len(V), bool), engine_type="great_circle_stub")
    import core_analysis as ca

    no_mask = bool(getattr(args, "no_mask", False))
    if not no_mask:
        check_label_matches_surface(args)
    an = ca.FastCorticalWiringAnalysis.from_freesurfer(
        args.subjects_dir, args.subject, hemi=args.hemi, surf_type=args.surf_type,
        engine_type=args.engine, custom_label=args.custom_label, no_mask=no_mask,
        engine_kwargs={"use_robust": True} if getattr(args, "use_robust", False) else None,
    )
    return ca, an


def check_label_matches_surface(args):
    """Refuse to apply a label written for a different mesh (e.g. native label on a decimated surface).

    io_utils silently drops out-of-range label indices, which on a remeshed surface
    produces an arbitrary scattered mask instead of an error.
    """
    import os

    import nibabel as nib

    surf = os.path.join(args.subjects_dir, args.subject, "surf", f"{args.hemi}.{args.surf_type}")
    n_v = nib.freesurfer.read_geometry(surf)[0].shape[0]
    label_dir = os.path.join(args.subjects_dir, args.subject, "label")
    for name in ([args.custom_label] if args.custom_label else []) + ["cortex"]:
        lab = os.path.join(label_dir, f"{args.hemi}.{name}.label")
        if os.path.exists(lab):
            idx = np.asarray(nib.freesurfer.read_label(lab))
            if idx.size and int(idx.max()) >= n_v:
                raise SystemExit(
                    f"ERROR: {lab} indexes vertices up to {int(idx.max())}, but {surf} has only {n_v} "
                    "vertices, so the label belongs to a different mesh. For cortex-only or "
                    "decimated surfaces pass --no-mask.")
            return


def pick_sources(an, n, seed=None):
    """n submesh sources: evenly spaced along BFS order (deterministic), or random with a seed."""
    order = np.asarray(an._bfs_order)
    n = min(int(n), order.size)
    if seed is None:
        return order[np.linspace(0, order.size - 1, n).astype(int)]
    return np.sort(np.random.default_rng(int(seed)).choice(order.size, n, replace=False))


def add_common_args(p):
    p.add_argument("--no-mask", action="store_true",
                   help="use every vertex (required for cortex-only / decimated surfaces)")
    p.add_argument("--seed", type=int, default=None,
                   help="draw random sources with this seed (default: fixed, evenly spaced along BFS order)")


def current_path(an, d, targets, extra_fracs, area_tol):
    """Replicates phases 3-5 + supplementary samples of compute_all_wiring_costs (cold start)."""
    np.take(d, an._f0, out=an._d0_buf)
    np.take(d, an._f1, out=an._d1_buf)
    np.take(d, an._f2, out=an._d2_buf)
    np.minimum(an._d0_buf, an._d1_buf, out=an._dmin_buf)
    np.minimum(an._dmin_buf, an._d2_buf, out=an._dmin_buf)
    np.maximum(an._d0_buf, an._d1_buf, out=an._dmax_buf)
    np.maximum(an._dmax_buf, an._d2_buf, out=an._dmax_buf)
    order = np.argsort(an._dmax_buf, kind="stable")
    sd = an._dmax_buf[order]
    cum = np.concatenate(([0.0], np.cumsum(an.face_areas[order])))
    radii, perims, n_eval = [], [], 0
    for T in targets:
        re = float(np.sqrt(T / np.pi))  # same cold-start bracket as the first vertex in the real loop
        r, _, hist = an._find_radius_for_area(
            d, T, tol=area_tol, dmin=an._dmin_buf, dmax=an._dmax_buf,
            r_init=re, r_lower=0.6 * re, r_upper=1.4 * re, delta0=0.4 * re, max_iter=50,
            sorted_dmax=sd, cumulative_inside_area=cum,
        )
        n_eval += len(hist)
        radii.append(r)
        perims.append(an._perimeter_at_radius(r, d, dmin=an._dmin_buf, dmax=an._dmax_buf) if np.isfinite(r) else np.nan)
    radii = np.array(radii)
    extra = []
    for i in range(len(radii) - 1):
        lo, hi = radii[i], radii[i + 1]
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo > 0:
            extra.extend(np.exp(np.log(lo) + extra_fracs * (np.log(hi) - np.log(lo))))
    if extra:
        an._area_inside_radius_vectorized(np.sort(np.array(extra)), d, dmin=an._dmin_buf, dmax=an._dmax_buf)
    return radii, np.array(perims), n_eval, (sd, cum)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subjects_dir", nargs="?")
    p.add_argument("subject", nargs="?")
    p.add_argument("--hemi", default="lh")
    p.add_argument("--surf-type", default="pial")
    p.add_argument("--custom-label", default=None)
    p.add_argument("--engine", default="potpourri", help="reference engine for distances")
    p.add_argument("--use-robust", action="store_true", help="potpourri3d intrinsic-Delaunay mode")
    add_common_args(p)
    p.add_argument("--synthetic", type=int, default=None, help="icosphere level for a self-test")
    p.add_argument("--n-sources", type=int, default=20)
    p.add_argument("--area-tol", type=float, default=0.01)
    p.add_argument("--n-extra", type=int, default=3, help="supplementary samples between scales")
    p.add_argument("--solve", action="store_true", help="also test ConsistentBatchHeatEngine")
    p.add_argument("--k", type=int, nargs="+", default=[1, 4, 16, 64])
    args = p.parse_args()
    if args.synthetic is None and (args.subjects_dir is None or args.subject is None):
        p.error("give SUBJECTS_DIR SUBJECT, or --synthetic LEVEL")

    ca, an = load_analysis(args)
    import fastcw_lean as fl

    if not (ca.NUMBA_AVAILABLE and fl.NUMBA_AVAILABLE):
        print("WARNING: Numba missing -> both paths run as pure Python; timings are meaningless.", file=sys.stderr)
    if an.engine_kwargs.get("use_robust"):
        print("NOTE: reference engine is in use_robust=True mode; the consistent batch engine targets "
              "use_robust=False, so expect larger solve differences on this surface.")

    Vs, Fs = an.vertices, an.faces
    nF = Fs.shape[0]
    area, g11, g12, g22 = fl.precompute_face_metric(Vs, Fs)
    width = fl.default_bin_width(Vs, Fs)
    targets = np.array(DEFAULT_SCALES) * float(an.vertex_areas_sub.sum())
    fr = np.arange(1, args.n_extra + 1) / (args.n_extra + 1.0)
    bufs = (np.empty(nF), np.empty(nF, np.int32), np.empty(nF, np.int32))
    ro, po = np.empty(len(targets)), np.empty(len(targets))
    er = np.empty((len(targets) - 1) * len(fr))
    ea = np.empty_like(er)
    sources = pick_sources(an, args.n_sources, args.seed)

    # JIT warm-up for both paths
    d = an._compute_geodesic_distances_from_subvertex(sources[0])
    fl.per_source_geometry(d, Fs, area, g11, g12, g22, width, targets, fr, 1e-10, 60, *bufs, ro, po, er, ea)
    current_path(an, d, targets[:2], fr, args.area_tol)

    t_cur, t_new, ev_cur, ev_new = [], [], [], []
    worst = dict(area=0.0, perim=0.0, samples=0.0, radius_vs_bisection=0.0)
    for src in sources:
        d = an._compute_geodesic_distances_from_subvertex(src)
        t0 = time.perf_counter()
        rc, pc, nc, (sd, cum) = current_path(an, d, targets, fr, args.area_tol)
        t1 = time.perf_counter()
        n_new = fl.per_source_geometry(d, Fs, area, g11, g12, g22, width, targets, fr, 1e-10, 60,
                                       *bufs, ro, po, er, ea)
        t2 = time.perf_counter()
        t_cur.append(t1 - t0)
        t_new.append(t2 - t1)
        ev_cur.append(nc)
        ev_new.append(n_new)
        ok = np.isfinite(ro)
        # Evaluate the current clipping code at the new exact radii: should hit the targets.
        a_at = np.array([an._area_inside_radius(r, d, dmin=an._dmin_buf, dmax=an._dmax_buf,
                                                sorted_dmax=sd, cumulative_inside_area=cum) for r in ro[ok]])
        p_at = np.array([an._perimeter_at_radius(r, d, dmin=an._dmin_buf, dmax=an._dmax_buf) for r in ro[ok]])
        worst["area"] = max(worst["area"], float(np.max(np.abs(a_at - targets[ok]) / targets[ok])))
        worst["perim"] = max(worst["perim"], float(np.max(np.abs(p_at - po[ok]) / np.maximum(p_at, 1e-12))))
        both = ok & np.isfinite(rc)
        worst["radius_vs_bisection"] = max(worst["radius_vs_bisection"],
                                           float(np.max(np.abs(rc[both] - ro[both]) / ro[both])))
        good = np.isfinite(er)
        if good.any():
            idx = np.argsort(er[good])
            ref = an._area_inside_radius_vectorized(er[good][idx], d, dmin=an._dmin_buf, dmax=an._dmax_buf)
            worst["samples"] = max(worst["samples"], float(np.max(np.abs(ref - ea[good][idx]) / ref)))

    print(f"\n=== Geometry: {len(sources)} sources, {Vs.shape[0]} vertices, {nF} faces, bin width {width:.3f} ===")
    print(f"current path : median {1e3 * np.median(t_cur):8.2f} ms/source, {np.mean(ev_cur):.0f} full area evals")
    print(f"lean path    : median {1e3 * np.median(t_new):8.2f} ms/source, {np.mean(ev_new):.0f} band evals")
    print(f"speed-up     : {np.median(t_cur) / np.median(t_new):.1f}x (single process, uncontended)")
    print("agreement (max relative):")
    print(f"  current clipping area at exact radius vs target : {worst['area']:.1e}")
    print(f"  current clipping perimeter vs coarea perimeter  : {worst['perim']:.1e}")
    print(f"  supplementary sample areas                      : {worst['samples']:.1e}")
    print(f"  bisection radius vs exact radius (<= ~0.5% expected at 1% area tol): {100 * worst['radius_vs_bisection']:.2f}%")

    if not args.solve:
        return

    print("\n=== Solve ===")
    t0 = time.perf_counter()
    eng = fl.ConsistentBatchHeatEngine(Vs, Fs, factor="auto")
    backend = type(eng._heat).__name__
    print(f"ConsistentBatchHeatEngine built in {time.perf_counter() - t0:.1f}s (factor backend: {backend})")
    if backend != "_Cholmod":
        print("WARNING: not using CHOLMOD; SciPy SuperLU is a poor proxy (compute-bound solves), "
              "so amortisation numbers below will understate CHOLMOD's.")
    kmax = min(max(args.k), len(sources))
    ref_src = sources[:min(8, kmax)]
    D = eng.compute_distance_batch(ref_src)
    t0 = time.perf_counter()
    Dref = np.vstack([an._compute_geodesic_distances_from_subvertex(s) for s in ref_src])
    t_ref = (time.perf_counter() - t0) / len(ref_src)
    h = np.median(an.face_L)
    far = Dref > 5 * h
    print(f"vs reference engine '{an.distance_engine.name}' on {len(ref_src)} sources:")
    print(f"  max |diff| {np.max(np.abs(D - Dref)):.3e} mm, median rel diff (d > 5h) "
          f"{np.median(np.abs(D - Dref)[far] / Dref[far]):.2e}, max rel diff {np.max(np.abs(D - Dref)[far] / Dref[far]):.2e}")
    print(f"reference single-source: {1e3 * t_ref:.2f} ms/source")
    print("   k   heat   fused_div   poisson   total  (ms/source)")
    for k in args.k:
        k = min(k, len(sources))
        tm = {}
        reps = max(1, 32 // k)
        for _ in range(reps):
            eng.compute_distance_batch(sources[:k], timings=tm)
        n = reps * k
        tot = sum(tm.values())
        print(f"{k:4d}  {1e3 * tm['heat_solve'] / n:6.2f}   {1e3 * tm['fused_div'] / n:8.2f}   "
              f"{1e3 * tm['poisson_solve'] / n:7.2f}  {1e3 * tot / n:6.2f}")


if __name__ == "__main__":
    main()
