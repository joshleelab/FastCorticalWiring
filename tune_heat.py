#!/usr/bin/env python3
"""
Choose a heat-method configuration by its error against exact geodesics
(pygeodesic) in the quantities FastCW reports: distances, MSD (unweighted and
area-weighted), and the 12 radii and perimeters.

Configurations: potpourri3d use_robust=False at t_coef=1 (current default, as a
baseline) and use_robust=True at each --t-coefs value. Exact distances are
computed once per source and cached (--exact-cache) so reruns are cheap.

  python tune_heat.py $S $SUB --hemi lh --surf-type pial.cortexonly.qd.n80000 --no-mask \
      --n-sources 8 --t-coefs 1 0.5 0.25 0.1 --exact-cache exact_lh_n80000.npz

Signed errors are (heat - exact) / exact: a consistent sign means a bias,
which matters less for between-subject comparisons than scatter does.
"""
import argparse
import os
import time

import numpy as np

import compare_fastcw_lean as cf
import fastcw_lean as fl
from screen_heat import field_health


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subjects_dir", nargs="?")
    p.add_argument("subject", nargs="?")
    p.add_argument("--hemi", default="lh")
    p.add_argument("--surf-type", default="pial")
    p.add_argument("--custom-label", default=None)
    p.add_argument("--engine", default="potpourri")
    p.add_argument("--synthetic", type=int, default=None)
    p.add_argument("--n-sources", type=int, default=8)
    p.add_argument("--t-coefs", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    p.add_argument("--diffusion-mm", type=float, nargs="+", default=None,
                   help="specify robust configs by diffusion length sqrt(t) in mm instead; "
                        "converted per mesh as t_coef = (L / mean_edge_length)^2")
    p.add_argument("--exact-cache", default=None, help="npz file to store/reuse exact distances")
    p.add_argument("--scales", type=float, nargs="+", default=None,
                   help="area fractions to evaluate (default: the 12 FastCW default scales)")
    p.add_argument("--per-scale", action="store_true",
                   help="also print signed radius/perimeter error per scale and the error in the "
                        "fitted scaling exponent d ln r / d ln A")
    cf.add_common_args(p)
    args = p.parse_args()
    args.use_robust = False
    ca, an = cf.load_analysis(args)
    V, F = an.vertices, an.faces
    h = float(np.median(an.face_L))
    sources = cf.pick_sources(an, args.n_sources, args.seed)

    # ---- exact distances (cached)
    D_exact = None
    if args.exact_cache and os.path.exists(args.exact_cache):
        z = np.load(args.exact_cache)
        if z["D"].shape == (len(sources), V.shape[0]) and np.array_equal(z["sources"], sources):
            D_exact = z["D"]
            print(f"exact distances loaded from {args.exact_cache}")
    if D_exact is None:
        if args.synthetic is not None:
            exact = an.distance_engine
        else:
            from distance_engines import create_distance_engine

            exact = create_distance_engine("pygeodesic", V, F)
        t0 = time.perf_counter()
        D_exact = np.vstack([np.asarray(exact.compute_distance(int(s)), float) for s in sources])
        print(f"exact distances: {len(sources)} sources in {time.perf_counter() - t0:.0f}s")
        if args.exact_cache:
            np.savez_compressed(args.exact_cache, D=D_exact, sources=sources)

    # ---- reported quantities
    nF = F.shape[0]
    area, g11, g12, g22 = fl.precompute_face_metric(V, F)
    width = fl.default_bin_width(V, F)
    scales = np.array(args.scales if args.scales else cf.DEFAULT_SCALES, dtype=float)
    scales.sort()
    T = scales * float(an.vertex_areas_sub.sum())
    bufs = (np.empty(nF), np.empty(nF, np.int32), np.empty(nF, np.int32))
    empty = np.empty(0)
    w_vert = an.vertex_areas_sub

    def local(d):
        ro, po = np.empty(len(T)), np.empty(len(T))
        fl.per_source_geometry(d, F, area, g11, g12, g22, width, T, empty, 1e-10, 60, *bufs, ro, po, empty, empty)
        return ro, po

    def msd(d):
        v = (d > an.eps) & np.isfinite(d)
        return float(np.mean(d[v])), float(np.sum(d[v] * w_vert[v]) / np.sum(w_vert[v]))

    ref = [(local(d), msd(d)) for d in D_exact]

    def slope(r):
        ok = np.isfinite(r) & (r > 0)
        return np.polyfit(np.log(T[ok]), np.log(r[ok]), 1)[0] if ok.sum() >= 2 else np.nan

    slope_ref = np.array([slope(rp[0][0]) for rp in ref])
    per_scale_rows = []

    def make(t_coef, robust):
        if args.synthetic is not None:
            e = fl.ConsistentBatchHeatEngine(V, F, t_coef=t_coef, factor="auto")
            return e.compute_distance
        import potpourri3d as pp3d

        s = pp3d.MeshHeatMethodDistanceSolver(V, F, t_coef=t_coef, use_robust=robust)
        return lambda src: np.asarray(s.compute_distance(int(src)), float)

    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64)
    E.sort(axis=1)
    E = np.unique(E, axis=0)
    mean_edge = float(np.mean(np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1)))
    if args.diffusion_mm:
        t_list = [(float(L) / mean_edge) ** 2 for L in args.diffusion_mm]
    else:
        t_list = [float(t) for t in args.t_coefs]
    configs = [("non-robust", 1.0, False)] + [("robust", t, True) for t in t_list]
    print(f"\n{V.shape[0]} vertices, {len(sources)} sources, median edge {h:.3f} mm, mean edge {mean_edge:.3f} mm. "
          "Errors are relative to exact; |x| = median absolute, sgn = median signed, max = worst.")
    hdr = (f"{'config':>27} {'ms':>6} {'flag':>4} | {'dist |x|':>8} {'sgn':>8} {'p95':>7} | "
           f"{'MSDw |x|':>8} {'sgn':>8} {'max':>7} | {'MSDu |x|':>8} | {'rad |x|':>8} {'sgn':>8} {'max':>7} | "
           f"{'per |x|':>8} {'sgn':>8} {'max':>7}")
    print(hdr)
    for name, t, rob in configs:
        solve = make(t, rob)
        solve(int(sources[0]))  # warm-up
        dist_err, msdw, msdu, times, flags = [], [], [], [], 0
        R = np.full((len(sources), len(T)), np.nan)
        P = np.full((len(sources), len(T)), np.nan)
        dslope = np.full(len(sources), np.nan)
        for k, src in enumerate(sources):
            t0 = time.perf_counter()
            d = solve(int(src))
            times.append(time.perf_counter() - t0)
            de = D_exact[k]
            flags += int(field_health(d, V, int(src), h)[2])
            m = np.isfinite(de) & np.isfinite(d) & (de > 5 * h)
            dist_err.append((d[m] - de[m]) / de[m])
            (rx, px), (mux, mwx) = ref[k]
            (rh, ph), (muh, mwh) = local(d), msd(d)
            msdw.append((mwh - mwx) / mwx)
            msdu.append((muh - mux) / mux)
            ok = np.isfinite(rx) & np.isfinite(rh)
            R[k, ok] = (rh[ok] - rx[ok]) / rx[ok]
            okp = ok & (px > 0)
            P[k, okp] = (ph[okp] - px[okp]) / px[okp]
            dslope[k] = slope(rh) - slope_ref[k]
        de_all = np.concatenate(dist_err)
        rad_all, per_all = R[np.isfinite(R)], P[np.isfinite(P)]
        msdw, msdu = np.array(msdw), np.array(msdu)
        label = f"{name} t={t:.3g} ({np.sqrt(t) * mean_edge:.2f}mm)"
        per_scale_rows.append((label, R, P, dslope))
        print(f"{label:>27} {1e3 * np.median(times):6.1f} {flags:4d} | "
              f"{np.median(np.abs(de_all)):8.2e} {np.median(de_all):+8.1e} {np.percentile(np.abs(de_all), 95):7.1e} | "
              f"{np.median(np.abs(msdw)):8.2e} {np.median(msdw):+8.1e} {np.max(np.abs(msdw)):7.1e} | "
              f"{np.median(np.abs(msdu)):8.2e} | "
              f"{np.median(np.abs(rad_all)):8.2e} {np.median(rad_all):+8.1e} {np.max(np.abs(rad_all)):7.1e} | "
              f"{np.median(np.abs(per_all)):8.2e} {np.median(per_all):+8.1e} {np.max(np.abs(per_all)):7.1e}", flush=True)

    if args.per_scale:
        cols = " ".join(f"{100 * s:>6.3g}" for s in scales)
        for title, idx in (("radius", 1), ("perimeter", 2)):
            print(f"\nmedian signed {title} error (%) by scale (area % of cortex):")
            print(f"{'config':>27}  {cols}")
            for row in per_scale_rows:
                med = np.nanmedian(row[idx], axis=0)
                print(f"{row[0]:>27}  " + " ".join(f"{100 * v:+6.2f}" for v in med))
        print(f"\nscaling exponent d ln r / d ln A fitted per source over these scales "
              f"(exact: median {np.nanmedian(slope_ref):.4f}, range {np.nanmin(slope_ref):.4f}-{np.nanmax(slope_ref):.4f})")
        print(f"{'config':>27}  {'median err':>10} {'max |err|':>10}")
        for row in per_scale_rows:
            print(f"{row[0]:>27}  {np.nanmedian(row[3]):+10.4f} {np.nanmax(np.abs(row[3])):10.4f}")


if __name__ == "__main__":
    main()
