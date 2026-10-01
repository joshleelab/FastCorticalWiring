#!/usr/bin/env python3
"""
Localise a radius/area mismatch between the current FastCW path and fastcw_lean.

For every (source, scale) it evaluates the enclosed area at both the bisection
radius (r_cur) and the lean exact radius (r_lean) with four evaluators:
  lean   : fastcw_lean.area_at            (prefix sums + band)
  cum    : _area_inside_radius with sorted_dmax/cumulative (production path)
  plain  : _area_inside_radius without sorted_dmax/cumulative (mask sum)
  vec    : _area_inside_radius_vectorized  (full-pass Numba kernel)
The evaluator that disagrees with the other three is the bug. The worst case
is saved to mismatch_case.npz (distance field + indices) for offline analysis.

  python diagnose_mismatch.py $SUBJECTS_DIR <subject> --hemi lh --n-sources 50
  python diagnose_mismatch.py ... --exact-check 3   # also compare pp3d and lean solver to pygeodesic
"""
import argparse
import sys

import numpy as np

import compare_fastcw_lean as cf
import fastcw_lean as fl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("subjects_dir", nargs="?")
    p.add_argument("subject", nargs="?")
    p.add_argument("--hemi", default="lh")
    p.add_argument("--surf-type", default="pial")
    p.add_argument("--custom-label", default=None)
    p.add_argument("--engine", default="potpourri")
    p.add_argument("--use-robust", action="store_true", help="potpourri3d intrinsic-Delaunay mode")
    cf.add_common_args(p)
    p.add_argument("--synthetic", type=int, default=None)
    p.add_argument("--n-sources", type=int, default=50)
    p.add_argument("--exact-check", type=int, default=0, help="number of sources to compare against pygeodesic")
    args = p.parse_args()
    ca, an = cf.load_analysis(args)

    Vs, Fs = an.vertices, an.faces
    nF = Fs.shape[0]
    area, g11, g12, g22 = fl.precompute_face_metric(Vs, Fs)
    width = fl.default_bin_width(Vs, Fs)
    total_v = float(an.vertex_areas_sub.sum())
    T = np.array(cf.DEFAULT_SCALES) * total_v
    fr = np.array([0.25, 0.5, 0.75])
    dmax_b, order_b, long_b = np.empty(nF), np.empty(nF, np.int32), np.empty(nF, np.int32)
    print(f"Numba: core_analysis={ca.NUMBA_AVAILABLE} lean={fl.NUMBA_AVAILABLE} | faces {nF} | width {width:.3f}")
    print(f"area totals: vertex_areas_sub {total_v:.3f}  their face_areas {an.face_areas.sum():.3f}  "
          f"lean face areas {area.sum():.3f}  (zero-area faces in theirs: {int(np.sum(an.face_areas == 0))})")

    sources = cf.pick_sources(an, args.n_sources, args.seed)
    rows, worst = [], None
    for src in sources:
        d = an._compute_geodesic_distances_from_subvertex(src)
        rc, _, _, (sd, cum) = cf.current_path(an, d, T, fr, 0.01)
        nb, ptr, pre, nl = fl.build_distance_buckets(d, Fs, area, width, dmax_b, order_b, long_b)
        stats = dict(src=int(src), dmin=float(np.nanmin(d)), dmax=float(np.nanmax(d)), d_src=float(d[src]),
                     n_nan=int(np.sum(~np.isfinite(d))), n_neg=int(np.sum(d < 0)), nbins=int(nb),
                     n_long=int(nl), prefix_total=float(pre[nb]))
        for s, t in enumerate(T):
            rl, nev = fl.solve_radius(t, d, Fs, area, width, nb, ptr, pre, order_b, dmax_b, long_b, nl, 1e-10, 60)

            def ev(r):
                if not np.isfinite(r):
                    return (np.nan,) * 4
                a_lean, _ = fl.area_at(r, d, Fs, area, width, nb, ptr, pre, order_b, dmax_b, long_b, nl)
                a_cum = an._area_inside_radius(r, d, dmin=an._dmin_buf, dmax=an._dmax_buf,
                                               sorted_dmax=sd, cumulative_inside_area=cum)
                a_plain = an._area_inside_radius(r, d, dmin=an._dmin_buf, dmax=an._dmax_buf)
                a_vec = float(an._area_inside_radius_vectorized(np.array([r]), d)[0])
                return a_lean, a_cum, a_plain, a_vec

            at_l, at_c = ev(rl), ev(rc[s])
            vals = np.array(at_l + at_c)
            bad_eval = np.nanmax(np.abs(vals[:4] - np.nanmedian(vals[:4]))) / t if np.isfinite(rl) else 0.0
            rdiff = abs(rc[s] - rl) / rl if np.isfinite(rl) and np.isfinite(rc[s]) else np.nan
            lean_miss = abs(at_l[0] - t) / t if np.isfinite(at_l[0]) else 0.0
            row = (stats, s, t, rl, nev, rc[s], at_l, at_c, bad_eval, rdiff)
            rows.append(row)
            is_bad = bad_eval > 1e-6 or lean_miss > 1e-6 or (np.isfinite(rdiff) and rdiff > 0.01)
            if is_bad:
                score = max(bad_eval, lean_miss, rdiff if np.isfinite(rdiff) else 0.0)
                if worst is None or score > worst[0]:
                    worst = (score, src, s, d.copy(), rl, rc[s])

    flagged = [r for r in rows if r[8] > 1e-6 or (np.isfinite(r[9]) and r[9] > 0.01)
               or (np.isfinite(r[6][0]) and abs(r[6][0] - r[2]) / r[2] > 1e-6)]
    print(f"\n{len(flagged)} of {len(rows)} (source, scale) pairs flagged")
    names = ("lean", "cum", "plain", "vec")
    for stats, s, t, rl, nev, rcur, at_l, at_c, bad, rdiff in flagged[:8]:
        print(f"\nsource {stats['src']} scale #{s} target {t:.2f}  "
              f"d[min,max]=[{stats['dmin']:.4g},{stats['dmax']:.1f}] d[src]={stats['d_src']:.3g} "
              f"NaN={stats['n_nan']} neg={stats['n_neg']} bins={stats['nbins']} long={stats['n_long']} "
              f"prefix_total={stats['prefix_total']:.2f}")
        print(f"  r_lean {rl:.4f} ({nev} evals)   r_cur {rcur:.4f}   radius diff {100 * rdiff:.2f}%")
        print("  area / target at r_lean : " + "  ".join(f"{n}={a / t:.6f}" for n, a in zip(names, at_l)))
        print("  area / target at r_cur  : " + "  ".join(f"{n}={a / t:.6f}" for n, a in zip(names, at_c)))
    if worst is not None:
        np.savez_compressed("mismatch_case.npz", d=worst[3], src=worst[1], scale_idx=worst[2],
                            r_lean=worst[4], r_cur=worst[5], faces=Fs, width=width)
        print("\nSaved worst case to mismatch_case.npz")
    elif not flagged:
        print("No mismatches: all evaluators agree and both solvers hit their targets.")

    if args.exact_check:
        from distance_engines import create_distance_engine

        # synthetic self-test: the reference engine is already the analytic geodesic
        exact = an.distance_engine if args.synthetic is not None else create_distance_engine("pygeodesic", Vs, Fs)
        lean = fl.ConsistentBatchHeatEngine(Vs, Fs, factor="auto")
        h = np.median(an.face_L)
        print(f"\nAccuracy vs exact (pygeodesic), {args.exact_check} sources, d > 5h:")
        for src in sources[:args.exact_check]:
            de = np.asarray(exact.compute_distance(int(src)), float)
            dp = an._compute_geodesic_distances_from_subvertex(src)
            dl = lean.compute_distance(int(src))
            fin = np.isfinite(de) & np.isfinite(dp) & np.isfinite(dl)
            m = fin & (de > 5 * h)
            print(f"  source {src}: non-finite exact {int(np.sum(~np.isfinite(de)))}, "
                  f"heat {int(np.sum(~np.isfinite(dp)))}/{int(np.sum(~np.isfinite(dl)))} (of {de.size}) | "
                  f"median rel err  {args.engine}={np.median(np.abs(dp - de)[m] / de[m]):.2e}  "
                  f"lean={np.median(np.abs(dl - de)[m] / de[m]):.2e}   "
                  f"max abs  {args.engine}={np.max(np.abs(dp - de)[fin]):.2f}  lean={np.max(np.abs(dl - de)[fin]):.2f} mm")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
