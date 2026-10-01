#!/usr/bin/env python3
"""
Does the reduced-mesh pipeline reproduce native-mesh results?

Picks sources on the reduced surface, matches each to the nearest native
cortical vertex, and computes radii, perimeters (the 12 scales, as fractions
of each mesh's own cortical area, exactly as the pipeline defines them),
area-weighted MSD and the fitted scaling exponent from:
  native exact        (reference: pygeodesic on the native mesh)
  reduced exact       (pygeodesic on the reduced mesh: the effect of decimation alone)
  reduced heat L      (the reduced pipeline end to end: decimation + robust heat)
  native heat L       (the full-resolution pipeline, for comparison)
All rows are signed errors relative to native exact. Unweighted MSD is not
reported because it depends on vertex density by definition.

  python crossmesh_check.py $S $SUB --hemi lh --reduced-surf pial.cortexonly.qd.n80000 \
      --n-sources 8 --diffusion-mm 0.6 0.7 --cache crossmesh_lh.npz
"""
import argparse
import contextlib
import io
import os
import time
import types

import numpy as np

import compare_fastcw_lean as cf
import fastcw_lean as fl


class Geo:
    """Per-mesh geometry state for fastcw_lean evaluation."""

    def __init__(self, an, scales):
        self.an = an
        V, F = an.vertices, an.faces
        self.area, self.g11, self.g12, self.g22 = fl.precompute_face_metric(V, F)
        self.width = fl.default_bin_width(V, F)
        nF = F.shape[0]
        self.bufs = (np.empty(nF), np.empty(nF, np.int32), np.empty(nF, np.int32))
        self.total = float(an.vertex_areas_sub.sum())
        self.T = scales * self.total
        E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64)
        E.sort(axis=1)
        E = np.unique(E, axis=0)
        self.mean_edge = float(np.mean(np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1)))

    def metrics(self, d):
        an = self.an
        ro, po, e = np.empty(len(self.T)), np.empty(len(self.T)), np.empty(0)
        fl.per_source_geometry(d, an.faces, self.area, self.g11, self.g12, self.g22, self.width, self.T,
                               e, 1e-10, 60, *self.bufs, ro, po, e, e)
        v = (d > an.eps) & np.isfinite(d)
        w = an.vertex_areas_sub
        msdw = float(np.sum(d[v] * w[v]) / np.sum(w[v]))
        ok = np.isfinite(ro) & (ro > 0)
        slope = np.polyfit(np.log(self.T[ok]), np.log(ro[ok]), 1)[0] if ok.sum() >= 2 else np.nan
        return ro, po, msdw, slope


def load_pair(args):
    if args.synthetic is not None:
        import lean_testkit as tk

        tk.install_stubs()
        import core_analysis as ca

        out = []
        for level in (args.synthetic + 1, args.synthetic):  # native = finer, reduced = coarser
            V, F = tk.icosphere(level)
            with contextlib.redirect_stdout(io.StringIO()):
                out.append(ca.FastCorticalWiringAnalysis(V * 70.0, F, np.ones(len(V), bool),
                                                         engine_type="great_circle_stub"))
        return out
    base = dict(subjects_dir=args.subjects_dir, subject=args.subject, hemi=args.hemi, custom_label=args.custom_label,
                engine="potpourri", synthetic=None, use_robust=False)
    an_n = cf.load_analysis(types.SimpleNamespace(**base, surf_type=args.native_surf, no_mask=False))[1]
    an_r = cf.load_analysis(types.SimpleNamespace(**base, surf_type=args.reduced_surf, no_mask=True))[1]
    return an_n, an_r


def exact_solver(an, synthetic):
    if synthetic:
        return an.distance_engine.compute_distance
    from distance_engines import create_distance_engine

    e = create_distance_engine("pygeodesic", an.vertices, an.faces)
    return lambda s: np.asarray(e.compute_distance(int(s)), float)


def heat_solver(an, geo, L, synthetic):
    t_coef = (float(L) / geo.mean_edge) ** 2
    if synthetic:
        e = fl.ConsistentBatchHeatEngine(an.vertices, an.faces, t_coef=t_coef, factor="auto")
        return e.compute_distance
    import potpourri3d as pp3d

    s = pp3d.MeshHeatMethodDistanceSolver(an.vertices, an.faces, t_coef=t_coef, use_robust=True)
    return lambda src: np.asarray(s.compute_distance(int(src)), float)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subjects_dir", nargs="?")
    p.add_argument("subject", nargs="?")
    p.add_argument("--hemi", default="lh")
    p.add_argument("--native-surf", default="pial")
    p.add_argument("--reduced-surf", default="pial.cortexonly.qd.n80000")
    p.add_argument("--custom-label", default=None)
    p.add_argument("--synthetic", type=int, default=None, help="self-test: icosphere levels N+1 (native) and N")
    p.add_argument("--n-sources", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--diffusion-mm", type=float, nargs="+", default=[0.6, 0.7])
    p.add_argument("--cache", default=None, help="npz file to store/reuse exact distances")
    args = p.parse_args()
    synthetic = args.synthetic is not None
    if not synthetic and (args.subjects_dir is None or args.subject is None):
        p.error("give SUBJECTS_DIR SUBJECT, or --synthetic LEVEL")

    an_n, an_r = load_pair(args)
    scales = np.array(cf.DEFAULT_SCALES, dtype=float)
    gn, gr = Geo(an_n, scales), Geo(an_r, scales)

    from scipy.spatial import cKDTree

    src_r = cf.pick_sources(an_r, args.n_sources, args.seed)
    map_dist, src_n = cKDTree(an_n.vertices).query(an_r.vertices[src_r])
    src_n = np.asarray(src_n, dtype=np.int64)

    Dn = Dr = None
    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        if np.array_equal(z["src_r"], src_r) and z["Dn"].shape[1] == an_n.vertices.shape[0] \
                and z["Dr"].shape[1] == an_r.vertices.shape[0]:
            Dn, Dr = z["Dn"], z["Dr"]
            print(f"exact distances loaded from {args.cache}")
    if Dn is None:
        t0 = time.perf_counter()
        en, er = exact_solver(an_n, synthetic), exact_solver(an_r, synthetic)
        Dn = np.vstack([en(int(s)) for s in src_n])
        Dr = np.vstack([er(int(s)) for s in src_r])
        print(f"exact distances on both meshes: {len(src_r)} sources in {time.perf_counter() - t0:.0f}s")
        if args.cache:
            np.savez_compressed(args.cache, Dn=Dn, Dr=Dr, src_r=src_r, src_n=src_n)

    print(f"\nnative : {an_n.vertices.shape[0]} vertices, mean edge {gn.mean_edge:.3f} mm, cortical area {gn.total:.0f} mm^2")
    print(f"reduced: {an_r.vertices.shape[0]} vertices, mean edge {gr.mean_edge:.3f} mm, cortical area {gr.total:.0f} mm^2 "
          f"({100 * (gr.total / gn.total - 1):+.2f}% vs native)")
    print(f"source matching distance: median {np.median(map_dist):.3f} mm, max {np.max(map_dist):.3f} mm")

    ref = [gn.metrics(d) for d in Dn]
    rows = [("reduced exact", [gr.metrics(d) for d in Dr])]
    for L in args.diffusion_mm:
        hr = heat_solver(an_r, gr, L, synthetic)
        rows.append((f"reduced heat {L:g}mm", [gr.metrics(hr(int(s))) for s in src_r]))
    for L in args.diffusion_mm:
        hn = heat_solver(an_n, gn, L, synthetic)
        rows.append((f"native heat {L:g}mm", [gn.metrics(hn(int(s))) for s in src_n]))

    def rel(a, b):
        with np.errstate(invalid="ignore", divide="ignore"):
            return (a - b) / b

    cols = " ".join(f"{100 * s:>6.3g}" for s in scales)
    for title, idx in (("radius", 0), ("perimeter", 1)):
        print(f"\nmedian signed {title} error (%) vs native exact, by scale (area % of cortex):")
        print(f"{'':>20}  {cols}")
        for name, mets in rows:
            err = np.array([rel(m[idx], r[idx]) for m, r in zip(mets, ref)])
            print(f"{name:>20}  " + " ".join(f"{100 * v:+6.2f}" for v in np.nanmedian(err, axis=0)))
    ex = np.array([r[3] for r in ref])
    print(f"\nvs native exact (exponent: median {np.nanmedian(ex):.4f}, range {np.nanmin(ex):.4f}-{np.nanmax(ex):.4f})")
    print(f"{'':>20}  {'MSDw sgn':>9} {'max':>7}   {'exponent err':>12} {'max |err|':>10}")
    for name, mets in rows:
        m = np.array([rel(x[2], r[2]) for x, r in zip(mets, ref)])
        s = np.array([x[3] - r[3] for x, r in zip(mets, ref)])
        print(f"{name:>20}  {100 * np.median(m):+8.2f}% {100 * np.max(np.abs(m)):6.2f}%   "
              f"{np.nanmedian(s):+12.4f} {np.nanmax(np.abs(s)):10.4f}")


if __name__ == "__main__":
    main()
