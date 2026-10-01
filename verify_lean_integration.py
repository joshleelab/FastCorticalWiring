#!/usr/bin/env python3
"""
Acceptance test for FastCW lean mode (lean geometry, robust-only potpourri3d).

Runs the integrated FastCorticalWiringAnalysis.compute_all_wiring_costs on a
vertex subset and checks every output against the legacy clipping code on the
same analysis object, using the same distance fields:

  radii       legacy _area_inside_radius(radius) reproduces each target area
  perimeters  equal legacy _perimeter_at_radius at the same radius
  samples     areas equal legacy _area_inside_radius_vectorized; solved pairs present
  MSD         equals a direct recomputation from the distance field
  health      a corrupted distance field is flagged, written as NaN, and warned about
  engine      potpourri runs robust, t_coef = (L / mean_edge)^2, forbidden settings raise
  outputs     provenance() is complete

Exits with status 1 if any check fails.

Real subject (run from the FastCW directory):
  python verify_lean_integration.py $SUBJECTS_DIR <subject> --hemi lh \\
      --surf-type pial.cortexonly.qd.n80000 --n-sources 20
Synthetic self-test (no data, no potpourri3d needed):
  python verify_lean_integration.py --synthetic 5
"""
import argparse
import contextlib
import io
import sys
import types
import warnings

import numpy as np

FAILURES = []
F32 = 2e-6  # outputs are stored as float32


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)


def build(args):
    import core_analysis as ca

    if args.synthetic is not None:
        import lean_testkit as tk

        tk.install_stubs()
        V, F = tk.icosphere(args.synthetic)
        V = tk.bumpy(V)
        keep = np.arccos(np.clip(V[:, 2] / np.linalg.norm(V, axis=1), -1, 1)) > 0.5  # cut a cap: boundary
        with contextlib.redirect_stdout(io.StringIO()):
            return ca, ca.FastCorticalWiringAnalysis(V, F, keep, engine_type="euclid_stub")
    return ca, ca.FastCorticalWiringAnalysis.from_freesurfer(
        args.subjects_dir, args.subject, hemi=args.hemi, surf_type=args.surf_type,
        engine_type="potpourri", no_mask=args.no_mask,
        engine_kwargs={"diffusion_length_mm": args.diffusion_length_mm},
    )


def check_engine_rules(real):
    import distance_engines as de
    import lean_geometry as lg
    import lean_testkit as tk

    V, F = tk.icosphere(3)
    V = V * 70.0
    captured = {}
    fake = types.ModuleType("potpourri3d")

    class FakeSolver:
        def __init__(self, V_, F_, **kw):
            captured.update(kw)

        def compute_distance(self, s):
            return np.zeros(len(V))

    fake.MeshHeatMethodDistanceSolver = FakeSolver
    saved = sys.modules.get("potpourri3d")
    sys.modules["potpourri3d"] = fake
    try:
        e = de.PotpourriDistanceEngine(V, F, allow_eigen_fallback=True)
        expect_t = (0.7 / lg.mean_edge_length(V, F)) ** 2
        check("engine: default is robust with diffusion 0.7 mm",
              captured.get("use_robust") is True and abs(captured.get("t_coef", -1) - expect_t) < 1e-12
              and e.diffusion_length_mm == 0.7,
              f"use_robust={captured.get('use_robust')}, t_coef={captured.get('t_coef')}, expected {expect_t:.6g}")
        for bad_kw, label in (({"use_robust": False}, "use_robust=False"), ({"t_coef": 1.0}, "t_coef"),
                              ({"diffusion_length_mm": 0.4}, "diffusion 0.4 mm")):
            try:
                de.PotpourriDistanceEngine(V, F, allow_eigen_fallback=True, **bad_kw)
                check(f"engine: {label} is rejected", False)
            except ValueError:
                check(f"engine: {label} is rejected", True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            de.PotpourriDistanceEngine(V, F, allow_eigen_fallback=True, diffusion_length_mm=0.6)
        check("engine: diffusion 0.6 mm warns", any(issubclass(x.category, RuntimeWarning) for x in w))
        check("engine: batch_heat removed from registry", "batch_heat" not in de.ENGINE_REGISTRY)
    finally:
        if saved is None:
            sys.modules.pop("potpourri3d", None)
        else:
            sys.modules["potpourri3d"] = saved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subjects_dir", nargs="?")
    p.add_argument("subject", nargs="?")
    p.add_argument("--hemi", default="lh")
    p.add_argument("--surf-type", default="pial")
    p.add_argument("--no-mask", action="store_true")
    p.add_argument("--diffusion-length-mm", type=float, default=0.7)
    p.add_argument("--n-sources", type=int, default=20)
    p.add_argument("--synthetic", type=int, default=None)
    args = p.parse_args()
    if args.synthetic is None and (args.subjects_dir is None or args.subject is None):
        p.error("give SUBJECTS_DIR SUBJECT, or --synthetic LEVEL")
    if args.synthetic is not None:
        import lean_testkit as tk

        tk.install_stubs()

    ca, an = build(args)
    check("core_analysis no longer silences all warnings",
          not any(f[0] == "ignore" and f[1] is None and f[2] is Warning and f[3] is None for f in warnings.filters))

    order = np.asarray(an._bfs_order)
    pick = order[np.linspace(0, order.size - 1, args.n_sources).astype(int)]
    subset = an.sub_to_orig[pick]
    scales = [0.002, 0.00267988, 0.00359088, 0.00481157, 0.00644721, 0.00863888,
              0.01157558, 0.01551059, 0.02078326, 0.02784833, 0.03731509, 0.05]
    with contextlib.redirect_stdout(io.StringIO()):
        an.compute_all_wiring_costs(scale=scales, vertex_subset=subset, n_samples_between_scales=3)
    T = np.array(scales) * float(an.vertex_areas_sub.sum())

    worst = dict(area=0.0, perim=0.0, samp=0.0, msd=0.0)
    missing_pairs = 0
    n_nan = 0
    for sub_idx, orig in zip(pick, subset):
        d = an._compute_geodesic_distances_from_subvertex(int(sub_idx))
        if an.field_flag[orig]:
            continue
        for s, t in zip(scales, T):
            r = float(an.radius_function[float(s)][orig])
            if not np.isfinite(r):
                n_nan += 1
                continue
            worst["area"] = max(worst["area"], float(rel(an._area_inside_radius(r, d), t)))
            worst["perim"] = max(worst["perim"], float(rel(an.perimeter_function[float(s)][orig],
                                                           an._perimeter_at_radius(r, d))))
        rs, as_ = an.get_vertex_samples(orig)
        if rs.size:
            legacy = an._area_inside_radius_vectorized(np.sort(rs.astype(np.float64)), d)
            worst["samp"] = max(worst["samp"], float(np.max(rel(as_[np.argsort(rs)], legacy))))
        for t in T:
            if not np.any(rel(as_, t) < F32):
                missing_pairs += 1
        v = (d > an.eps) & np.isfinite(d)
        worst["msd"] = max(worst["msd"], float(rel(an.msd_unweighted[orig], np.mean(d[v]))),
                           float(rel(an.msd_weighted[orig],
                                     np.sum(d[v] * an.vertex_areas_sub[v]) / np.sum(an.vertex_areas_sub[v]))))

    tol = 5 * F32
    check("radii reproduce target areas (legacy clipping)", worst["area"] < tol, f"max rel {worst['area']:.1e}")
    check("perimeters equal legacy perimeter kernel", worst["perim"] < tol, f"max rel {worst['perim']:.1e}")
    check("sample areas equal legacy vectorized kernel", worst["samp"] < tol, f"max rel {worst['samp']:.1e}")
    check("every solved (radius, target) pair is stored", missing_pairs == 0, f"{missing_pairs} missing")
    check("MSD (unweighted and weighted) recomputes exactly", worst["msd"] < tol, f"max rel {worst['msd']:.1e}")
    check("no NaN radii on healthy fields", n_nan == 0, f"{n_nan} NaN")
    check("no fields flagged on this run", int(an.field_flag[subset].sum()) == 0,
          f"{int(an.field_flag[subset].sum())} flagged")

    prov = an.provenance()
    need = {"geometry_method", "engine", "use_robust", "diffusion_length_mm", "t_coef",
            "mean_edge_length_mm", "n_flagged_fields"}
    check("provenance() has all fields", need <= set(prov), f"missing {sorted(need - set(prov))}")
    if args.synthetic is None:
        check("potpourri engine is robust at the requested diffusion length",
              prov["use_robust"] is True and abs(prov["diffusion_length_mm"] - args.diffusion_length_mm) < 1e-12,
              str(prov))

    # Health check: corrupt one source's field and confirm it is flagged and NaN'd
    victim_sub, victim = int(pick[0]), int(subset[0])
    real_solver = an._compute_geodesic_distances_from_subvertex

    def corrupted(sub_idx):
        d = real_solver(sub_idx)
        if int(sub_idx) == victim_sub:
            d = np.minimum(d, 0.2 * np.nanmax(d))  # truncated field, like the non-robust failures
        return d

    an._compute_geodesic_distances_from_subvertex = corrupted
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        an.compute_all_wiring_costs(scale=scales, vertex_subset=subset[:3], n_samples_between_scales=3)
    an._compute_geodesic_distances_from_subvertex = real_solver
    all_nan = all(np.isnan(an.radius_function[float(s)][victim]) for s in scales) and np.isnan(an.msd_weighted[victim])
    check("corrupted field is flagged, NaN'd and warned about",
          an.field_flag[victim] == 1 and all_nan and an.n_flagged_fields == 1
          and any(issubclass(x.category, RuntimeWarning) for x in w))
    check("healthy neighbours of a flagged source are unaffected",
          all(an.field_flag[o] == 0 and np.isfinite(an.msd_weighted[o]) for o in subset[1:3]))

    check_engine_rules(args.synthetic is None)

    print("\nALL CHECKS PASSED" if not FAILURES else f"\n{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
