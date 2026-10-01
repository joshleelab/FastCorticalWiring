#!/usr/bin/env python3
"""Self-contained correctness tests for fastcw_lean against core_analysis (no subject data needed).

Checks, on a folded synthetic surface (~41k vertices):
  1. exact area/perimeter vs the existing clipping code at random radii, smooth fields
  2. same under a non-Lipschitz field that forces thousands of faces onto the long-face list
  3. full per-source pass (12 scales + supplementary samples) vs the existing code
  4. NaN handling
Run from the FastCW directory:  python test_fastcw_lean.py
"""
import io, contextlib
import numpy as np
import lean_testkit as tk

tk.install_stubs()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import core_analysis as ca
    V, F = tk.icosphere(6)
    an = ca.FastCorticalWiringAnalysis(tk.bumpy(V), F, np.ones(len(V), bool), engine_type="euclid_stub")
import fastcw_lean as fl

Vs, Fs = an.vertices, an.faces
nF = len(Fs)
area, g11, g12, g22 = fl.precompute_face_metric(Vs, Fs)
w = fl.default_bin_width(Vs, Fs)
bufs = (np.empty(nF), np.empty(nF, np.int32), np.empty(nF, np.int32))
rng = np.random.default_rng(1)
TOL = 1e-12
failures = []

def check_field(d, label):
    nb, ptr, pre, nl = fl.build_distance_buckets(d, Fs, area, w, *bufs)
    wa = wp = 0.0
    for r in rng.uniform(0.5, np.nanmax(d) * 0.8, 12):
        at, pt = an._area_inside_radius(r, d), an._perimeter_at_radius(r, d)
        am, _ = fl.area_at(r, d, Fs, area, w, nb, ptr, pre, bufs[1], bufs[0], bufs[2], nl)
        pm = fl.perimeter_at(r, d, Fs, area, g11, g12, g22, w, nb, ptr, bufs[1], bufs[0], bufs[2], nl)
        wa, wp = max(wa, abs(am - at) / at), max(wp, abs(pm - pt) / pt)
    status = "ok" if max(wa, wp) < TOL else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label:40s} long faces {nl:5d}/{nF}  area {wa:.1e}  perimeter {wp:.1e}")

for s in rng.choice(len(Vs), 3, replace=False):
    check_field(an._compute_geodesic_distances_from_subvertex(s), f"smooth field, source {s}")
d = an._compute_geodesic_distances_from_subvertex(7).copy()
bad = rng.choice(len(d), len(d) // 50, replace=False)
d[bad] += rng.uniform(0, 15, bad.size)
check_field(d, "non-Lipschitz field (2% vertices +0-15 mm)")

scales = np.array([0.002, 0.00267988, 0.00359088, 0.00481157, 0.00644721, 0.00863888,
                   0.01157558, 0.01551059, 0.02078326, 0.02784833, 0.03731509, 0.05])
T = scales * float(an.vertex_areas_sub.sum())
fr = np.array([0.25, 0.5, 0.75])
ro, po = np.empty(12), np.empty(12)
er, ea = np.empty(33), np.empty(33)
for label, dd in (("smooth", an._compute_geodesic_distances_from_subvertex(123)), ("non-Lipschitz", d)):
    n = fl.per_source_geometry(dd, Fs, area, g11, g12, g22, w, T, fr, 1e-12, 60, *bufs, ro, po, er, ea)
    a_at = np.array([an._area_inside_radius(r, dd) for r in ro])
    p_at = np.array([an._perimeter_at_radius(r, dd) for r in ro])
    ext = an._area_inside_radius_vectorized(er, dd)
    errs = (np.max(np.abs(a_at - T) / T), np.max(np.abs(p_at - po) / p_at), np.max(np.abs(ext - ea) / ext))
    status = "ok" if max(errs) < 1e-10 else "FAIL"
    if status == "FAIL":
        failures.append(f"per_source_geometry {label}")
    print(f"[{status}] per_source_geometry, {label:14s} {n:3d} band evals  "
          f"area@r {errs[0]:.1e}  perimeter {errs[1]:.1e}  samples {errs[2]:.1e}")

nb, ptr, pre, nl = fl.build_distance_buckets(d, Fs, area, w, *bufs)
a_nan, _ = fl.area_at(np.nan, d, Fs, area, w, nb, ptr, pre, bufs[1], bufs[0], bufs[2], nl)
r_nan, _ = fl.solve_radius(np.nan, d, Fs, area, w, nb, ptr, pre, bufs[1], bufs[0], bufs[2], nl, 1e-12, 60)
r_big, _ = fl.solve_radius(pre[nb] * 2, d, Fs, area, w, nb, ptr, pre, bufs[1], bufs[0], bufs[2], nl, 1e-12, 60)
ok = a_nan == 0.0 and np.isnan(r_nan) and np.isnan(r_big)
if not ok:
    failures.append("nan handling")
print(f"[{'ok' if ok else 'FAIL'}] NaN radius -> 0 area; NaN or infeasible target -> NaN radius")
print("\nALL PASSED" if not failures else f"\nFAILED: {failures}")
