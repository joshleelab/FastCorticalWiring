#!/usr/bin/env python3
"""
Generate analytic validation surfaces and validate fastcw_pp3d.

Surfaces generated (FreeSurfer layout under --base-dir):
  - sphere_R100_fs5/fs6/fs7
  - sphere_R30_fs5/fs6/fs7
  - plane_large_fs5/fs6/fs7
  - cylinder_R50_fs5/fs6/fs7

Validation checks:
  - radius inversion error against analytic target radius
  - perimeter error against analytic perimeter
  - consistency check: dA/dr (finite-difference) vs measured perimeter
"""

import argparse
import os

import nibabel as nib
import numpy as np
from scipy.spatial import Delaunay

from core_analysis import FastCorticalWiringAnalysis


HEMI = "lh"
AREA_FRACTIONS = [0.01, 0.03, 0.05, 0.10]
MIN_VERTICES_VALIDATE = 10_000
MIN_RADIUS_EDGES = 5.0
N_TEST_VERTICES_SURFACE = 30

PLANE_EXTENTS = (-200.0, 200.0, -200.0, 200.0)
CYL_RADIUS = 50.0
CYL_HEIGHT = 300.0
CYL_LOCAL_CAP = 0.3 * np.pi * CYL_RADIUS

RES_LEVELS = {
    "fs5": {
        "sphere_subdiv": 5,
        "plane_spacing": 4.0,
        "cyl_n_theta": 100,
        "cyl_n_z": 100,
    },
    "fs6": {
        "sphere_subdiv": 6,
        "plane_spacing": 2.0,
        "cyl_n_theta": 200,
        "cyl_n_z": 200,
    },
    "fs7": {
        "sphere_subdiv": 7,
        "plane_spacing": 1.0,
        "cyl_n_theta": 400,
        "cyl_n_z": 400,
    },
}


def expected_subjects():
    out = []
    for tier in RES_LEVELS:
        out.extend(
            [
                f"sphere_R100_{tier}",
                f"sphere_R30_{tier}",
                f"plane_large_{tier}",
                f"cylinder_R50_{tier}",
            ]
        )
    return tuple(out)


EXPECTED_SUBJECTS = expected_subjects()


def subject_tier(subject_id):
    return subject_id.rsplit("_", 1)[-1]


def subject_family(subject_id):
    if subject_id.startswith("sphere_R100"):
        return "sphere_R100"
    if subject_id.startswith("sphere_R30"):
        return "sphere_R30"
    if subject_id.startswith("plane_large"):
        return "plane_large"
    if subject_id.startswith("cylinder_R50"):
        return "cylinder_R50"
    return "unknown"


def make_icosphere(radius, subdivisions):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int32,
    )

    verts_list = [v.copy() for v in verts]
    faces_list = [tuple(f) for f in faces]
    for _ in range(int(subdivisions)):
        midpoint_cache = {}
        new_faces = []

        def midpoint(i, j):
            key = (i, j) if i < j else (j, i)
            if key in midpoint_cache:
                return midpoint_cache[key]
            m = 0.5 * (verts_list[i] + verts_list[j])
            m /= np.linalg.norm(m)
            idx = len(verts_list)
            verts_list.append(m)
            midpoint_cache[key] = idx
            return idx

        for i, j, k in faces_list:
            a = midpoint(i, j)
            b = midpoint(j, k)
            c = midpoint(k, i)
            new_faces.extend(((i, a, c), (j, b, a), (k, c, b), (a, b, c)))
        faces_list = new_faces

    out_v = np.asarray(verts_list, dtype=np.float64) * float(radius)
    out_f = np.asarray(faces_list, dtype=np.int32)
    return out_v, out_f


def make_plane_grid(xmin=-200.0, xmax=200.0, ymin=-200.0, ymax=200.0, spacing=5.0):
    xs = np.arange(xmin, xmax + 0.5 * spacing, spacing, dtype=np.float64)
    ys = np.arange(ymin, ymax + 0.5 * spacing, spacing, dtype=np.float64)
    nx = xs.size
    ny = ys.size
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    verts = np.column_stack((xv.ravel(), yv.ravel(), np.zeros(nx * ny, dtype=np.float64)))
    faces = Delaunay(verts[:, :2]).simplices.astype(np.int32)
    return verts, faces


def make_cylinder(radius=50.0, height=300.0, n_theta=128, n_z=100):
    thetas = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False, dtype=np.float64)
    zs = np.linspace(-0.5 * height, 0.5 * height, int(n_z), dtype=np.float64)
    tt, zz = np.meshgrid(thetas, zs, indexing="xy")
    xx = radius * np.cos(tt)
    yy = radius * np.sin(tt)
    verts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    faces = []
    for iz in range(n_z - 1):
        row0 = iz * n_theta
        row1 = (iz + 1) * n_theta
        for it in range(n_theta):
            jt = (it + 1) % n_theta
            a = row0 + it
            b = row0 + jt
            c = row1 + it
            d = row1 + jt
            faces.append((a, b, d))
            faces.append((a, d, c))
    return verts.astype(np.float64), np.asarray(faces, dtype=np.int32)


def sphere_target_points(radius):
    r3 = radius / np.sqrt(3.0)
    return np.array(
        [
            (0, 0, radius),
            (0, 0, -radius),
            (radius, 0, 0),
            (-radius, 0, 0),
            (0, radius, 0),
            (0, -radius, 0),
            (r3, r3, r3),
            (-r3, r3, r3),
            (r3, -r3, r3),
            (r3, r3, -r3),
        ],
        dtype=np.float64,
    )


def nearest_unique_vertices(verts, targets):
    picked = []
    used = set()
    for t in targets:
        d2 = np.sum((verts - t) ** 2, axis=1)
        order = np.argsort(d2)
        found = None
        for idx in order:
            i = int(idx)
            if i not in used:
                found = i
                break
        if found is None:
            raise RuntimeError("Failed to pick unique nearest vertices for targets.")
        used.add(found)
        picked.append(found)
    return np.asarray(picked, dtype=np.int32)


def write_full_cortex_label(path, n_vertices):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Validation full-cortex label\n")
        f.write(f"{int(n_vertices)}\n")
        for i in range(int(n_vertices)):
            f.write(f"{i} 0.0 0.0 0.0 0.0\n")


def write_fs_subject(base_dir, subject_id, verts, faces, test_indices):
    subj = os.path.join(base_dir, subject_id)
    surf_dir = os.path.join(subj, "surf")
    label_dir = os.path.join(subj, "label")
    os.makedirs(surf_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    surf_path = os.path.join(surf_dir, f"{HEMI}.pial")
    nib.freesurfer.write_geometry(surf_path, verts.astype(np.float64), faces.astype(np.int32))

    label_path = os.path.join(label_dir, f"{HEMI}.cortex.label")
    write_full_cortex_label(label_path, verts.shape[0])

    idx_path = os.path.join(subj, f"{subject_id}_test_vertices.txt")
    np.savetxt(idx_path, np.asarray(test_indices, dtype=np.int32), fmt="%d")


def has_required_layout(base_dir):
    for subject_id in EXPECTED_SUBJECTS:
        subj = os.path.join(base_dir, subject_id)
        surf = os.path.join(subj, "surf", f"{HEMI}.pial")
        label = os.path.join(subj, "label", f"{HEMI}.cortex.label")
        idx = os.path.join(subj, f"{subject_id}_test_vertices.txt")
        if not (os.path.exists(surf) and os.path.exists(label) and os.path.exists(idx)):
            return False
    return True


def sphere_area(radius, r):
    return 2.0 * np.pi * (radius ** 2) * (1.0 - np.cos(r / radius))


def sphere_perimeter(radius, r):
    return 2.0 * np.pi * radius * np.sin(r / radius)


def plane_area(r):
    return np.pi * (r ** 2)


def plane_perimeter(r):
    return 2.0 * np.pi * r


def sphere_r_from_area_fraction(radius, frac):
    f = float(frac)
    if not (0.0 < f <= 0.5):
        raise ValueError(f"Sphere fraction must be in (0,0.5], got {f}")
    return float(float(radius) * np.arccos(1.0 - 2.0 * f))


def plane_r_from_area_fraction(total_area, frac):
    return float(np.sqrt((float(frac) * float(total_area)) / np.pi))


def median_edge_length(vertices, faces):
    e01 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    e12 = vertices[faces[:, 2]] - vertices[faces[:, 1]]
    e20 = vertices[faces[:, 0]] - vertices[faces[:, 2]]
    lengths = np.concatenate(
        (
            np.linalg.norm(e01, axis=1),
            np.linalg.norm(e12, axis=1),
            np.linalg.norm(e20, axis=1),
        )
    )
    return float(np.median(lengths))


def finite_diff_delta(median_edge, radius):
    base = max(0.5 * float(median_edge), 0.5)
    return float(max(1e-3, min(base, 0.1 * max(float(radius), 1e-3))))


def plane_dist_to_boundary(v_plane, xmin, xmax, ymin, ymax):
    x = v_plane[:, 0]
    y = v_plane[:, 1]
    return np.minimum.reduce((x - xmin, xmax - x, y - ymin, ymax - y))


def radius_specs_for_surface(kind, *, R=None, total_area=None, max_r_cap=None, min_radius=None):
    specs = []
    for frac in AREA_FRACTIONS:
        if kind == "sphere":
            r_val = sphere_r_from_area_fraction(float(R), frac)
        elif kind == "plane":
            r_val = plane_r_from_area_fraction(float(total_area), frac)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        if max_r_cap is not None and r_val >= float(max_r_cap):
            continue
        if min_radius is not None and r_val < float(min_radius):
            continue
        specs.append({"fraction": float(frac), "radius": float(r_val)})
    return specs


def pick_plane_test_vertices(v_plane, faces, xmin, xmax, ymin, ymax, r_max, rng, n=30):
    L = median_edge_length(v_plane, faces)
    required = float(r_max) + 2.0 * L
    dist_to_boundary = plane_dist_to_boundary(v_plane, xmin, xmax, ymin, ymax)
    candidates = np.where(dist_to_boundary >= required)[0]
    if candidates.size < n:
        # Relax to still keep disc interior.
        candidates = np.where(dist_to_boundary >= float(r_max))[0]
    if candidates.size < n:
        raise RuntimeError(f"Plane: not enough interior vertices ({candidates.size}) for r_max={r_max}")
    return rng.choice(candidates, size=n, replace=False)


def pick_cylinder_test_vertices(v_cyl, faces, height, r_max, rng, n=30):
    L = median_edge_length(v_cyl, faces)
    required = float(r_max) + 2.0 * L
    z = np.abs(v_cyl[:, 2])
    half_height = 0.5 * float(height)
    candidates = np.where(z <= (half_height - required))[0]
    if candidates.size < n:
        # Relax to still keep disc interior.
        candidates = np.where(z <= (half_height - float(r_max)))[0]
    if candidates.size < n:
        raise RuntimeError(f"Cylinder: not enough safe vertices ({candidates.size}) for r_max={r_max}")
    return rng.choice(candidates, size=n, replace=False)


def generate_surfaces_fs_like(base_dir, seed=12345):
    rng = np.random.default_rng(int(seed))
    os.makedirs(base_dir, exist_ok=True)

    xmin, xmax, ymin, ymax = PLANE_EXTENTS
    plane_total_area = float((xmax - xmin) * (ymax - ymin))
    r_max_plane = max(plane_r_from_area_fraction(plane_total_area, frac) for frac in AREA_FRACTIONS)

    cyl_total_area = float((2.0 * np.pi * CYL_RADIUS) * CYL_HEIGHT)
    r_max_cyl = min(
        CYL_LOCAL_CAP,
        max(plane_r_from_area_fraction(cyl_total_area, frac) for frac in AREA_FRACTIONS),
    )

    for tier, params in RES_LEVELS.items():
        v_s100, f_s100 = make_icosphere(radius=100.0, subdivisions=params["sphere_subdiv"])
        idx_s100 = nearest_unique_vertices(v_s100, sphere_target_points(100.0))
        write_fs_subject(base_dir, f"sphere_R100_{tier}", v_s100, f_s100, idx_s100)

        v_s30, f_s30 = make_icosphere(radius=30.0, subdivisions=params["sphere_subdiv"])
        idx_s30 = nearest_unique_vertices(v_s30, sphere_target_points(30.0))
        write_fs_subject(base_dir, f"sphere_R30_{tier}", v_s30, f_s30, idx_s30)

        v_plane, f_plane = make_plane_grid(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            spacing=params["plane_spacing"],
        )
        idx_plane = pick_plane_test_vertices(
            v_plane,
            f_plane,
            xmin,
            xmax,
            ymin,
            ymax,
            r_max_plane,
            rng,
            n=N_TEST_VERTICES_SURFACE,
        )
        write_fs_subject(base_dir, f"plane_large_{tier}", v_plane, f_plane, idx_plane)

        v_cyl, f_cyl = make_cylinder(
            radius=CYL_RADIUS,
            height=CYL_HEIGHT,
            n_theta=params["cyl_n_theta"],
            n_z=params["cyl_n_z"],
        )
        idx_cyl = pick_cylinder_test_vertices(v_cyl, f_cyl, CYL_HEIGHT, r_max_cyl, rng, n=N_TEST_VERTICES_SURFACE)
        write_fs_subject(base_dir, f"cylinder_R50_{tier}", v_cyl, f_cyl, idx_cyl)

    print(f"Generated FS-like validation surfaces in: {base_dir}")


def load_test_indices(base_dir, subject_id):
    path = os.path.join(base_dir, subject_id, f"{subject_id}_test_vertices.txt")
    arr = np.loadtxt(path, dtype=np.int64)
    arr = np.atleast_1d(arr)
    return arr.astype(np.int32)


def build_surface_cfg(analysis_objects_by_name):
    cfg = {}
    for subject_id, analysis in analysis_objects_by_name.items():
        med_edge = median_edge_length(analysis.vertices, analysis.faces)
        min_radius = MIN_RADIUS_EDGES * med_edge

        if subject_id.startswith("sphere_R100"):
            specs = radius_specs_for_surface("sphere", R=100.0, min_radius=min_radius)
            cfg[subject_id] = {
                "shape": "sphere",
                "radius": 100.0,
                "radius_specs": specs,
                "area_fn": lambda r: sphere_area(100.0, r),
                "perim_fn": lambda r: sphere_perimeter(100.0, r),
                "radius_tol": 0.03,
                "perim_tol": 0.02,
                "antipode_tol": 0.02,
            }
        elif subject_id.startswith("sphere_R30"):
            specs = radius_specs_for_surface("sphere", R=30.0, min_radius=min_radius)
            cfg[subject_id] = {
                "shape": "sphere",
                "radius": 30.0,
                "radius_specs": specs,
                "area_fn": lambda r: sphere_area(30.0, r),
                "perim_fn": lambda r: sphere_perimeter(30.0, r),
                "radius_tol": 0.05,
                "perim_tol": 0.03,
                "antipode_tol": 0.05,
            }
        elif subject_id.startswith("plane_large"):
            xmin, xmax, ymin, ymax = PLANE_EXTENTS
            total_area = float((xmax - xmin) * (ymax - ymin))
            specs = radius_specs_for_surface("plane", total_area=total_area, min_radius=min_radius)
            cfg[subject_id] = {
                "shape": "plane",
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "radius_specs": specs,
                "area_fn": plane_area,
                "perim_fn": plane_perimeter,
                "radius_tol": 0.03,
                "perim_tol": 0.02,
            }
        elif subject_id.startswith("cylinder_R50"):
            total_area = float((2.0 * np.pi * CYL_RADIUS) * CYL_HEIGHT)
            specs = radius_specs_for_surface(
                "plane",
                total_area=total_area,
                max_r_cap=CYL_LOCAL_CAP,
                min_radius=min_radius,
            )
            cfg[subject_id] = {
                "shape": "cylinder",
                "radius": CYL_RADIUS,
                "height": CYL_HEIGHT,
                "radius_specs": specs,
                "area_fn": plane_area,
                "perim_fn": plane_perimeter,
                "radius_tol": 0.03,
                "perim_tol": 0.02,
                "max_radius": CYL_LOCAL_CAP,
                "local_radius_max": CYL_LOCAL_CAP,
            }
        else:
            continue

        # prefer 5% for integration
        integration_spec = next((s for s in cfg[subject_id]["radius_specs"] if abs(s["fraction"] - 0.05) < 1e-12), None)
        if integration_spec is None and cfg[subject_id]["radius_specs"]:
            integration_spec = cfg[subject_id]["radius_specs"][len(cfg[subject_id]["radius_specs"]) // 2]
        cfg[subject_id]["integration_spec"] = integration_spec
    return cfg


def validate_surface(base_dir, subject_id, cfg, run_integration=False):
    print(f"\n=== Validating {subject_id} ===")
    analysis = FastCorticalWiringAnalysis.from_freesurfer(base_dir, subject_id, hemi=HEMI, surf_type="pial")
    n_vertices = int(analysis.n_vertices)
    if n_vertices < MIN_VERTICES_VALIDATE:
        print(f"SKIP: mesh too small for reliable local discs (nV={n_vertices} < {MIN_VERTICES_VALIDATE})")
        return {
            "surface": subject_id,
            "tier": subject_tier(subject_id),
            "n_vertices": n_vertices,
            "skipped": True,
            "skip_reason": "mesh_too_small",
            "n_cases": 0,
            "pass_all": True,
        }, []

    test_vertices = load_test_indices(base_dir, subject_id)
    median_edge = median_edge_length(analysis.vertices, analysis.faces)
    min_radius = MIN_RADIUS_EDGES * median_edge
    margin = 2.0 * median_edge
    tv = analysis.vertices_full[test_vertices]
    dist_to_boundary_tv = None
    cyl_z_tv = None
    if cfg.get("shape") == "plane":
        xmin = float(cfg["xmin"])
        xmax = float(cfg["xmax"])
        ymin = float(cfg["ymin"])
        ymax = float(cfg["ymax"])
        dist_to_boundary_tv = np.minimum.reduce(
            (
                tv[:, 0] - xmin,
                xmax - tv[:, 0],
                tv[:, 1] - ymin,
                ymax - tv[:, 1],
            )
        )
    elif cfg.get("shape") == "cylinder":
        cyl_z_tv = np.abs(tv[:, 2])

    if "max_radius" in cfg and cfg["radius_specs"]:
        max_requested = max(spec["radius"] for spec in cfg["radius_specs"])
        if max_requested >= float(cfg["max_radius"]):
            raise ValueError(f"{subject_id}: requested radius {max_requested} violates max_radius={cfg['max_radius']}")

    pass_geodesic = True
    antipode_rel_err = np.nan
    if cfg.get("shape") == "sphere":
        R = float(cfg["radius"])
        north_target = np.array([0.0, 0.0, R], dtype=np.float64)
        south_target = np.array([0.0, 0.0, -R], dtype=np.float64)
        north_idx = int(np.argmin(np.sum((analysis.vertices_full - north_target) ** 2, axis=1)))
        south_idx = int(np.argmin(np.sum((analysis.vertices_full - south_target) ** 2, axis=1)))
        north_sub = int(analysis.orig_to_sub[north_idx])
        south_sub = int(analysis.orig_to_sub[south_idx])
        if north_sub >= 0 and south_sub >= 0:
            d_north = analysis._compute_geodesic_distances_from_subvertex(north_sub)
            d_antipode = float(d_north[south_sub])
            d_true = float(np.pi * R)
            antipode_rel_err = abs(d_antipode - d_true) / max(1e-12, d_true)
            pass_geodesic = antipode_rel_err <= float(cfg.get("antipode_tol", 0.03))
        else:
            pass_geodesic = False

    rows = []
    skipped_boundary = 0
    for tv_i, orig_idx in enumerate(test_vertices):
        sub_idx = int(analysis.orig_to_sub[int(orig_idx)])
        if sub_idx < 0:
            continue
        d_sub = analysis._compute_geodesic_distances_from_subvertex(sub_idx)
        df = d_sub[analysis.faces]
        dmin = df.min(axis=1)
        dmax = df.max(axis=1)

        for spec in cfg["radius_specs"]:
            frac = float(spec["fraction"])
            r_true = float(spec["radius"])
            if r_true < min_radius:
                continue
            if cfg.get("shape") == "plane" and dist_to_boundary_tv is not None:
                if dist_to_boundary_tv[tv_i] < (r_true + margin):
                    skipped_boundary += 1
                    continue
            if cfg.get("shape") == "cylinder" and cyl_z_tv is not None:
                if cyl_z_tv[tv_i] > (0.5 * float(cfg["height"]) - r_true - margin):
                    skipped_boundary += 1
                    continue

            target_area = cfg["area_fn"](r_true)
            p_true = cfg["perim_fn"](r_true)
            r_measured, _bcount, _history = analysis._find_radius_for_area(
                d_sub,
                target_area,
                tol=1e-3,
                dmin=dmin,
                dmax=dmax,
            )
            if not np.isfinite(r_measured):
                continue

            p_measured = analysis._perimeter_at_radius(r_measured, d_sub, dmin=dmin, dmax=dmax)

            delta_eff = finite_diff_delta(median_edge, r_measured)
            a_plus = analysis._area_inside_radius(r_measured + delta_eff, d_sub, dmin=dmin, dmax=dmax)
            a_minus = analysis._area_inside_radius(max(0.0, r_measured - delta_eff), d_sub, dmin=dmin, dmax=dmax)
            finite_diff = (a_plus - a_minus) / (2.0 * delta_eff)

            radius_err = abs(r_measured - r_true) / max(1e-12, abs(r_true))
            perim_err = abs(p_measured - p_true) / max(1e-12, abs(p_true))
            consistency_err = abs(finite_diff - p_measured) / max(1e-12, abs(p_measured))

            rows.append(
                {
                    "surface": subject_id,
                    "tier": subject_tier(subject_id),
                    "family": subject_family(subject_id),
                    "vertex": int(orig_idx),
                    "area_fraction": frac,
                    "r_true": float(r_true),
                    "r_measured": float(r_measured),
                    "p_true": float(p_true),
                    "p_measured": float(p_measured),
                    "radius_error": float(radius_err),
                    "perimeter_error": float(perim_err),
                    "consistency_error": float(consistency_err),
                }
            )

    if not rows:
        print("SKIP: no valid radius cases after mesh-scale gating")
        return {
            "surface": subject_id,
            "tier": subject_tier(subject_id),
            "n_vertices": n_vertices,
            "skipped": True,
            "skip_reason": "no_valid_radii",
            "n_cases": 0,
            "pass_all": True,
        }, rows

    radius_errors = np.array([r["radius_error"] for r in rows], dtype=np.float64)
    perim_errors = np.array([r["perimeter_error"] for r in rows], dtype=np.float64)
    consistency_errors = np.array([r["consistency_error"] for r in rows], dtype=np.float64)

    radius_max = float(np.max(radius_errors))
    perim_max = float(np.max(perim_errors))
    consistency_max = float(np.max(consistency_errors))

    pass_radius = radius_max <= cfg["radius_tol"]
    pass_perimeter = perim_max <= cfg["perim_tol"]
    pass_consistency = consistency_max <= 0.02
    pass_integration = True
    int_radius_max = np.nan
    int_perim_max = np.nan

    if run_integration and cfg.get("integration_spec") is not None:
        ref_frac = float(cfg["integration_spec"]["fraction"])
        ref_r = float(cfg["integration_spec"]["radius"])
        ref_area = float(cfg["area_fn"](ref_r))
        ref_perim = float(cfg["perim_fn"](ref_r))
        scale = ref_area / float(np.sum(analysis.vertex_areas_sub))
        analysis.compute_all_wiring_costs(scale=scale, area_tol=1e-3)
        scale_key = FastCorticalWiringAnalysis.normalize_scales(scale)[0]

        eligible = np.ones(test_vertices.shape[0], dtype=bool)
        if cfg.get("shape") == "plane":
            xmin = float(cfg["xmin"])
            xmax = float(cfg["xmax"])
            ymin = float(cfg["ymin"])
            ymax = float(cfg["ymax"])
            tv = analysis.vertices_full[test_vertices]
            dist_to_boundary = np.minimum.reduce(
                (
                    tv[:, 0] - xmin,
                    xmax - tv[:, 0],
                    tv[:, 1] - ymin,
                    ymax - tv[:, 1],
                )
            )
            eligible = dist_to_boundary >= (ref_r + margin)
        elif cfg.get("shape") == "cylinder":
            H = float(cfg["height"])
            local_max = float(cfg.get("local_radius_max", np.inf))
            if ref_r > local_max:
                eligible[:] = False
            else:
                z = np.abs(analysis.vertices_full[test_vertices, 2])
                eligible = z <= (0.5 * H - ref_r - margin)

        eligible_vertices = test_vertices[eligible]
        r_vals = analysis.radius_function[scale_key][eligible_vertices]
        p_vals = analysis.perimeter_function[scale_key][eligible_vertices]
        valid_mask = np.isfinite(r_vals) & np.isfinite(p_vals)
        if eligible_vertices.size > 0 and np.any(valid_mask):
            int_radius_err = np.abs(r_vals[valid_mask] - ref_r) / max(1e-12, abs(ref_r))
            int_perim_err = np.abs(p_vals[valid_mask] - ref_perim) / max(1e-12, abs(ref_perim))
            int_radius_max = float(np.max(int_radius_err))
            int_perim_max = float(np.max(int_perim_err))
            pass_integration = (int_radius_max <= cfg["radius_tol"]) and (int_perim_max <= cfg["perim_tol"])
        else:
            pass_integration = False

    pass_all = pass_radius and pass_perimeter and pass_consistency and pass_integration and pass_geodesic

    # Explicit 5% summary
    frac5 = [r for r in rows if abs(float(r["area_fraction"]) - 0.05) < 1e-12]
    if frac5:
        frac5_radius = np.array([r["radius_error"] for r in frac5], dtype=np.float64)
        frac5_perim = np.array([r["perimeter_error"] for r in frac5], dtype=np.float64)
        frac5_cons = np.array([r["consistency_error"] for r in frac5], dtype=np.float64)
        frac5_radius_max = float(np.max(frac5_radius))
        frac5_radius_mean = float(np.mean(frac5_radius))
        frac5_perim_max = float(np.max(frac5_perim))
        frac5_perim_mean = float(np.mean(frac5_perim))
        frac5_cons_max = float(np.max(frac5_cons))
        frac5_cons_mean = float(np.mean(frac5_cons))
    else:
        frac5_radius_max = np.nan
        frac5_radius_mean = np.nan
        frac5_perim_max = np.nan
        frac5_perim_mean = np.nan
        frac5_cons_max = np.nan
        frac5_cons_mean = np.nan

    print(f"Cases: {len(rows)}")
    print(
        f"Radius err max/mean: {radius_max:.4%} / {float(np.mean(radius_errors)):.4%} "
        f"(tol {cfg['radius_tol']:.2%})"
    )
    print(
        f"Perim  err max/mean: {perim_max:.4%} / {float(np.mean(perim_errors)):.4%} "
        f"(tol {cfg['perim_tol']:.2%})"
    )
    print(
        f"dA/dr consistency max/mean: {consistency_max:.4%} / {float(np.mean(consistency_errors)):.4%} "
        "(tol 2.00%)"
    )
    print(
        f"5% patch radius err max/mean: {frac5_radius_max:.4%} / {frac5_radius_mean:.4%}; "
        f"perim err max/mean: {frac5_perim_max:.4%} / {frac5_perim_mean:.4%}; "
        f"consistency max/mean: {frac5_cons_max:.4%} / {frac5_cons_mean:.4%}"
    )
    print(f"Skipped boundary-intersecting cases: {skipped_boundary}")
    if cfg.get("shape") == "sphere":
        print(
            f"Antipode geodesic rel error: {antipode_rel_err:.4%} "
            f"(tol {float(cfg.get('antipode_tol', 0.03)):.2%})"
        )
    if run_integration and cfg.get("integration_spec") is not None:
        print(
            f"Integration (compute_all_wiring_costs @ f={ref_frac:.2%}, r={ref_r:.2f}): "
            f"radius max={int_radius_max:.4%}, perim max={int_perim_max:.4%}"
        )
    print(f"PASS: {pass_all}")

    summary = {
        "surface": subject_id,
        "tier": subject_tier(subject_id),
        "family": subject_family(subject_id),
        "n_vertices": n_vertices,
        "skipped": False,
        "n_cases": len(rows),
        "skipped_boundary_cases": int(skipped_boundary),
        "radius_error_max": radius_max,
        "perimeter_error_max": perim_max,
        "consistency_error_max": consistency_max,
        "radius_error_f05_max": frac5_radius_max,
        "radius_error_f05_mean": frac5_radius_mean,
        "perimeter_error_f05_max": frac5_perim_max,
        "perimeter_error_f05_mean": frac5_perim_mean,
        "consistency_error_f05_max": frac5_cons_max,
        "consistency_error_f05_mean": frac5_cons_mean,
        "pass_radius": pass_radius,
        "pass_perimeter": pass_perimeter,
        "pass_consistency": pass_consistency,
        "integration_radius_error_max": int_radius_max,
        "integration_perimeter_error_max": int_perim_max,
        "pass_integration": pass_integration,
        "antipode_rel_error": antipode_rel_err,
        "pass_geodesic": pass_geodesic,
        "pass_all": pass_all,
    }
    return summary, rows


def write_csv(path, rows):
    header = [
        "surface",
        "tier",
        "family",
        "vertex",
        "area_fraction",
        "r_true",
        "r_measured",
        "p_true",
        "p_measured",
        "radius_error",
        "perimeter_error",
        "consistency_error",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(
                f"{r['surface']},{r['tier']},{r['family']},{r['vertex']},{r['area_fraction']:.8f},"
                f"{r['r_true']:.8f},{r['r_measured']:.8f},{r['p_true']:.8f},{r['p_measured']:.8f},"
                f"{r['radius_error']:.10f},{r['perimeter_error']:.10f},{r['consistency_error']:.10f}\n"
            )


def write_tier_summary_csv(path, summaries):
    header = [
        "tier",
        "family",
        "n_surfaces",
        "radius_error_f05_max",
        "radius_error_f05_mean",
        "perimeter_error_f05_max",
        "perimeter_error_f05_mean",
        "consistency_error_f05_max",
        "consistency_error_f05_mean",
    ]
    groups = {}
    for s in summaries:
        if s.get("skipped", False):
            continue
        key = (s["tier"], s["family"])
        groups.setdefault(key, []).append(s)

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for (tier, family), items in sorted(groups.items()):
            r5_max = np.nanmax(np.array([x["radius_error_f05_max"] for x in items], dtype=np.float64))
            r5_mean = np.nanmean(np.array([x["radius_error_f05_mean"] for x in items], dtype=np.float64))
            p5_max = np.nanmax(np.array([x["perimeter_error_f05_max"] for x in items], dtype=np.float64))
            p5_mean = np.nanmean(np.array([x["perimeter_error_f05_mean"] for x in items], dtype=np.float64))
            c5_max = np.nanmax(np.array([x["consistency_error_f05_max"] for x in items], dtype=np.float64))
            c5_mean = np.nanmean(np.array([x["consistency_error_f05_mean"] for x in items], dtype=np.float64))
            f.write(
                f"{tier},{family},{len(items)},{r5_max:.10f},{r5_mean:.10f},{p5_max:.10f},{p5_mean:.10f},{c5_max:.10f},{c5_mean:.10f}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Generate and validate FS-like synthetic surfaces for fastcw_pp3d.")
    parser.add_argument("--base-dir", default="test_surfaces_fs_like", help="Root folder for generated test subjects.")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed for random vertex selection.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate surfaces even if directory exists.")
    parser.add_argument(
        "--run-integration",
        action="store_true",
        help="Also run compute_all_wiring_costs integration checks (slow on high-resolution meshes).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    if args.regenerate or (not os.path.exists(base_dir)) or (not has_required_layout(base_dir)):
        generate_surfaces_fs_like(base_dir, seed=args.seed)
    else:
        print(f"Using existing surfaces in: {base_dir}")

    analysis_objects = {}
    for surface_name in EXPECTED_SUBJECTS:
        analysis_objects[surface_name] = FastCorticalWiringAnalysis.from_freesurfer(
            base_dir, surface_name, hemi=HEMI, surf_type="pial"
        )

    surface_cfg = build_surface_cfg(analysis_objects)

    summaries = []
    all_rows = []
    for surface_name in EXPECTED_SUBJECTS:
        cfg = surface_cfg[surface_name]
        summary, rows = validate_surface(base_dir, surface_name, cfg, run_integration=args.run_integration)
        summaries.append(summary)
        all_rows.extend(rows)

    os.makedirs(base_dir, exist_ok=True)
    detailed_csv = os.path.join(base_dir, "validation_results_detailed.csv")
    write_csv(detailed_csv, all_rows)
    print(f"\nWrote detailed results: {detailed_csv}")

    tier_csv = os.path.join(base_dir, "validation_results_tiers_f05.csv")
    write_tier_summary_csv(tier_csv, summaries)
    print(f"Wrote tier 5% summary: {tier_csv}")

    not_skipped = [s for s in summaries if not s.get("skipped", False)]
    all_pass = all(s["pass_all"] for s in not_skipped) and len(not_skipped) > 0
    print("\n=== Overall ===")
    for s in summaries:
        if s.get("skipped", False):
            print(f"{s['surface']}: SKIP ({s.get('skip_reason', 'unknown')})")
        else:
            print(f"{s['surface']}: PASS={s['pass_all']} (cases={s['n_cases']})")
    print(f"ALL PASS (non-skipped): {all_pass}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
