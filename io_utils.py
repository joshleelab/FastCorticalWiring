#!/usr/bin/env python3
"""I/O helpers for FastCorticalWiring.

This module isolates file-format handling from solver code.
"""

import os
import warnings

import numpy as np

try:
    import nibabel as nib
except Exception:
    nib = None


def _require_nibabel():
    if nib is None:
        raise ImportError("nibabel is required for neuroimaging I/O. Install with: pip install nibabel")


def infer_output_basename(surface_path, fallback="surface"):
    """Infer a stable output basename from a surface filename."""
    if not surface_path:
        return fallback
    name = os.path.basename(surface_path)
    for suffix in (".surf.gii", ".shape.gii", ".func.gii", ".gii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    root, _ = os.path.splitext(name)
    return root if root else fallback


def _normalize_region_name(name):
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="ignore")
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _read_freesurfer_label_mask(label_path, n_vertices):
    _require_nibabel()
    cortex_mask = np.zeros(int(n_vertices), dtype=bool)
    try:
        cortex_idx = nib.freesurfer.read_label(label_path)
        cortex_idx = np.asarray(cortex_idx, dtype=np.int64)
    except Exception:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            raise ValueError(f"Malformed label file: {label_path}")
        n_vertices_in_label = int(lines[1].strip())
        cortex_idx = []
        for i in range(2, 2 + n_vertices_in_label):
            if i >= len(lines):
                break
            fields = lines[i].split()
            if not fields:
                continue
            cortex_idx.append(int(fields[0]))
        cortex_idx = np.asarray(cortex_idx, dtype=np.int64)

    valid = (cortex_idx >= 0) & (cortex_idx < cortex_mask.size)
    cortex_mask[cortex_idx[valid]] = True
    return cortex_mask


def _read_freesurfer_annot_mask(annot_path, n_vertices):
    _require_nibabel()
    labels, ctab, names = nib.freesurfer.read_annot(annot_path, orig_ids=True)
    labels = np.asarray(labels)
    if labels.shape[0] != int(n_vertices):
        raise ValueError(
            f"Annotation vertex count mismatch: annot has {labels.shape[0]}, mesh has {n_vertices}"
        )

    if ctab.ndim == 2 and ctab.shape[1] >= 5 and ctab.shape[0] == len(names):
        codes = ctab[:, 4]
    else:
        codes = np.arange(len(names), dtype=np.int32)

    name_to_code = {}
    for i, raw_name in enumerate(names):
        name_to_code[_normalize_region_name(raw_name)] = int(codes[i])

    exclude_tokens = ("unknown", "corpuscallosum", "medialwall")
    exclude_codes = {
        code for norm_name, code in name_to_code.items() if any(tok in norm_name for tok in exclude_tokens)
    }

    cortex_mask = np.ones(int(n_vertices), dtype=bool)
    for code in exclude_codes:
        cortex_mask[labels == code] = False
    cortex_mask[labels == -1] = False
    return cortex_mask


def _load_freesurfer_surface(surface_path):
    _require_nibabel()
    if not os.path.exists(surface_path):
        raise FileNotFoundError(f"Surface file not found: {surface_path}")
    vertices, faces = nib.freesurfer.read_geometry(surface_path)
    return vertices.astype(np.float64), faces.astype(np.int32)


def _load_fslr_surface(surface_path):
    _require_nibabel()
    if not os.path.exists(surface_path):
        raise FileNotFoundError(f"Surface file not found: {surface_path}")

    img = nib.load(surface_path)
    if not isinstance(img, nib.gifti.GiftiImage):
        raise ValueError(f"Expected GIFTI surface file, got: {surface_path}")

    pointset_code = nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]
    triangle_code = nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]

    pointsets = [da for da in img.darrays if int(da.intent) == int(pointset_code)]
    triangles = [da for da in img.darrays if int(da.intent) == int(triangle_code)]
    if len(pointsets) != 1 or len(triangles) != 1:
        raise ValueError(
            f"Malformed GIFTI surface '{surface_path}': expected exactly one POINTSET and one TRIANGLE array"
        )

    vertices = np.asarray(pointsets[0].data, dtype=np.float64)
    faces = np.asarray(triangles[0].data, dtype=np.int32)
    return vertices, faces


def _load_fslr_mask(mask_path, n_vertices):
    _require_nibabel()
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    img = nib.load(mask_path)
    if not isinstance(img, nib.gifti.GiftiImage):
        raise ValueError(f"Expected GIFTI mask file, got: {mask_path}")
    if len(img.darrays) == 0:
        raise ValueError(f"GIFTI mask contains no data arrays: {mask_path}")

    data = np.asarray(img.darrays[0].data)
    data = np.squeeze(data)
    if data.ndim != 1:
        raise ValueError(f"Mask must be a 1D scalar GIFTI array, got shape {tuple(data.shape)}: {mask_path}")
    if data.shape[0] != int(n_vertices):
        raise ValueError(
            f"Mask vertex count mismatch: mask has {data.shape[0]}, mesh has {n_vertices} ({mask_path})"
        )
    return data > 0


def load_surface_and_mask(
    *,
    standard,
    surface_path=None,
    mask_path=None,
    subject_dir=None,
    subject_id=None,
    hemi="lh",
    surf_type="pial",
    custom_label=None,
    no_mask=False,
):
    """Load mesh + cortex mask for FreeSurfer or fsLR workflows."""
    standard = str(standard).lower()
    metadata = {
        "standard": standard,
        "subject_dir": subject_dir,
        "subject_id": subject_id,
        "hemi": hemi,
        "surf_type": surf_type,
        "custom_label": custom_label,
    }

    if standard == "freesurfer":
        legacy_mode = False
        if surface_path is not None:
            surf_dir = os.path.dirname(os.path.abspath(surface_path))
            if os.path.basename(surf_dir) == "surf":
                subject_root = os.path.dirname(surf_dir)
                inferred_subject_id = os.path.basename(subject_root)
                inferred_subject_dir = os.path.dirname(subject_root)
                if subject_id is None:
                    subject_id = inferred_subject_id
                if subject_dir is None:
                    subject_dir = inferred_subject_dir

                metadata["subject_dir"] = subject_dir
                metadata["subject_id"] = subject_id

        if surface_path is None:
            if subject_dir is None or subject_id is None:
                raise ValueError("FreeSurfer loading requires either --surface or both subject_dir and subject_id.")
            legacy_mode = True
            surface_path = os.path.join(subject_dir, subject_id, "surf", f"{hemi}.{surf_type}")

        vertices, faces = _load_freesurfer_surface(surface_path)
        n_vertices = vertices.shape[0]
        metadata["surface_path"] = surface_path
        metadata["legacy_mode"] = legacy_mode
        metadata["output_basename"] = f"{subject_id}_{hemi}_{surf_type}" if subject_id else infer_output_basename(surface_path)

        if no_mask:
            cortex_mask = np.ones(n_vertices, dtype=bool)
            metadata["mask_source"] = "all_vertices_no_mask"
        elif mask_path:
            mask_ext = os.path.splitext(mask_path)[1].lower()
            if mask_ext == ".label":
                cortex_mask = _read_freesurfer_label_mask(mask_path, n_vertices)
                metadata["mask_source"] = f"explicit_label:{mask_path}"
            elif mask_ext == ".annot":
                cortex_mask = _read_freesurfer_annot_mask(mask_path, n_vertices)
                metadata["mask_source"] = f"explicit_annot:{mask_path}"
            else:
                raise ValueError(
                    f"Unsupported FreeSurfer mask file extension '{mask_ext}'. Use .label or .annot."
                )
        else:
            label_dir = None
            if subject_dir and subject_id:
                label_dir = os.path.join(subject_dir, subject_id, "label")
            elif surface_path:
                maybe_label = os.path.join(os.path.dirname(os.path.dirname(surface_path)), "label")
                if os.path.isdir(maybe_label):
                    label_dir = maybe_label

            cortex_mask = None
            if label_dir:
                if custom_label:
                    candidate = os.path.join(label_dir, f"{hemi}.{custom_label}.label")
                    if os.path.exists(candidate):
                        cortex_mask = _read_freesurfer_label_mask(candidate, n_vertices)
                        metadata["mask_source"] = f"custom_label:{candidate}"
                if cortex_mask is None:
                    candidate = os.path.join(label_dir, f"{hemi}.cortex.label")
                    if os.path.exists(candidate):
                        cortex_mask = _read_freesurfer_label_mask(candidate, n_vertices)
                        metadata["mask_source"] = f"default_label:{candidate}"
                if cortex_mask is None:
                    candidate = os.path.join(label_dir, f"{hemi}.aparc.annot")
                    if os.path.exists(candidate):
                        warnings.warn("Cortex label not found; using aparc annotation fallback.", RuntimeWarning)
                        cortex_mask = _read_freesurfer_annot_mask(candidate, n_vertices)
                        metadata["mask_source"] = f"aparc_fallback:{candidate}"

            if cortex_mask is None:
                raise FileNotFoundError(
                    "No cortical mask found for FreeSurfer surface. "
                    "Provide --mask, use --custom-label if applicable, or pass --no-mask explicitly."
                )

        if subject_dir and subject_id:
            mgh_template = os.path.join(subject_dir, subject_id, "mri", "orig.mgz")
            if not os.path.exists(mgh_template):
                mgh_template = os.path.join(subject_dir, subject_id, "mri", "T1.mgz")
            if os.path.exists(mgh_template):
                metadata["mgh_template_path"] = mgh_template

        return vertices, faces, cortex_mask.astype(bool), metadata

    if standard == "fslr":
        if surface_path is None:
            raise ValueError("fsLR loading requires --surface pointing to a .surf.gii file.")
        vertices, faces = _load_fslr_surface(surface_path)
        n_vertices = vertices.shape[0]
        metadata["surface_path"] = surface_path
        metadata["legacy_mode"] = False
        metadata["output_basename"] = infer_output_basename(surface_path)

        if no_mask:
            cortex_mask = np.ones(n_vertices, dtype=bool)
            metadata["mask_source"] = "all_vertices_no_mask"
        else:
            if not mask_path:
                raise ValueError("fsLR mode requires --mask (GIFTI scalar) or --no-mask.")
            cortex_mask = _load_fslr_mask(mask_path, n_vertices)
            metadata["mask_source"] = f"gifti_mask:{mask_path}"
        return vertices, faces, cortex_mask.astype(bool), metadata

    raise ValueError(f"Unsupported standard '{standard}'. Expected 'freesurfer' or 'fslr'.")


def save_results_csv(
    output_dir,
    csv_filename,
    cortex_mask,
    msd_unweighted,
    msd_weighted,
    radius_function,
    perimeter_function,
):
    """Write tabular results CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, csv_filename)

    n_vertices = int(len(cortex_mask))
    if len(msd_unweighted) != n_vertices or len(msd_weighted) != n_vertices:
        raise ValueError("MSD arrays must match cortex_mask length for CSV output.")
    if (
        not isinstance(radius_function, dict)
        or not isinstance(perimeter_function, dict)
    ):
        raise ValueError("radius_function and perimeter_function must be dicts keyed by scale.")
    keys = set(radius_function.keys())
    if keys != set(perimeter_function.keys()):
        raise ValueError("radius_function and perimeter_function must share the same scale keys.")

    scale_keys = list(radius_function.keys())
    radius_arrays = {}
    perimeter_arrays = {}
    for scale_key in scale_keys:
        r_arr = np.asarray(radius_function[scale_key])
        p_arr = np.asarray(perimeter_function[scale_key])
        if (
            r_arr.shape[0] != n_vertices
            or p_arr.shape[0] != n_vertices
        ):
            raise ValueError("All metric arrays must match cortex_mask length for CSV output.")
        radius_arrays[scale_key] = r_arr
        perimeter_arrays[scale_key] = p_arr

    with open(path, "w", encoding="utf-8") as f:
        header = ["vertex_id", "is_cortex", "msd_unweighted", "msd_weighted"]
        for scale_key in scale_keys:
            token = format(float(scale_key), "g")
            header.append(f"radius_{token}")
            header.append(f"perimeter_{token}")
        f.write(",".join(header) + "\n")
        for i in range(n_vertices):
            fields = [
                str(i),
                str(int(bool(cortex_mask[i]))),
                str(float(msd_unweighted[i])),
                str(float(msd_weighted[i])),
            ]
            for scale_key in scale_keys:
                fields.append(str(float(radius_arrays[scale_key][i])))
                fields.append(str(float(perimeter_arrays[scale_key][i])))
            f.write(",".join(fields) + "\n")
    return path


def save_results_mgh(output_dir, filename_template, metrics, n_vertices, template_mgz_path=None):
    """Write scalar overlays in FreeSurfer MGH format."""
    _require_nibabel()
    os.makedirs(output_dir, exist_ok=True)
    mgh_ns = getattr(getattr(nib, "freesurfer", None), "mghformat", None)
    MGHHeaderCls = getattr(nib, "MGHHeader", None)
    MGHImageCls = getattr(nib, "MGHImage", None)
    if MGHHeaderCls is None and mgh_ns is not None:
        MGHHeaderCls = getattr(mgh_ns, "MGHHeader", None)
    if MGHImageCls is None and mgh_ns is not None:
        MGHImageCls = getattr(mgh_ns, "MGHImage", None)
    if MGHHeaderCls is None or MGHImageCls is None:
        raise ImportError("Installed nibabel build does not expose FreeSurfer MGH classes.")

    if template_mgz_path and os.path.exists(template_mgz_path):
        template_img = nib.load(template_mgz_path)
        template_affine = template_img.affine
        template_header = template_img.header
    else:
        template_affine = np.array(
            [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        template_header = MGHHeaderCls()
        template_header.set_data_shape([int(n_vertices), 1, 1])

    written = []
    for metric_name, data_array in metrics.items():
        arr = np.asarray(data_array)
        if arr.shape[0] != int(n_vertices):
            raise ValueError(
                f"Metric '{metric_name}' length mismatch for MGH output: {arr.shape[0]} vs {n_vertices}"
            )
        if not np.any(np.isfinite(arr)):
            continue

        payload = arr.astype(np.float32).reshape((int(n_vertices), 1, 1))
        payload[~np.isfinite(payload)] = 0

        header = template_header.copy()
        header.set_data_shape(payload.shape)
        header.set_data_dtype(np.float32)

        out_path = os.path.join(output_dir, filename_template.format(metric=metric_name))
        nib.save(MGHImageCls(payload, template_affine, header), out_path)
        written.append(out_path)

    return written


def save_results_gifti(output_dir, filename_template, metrics, n_vertices):
    """Write scalar overlays in GIFTI format (one file per metric)."""
    _require_nibabel()
    os.makedirs(output_dir, exist_ok=True)
    shape_intent = nib.nifti1.intent_codes["NIFTI_INTENT_SHAPE"]

    written = []
    for metric_name, data_array in metrics.items():
        arr = np.asarray(data_array, dtype=np.float32)
        if arr.shape[0] != int(n_vertices):
            raise ValueError(
                f"Metric '{metric_name}' length mismatch for GIFTI output: {arr.shape[0]} vs {n_vertices}"
            )
        if not np.any(np.isfinite(arr)):
            continue

        da = nib.gifti.GiftiDataArray(data=arr, intent=int(shape_intent))
        da.meta = nib.gifti.GiftiMetaData.from_dict({"Name": metric_name})
        img = nib.gifti.GiftiImage(darrays=[da])

        out_path = os.path.join(output_dir, filename_template.format(metric=metric_name))
        nib.save(img, out_path)
        written.append(out_path)

    return written


def save_analysis_npz(output_dir, npz_filename, analysis, n_samples_between_scales=None):
    """Write scalar metrics plus sampled radius/area pairs to a compressed NPZ."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, npz_filename)

    payload = dict(analysis.get_metric_arrays())
    payload.update(
        {
            "fastcw_output_schema_version": np.asarray(2, dtype=np.int32),
            "sample_storage_layout": np.asarray("csr_flat_v1"),
            "sample_storage_note": np.asarray(
                "sample_radii_flat/sample_areas_flat are CSR-style rows indexed by sample_indptr[v]:sample_indptr[v+1]."
            ),
            "sample_radii_flat": np.asarray(analysis.sample_radii_flat, dtype=np.float32),
            "sample_areas_flat": np.asarray(analysis.sample_areas_flat, dtype=np.float32),
            "sample_indptr": np.asarray(analysis.sample_indptr, dtype=np.int64),
            "sample_scales_solved": np.asarray(analysis.active_scales, dtype=np.float64),
            "n_samples_between_scales": np.asarray(
                int(
                    analysis.n_samples_between_scales
                    if n_samples_between_scales is None
                    else n_samples_between_scales
                ),
                dtype=np.int32,
            ),
            "boundary_cap_fraction": np.asarray(
                np.nan if getattr(analysis, "boundary_cap_fraction", None) is None else float(analysis.boundary_cap_fraction),
                dtype=np.float64,
            ),
            "cortex_mask": np.asarray(analysis.cortex_mask_full, dtype=bool),
            "sub_to_orig": np.asarray(analysis.sub_to_orig, dtype=np.int32),
        }
    )
    np.savez_compressed(path, **payload)
    return path


def load_sampled_pairs(npz_path):
    """
    Load sampled radius/area pairs from a FastCW compressed NPZ.

    Downstream refitting code should use sample_indptr[v]:sample_indptr[v + 1]
    to slice sample_radii_flat and sample_areas_flat.
    """
    data = np.load(npz_path, allow_pickle=False)
    required = ("sample_radii_flat", "sample_areas_flat", "sample_indptr")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"NPZ file is missing sampled-pair arrays: {', '.join(missing)}")
    return data

def random_vertex_sampling(valid_mask, k, random_state=None):
    """Uniform random sampling over valid vertices without replacement."""
    valid_indices = np.where(valid_mask)[0]
    n_valid = int(valid_indices.size)
    if k >= n_valid:
        return valid_indices.tolist()
    rng = np.random.default_rng(random_state)
    picked = rng.choice(valid_indices, size=int(k), replace=False)
    return np.asarray(picked, dtype=np.int64).tolist()


def stratified_vertex_sampling(vertices, valid_mask, k, n_bins=None, random_state=None):
    """
    Lightweight stratification over cortical vertex order.

    This is intentionally cheap and reproducible; cortical order is only a proxy for
    spatial uniformity and not a geodesic stratifier.
    """
    _ = vertices  # Reserved for future upgrades (e.g., coordinate-driven binning).
    valid_indices = np.where(valid_mask)[0]
    n_valid = int(valid_indices.size)
    if k >= n_valid:
        return valid_indices.tolist()

    k = int(k)
    if n_bins is None:
        n_bins = max(8, min(256, int(np.sqrt(k))))
    n_bins = max(1, min(int(n_bins), n_valid))

    bins = np.array_split(valid_indices, n_bins)
    bin_sizes = np.array([b.size for b in bins], dtype=np.int64)
    proportions = (bin_sizes / float(n_valid)) * float(k)
    alloc = np.floor(proportions).astype(np.int64)
    alloc = np.minimum(alloc, bin_sizes)

    target_remaining = k - int(np.sum(alloc))
    if target_remaining > 0:
        frac = proportions - alloc
        order = np.argsort(-frac)
        for idx in order:
            if target_remaining <= 0:
                break
            cap = int(bin_sizes[idx] - alloc[idx])
            if cap <= 0:
                continue
            take = min(cap, target_remaining)
            alloc[idx] += take
            target_remaining -= take

    rng = np.random.default_rng(random_state)
    chosen_parts = []
    for i, b in enumerate(bins):
        n_take = int(alloc[i])
        if n_take <= 0:
            continue
        if n_take >= b.size:
            chosen_parts.append(np.asarray(b, dtype=np.int64))
        else:
            chosen_parts.append(np.asarray(rng.choice(b, size=n_take, replace=False), dtype=np.int64))

    if chosen_parts:
        chosen = np.concatenate(chosen_parts)
    else:
        chosen = np.empty(0, dtype=np.int64)

    if chosen.size < k:
        need = int(k - chosen.size)
        available = np.setdiff1d(valid_indices, chosen, assume_unique=False)
        top_up = rng.choice(available, size=need, replace=False)
        chosen = np.concatenate([chosen, np.asarray(top_up, dtype=np.int64)])
    elif chosen.size > k:
        trim_idx = rng.choice(np.arange(chosen.size), size=k, replace=False)
        chosen = chosen[trim_idx]

    return np.asarray(chosen, dtype=np.int64).tolist()


def farthest_point_sampling(vertices, k, valid_mask):
    """Spatially uniform Euclidean FPS restricted to valid/cortical vertices."""
    valid_indices = np.where(valid_mask)[0]
    valid_vertices = vertices[valid_indices]

    if k >= len(valid_indices):
        return valid_indices.tolist()

    sampled_local = np.zeros(int(k), dtype=np.int64)
    distances = np.full(valid_vertices.shape[0], np.inf)
    farthest_local = 0

    for i in range(int(k)):
        sampled_local[i] = farthest_local
        dist = np.sum((valid_vertices - valid_vertices[farthest_local]) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest_local = int(np.argmax(distances))

    return valid_indices[sampled_local].tolist()


def select_vertex_subset(
    vertices,
    cortex_mask,
    *,
    sample_frac=None,
    sample_count=None,
    sample_method="stratified",
    random_state=None,
):
    """Select cortical vertex subset using method-based dispatch."""
    valid_indices = np.where(cortex_mask)[0]
    n_valid = int(valid_indices.size)
    if n_valid == 0:
        return []

    if sample_frac is not None and sample_count is not None:
        raise ValueError("sample_frac and sample_count are mutually exclusive.")

    if sample_count is not None:
        k = int(sample_count)
        if k <= 0:
            raise ValueError("sample_count must be a positive integer.")
    elif sample_frac is not None:
        frac = float(sample_frac)
        if frac <= 0.0 or frac > 1.0:
            raise ValueError("sample_frac must satisfy 0 < sample_frac <= 1.")
        k = max(1, int(round(frac * n_valid)))
    else:
        return None

    k = min(k, n_valid)
    method = str(sample_method or "stratified").lower()
    if method == "stratified":
        return stratified_vertex_sampling(vertices, cortex_mask, k, random_state=random_state)
    if method == "random":
        return random_vertex_sampling(cortex_mask, k, random_state=random_state)
    if method == "fps":
        return farthest_point_sampling(vertices, k, cortex_mask)
    raise ValueError(f"Unknown sample_method: {sample_method}")
