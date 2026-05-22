#!/usr/bin/env python3
"""Tests for sampling methods and sampling dispatch semantics."""

import tempfile
import unittest
from unittest import mock

import numpy as np

import fastcw
from io_utils import farthest_point_sampling, select_vertex_subset


class _StubAnalysis:
    last_compute_kwargs = None

    @staticmethod
    def normalize_scales(scale):
        from core_analysis import FastCorticalWiringAnalysis

        return FastCorticalWiringAnalysis.normalize_scales(scale)

    @staticmethod
    def scale_token(scale):
        from core_analysis import FastCorticalWiringAnalysis

        return FastCorticalWiringAnalysis.scale_token(scale)

    def __init__(self, _vertices, _faces, cortex_mask, **_kwargs):
        n = int(cortex_mask.shape[0])
        self.cortex_mask_full = np.asarray(cortex_mask, dtype=bool)
        self.n_vertices_full = n
        self.msd_unweighted = np.full(n, np.nan, dtype=np.float32)
        self.msd_weighted = np.full(n, np.nan, dtype=np.float32)
        self.active_scales = (0.05,)
        self.radius_function = {0.05: np.full(n, np.nan, dtype=np.float32)}
        self.perimeter_function = {0.05: np.full(n, np.nan, dtype=np.float32)}
        self.sample_radii_flat = np.empty(0, dtype=np.float32)
        self.sample_areas_flat = np.empty(0, dtype=np.float32)
        self.sample_indptr = np.zeros(n + 1, dtype=np.int64)
        self.metadata = {}

    def compute_all_wiring_costs(self, **kwargs):
        _StubAnalysis.last_compute_kwargs = dict(kwargs)
        return (self.msd_unweighted, self.msd_weighted), self.radius_function, self.perimeter_function

    def get_metric_arrays(self):
        return {
            "msd_unweighted": self.msd_unweighted,
            "msd_weighted": self.msd_weighted,
            "radius_0.05": self.radius_function[0.05],
            "perimeter_0.05": self.perimeter_function[0.05],
        }


class SamplingMethodTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array(
            [[float(i), 0.0, 0.0] for i in range(10)],
            dtype=np.float64,
        )
        self.valid_mask = np.array([True, False, True, True, False, True, True, True, False, True], dtype=bool)

    def test_sampling_methods_return_unique_valid_indices(self):
        for method in ("stratified", "random", "fps"):
            subset = select_vertex_subset(
                self.vertices,
                self.valid_mask,
                sample_count=4,
                sample_method=method,
                random_state=0,
            )
            self.assertEqual(len(subset), len(set(subset)))
            self.assertTrue(all(self.valid_mask[idx] for idx in subset))

    def test_sample_frac_is_based_on_cortical_count(self):
        subset = select_vertex_subset(
            self.vertices,
            self.valid_mask,
            sample_frac=0.4,
            sample_method="stratified",
            random_state=0,
        )
        expected = max(1, int(round(0.4 * int(np.count_nonzero(self.valid_mask)))))
        self.assertEqual(len(subset), expected)

    def test_sample_size_saturates_to_all_cortical_vertices(self):
        subset_by_count = select_vertex_subset(
            self.vertices,
            self.valid_mask,
            sample_count=10_000,
            sample_method="random",
            random_state=0,
        )
        subset_by_frac = select_vertex_subset(
            self.vertices,
            self.valid_mask,
            sample_frac=1.0,
            sample_method="stratified",
            random_state=0,
        )
        expected = set(np.where(self.valid_mask)[0].tolist())
        self.assertEqual(set(subset_by_count), expected)
        self.assertEqual(set(subset_by_frac), expected)

    def test_sample_frac_and_sample_count_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            select_vertex_subset(
                self.vertices,
                self.valid_mask,
                sample_frac=0.5,
                sample_count=3,
                sample_method="stratified",
            )

    def test_fps_masked_matches_previous_semantics_on_small_example(self):
        def old_fps(vertices, k, valid_mask):
            valid_indices = np.where(valid_mask)[0]
            if k >= len(valid_indices):
                return valid_indices.tolist()

            sampled_indices = np.zeros(k, dtype=int)
            distances = np.full(vertices.shape[0], np.inf)
            farthest = valid_indices[0]

            for i in range(k):
                sampled_indices[i] = farthest
                dist = np.sum((vertices - vertices[farthest]) ** 2, axis=1)
                distances = np.minimum(distances, dist)

                masked_distances = distances.copy()
                masked_distances[~valid_mask] = -1.0
                farthest = np.argmax(masked_distances)

            return sampled_indices.tolist()

        k = 4
        old_subset = old_fps(self.vertices, k, self.valid_mask)
        new_subset = farthest_point_sampling(self.vertices, k, self.valid_mask)
        self.assertEqual(new_subset, old_subset)


class SamplingDispatchTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[float(i), 0.0, 0.0] for i in range(12)], dtype=np.float64)
        self.faces = np.array([[0, 2, 3], [3, 5, 6], [6, 7, 9]], dtype=np.int32)
        self.cortex_mask = np.array(
            [True, False, True, True, False, True, True, True, False, True, False, True],
            dtype=bool,
        )
        self.metadata = {
            "legacy_mode": False,
            "surface_path": "/tmp/mock.surf.gii",
            "output_basename": "mock_surface",
            "hemi": "lh",
            "surf_type": "pial",
        }

    def _run_single(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "io_utils.load_surface_and_mask",
                return_value=(self.vertices, self.faces, self.cortex_mask, self.metadata),
            ):
                with mock.patch("fastcw.FastCorticalWiringAnalysis", _StubAnalysis):
                    with mock.patch("fastcw._save_analysis_outputs", return_value=[]):
                        return fastcw._run_single_surface(
                            standard="freesurfer",
                            surface_path="/tmp/mock.surf",
                            mask_path=None,
                            output_dir=tmpdir,
                            output_basename=None,
                            subject_dir=None,
                            subject_id=None,
                            hemi="lh",
                            surf_type="pial",
                            custom_label=None,
                            no_mask=False,
                            output_format="csv",
                            engine_type="potpourri",
                            engine_kwargs={},
                            scale=0.05,
                            area_tol=0.01,
                            eps=1e-6,
                            overwrite=True,
                            **kwargs,
                        )

    def test_sampling_happens_after_mask_and_indices_are_cortical(self):
        self._run_single(sample=0.4, sample_method="stratified", vertex_list=None)
        subset = _StubAnalysis.last_compute_kwargs["vertex_subset"]
        self.assertTrue(all(self.cortex_mask[idx] for idx in subset))

    def test_vertex_list_has_priority_over_sampling(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("2\n9\n")
            path = tmp.name
        try:
            self._run_single(vertex_list=path, sample=0.4, sample_method="random")
            self.assertEqual(_StubAnalysis.last_compute_kwargs["vertex_subset"], [2, 9])
        finally:
            import os

            os.unlink(path)

    def test_fraction_suffix_uses_method_and_frac(self):
        captured = {}

        def _capture_naming(metadata, output_dir, output_basename=None, suffix=""):
            captured["suffix"] = suffix
            return "x.csv", "x.{metric}"

        with mock.patch("fastcw._resolve_naming", side_effect=_capture_naming):
            self._run_single(sample=0.4, sample_method="stratified", vertex_list=None)
        self.assertEqual(captured["suffix"], "_sample-stratified-frac40p")

    def test_count_suffix_uses_method_and_count(self):
        captured = {}

        def _capture_naming(metadata, output_dir, output_basename=None, suffix=""):
            captured["suffix"] = suffix
            return "x.csv", "x.{metric}"

        with mock.patch("fastcw._resolve_naming", side_effect=_capture_naming):
            self._run_single(sample=5, sample_kind="count", sample_method="fps", vertex_list=None)
        self.assertEqual(captured["suffix"], "_sample-fps-n5")

    def test_sample_defaults_to_fraction_mode(self):
        self._run_single(sample=0.4, sample_method="stratified", vertex_list=None)
        subset = _StubAnalysis.last_compute_kwargs["vertex_subset"]
        n_valid = int(np.count_nonzero(self.cortex_mask))
        expected = max(1, int(round(0.4 * n_valid)))
        self.assertEqual(len(subset), expected)

    def test_run_single_rejects_frac_and_count_together(self):
        with self.assertRaises(ValueError):
            self._run_single(sample_frac=0.4, sample_count=3, sample_method="stratified", vertex_list=None)


if __name__ == "__main__":
    unittest.main()
