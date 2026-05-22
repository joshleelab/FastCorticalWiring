#!/usr/bin/env python3
"""Lightweight smoke tests for the pygeodesic backend."""

import unittest

import numpy as np

from core_analysis import FastCorticalWiringAnalysis
from distance_engines import create_distance_engine


def _make_octahedron():
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int32,
    )
    mask = np.ones(vertices.shape[0], dtype=bool)
    return vertices, faces, mask


class PyGeodesicBackendSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pygeodesic.geodesic  # noqa: F401

            cls._has_pygeodesic = True
        except Exception:
            cls._has_pygeodesic = False

    def setUp(self):
        if not self._has_pygeodesic:
            self.skipTest("pygeodesic not installed")
        self.vertices, self.faces, self.mask = _make_octahedron()

    def test_factory_constructs_pygeodesic_engine(self):
        engine = create_distance_engine("pygeodesic", self.vertices, self.faces)
        self.assertEqual(engine.name, "pygeodesic")

    def test_compute_distance_shape_dtype_and_finiteness(self):
        engine = create_distance_engine("pygeodesic", self.vertices, self.faces)
        d = engine.compute_distance(0)
        self.assertEqual(d.shape, (self.vertices.shape[0],))
        self.assertEqual(d.dtype, np.float64)
        self.assertTrue(d.flags["C_CONTIGUOUS"])
        self.assertAlmostEqual(float(d[0]), 0.0, places=7)
        self.assertTrue(np.all(np.isfinite(d)))

    def test_shared_pipeline_runs_with_pygeodesic(self):
        analysis = FastCorticalWiringAnalysis(
            self.vertices,
            self.faces,
            self.mask,
            engine_type="pygeodesic",
            eps=1e-6,
            metadata={"subject_id": "synthetic", "hemi": "lh", "surf_type": "unit_octa"},
        )
        analysis.compute_all_wiring_costs(scale=0.2, area_tol=0.1)

        scale_key = FastCorticalWiringAnalysis.normalize_scales(0.2)[0]
        self.assertIn(scale_key, analysis.radius_function)
        self.assertIn(scale_key, analysis.perimeter_function)
        self.assertTrue(np.any(np.isfinite(analysis.radius_function[scale_key])))
        self.assertTrue(np.any(np.isfinite(analysis.perimeter_function[scale_key])))


if __name__ == "__main__":
    unittest.main()
