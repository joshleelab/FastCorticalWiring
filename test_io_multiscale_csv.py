#!/usr/bin/env python3
"""Tests for multi-scale CSV export layout."""

import os
import tempfile
import unittest

import numpy as np

from io_utils import load_sampled_pairs, save_analysis_npz, save_results_csv


class MultiScaleCsvTests(unittest.TestCase):
    def test_csv_contains_dynamic_scale_columns(self):
        cortex_mask = np.array([True, False, True], dtype=bool)
        msd_unweighted = np.array([1.0, np.nan, 2.0], dtype=np.float32)
        msd_weighted = np.array([1.5, np.nan, 2.5], dtype=np.float32)
        radius = {
            0.001: np.array([0.1, np.nan, 0.2], dtype=np.float32),
            0.05: np.array([1.1, np.nan, 1.2], dtype=np.float32),
        }
        perimeter = {
            0.001: np.array([0.3, np.nan, 0.4], dtype=np.float32),
            0.05: np.array([1.3, np.nan, 1.4], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = save_results_csv(
                tmpdir,
                "results.csv",
                cortex_mask,
                msd_unweighted,
                msd_weighted,
                radius,
                perimeter,
            )
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual(
            lines[0],
            "vertex_id,is_cortex,msd_unweighted,msd_weighted,radius_0.001,perimeter_0.001,radius_0.05,perimeter_0.05",
        )
        row0 = lines[1].split(",")
        self.assertEqual(row0[0], "0")
        self.assertEqual(row0[1], "1")
        self.assertAlmostEqual(float(row0[2]), 1.0, places=6)
        self.assertAlmostEqual(float(row0[3]), 1.5, places=6)
        self.assertAlmostEqual(float(row0[4]), 0.1, places=6)
        self.assertAlmostEqual(float(row0[5]), 0.3, places=6)
        self.assertAlmostEqual(float(row0[6]), 1.1, places=6)
        self.assertAlmostEqual(float(row0[7]), 1.3, places=6)


class SampledPairsNpzTests(unittest.TestCase):
    def test_npz_roundtrip_preserves_sample_arrays(self):
        class _Analysis:
            active_scales = (0.001, 0.05)
            n_samples_between_scales = 3
            cortex_mask_full = np.array([True, False, True], dtype=bool)
            sub_to_orig = np.array([0, 2], dtype=np.int32)
            sample_radii_flat = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            sample_areas_flat = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            sample_indptr = np.array([0, 2, 2, 3], dtype=np.int64)

            def get_metric_arrays(self):
                return {
                    "msd_unweighted": np.array([1.0, np.nan, 2.0], dtype=np.float32),
                    "msd_weighted": np.array([1.5, np.nan, 2.5], dtype=np.float32),
                }

            def get_vertex_samples(self, orig_idx):
                lo = int(self.sample_indptr[orig_idx])
                hi = int(self.sample_indptr[orig_idx + 1])
                return self.sample_radii_flat[lo:hi], self.sample_areas_flat[lo:hi]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_analysis_npz(tmpdir, "samples.npz", _Analysis())
            loaded = load_sampled_pairs(path)
            try:
                np.testing.assert_array_equal(loaded["sample_radii_flat"], _Analysis.sample_radii_flat)
                np.testing.assert_array_equal(loaded["sample_areas_flat"], _Analysis.sample_areas_flat)
                np.testing.assert_array_equal(loaded["sample_indptr"], _Analysis.sample_indptr)
                np.testing.assert_array_equal(loaded["sample_scales_solved"], np.array([0.001, 0.05], dtype=np.float64))
                self.assertEqual(int(loaded["sample_indptr"][-1]), len(loaded["sample_radii_flat"]))
                self.assertEqual(len(loaded["sample_radii_flat"]), len(loaded["sample_areas_flat"]))
            finally:
                loaded.close()


if __name__ == "__main__":
    unittest.main()
