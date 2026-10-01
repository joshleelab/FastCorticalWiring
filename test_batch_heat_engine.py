#!/usr/bin/env python3
"""Regression checks for removing the old batched distance interface."""

import unittest

import numpy as np

import distance_engines


class SingleSourceEngineTests(unittest.TestCase):
    def test_batch_engine_and_interface_are_removed(self):
        self.assertNotIn("batch_heat", distance_engines.ENGINE_REGISTRY)
        self.assertFalse(hasattr(distance_engines, "BatchHeatDistanceEngine"))
        self.assertFalse(hasattr(distance_engines.BaseDistanceEngine, "compute_distance_batch"))
        self.assertFalse(hasattr(distance_engines.BaseDistanceEngine, "supports_batching"))

    def test_batch_engine_name_is_rejected(self):
        vertices = np.zeros((3, 3), dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "Unknown engine_type 'batch_heat'"):
            distance_engines.create_distance_engine("batch_heat", vertices, faces)


if __name__ == "__main__":
    unittest.main()
