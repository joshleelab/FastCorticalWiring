#!/usr/bin/env python3
"""Synthetic meshes and reference engines for testing fastcw_lean without FreeSurfer data."""

import sys
import types

import numpy as np


def install_stubs():
    """Register reference engines with distance_engines; stub tqdm only if it is missing."""
    try:
        import tqdm  # noqa: F401
    except Exception:
        tq = types.ModuleType("tqdm")

        class _T:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def update(self, n=1):
                pass

            @staticmethod
            def write(s):
                print(s)

        tq.tqdm = _T
        sys.modules["tqdm"] = tq

    import distance_engines as de

    class GreatCircleStub(de.BaseDistanceEngine):
        """Exact geodesic distance on a sphere centred at the origin."""

        @property
        def name(self):
            return "great_circle_stub"

        def compute_distance(self, s):
            V = self.vertices
            R = float(np.mean(np.linalg.norm(V, axis=1)))
            c = np.clip((V @ V[int(s)]) / (R * R), -1.0, 1.0)
            return R * np.arccos(c)

    class EuclidStub(de.BaseDistanceEngine):
        """Euclidean distance: linear per face, enough for area/perimeter checks."""

        @property
        def name(self):
            return "euclid_stub"

        def compute_distance(self, s):
            return np.linalg.norm(self.vertices - self.vertices[int(s)], axis=1)

    de.ENGINE_REGISTRY["great_circle_stub"] = GreatCircleStub
    de.ENGINE_REGISTRY["euclid_stub"] = EuclidStub


def icosphere(level):
    """Unit icosphere; level 5/6/7 ~ fsaverage5/6/fsaverage vertex counts."""
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = np.array(V, float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    F = np.array(F, np.int64)
    for _ in range(level):
        e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        e.sort(1)
        ue, inv = np.unique(e, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        mid = V[ue[:, 0]] + V[ue[:, 1]]
        mid /= np.linalg.norm(mid, axis=1)[:, None]
        base = V.shape[0]
        V = np.vstack([V, mid])
        nF = F.shape[0]
        a = inv[:nF] + base
        b = inv[nF:2 * nF] + base
        c = inv[2 * nF:] + base
        F = np.vstack([np.c_[F[:, 0], a, c], np.c_[F[:, 1], b, a], np.c_[F[:, 2], c, b], np.c_[a, b, c]])
    return V, F.astype(np.int32)


def bumpy(V, R=70.0, amp=0.12):
    """Radially folded sphere (mm) for irregular triangles and long faces."""
    th = np.arccos(np.clip(V[:, 2], -1, 1))
    ph = np.arctan2(V[:, 1], V[:, 0])
    rr = R * (1 + amp * np.sin(6 * th) * np.cos(5 * ph) + 0.5 * amp * np.cos(9 * th + 2 * ph))
    return V * rr[:, None]
