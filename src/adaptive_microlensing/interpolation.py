"""Delaunay index of one region: locating queries and applying the hit rule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from .mpd import js_distances, pairwise_js_distances

LOCATION_TOLERANCE = 1e-10
COINCIDENCE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Location:
    """A query expressed in its containing tetrahedron."""

    simplex_index: int
    vertices: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class HitEvaluation:
    """Outcome of the hit rule in one tetrahedron."""

    vertex_distances: np.ndarray
    matched_entry_id: int
    interpolated_distance: float
    query_quantile: float
    matched_quantile: float
    threshold: float
    is_hit: bool

    @property
    def margin(self) -> float:
        """How far the interpolated distance lies below the threshold."""
        return self.threshold - self.interpolated_distance


def barycentric_weights(delaunay: Delaunay, simplex_index: int, point: np.ndarray, tol: float) -> np.ndarray:
    """Barycentric weights of ``point`` in a simplex, cleaned up at faces, edges and vertices."""
    transform = delaunay.transform[simplex_index]
    ndim = delaunay.ndim
    first = transform[:ndim] @ (point - transform[ndim])
    weights = np.concatenate([first, [1.0 - first.sum()]])
    if not np.all(np.isfinite(weights)):
        raise RuntimeError(f"Non-finite barycentric weights {weights.tolist()} for {point.tolist()}.")
    weights[np.abs(weights) <= tol] = 0.0
    weights[np.abs(weights - 1.0) <= tol] = 1.0
    if np.any(weights < -tol):
        raise RuntimeError(f"Substantially negative barycentric weights {weights.tolist()}.")
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError(f"Barycentric weights have non-positive sum: {weights.tolist()}.")
    return weights / total


class RegionIndex:
    """Delaunay mesh over all of a region's entries, valid and invalid.

    Parameters
    ----------
    points : array of shape (n, 3)
        ``(kappa, gamma, s)`` of every entry, in ``entry_id`` order.
    valid : array of shape (n,)
        Whether each entry is valid.
    quantiles : array of shape (n,)
        Intrinsic MPD-distance quantile of each entry (NaN if invalid).
    mpds : array of shape (n, m), optional
        Bank MPDs. Without them the index can locate points and interpolate
        quantiles, but cannot apply the hit rule.
    distances : array of shape (n, n), optional
        Precomputed JS distances between the MPDs.
    """

    def __init__(
        self,
        points: np.ndarray,
        valid: np.ndarray,
        quantiles: np.ndarray,
        mpds: np.ndarray | None = None,
        distances: np.ndarray | None = None,
    ) -> None:
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.valid = np.asarray(valid, dtype=bool)
        self.quantiles = np.asarray(quantiles, dtype=float)
        self.mpds = None if mpds is None else np.asarray(mpds, dtype=float)
        if distances is None and self.mpds is not None:
            distances = pairwise_js_distances(self.mpds, self.valid)
        self.distances = distances
        self.ready = bool(len(self.points) >= 4 and np.linalg.matrix_rank(self.points - self.points[0]) == 3)
        self.delaunay = Delaunay(self.points) if self.ready else None

    def extended(
        self, point: np.ndarray, valid: bool, quantile: float, mpd: np.ndarray | None = None
    ) -> RegionIndex:
        """A new index with one more entry appended; only the new distances are computed."""
        points = np.vstack([self.points, np.asarray(point, dtype=float)])
        valid_all = np.append(self.valid, bool(valid))
        quantiles = np.append(self.quantiles, float(quantile))
        if self.mpds is None or self.distances is None:
            return RegionIndex(points, valid_all, quantiles)
        n = len(self.points)
        row = np.full(self.mpds.shape[1], np.nan) if mpd is None else np.asarray(mpd, dtype=float)
        distances = np.full((n + 1, n + 1), np.nan)
        distances[:n, :n] = self.distances
        if valid:
            rows = np.flatnonzero(self.valid)
            if rows.size:
                new = js_distances(row, self.mpds[rows])
                distances[n, rows] = new
                distances[rows, n] = new
            distances[n, n] = 0.0
        return RegionIndex(points, valid_all, quantiles, np.vstack([self.mpds, row]), distances)

    def coincident(self, point: np.ndarray, atol: float = COINCIDENCE_TOLERANCE) -> int | None:
        """Entry ID of an entry at ``point`` (every coordinate within ``atol``), if any."""
        if len(self.points) == 0:
            return None
        matches = np.flatnonzero(np.all(np.abs(self.points - np.asarray(point)) <= atol, axis=1))
        return int(matches[0]) if matches.size else None

    def locate(self, point: np.ndarray, tol: float = LOCATION_TOLERANCE) -> Location | None:
        """The tetrahedron containing ``point`` and its barycentric weights, or None outside the hull."""
        if self.delaunay is None:
            return None
        query = np.asarray(point, dtype=float)
        simplex_index = int(self.delaunay.find_simplex(query, tol=tol))
        if simplex_index < 0:
            return None
        vertices = np.asarray(self.delaunay.simplices[simplex_index], dtype=int).copy()
        weights = barycentric_weights(self.delaunay, simplex_index, query, tol)
        return Location(simplex_index=simplex_index, vertices=vertices, weights=weights)

    def simplex_is_valid(self, location: Location) -> bool:
        """Whether every vertex of the tetrahedron is a valid entry."""
        return bool(np.all(self.valid[location.vertices]))

    def interpolate_quantile(self, point: np.ndarray) -> float | None:
        """Piecewise-linear intrinsic quantile at ``point``.

        Returns None unless a fully valid tetrahedron contains the point.
        """
        location = self.locate(point)
        if location is None or not self.simplex_is_valid(location):
            return None
        return float(location.weights @ self.quantiles[location.vertices])

    def evaluate(self, location: Location) -> HitEvaluation:
        """Apply the hit rule in a fully valid tetrahedron."""
        if self.distances is None:
            raise RuntimeError("This index has no MPD distances; finalize the region first.")
        rows = location.vertices
        local = self.distances[np.ix_(rows, rows)]
        if not np.all(np.isfinite(local)):
            raise RuntimeError(f"Incomplete distance matrix for entries {rows.tolist()}.")
        # local[j, i] = d(v_j, v_i), so element i is sum_j lambda_j d(v_j, v_i).
        vertex_distances = location.weights @ local
        nearest = int(np.argmin(vertex_distances))
        interpolated = float(vertex_distances[nearest])
        query_quantile = float(location.weights @ self.quantiles[rows])
        matched_quantile = float(self.quantiles[rows[nearest]])
        threshold = max(query_quantile, matched_quantile)
        return HitEvaluation(
            vertex_distances=vertex_distances,
            matched_entry_id=int(rows[nearest]),
            interpolated_distance=interpolated,
            query_quantile=query_quantile,
            matched_quantile=matched_quantile,
            threshold=threshold,
            is_hit=bool(interpolated <= threshold),
        )
