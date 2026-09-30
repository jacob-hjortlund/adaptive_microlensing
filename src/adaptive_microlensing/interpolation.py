"""Delaunay index of one region: locating queries and applying the hit rule.

:class:`RegionIndex` triangulates all of a region's entries, valid and invalid, with
:class:`scipy.spatial.Delaunay`. It keeps each entry's validity and intrinsic quantile
and, once the region is finalized, its bank MPD and the JS distances between the MPDs of
all valid entries. :meth:`RegionIndex.locate` finds the tetrahedron that contains a query
and returns it as a :class:`Location`, with weights from :func:`barycentric_weights`.
:meth:`RegionIndex.evaluate` applies the hit rule in that tetrahedron and returns a
:class:`HitEvaluation`.

:class:`MapBank` keeps one index per region. It builds the index when the region is
loaded, imported or finalized, and grows it with :meth:`RegionIndex.extended` each time
it commits an entry. :meth:`MapBank.query` answers queries with it, and
:meth:`MapBank.build` uses :meth:`RegionIndex.interpolate_quantile` to measure how well
the entries so far predict the intrinsic quantile of each new entry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from .mpd import js_distances, pairwise_js_distances

#: Default tolerance of :meth:`RegionIndex.locate`, in barycentric weight.
LOCATION_TOLERANCE = 1e-10
#: Default tolerance of :meth:`RegionIndex.coincident`, on each coordinate.
COINCIDENCE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Location:
    """A query expressed in its containing tetrahedron.

    :meth:`RegionIndex.locate` makes it, and :meth:`RegionIndex.evaluate` and
    :meth:`MapBank.query` read it.

    Parameters
    ----------
    simplex_index : int
        Index of the tetrahedron in the index's Delaunay triangulation.
    vertices : numpy.ndarray
        Integer array of shape ``(4,)``: the entry IDs (row positions in
        :attr:`RegionIndex.points`) of the tetrahedron's vertices, in the
        triangulation's vertex order.
    weights : numpy.ndarray
        Array of shape ``(4,)`` of the query's barycentric weights, in the order of
        ``vertices``. They are non-negative and sum to one.
    """

    simplex_index: int
    vertices: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class HitEvaluation:
    """Outcome of the hit rule in one tetrahedron.

    The matched entry is the passing vertex nearest in MPD distance on a hit, and the
    vertex closest to passing on a miss. ``interpolated_distance``,
    ``matched_quantile``, ``threshold`` and :attr:`margin` describe that vertex.
    :meth:`RegionIndex.evaluate` makes it, and :meth:`MapBank.query` copies it into a
    :class:`QueryResult`.

    Parameters
    ----------
    vertex_distances : numpy.ndarray
        Array of shape ``(4,)`` of the interpolated JS distances from the query to each
        vertex, in the order of :attr:`Location.vertices`.
    matched_entry_id : int
        Entry ID of the matched vertex.
    interpolated_distance : float
        Interpolated JS distance from the query to the matched vertex.
    query_quantile : float
        Intrinsic quantile at the query: the barycentric interpolation of the vertices'
        intrinsic quantiles.
    matched_quantile : float
        Intrinsic quantile of the matched entry.
    threshold : float
        Largest distance at which the matched vertex passes,
        ``max(query_quantile, matched_quantile)``.
    is_hit : bool
        Whether any vertex passes the hit rule.
    """

    vertex_distances: np.ndarray
    matched_entry_id: int
    interpolated_distance: float
    query_quantile: float
    matched_quantile: float
    threshold: float
    is_hit: bool

    @property
    def margin(self) -> float:
        """Amount by which the matched vertex's distance lies below its threshold.

        It is ``threshold - interpolated_distance``: zero or positive on a hit. On a miss
        it is negative, and minus the smallest gap to a hit over the four vertices.
        """
        return self.threshold - self.interpolated_distance


def barycentric_weights(delaunay: Delaunay, simplex_index: int, point: np.ndarray, tol: float) -> np.ndarray:
    """Return the barycentric weights of ``point`` in a simplex, cleaned up at faces, edges and vertices.

    The weights come from the simplex's affine transform in ``delaunay.transform``.
    Weights within ``tol`` of zero or one are snapped to exactly zero or one, so a point
    on a face, an edge or a vertex gets exact weights, and the weights are then divided
    by their sum.

    Parameters
    ----------
    delaunay : scipy.spatial.Delaunay
        Triangulation that contains the simplex.
    simplex_index : int
        Index of the simplex in ``delaunay``, as returned by
        :meth:`scipy.spatial.Delaunay.find_simplex`.
    point : numpy.ndarray
        Point of shape ``(ndim,)``, inside the simplex or within ``tol`` of it.
    tol : float
        Tolerance in barycentric weight. Weights within ``tol`` of 0 or 1 are snapped,
        and a weight below ``-tol`` is an error.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(ndim + 1,)`` of non-negative weights that sum to one, in the
        vertex order of ``delaunay.simplices[simplex_index]``.

    Raises
    ------
    RuntimeError
        If a weight is not finite, as for a degenerate simplex, whose transform scipy
        fills with NaN; if a weight is below ``-tol``, so that the point lies outside the
        simplex; or if the cleaned-up weights do not have a positive sum.
    """
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

    Invalid entries stay in the mesh, and a tetrahedron with an invalid vertex is never
    used to interpolate (see :meth:`simplex_is_valid`), so the index does not interpolate
    across places where map generation failed. The index locates queries, interpolates
    the intrinsic quantile and, once the region is finalized and its MPDs are known,
    applies the hit rule. :class:`MapBank` keeps one index per region.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape ``(n, 3)`` of the ``(kappa, gamma, s)`` of every entry, in
        ``entry_id`` order, so that row positions are entry IDs.
    valid : numpy.ndarray
        Boolean array of shape ``(n,)``, true for the valid entries.
    quantiles : numpy.ndarray
        Array of shape ``(n,)`` of each entry's intrinsic MPD-distance quantile, NaN if
        the entry is invalid.
    mpds : numpy.ndarray or None, optional
        Bank MPDs of shape ``(n, m)``, NaN in the rows of invalid entries. Without them,
        and without ``distances``, the index can locate points and interpolate
        quantiles, but cannot apply the hit rule. Default is ``None``.
    distances : numpy.ndarray or None, optional
        Precomputed JS distances of shape ``(n, n)`` between the MPDs. When ``None``,
        they are computed from ``mpds`` with
        :func:`~adaptive_microlensing.mpd.pairwise_js_distances` if ``mpds`` is given.

    Attributes
    ----------
    points : numpy.ndarray
        ``points`` as a float array of shape ``(n, 3)``.
    valid : numpy.ndarray
        ``valid`` as a boolean array of shape ``(n,)``.
    quantiles : numpy.ndarray
        ``quantiles`` as a float array of shape ``(n,)``.
    mpds : numpy.ndarray or None
        ``mpds`` as a float array of shape ``(n, m)``, or ``None``.
    distances : numpy.ndarray or None
        JS distances of shape ``(n, n)`` between the entries' MPDs, as given or as
        computed from ``mpds``, with NaN wherever an invalid entry is involved. ``None``
        when neither ``mpds`` nor ``distances`` was given.
    ready : bool
        Whether there are at least four points and they span three dimensions, so that
        they can be triangulated. A region answers queries only when its index is ready.
    delaunay : scipy.spatial.Delaunay or None
        Triangulation of ``points``, or ``None`` when the index is not ready.
    """

    def __init__(
        self,
        points: np.ndarray,
        valid: np.ndarray,
        quantiles: np.ndarray,
        mpds: np.ndarray | None = None,
        distances: np.ndarray | None = None,
    ) -> None:
        """Convert the inputs, compute any missing distances and triangulate the points when ready."""
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
        """Return a new index with one more entry appended, computing only the new distances.

        The new entry takes the next row position, which is its entry ID. When this index
        has MPDs and distances, its distances are copied and only those between the new
        entry and the valid old entries are computed, with
        :func:`~adaptive_microlensing.mpd.js_distances`, instead of the whole matrix. The
        triangulation is rebuilt from all the points. This index is not changed. The
        bank calls it each time it commits an entry.

        Parameters
        ----------
        point : numpy.ndarray
            ``(kappa, gamma, s)`` of the new entry, of shape ``(3,)``.
        valid : bool
            Whether the new entry is valid.
        quantile : float
            Intrinsic quantile of the new entry, NaN if it is invalid.
        mpd : numpy.ndarray or None, optional
            Bank MPD of the new entry, of shape ``(m,)``. When ``None``, a row of NaN is
            stored, and a valid entry then gets NaN distances. It is ignored when this
            index has no MPDs or no distances.

        Returns
        -------
        RegionIndex
            Index of the ``n + 1`` entries. It has MPDs and distances only when this
            index has both. A valid new entry is at distance zero from itself; an invalid
            one has NaN in its row and column.
        """
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
        """Return the entry ID of an entry at ``point``, if there is one.

        An entry is at ``point`` when each of its coordinates is within ``atol`` of the
        point's. The check does not use the triangulation, so it also works on an index
        that is not ready. :meth:`MapBank.query` calls it before :meth:`locate`, so a
        query at an existing entry is answered by that entry.

        Parameters
        ----------
        point : numpy.ndarray
            ``(kappa, gamma, s)`` of shape ``(3,)``.
        atol : float, optional
            Absolute tolerance on each coordinate. Default is ``COINCIDENCE_TOLERANCE``,
            ``1e-12``.

        Returns
        -------
        int or None
            The lowest entry ID of an entry at ``point``, or ``None`` when there is none.
        """
        if len(self.points) == 0:
            return None
        matches = np.flatnonzero(np.all(np.abs(self.points - np.asarray(point)) <= atol, axis=1))
        return int(matches[0]) if matches.size else None

    def locate(self, point: np.ndarray, tol: float = LOCATION_TOLERANCE) -> Location | None:
        """Return the tetrahedron that contains ``point`` and the point's barycentric weights in it.

        The search uses :meth:`scipy.spatial.Delaunay.find_simplex` with tolerance
        ``tol``, so a point up to ``tol`` outside a tetrahedron, in barycentric weight,
        still counts as inside it, and a point on the boundary of the mesh is found
        despite rounding. The same ``tol`` cleans up the weights; see
        :func:`barycentric_weights`.

        Parameters
        ----------
        point : numpy.ndarray
            ``(kappa, gamma, s)`` of shape ``(3,)``.
        tol : float, optional
            Tolerance in barycentric weight. Default is ``LOCATION_TOLERANCE``,
            ``1e-10``.

        Returns
        -------
        Location or None
            The containing tetrahedron, its vertices and the weights, or ``None`` when
            the index is not ready or the point lies outside the convex hull of the
            entries.

        Raises
        ------
        RuntimeError
            If the weights cannot be cleaned up; see :func:`barycentric_weights`.
        """
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
        """Return whether every vertex of the tetrahedron is a valid entry.

        Only a fully valid tetrahedron is used to interpolate the intrinsic quantile or
        to apply the hit rule.

        Parameters
        ----------
        location : Location
            Tetrahedron from :meth:`locate`.

        Returns
        -------
        bool
            ``True`` when all four vertices are valid entries.
        """
        return bool(np.all(self.valid[location.vertices]))

    def interpolate_quantile(self, point: np.ndarray) -> float | None:
        """Return the piecewise-linear intrinsic quantile at ``point``.

        The quantile is the barycentric interpolation of the vertices' intrinsic
        quantiles in the tetrahedron that contains the point. :meth:`MapBank.build`
        compares it with the quantile measured at each new point to get the entry's
        residuals.

        Parameters
        ----------
        point : numpy.ndarray
            ``(kappa, gamma, s)`` of shape ``(3,)``.

        Returns
        -------
        float or None
            The interpolated quantile. ``None`` unless a fully valid tetrahedron contains
            the point, which includes the case of an index that is not ready.

        Raises
        ------
        RuntimeError
            If the weights cannot be cleaned up; see :func:`barycentric_weights`.
        """
        location = self.locate(point)
        if location is None or not self.simplex_is_valid(location):
            return None
        return float(location.weights @ self.quantiles[location.vertices])

    def evaluate(self, location: Location) -> HitEvaluation:
        """Apply the hit rule in a fully valid tetrahedron.

        A vertex passes when its interpolated distance is at most the larger of the
        query's and its own intrinsic quantile. On a hit the match is the passing vertex
        with the smallest distance; on a miss it is the vertex closest to passing.
        :meth:`MapBank.query` calls it once it has checked that the tetrahedron is fully
        valid.

        Parameters
        ----------
        location : Location
            Tetrahedron that contains the query, from :meth:`locate`.

        Returns
        -------
        HitEvaluation
            Whether the query is a hit, the matched vertex, and the distances, quantiles
            and threshold that decided it.

        Raises
        ------
        RuntimeError
            If the index has no MPD distances, because the region is not finalized, or
            if a distance between the tetrahedron's vertices is not finite, as when a
            vertex is invalid.

        Notes
        -----
        Write ``w_j`` for the barycentric weights of the vertices ``v_j``, ``q_j`` for
        their intrinsic quantiles and ``d(v_j, v_i)`` for the JS distances between their
        MPDs. The interpolated distance from the query to vertex ``i`` is
        ``d_i = sum_j w_j d(v_j, v_i)``, the query's quantile is ``q = sum_j w_j q_j``,
        and vertex ``i`` passes when ``d_i <= max(q, q_i)``: the query lies within the
        noise of either map. On a miss, the reported vertex is the one with the largest
        ``max(q, q_i) - d_i``, so :attr:`HitEvaluation.margin` is negative.

        Every vertex is tested, so a hit may match a vertex that is not the nearest.
        This deliberately differs from the legacy script, which tested only the vertex
        with the smallest interpolated distance.
        """
        if self.distances is None:
            raise RuntimeError("This index has no MPD distances; finalize the region first.")
        rows = location.vertices
        local = self.distances[np.ix_(rows, rows)]
        if not np.all(np.isfinite(local)):
            raise RuntimeError(f"Incomplete distance matrix for entries {rows.tolist()}.")
        # local[j, i] = d(v_j, v_i), so element i is sum_j lambda_j d(v_j, v_i).
        vertex_distances = location.weights @ local
        query_quantile = float(location.weights @ self.quantiles[rows])
        thresholds = np.maximum(query_quantile, self.quantiles[rows])
        passing = vertex_distances <= thresholds
        is_hit = bool(passing.any())
        if is_hit:
            matched = int(np.argmin(np.where(passing, vertex_distances, np.inf)))
        else:
            matched = int(np.argmax(thresholds - vertex_distances))
        return HitEvaluation(
            vertex_distances=vertex_distances,
            matched_entry_id=int(rows[matched]),
            interpolated_distance=float(vertex_distances[matched]),
            query_quantile=query_quantile,
            matched_quantile=float(self.quantiles[rows[matched]]),
            threshold=float(thresholds[matched]),
            is_hit=is_hit,
        )
