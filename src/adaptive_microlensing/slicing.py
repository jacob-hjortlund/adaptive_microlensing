"""Slices of a bank: its tetrahedral mesh cut by a plane, and the hit rule on a grid in a plane.

A plane holds one of the coordinates ``kappa``, ``gamma`` or ``s`` fixed at a value inside
the bank's domain (:func:`resolve_plane`). The other two coordinates, in the order of
``AXES``, are its in-plane axes. Two views of a bank in such a plane are provided:

- :func:`slice_region` cuts one region's Delaunay mesh of entries with the plane. The
  result, a :class:`RegionSlice`, is a triangle mesh of the cross-section that carries the
  intrinsic quantile, interpolated exactly as :meth:`MapBank.query` interpolates it. It
  reads only the region's in-memory index, so the region need not be finalized.
- :func:`coverage_grid` runs :meth:`MapBank.query_many` on a regular grid of points in the
  plane. The result, a :class:`CoverageGrid`, holds each node's status, region, hit flag,
  margin and interpolated distance.

Neither writes to the bank. :mod:`adaptive_microlensing.plotting` draws both views.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import DomainSpec

if TYPE_CHECKING:
    from .bank import MapBank

logger = logging.getLogger("adaptive_microlensing")

#: Names of the three coordinates of a bank point, in the column order of its
#: ``(kappa, gamma, s)`` arrays.
AXES = ("kappa", "gamma", "s")
#: A vertex within this fraction of the domain's axis range from the plane lies on it.
PLANE_TOLERANCE = 1e-12
#: Slice triangles with at most this fraction of the plane's domain area are dropped.
AREA_TOLERANCE = 1e-12


def axis_range(domain: DomainSpec, axis: str) -> tuple[float, float]:
    """Return the domain's closed range on one axis.

    The range is ``kappa_range``, ``gamma_range`` or ``s_range`` of ``domain``. Only the
    box enters; the magnification cap does not.

    Parameters
    ----------
    domain : DomainSpec
        Domain of the bank.
    axis : str
        Name of the axis, one of ``AXES``.

    Returns
    -------
    tuple of (float, float)
        ``(lower, upper)`` bounds of the axis. The domain includes both.

    Raises
    ------
    KeyError
        If ``axis`` is not one of ``AXES``.
    """
    ranges = {"kappa": domain.kappa_range, "gamma": domain.gamma_range, "s": domain.s_range}
    return ranges[axis]


def resolve_plane(
    domain: DomainSpec,
    *,
    s: float | None = None,
    kappa: float | None = None,
    gamma: float | None = None,
) -> tuple[str, float, tuple[str, str]]:
    """Return the plane's axis, its value and the two in-plane axes, from exactly one of s, kappa or gamma.

    This validates the plane arguments of :func:`slice_region` and :func:`coverage_grid`.
    The plane holds the given coordinate fixed, and the other two coordinates span it.

    Parameters
    ----------
    domain : DomainSpec
        Domain of the bank. The value must lie in its closed range on the plane's axis.
    s : float or None, optional
        Smooth-matter fraction of an ``s`` plane.
    kappa : float or None, optional
        Total convergence of a ``kappa`` plane.
    gamma : float or None, optional
        Shear of a ``gamma`` plane.

    Returns
    -------
    axis : str
        Name of the fixed coordinate, one of ``AXES``.
    value : float
        Value of that coordinate on the plane.
    plane_axes : tuple of (str, str)
        The other two axes, in the order of ``AXES``: ``("kappa", "gamma")`` for an ``s``
        plane, ``("gamma", "s")`` for a ``kappa`` plane and ``("kappa", "s")`` for a
        ``gamma`` plane.

    Raises
    ------
    ValueError
        If not exactly one of ``s``, ``kappa`` and ``gamma`` is given, or if the value is
        not finite or lies outside the domain's range on its axis. Only the box is checked,
        not the magnification cap.
    """
    candidates = (("kappa", kappa), ("gamma", gamma), ("s", s))
    given = {name: value for name, value in candidates if value is not None}
    if len(given) != 1:
        raise ValueError(f"Give exactly one of s, kappa or gamma; got {sorted(given) or 'none'}.")
    ((axis, raw),) = given.items()
    value = float(raw)
    lower, upper = axis_range(domain, axis)
    if not (math.isfinite(value) and lower <= value <= upper):
        raise ValueError(f"{axis}={raw!r} lies outside the domain range [{lower}, {upper}].")
    first, second = (name for name in AXES if name != axis)
    return axis, value, (first, second)


@dataclass(frozen=True)
class RegionSlice:
    """One region's tetrahedral mesh cut by the plane ``axis = value``.

    ``points`` are in-plane coordinates along ``plane_axes``. Each triangle is part of
    one tetrahedron, ``simplex_index``, and the quantile is linear on it, exactly as
    ``MapBank.query`` interpolates it. Made by :func:`slice_region`. The triangles cover
    the mesh's cross-section once, and neighbouring triangles share their points.

    Parameters
    ----------
    region : str
        Name of the sliced region, one of ``REGIONS``.
    axis : str
        The coordinate held fixed, one of ``AXES``.
    value : float
        Value of ``axis`` on the plane.
    plane_axes : tuple of (str, str)
        The other two coordinates, in the order of ``AXES``. They are the columns of
        ``points``.
    points : numpy.ndarray
        Array of shape ``(p, 2)`` of the slice's points, in in-plane coordinates. Each is
        where an edge of the mesh crosses the plane, or an entry that lies on the plane.
    triangles : numpy.ndarray
        Integer array of shape ``(t, 3)`` of indices into ``points``. Each triangle is one
        tetrahedron's whole cross-section, or one of the two triangles into which a
        quadrilateral cross-section is split.
    quantiles : numpy.ndarray
        Array of shape ``(p,)`` of the intrinsic quantile at each point: an entry's own
        quantile at an entry, and otherwise the linear interpolation between the two ends
        of the crossing edge, which is NaN when either end is an invalid entry.
    valid : numpy.ndarray
        Boolean array of shape ``(t,)``, ``True`` where all four vertices of the triangle's
        tetrahedron are valid entries. This is the condition for :meth:`MapBank.query` to
        apply the hit rule in the tetrahedron rather than answer ``"invalid_simplex"``.
    simplex_index : numpy.ndarray
        Integer array of shape ``(t,)`` of the index of each triangle's tetrahedron in the
        region's Delaunay mesh, as reported by ``QueryResult.simplex_index``.
    """

    region: str
    axis: str
    value: float
    plane_axes: tuple[str, str]
    points: np.ndarray
    triangles: np.ndarray
    quantiles: np.ndarray
    valid: np.ndarray
    simplex_index: np.ndarray


def _cut_edges(n_negative: int) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Return a tetrahedron's edges that cross the plane and the triangles that their crossings span.

    The tetrahedron's vertices are numbered ``0`` to ``3``, with the ``n_negative``
    vertices on the negative side of the plane first (see :func:`slice_region`). An edge
    crosses the plane when it joins a negative vertex to a positive one, and the crossing
    points of all such edges make the tetrahedron's cross-section: a triangle when one or
    three vertices are negative, and a convex quadrilateral, split into two triangles, when
    two are.

    Parameters
    ----------
    n_negative : int
        Number of vertices on the negative side: 1, 2 or 3.

    Returns
    -------
    edges : list of tuple of (int, int)
        The crossing edges, as ``(negative, positive)`` pairs of vertex numbers.
    triangles : list of tuple of (int, int, int)
        The cross-section's triangles, as triples of positions in ``edges``.

    Notes
    -----
    With negative vertices ``a, b`` and positive ``c, d``, the crossings on ``ac``, ``ad``,
    ``bd`` and ``bc`` are listed in order around the quadrilateral, which is split along
    its diagonal from the ``ac`` crossing to the ``bd`` crossing.
    """
    if n_negative == 1:
        return [(0, 1), (0, 2), (0, 3)], [(0, 1, 2)]
    if n_negative == 3:
        return [(0, 3), (1, 3), (2, 3)], [(0, 1, 2)]
    # Negatives a, b and positives c, d: the crossings on ac, ad, bd, bc form a convex quadrilateral.
    return [(0, 2), (0, 3), (1, 3), (1, 2)], [(0, 1, 2), (0, 2, 3)]


def slice_region(
    bank: MapBank,
    region: str,
    *,
    s: float | None = None,
    kappa: float | None = None,
    gamma: float | None = None,
) -> RegionSlice:
    """Cut a region's tetrahedral mesh with the plane given by exactly one of s, kappa or gamma.

    The mesh is the Delaunay tetrahedralisation of all the region's entries, valid and
    invalid, in which :meth:`MapBank.query` locates queries. Every tetrahedron that the
    plane crosses contributes its cross-section, as one or two triangles, and the intrinsic
    quantile is interpolated linearly onto the crossing points. On a triangle of a fully
    valid tetrahedron, linear interpolation between its corners then gives the quantile
    that the bank interpolates at the same 3D point. Only the region's in-memory index is
    read, so the region need not be finalized, and nothing is written.

    Parameters
    ----------
    bank : MapBank
        The bank to slice. It may be open read-only.
    region : str
        Name of the region, one of ``REGIONS``.
    s : float or None, optional
        Slice at this smooth-matter fraction.
    kappa : float or None, optional
        Slice at this total convergence.
    gamma : float or None, optional
        Slice at this shear.

    Returns
    -------
    RegionSlice
        The cross-section. Its arrays are empty when the region has no mesh (fewer than
        four entries, or entries that do not span three dimensions) or when the plane
        misses the mesh.

    Raises
    ------
    ValueError
        If not exactly one of ``s``, ``kappa`` and ``gamma`` is given, if the value is not
        finite or lies outside the domain's range on its axis, or if ``region`` is not one
        of ``REGIONS``.

    Notes
    -----
    An entry within ``PLANE_TOLERANCE`` times the axis range of the plane counts as lying
    on it. Each entry is then on the negative or the positive side: entries below the plane
    are negative and entries on or above it are positive, except when no entry lies below
    the plane, in which case the entries on it are negative. This symbolic perturbation
    makes a face that lies in the plane come from exactly one of its two tetrahedra, and
    gives slices at the bottom and top of the mesh.

    Along an edge from a negative entry ``a`` to a positive entry ``b``, with signed
    distances ``d_a`` and ``d_b`` from the plane, the crossing lies at the fraction
    ``t = d_a / (d_a - d_b)`` of the way from ``a``, and both the in-plane point and the
    quantile are interpolated with this ``t``. A crossing at ``t = 0`` or ``t = 1`` is the
    entry itself. Crossings on the same edge, or at the same entry, are merged, so that
    neighbouring triangles share points. Triangles with an area of at most
    ``AREA_TOLERANCE`` times the area of the domain's box in the plane are dropped, and
    points that no triangle uses are removed.
    """
    domain = bank.config.domain
    axis, value, plane_axes = resolve_plane(domain, s=s, kappa=kappa, gamma=gamma)
    index = bank._region(region).index
    empty = RegionSlice(
        region,
        axis,
        value,
        plane_axes,
        np.empty((0, 2)),
        np.empty((0, 3), dtype=int),
        np.empty(0),
        np.empty(0, dtype=bool),
        np.empty(0, dtype=int),
    )
    if index.delaunay is None:
        return empty

    lower, upper = axis_range(domain, axis)
    points3d = index.points
    plane = points3d[:, [AXES.index(name) for name in plane_axes]]
    distance = points3d[:, AXES.index(axis)] - value
    distance[np.abs(distance) <= PLANE_TOLERANCE * (upper - lower)] = 0.0
    # Symbolic perturbation: a vertex on the plane counts as above it, except at the bottom
    # of the mesh, where nothing lies below and it counts as below. A face lying in the plane
    # is then produced by exactly one of its two tetrahedra, and bottom and top slices exist.
    positive = distance >= 0.0 if np.any(distance < 0.0) else distance > 0.0

    simplices = np.asarray(index.delaunay.simplices, dtype=int)
    n_positive = positive[simplices].sum(axis=1)
    negatives: list[np.ndarray] = []
    positives: list[np.ndarray] = []
    triangle_edges: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    offset = 0
    for n_negative in (1, 2, 3):
        rows = np.flatnonzero(n_positive == 4 - n_negative)
        if rows.size == 0:
            continue
        # Stable sort puts each tetrahedron's negative vertices first.
        order = np.argsort(positive[simplices[rows]], axis=1, kind="stable")
        vertices = np.take_along_axis(simplices[rows], order, axis=1)
        edges, triangles = _cut_edges(n_negative)
        n_edges = len(edges)
        negatives.append(vertices[:, [edge[0] for edge in edges]].ravel())
        positives.append(vertices[:, [edge[1] for edge in edges]].ravel())
        base = offset + n_edges * np.arange(rows.size)[:, np.newaxis]
        for triangle in triangles:
            triangle_edges.append(base + np.asarray(triangle))
            sources.append(rows)
        offset += n_edges * rows.size
    if not negatives:
        return empty

    neg = np.concatenate(negatives)
    pos = np.concatenate(positives)
    t = distance[neg] / (distance[neg] - distance[pos])
    # A crossing at t = 0 or 1 is the vertex itself; key it by the vertex so tetrahedra share it.
    n = len(points3d)
    at_neg, at_pos = t == 0.0, t == 1.0
    keys = np.where(at_neg, neg, np.where(at_pos, pos, n + np.minimum(neg, pos) * n + np.maximum(neg, pos)))
    unique_keys, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    t_first = t[first][:, np.newaxis]
    neg_first, pos_first = neg[first], pos[first]
    crossing = plane[neg_first] + t_first * (plane[pos_first] - plane[neg_first])
    q = index.quantiles
    crossing_q = q[neg_first] + t_first[:, 0] * (q[pos_first] - q[neg_first])
    on_vertex = unique_keys < n
    crossing[on_vertex] = plane[unique_keys[on_vertex]]
    crossing_q[on_vertex] = q[unique_keys[on_vertex]]

    triangles = inverse.ravel()[np.concatenate(triangle_edges)]
    simplex_index = np.concatenate(sources)
    corners = crossing[triangles]
    edge_a, edge_b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    box = (axis_range(domain, plane_axes[0]), axis_range(domain, plane_axes[1]))
    keep = area > AREA_TOLERANCE * (box[0][1] - box[0][0]) * (box[1][1] - box[1][0])
    triangles, simplex_index = triangles[keep], simplex_index[keep]

    used, compact = np.unique(triangles, return_inverse=True)
    return RegionSlice(
        region=region,
        axis=axis,
        value=value,
        plane_axes=plane_axes,
        points=crossing[used],
        triangles=compact.reshape(-1, 3),
        quantiles=crossing_q[used],
        valid=np.all(index.valid[simplices[simplex_index]], axis=1),
        simplex_index=simplex_index,
    )


@dataclass(frozen=True)
class CoverageGrid:
    """The hit rule evaluated by ``MapBank.query_many`` on an n x n grid in a plane.

    The 2D arrays are indexed ``[iy, ix]``, with ``x`` along ``plane_axes[0]`` and ``y``
    along ``plane_axes[1]``. Made by :func:`coverage_grid`. Node ``[iy, ix]`` is the point
    ``(x[ix], y[iy])`` of the plane and row ``iy * n + ix`` of ``table``.

    Parameters
    ----------
    axis : str
        The coordinate held fixed, one of ``AXES``.
    value : float
        Value of ``axis`` on the plane.
    plane_axes : tuple of (str, str)
        The other two coordinates, in the order of ``AXES``.
    domain : DomainSpec
        Domain of the bank. The grid spans its box in the plane.
    x : numpy.ndarray
        Array of shape ``(n,)`` of node coordinates along ``plane_axes[0]``, evenly spaced
        over the domain's closed range on that axis.
    y : numpy.ndarray
        Array of shape ``(n,)`` of node coordinates along ``plane_axes[1]``, evenly spaced
        over the domain's closed range on that axis.
    status : numpy.ndarray
        String array of shape ``(n, n)`` of each node's :class:`QueryStatus` value, such as
        ``"hit"``, ``"miss"``, ``"outside_hull"`` or ``"outside_domain"``.
    region : numpy.ndarray
        Object array of shape ``(n, n)`` of the region that each node is classified into,
        or ``None`` on a critical line. Nodes outside the domain still have a region.
    is_hit : numpy.ndarray
        Boolean array of shape ``(n, n)``, ``True`` where the query is a hit.
    margin : numpy.ndarray
        Float array of shape ``(n, n)`` of the hit rule's ``threshold - distance`` for the
        matched vertex: at least zero on a hit, negative on a miss and NaN for every other
        status.
    interpolated_distance : numpy.ndarray
        Float array of shape ``(n, n)`` of the interpolated JS distance from each node to
        its matched vertex: ``0`` where the node coincides with a valid entry, and NaN for
        statuses other than ``"hit"`` and ``"miss"``.
    table : pandas.DataFrame
        The full output of :meth:`MapBank.query_many`, one row per node: the node's
        coordinates, in columns named after the axes, followed by the query result
        columns.
    """

    axis: str
    value: float
    plane_axes: tuple[str, str]
    domain: DomainSpec
    x: np.ndarray
    y: np.ndarray
    status: np.ndarray
    region: np.ndarray
    is_hit: np.ndarray
    margin: np.ndarray
    interpolated_distance: np.ndarray
    table: pd.DataFrame


def coverage_grid(
    bank: MapBank,
    *,
    s: float | None = None,
    kappa: float | None = None,
    gamma: float | None = None,
    n: int = 200,
) -> CoverageGrid:
    """Query an n x n grid over the domain box in the plane given by exactly one of s, kappa or gamma.

    The grid spans the domain's box in the two in-plane axes, with ``n`` evenly spaced
    nodes along each, both ends included. Every node is queried with
    :meth:`MapBank.query_many`, with the domain enforced, so nodes beyond the magnification
    cap are ``"outside_domain"`` and nodes on a critical line are ``"critical_line"``. This
    makes ``n**2`` queries, logs their number and writes nothing.

    Parameters
    ----------
    bank : MapBank
        The bank to query. It may be open read-only.
    s : float or None, optional
        Put the grid in the plane at this smooth-matter fraction.
    kappa : float or None, optional
        Put the grid in the plane at this total convergence.
    gamma : float or None, optional
        Put the grid in the plane at this shear.
    n : int, optional
        Number of nodes along each in-plane axis, an integer of at least 2. Default is
        ``200``.

    Returns
    -------
    CoverageGrid
        The query results on the grid.

    Raises
    ------
    ValueError
        If not exactly one of ``s``, ``kappa`` and ``gamma`` is given, if the value is not
        finite or lies outside the domain's range on its axis, or if ``n`` is not an
        integer (a ``bool`` is refused) or is less than 2.
    """
    domain = bank.config.domain
    axis, value, plane_axes = resolve_plane(domain, s=s, kappa=kappa, gamma=gamma)
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n!r}.")
    x = np.linspace(*axis_range(domain, plane_axes[0]), int(n))
    y = np.linspace(*axis_range(domain, plane_axes[1]), int(n))
    xx, yy = np.meshgrid(x, y)
    nodes = pd.DataFrame({plane_axes[0]: xx.ravel(), plane_axes[1]: yy.ravel()})
    nodes[axis] = value
    logger.info("Coverage grid: %d queries at %s = %g", len(nodes), axis, value)
    table = bank.query_many(nodes)
    shape = xx.shape
    region = table["query_region"].astype(object)
    return CoverageGrid(
        axis=axis,
        value=value,
        plane_axes=plane_axes,
        domain=domain,
        x=x,
        y=y,
        status=table["interpolation_status"].to_numpy(dtype=str).reshape(shape),
        region=region.where(region.notna(), None).to_numpy(dtype=object).reshape(shape),
        is_hit=table["is_hit"].to_numpy(dtype=bool).reshape(shape),
        margin=table["distance_margin"].to_numpy(dtype=float).reshape(shape),
        interpolated_distance=table["interpolated_mpd_distance"].to_numpy(dtype=float).reshape(shape),
        table=table,
    )
