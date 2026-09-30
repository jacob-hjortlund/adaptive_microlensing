"""Slices of a bank: its tetrahedral mesh cut by a plane, and the hit rule on a grid in a plane."""

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

AXES = ("kappa", "gamma", "s")
#: A vertex within this fraction of the domain's axis range from the plane lies on it.
PLANE_TOLERANCE = 1e-12
#: Slice triangles with at most this fraction of the plane's domain area are dropped.
AREA_TOLERANCE = 1e-12


def axis_range(domain: DomainSpec, axis: str) -> tuple[float, float]:
    """The domain's closed range on one axis."""
    ranges = {"kappa": domain.kappa_range, "gamma": domain.gamma_range, "s": domain.s_range}
    return ranges[axis]


def resolve_plane(
    domain: DomainSpec,
    *,
    s: float | None = None,
    kappa: float | None = None,
    gamma: float | None = None,
) -> tuple[str, float, tuple[str, str]]:
    """The plane's axis, its value and the two in-plane axes, from exactly one of s, kappa or gamma."""
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
    ``MapBank.query`` interpolates it.
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
    """Crossing edges (negative, positive) and triangles over them; vertices sorted negatives first."""
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
    """Cut a region's tetrahedral mesh with the plane given by exactly one of s, kappa or gamma."""
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
    along ``plane_axes[1]``.
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
    """Query an n x n grid over the domain box in the plane given by exactly one of s, kappa or gamma."""
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
