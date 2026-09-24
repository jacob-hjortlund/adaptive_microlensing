"""Macro-lens quantities, image classification and region hulls."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import ConvexHull

from .config import DomainSpec

REGIONS = ("minima", "saddle", "maxima")
CRITICAL = "critical"
CRITICAL_TOLERANCE = 1e-10


def macro_magnification(kappa: float, gamma: float) -> float:
    """Signed macro magnification ``1 / ((1 - kappa)**2 - gamma**2)``; ``inf`` on a critical line.

    >>> macro_magnification(0.5, 0.0)
    4.0
    >>> macro_magnification(1.5, 0.5)
    inf
    """
    denominator = (1.0 - kappa) ** 2 - gamma**2
    if denominator == 0.0:
        return float("inf")
    return 1.0 / denominator


def classify_image(kappa: float, gamma: float, *, tol: float = CRITICAL_TOLERANCE) -> str:
    """Classify a macro image as ``"minima"``, ``"saddle"``, ``"maxima"`` or ``"critical"``.

    >>> classify_image(0.4, 0.3)
    'minima'
    >>> classify_image(1.0, 0.5)
    'saddle'
    >>> classify_image(1.8, 0.3)
    'maxima'
    >>> classify_image(0.5, 0.5)
    'critical'
    """
    kappa = float(kappa)
    gamma = abs(float(gamma))
    if not (np.isfinite(kappa) and np.isfinite(gamma)):
        raise ValueError(f"Expected finite kappa and gamma, got {kappa=} and {gamma=}.")
    radial = 1.0 - kappa
    if abs(abs(radial) - gamma) <= tol:
        return CRITICAL
    if radial > gamma:
        return "minima"
    if -radial > gamma:
        return "maxima"
    return "saddle"


def in_box(kappa: float, gamma: float, s: float, domain: DomainSpec) -> bool:
    """Whether the point lies in the domain's closed kappa/gamma/s box (gamma is not made absolute)."""
    return bool(
        domain.kappa_range[0] <= kappa <= domain.kappa_range[1]
        and domain.gamma_range[0] <= gamma <= domain.gamma_range[1]
        and domain.s_range[0] <= s <= domain.s_range[1]
    )


def in_domain(kappa: float, gamma: float, s: float, domain: DomainSpec) -> bool:
    """Whether the point lies in the box and satisfies ``|mu_macro| <= max_macro_magnification``."""
    return in_box(kappa, gamma, s, domain) and (
        abs(macro_magnification(kappa, gamma)) <= domain.max_macro_magnification
    )


def _hyperbolic_samples(lo: float, hi: float, eps: float, n: int) -> np.ndarray:
    """Sample [lo, hi] along the natural parameter of the boundary hyperbola.

    Writing the offset from the triple point as sqrt(eps) * sinh(t) spaces points
    tightly near the vertex, where the curvature lives, and loosely out along the
    asymptote, where the boundary is already almost straight. Endpoints are exact.
    """
    if n < 2:
        return np.array([lo, hi], dtype=float)
    if eps <= 0.0:
        return np.linspace(lo, hi, n)
    scale = np.sqrt(eps)
    t = np.linspace(np.arcsinh(lo / scale), np.arcsinh(hi / scale), n)
    return scale * np.sinh(t)


def _drop_collinear(vertices: np.ndarray, rel_tol: float = 1e-9) -> np.ndarray:
    """Remove hull vertices that sit on the chord between their neighbours.

    Every hull vertex becomes an evaluated bank entry before adaptive sampling
    starts, and each evaluation costs a full magnification map.
    """
    verts = np.asarray(vertices, dtype=float)
    if len(verts) < 4:
        return verts

    scale = np.ptp(verts, axis=0).max()
    tol = rel_tol * max(scale, 1.0)
    keep = np.ones(len(verts), dtype=bool)

    for i in range(len(verts)):
        prev = verts[(i - 1) % len(verts)]
        nxt = verts[(i + 1) % len(verts)]
        edge = nxt - prev
        norm = np.hypot(*edge)
        if norm == 0.0:
            continue
        offset = verts[i] - prev
        # Scalar z-component of the cross product of two 2D vectors.
        cross_z = edge[0] * offset[1] - edge[1] * offset[0]
        if abs(cross_z) / norm <= tol:
            keep[i] = False

    return verts[keep] if keep.sum() >= 3 else verts


def _satisfies(region: str, vertices: np.ndarray, eps: float, atol: float = 1e-9) -> bool:
    """Invariant check: every vertex is in the region and passes the mu cut."""
    kappa, gamma = np.asarray(vertices, dtype=float).T
    if region == "minima":
        return bool(np.all((1.0 - kappa) + atol >= np.sqrt(gamma**2 + eps)))
    if region == "maxima":
        return bool(np.all((kappa - 1.0) + atol >= np.sqrt(gamma**2 + eps)))
    return bool(np.all(gamma + atol >= np.sqrt((1.0 - kappa) ** 2 + eps)))


def region_hull_vertices(
    region: str,
    kappa_range: Sequence[float],
    gamma_range: Sequence[float],
    maximum_macro_mag: float | None = None,
    n_boundary: int = 6,
) -> np.ndarray | None:
    """Vertices of a convex polygon inscribed in ``region`` clipped to the kappa-gamma box.

    Returns ``None`` if the intersection is empty.
    """
    if region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}; expected one of {REGIONS}.")
    k_min, k_max = map(float, kappa_range)
    g_min, g_max = map(float, gamma_range)
    eps = 0.0 if maximum_macro_mag is None else 1.0 / float(maximum_macro_mag)

    points: list[tuple[float, float]] = []
    if region in ("minima", "maxima"):
        # The region is non-empty only up to the gamma where the hyperbola leaves
        # the box in kappa. Clip the sampling range to that, or a thin sliver of
        # region can fall between two samples and be missed entirely.
        reach = (1.0 - k_min) if region == "minima" else (k_max - 1.0)
        if reach <= 0.0:
            return None
        gamma_cap = np.sqrt(max(0.0, reach**2 - eps))
        g_hi = min(g_max, gamma_cap)
        if g_hi <= g_min:
            return None
        # Slice at constant gamma; the allowed kappa interval is analytic.
        for gamma in _hyperbolic_samples(g_min, g_hi, eps, n_boundary):
            offset = np.sqrt(gamma**2 + eps)
            if region == "minima":
                lo, hi = k_min, min(k_max, 1.0 - offset)
            else:
                lo, hi = max(k_min, 1.0 + offset), k_max
            # Clamp rather than test: at the clipped endpoint the two bounds coincide
            # analytically and a `>=` test can lose the most valuable vertex to rounding.
            points += [(lo, gamma), (max(hi, lo), gamma)]
    else:
        # Slice at constant kappa, sampled symmetrically about the vertex kappa = 1.
        reach = np.sqrt(max(0.0, g_max**2 - eps))
        off_lo = max(1.0 - k_max, -reach)
        off_hi = min(1.0 - k_min, reach)
        if off_hi <= off_lo:
            return None
        offsets = _hyperbolic_samples(off_lo, off_hi, eps, n_boundary)
        if off_lo <= 0.0 <= off_hi:  # resolve the vertex at kappa = 1 explicitly
            offsets = np.unique(np.concatenate([offsets, [0.0]]))
        for offset in offsets:
            kappa = 1.0 - offset
            lo = min(max(g_min, np.sqrt(offset**2 + eps)), g_max)  # clamp, see above
            points += [(kappa, lo), (kappa, g_max)]

    if len(points) < 3:
        return None

    array = np.asarray(points, dtype=float)
    hull = ConvexHull(array)
    if hull.volume <= 0.0:  # 'volume' is the enclosed area in 2D
        raise ValueError(f"Non-positive area for the {region} region.")

    vertices = _drop_collinear(array[hull.vertices])
    if not _satisfies(region, vertices, eps, atol=1e-9):
        raise ValueError(f"The {region} hull does not satisfy the magnification cut.")
    return vertices


def extrude(vertices: np.ndarray, s_range: Sequence[float]) -> np.ndarray:
    """Extrude a 2D polygon into a prism along the smooth-fraction axis.

    Valid because the image-type conditions involve only the total convergence,
    so the cross-section does not depend on the smooth fraction.
    """
    s_min, s_max = s_range
    return np.vstack(
        [
            np.column_stack([vertices, np.full(len(vertices), s_min)]),
            np.column_stack([vertices, np.full(len(vertices), s_max)]),
        ]
    )


def region_hull(region: str, domain: DomainSpec, n_boundary: int) -> np.ndarray | None:
    """Vertices of a region's 3D design hull, or ``None`` if the region misses the domain."""
    vertices = region_hull_vertices(
        region,
        domain.kappa_range,
        domain.gamma_range,
        maximum_macro_mag=domain.max_macro_magnification,
        n_boundary=n_boundary,
    )
    if vertices is None:
        return None
    return extrude(vertices, domain.s_range)
