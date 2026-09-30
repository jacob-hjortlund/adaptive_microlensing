"""Macro-lens quantities, image classification and region hulls.

The macro model at a lensed image has total convergence ``kappa`` and shear ``gamma``.
The Jacobian of its lens mapping has the eigenvalues ``1 - kappa + |gamma|`` and
``1 - kappa - |gamma|``, and the macro magnification is the inverse of their product,
``mu_macro = 1 / ((1 - kappa)**2 - gamma**2)`` (:func:`macro_magnification`). The signs of
the eigenvalues give the image type (:func:`classify_image`): a minimum when both are
positive, a saddle when their signs differ and a maximum when both are negative. On the
critical lines ``|1 - kappa| = |gamma|`` one eigenvalue vanishes and ``mu_macro`` diverges.
The smooth-matter fraction ``s`` enters none of this.

:func:`in_box` and :func:`in_domain` test whether a point lies in a bank's
:class:`DomainSpec`. :meth:`MapBank.query` uses them, together with :func:`classify_image`,
to send each query to its region or to refuse it.

:func:`region_hull` builds each region's design hull, which
:func:`adaptive_microlensing.design.region_convex_hull` hands to the adaptive learner as its
bounds. The cap ``|mu_macro| <= max_macro_magnification`` moves each region's boundary off
the critical lines onto one branch of the hyperbola
``|(1 - kappa)**2 - gamma**2| = 1 / max_macro_magnification``. The hull is a convex polygon
inscribed in the capped region and clipped to the kappa-gamma box
(:func:`region_hull_vertices`), extruded along ``s`` into a prism (:func:`extrude`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import ConvexHull

from .config import DomainSpec

#: The three macro-image types, in the package's fixed order; a bank has one region for each.
REGIONS = ("minima", "saddle", "maxima")
#: Label that :func:`classify_image` gives a point on a critical line. It is not a region.
CRITICAL = "critical"
#: Default absolute tolerance of :func:`classify_image` on ``| |1 - kappa| - |gamma| |``,
#: the smaller absolute eigenvalue, at or below which a point is on a critical line.
CRITICAL_TOLERANCE = 1e-10


def macro_magnification(kappa: float, gamma: float) -> float:
    """Return the signed macro magnification of an image.

    The magnification is ``1 / ((1 - kappa)**2 - gamma**2)``, the inverse of the product of
    the Jacobian eigenvalues. It is positive for minima and maxima, negative for saddles,
    and ``inf`` where the denominator is exactly zero, on a critical line.

    Parameters
    ----------
    kappa : float
        Total convergence of the macro model at the image.
    gamma : float
        Shear of the macro model at the image. Only ``gamma**2`` enters, so its sign does
        not matter.

    Returns
    -------
    float
        The signed macro magnification ``mu_macro``, or ``inf`` on a critical line.

    Notes
    -----
    No tolerance is applied. A point merely close to a critical line gets a large, finite
    magnification, whereas :func:`classify_image` calls it ``"critical"`` within its
    tolerance.

    Examples
    --------
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

    The image type follows from the signs of the Jacobian eigenvalues ``1 - kappa + |gamma|``
    and ``1 - kappa - |gamma|``. :meth:`MapBank.query` uses it to choose the region that
    answers a query, and refuses points that it calls ``"critical"``.

    Parameters
    ----------
    kappa : float
        Total convergence of the macro model at the image.
    gamma : float
        Shear of the macro model at the image. Its absolute value is used.
    tol : float, optional
        Absolute tolerance on ``| |1 - kappa| - |gamma| |`` within which the point is on a
        critical line. Default is ``CRITICAL_TOLERANCE`` (``1e-10``).

    Returns
    -------
    str
        The image type: one of ``REGIONS``, or ``CRITICAL`` on a critical line.

    Raises
    ------
    ValueError
        If ``kappa`` or ``gamma`` is not finite.

    Notes
    -----
    With ``radial = 1 - kappa``, the tests are made in this order:

    - ``"critical"`` if ``abs(abs(radial) - abs(gamma)) <= tol``. This quantity is the
      smaller of the two eigenvalues in absolute value, so ``tol`` is a tolerance on that
      eigenvalue, not on the magnification.
    - ``"minima"`` if ``radial > abs(gamma)``: both eigenvalues are positive.
    - ``"maxima"`` if ``-radial > abs(gamma)``: both eigenvalues are negative.
    - ``"saddle"`` otherwise: the eigenvalues have opposite signs.

    Examples
    --------
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
    """Return whether a point lies in the domain's closed kappa/gamma/s box.

    Both bounds of each range are included. ``gamma`` is compared as given, not made
    absolute, and the gamma range of a :class:`DomainSpec` cannot extend below zero, so a
    negative shear is never in the box. Unlike :func:`in_domain`, the magnification cap is
    not applied. :meth:`MapBank.query` reports this test as ``is_in_bounds``.

    Parameters
    ----------
    kappa : float
        Total convergence of the point.
    gamma : float
        Shear of the point.
    s : float
        Smooth-matter fraction of the point.
    domain : DomainSpec
        Domain whose ``kappa_range``, ``gamma_range`` and ``s_range`` make the box.

    Returns
    -------
    bool
        ``True`` if every coordinate lies within its closed range. A NaN coordinate is
        outside.
    """
    return bool(
        domain.kappa_range[0] <= kappa <= domain.kappa_range[1]
        and domain.gamma_range[0] <= gamma <= domain.gamma_range[1]
        and domain.s_range[0] <= s <= domain.s_range[1]
    )


def in_domain(kappa: float, gamma: float, s: float, domain: DomainSpec) -> bool:
    """Return whether a point lies in the domain: in its box and within its magnification cap.

    A point is in the domain when :func:`in_box` holds and
    ``|mu_macro| <= domain.max_macro_magnification``, with ``mu_macro`` from
    :func:`macro_magnification`. The cap removes a band around each critical line, bounded
    by the hyperbolae ``|(1 - kappa)**2 - gamma**2| = 1 / max_macro_magnification``, so a
    point on a critical line is never in the domain. :meth:`MapBank.query` refuses points
    outside the domain unless ``allow_outside_domain`` is set.

    Parameters
    ----------
    kappa : float
        Total convergence of the point.
    gamma : float
        Shear of the point.
    s : float
        Smooth-matter fraction of the point.
    domain : DomainSpec
        Domain that gives the box and the cap ``max_macro_magnification``.

    Returns
    -------
    bool
        ``True`` if the point lies in the box and its ``|mu_macro|`` is at most
        ``domain.max_macro_magnification``.
    """
    return in_box(kappa, gamma, s, domain) and (
        abs(macro_magnification(kappa, gamma)) <= domain.max_macro_magnification
    )


def _hyperbolic_samples(lo: float, hi: float, eps: float, n: int) -> np.ndarray:
    """Sample ``[lo, hi]`` evenly in the natural parameter of the boundary hyperbola.

    Under the magnification cap, each region's boundary is one branch of a hyperbola
    centred on the triple point ``(kappa, gamma) = (1, 0)``, where the critical lines cross.
    Along the branch, the sampled offset from the triple point (``gamma`` for minima and
    maxima, ``1 - kappa`` for saddles) is ``sqrt(eps) * sinh(t)``, and the other coordinate
    is ``sqrt(eps) * cosh(t)`` away from its value at the triple point. Spacing ``t`` evenly
    puts points tightly near the vertex, where the curvature lives, and loosely out along
    the asymptote, where the boundary is already almost straight.

    Parameters
    ----------
    lo : float
        Lower end of the sampled range, as an offset from the triple point.
    hi : float
        Upper end of the sampled range, as an offset from the triple point.
    eps : float
        ``1 / max_macro_magnification``. When it is not positive there is no cap, the
        boundary is a straight critical line and the samples are evenly spaced.
    n : int
        Number of samples. Below 2, only ``lo`` and ``hi`` are returned.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(max(n, 2),)`` of samples that run from ``lo`` to ``hi``. When
        ``eps > 0``, the end samples equal ``lo`` and ``hi`` only up to floating-point
        rounding.
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
    starts, and each evaluation costs a full magnification map, so vertices that do not
    change the polygon are dropped.

    Parameters
    ----------
    vertices : numpy.ndarray
        Array of shape ``(m, 2)`` of polygon vertices, in order around the polygon.
    rel_tol : float, optional
        Largest distance of a dropped vertex from the line through its two neighbours,
        relative to the larger of 1 and the polygon's widest coordinate extent. Default is
        ``1e-9``.

    Returns
    -------
    numpy.ndarray
        Float array of the vertices that are kept, in their original order. A polygon with
        fewer than four vertices, or one that would keep fewer than three, is returned
        unchanged.

    Notes
    -----
    Every vertex is tested against its neighbours in the input polygon, in one pass, so a
    run of collinear vertices is dropped together. A vertex whose two neighbours coincide
    is kept.
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
    """Return whether every vertex is in the region and passes the mu cut.

    This is the invariant check on the polygon that :func:`region_hull_vertices` builds.
    With ``eps = 1 / max_macro_magnification``, a vertex passes when
    ``1 - kappa >= sqrt(gamma**2 + eps)`` for minima, ``kappa - 1 >= sqrt(gamma**2 + eps)``
    for maxima, or ``gamma >= sqrt((1 - kappa)**2 + eps)`` for saddles, each relaxed by
    ``atol``. For ``gamma >= 0`` this means that the vertex lies in the region, or on its
    critical line when ``eps = 0``, with ``|mu_macro| <= 1 / eps``. The box is not checked.

    Parameters
    ----------
    region : str
        Name of the region. Any name other than ``"minima"`` or ``"maxima"`` is checked
        with the saddle condition.
    vertices : numpy.ndarray
        Array of shape ``(m, 2)`` of ``(kappa, gamma)`` vertices.
    eps : float
        ``1 / max_macro_magnification``, or ``0`` for no cap.
    atol : float, optional
        Absolute slack in each condition, to allow for rounding. Default is ``1e-9``.

    Returns
    -------
    bool
        ``True`` if every vertex passes.
    """
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
    """Return the vertices of a convex polygon inscribed in ``region`` clipped to the kappa-gamma box.

    The polygon lies in the part of ``region`` that satisfies
    ``|mu_macro| <= maximum_macro_mag`` and lies in the box ``kappa_range x gamma_range``.
    :func:`region_hull` extrudes it along ``s`` into the region's design hull.

    Parameters
    ----------
    region : str
        Name of the region, one of ``REGIONS``.
    kappa_range : sequence of float
        ``(minimum, maximum)`` of the total convergence.
    gamma_range : sequence of float
        ``(minimum, maximum)`` of the shear. The shear is taken to be non-negative, as in a
        :class:`DomainSpec`.
    maximum_macro_mag : float or None, optional
        Positive cap on ``|mu_macro|``. When ``None``, there is no cap and the polygon
        reaches the critical lines.
    n_boundary : int, optional
        Number of samples along the region's curved boundary (see Notes). Default is ``6``.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(m, 2)`` of ``(kappa, gamma)`` vertices, with ``m >= 3``, in
        counterclockwise order with ``kappa`` along the horizontal axis. ``None`` if the
        capped region does not intersect the box.

    Raises
    ------
    ValueError
        If ``region`` is not one of ``REGIONS``. Also, as internal consistency checks, if
        the polygon has no area or if a vertex fails :func:`_satisfies`.

    Notes
    -----
    With ``eps = 1 / maximum_macro_mag`` (``0`` without a cap), the capped regions are

    - minima: ``kappa <= 1 - sqrt(gamma**2 + eps)``;
    - maxima: ``kappa >= 1 + sqrt(gamma**2 + eps)``;
    - saddle: ``gamma >= sqrt((1 - kappa)**2 + eps)``.

    Each is bounded by one branch of the hyperbola ``|(1 - kappa)**2 - gamma**2| = eps``,
    with its vertex at ``(1 - sqrt(eps), 0)`` for minima, ``(1 + sqrt(eps), 0)`` for maxima
    and ``(1, sqrt(eps))`` for saddles, and the critical lines ``gamma = |1 - kappa|`` as
    asymptotes. Without a cap the boundaries are the critical lines themselves. Each capped
    region is convex for ``gamma >= 0``.

    Minima and maxima are sliced at ``n_boundary`` values of ``gamma``, from the bottom of
    the gamma range up to its top or to where the boundary leaves the kappa range, if that
    comes first. The allowed kappa interval at each ``gamma`` is known in closed form, and
    both its ends become points. Saddles are sliced at ``n_boundary`` values of ``kappa``,
    plus ``kappa = 1`` when it is in range, over the part of the kappa range where the
    boundary lies below the top of the gamma range; the points are the ends of the allowed
    gamma interval, from the boundary (or the bottom of the gamma range) to the top of the
    gamma range. The slices are spaced by :func:`_hyperbolic_samples`.

    The polygon is the convex hull of these points, with collinear vertices removed by
    :func:`_drop_collinear`. Every vertex lies, up to rounding, in the box and in the capped
    region, whose intersection is convex, so the whole polygon does: it is inscribed, and
    leaves out the thin slivers between its edges and the curved boundary.
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
    so the cross-section does not depend on the smooth fraction. The magnification cap
    likewise involves only the convergence and the shear, so every cross-section of a
    region's design hull at constant ``s`` is the same polygon.

    Parameters
    ----------
    vertices : numpy.ndarray
        Array of shape ``(m, 2)`` of the polygon's ``(kappa, gamma)`` vertices.
    s_range : sequence of float
        ``(s_min, s_max)``, the two ends of the prism.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2 * m, 3)`` of ``(kappa, gamma, s)`` points: the ``m`` vertices at
        ``s_min``, followed by the same vertices at ``s_max``.
    """
    s_min, s_max = s_range
    return np.vstack(
        [
            np.column_stack([vertices, np.full(len(vertices), s_min)]),
            np.column_stack([vertices, np.full(len(vertices), s_max)]),
        ]
    )


def region_hull(region: str, domain: DomainSpec, n_boundary: int) -> np.ndarray | None:
    """Return the vertices of a region's 3D design hull, or ``None`` if the region misses the domain.

    The hull is the polygon from :func:`region_hull_vertices`, for the domain's kappa and
    gamma ranges and its cap ``max_macro_magnification``, extruded over the domain's ``s``
    range by :func:`extrude`. :func:`adaptive_microlensing.design.region_convex_hull` turns
    it into the bounds of :meth:`MapBank.build`, and every vertex becomes a bank entry
    before adaptive sampling starts.

    Parameters
    ----------
    region : str
        Name of the region, one of ``REGIONS``.
    domain : DomainSpec
        Domain of the bank.
    n_boundary : int
        Number of samples along the region's curved boundary, usually
        ``DesignSpec.n_boundary``.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(2 * m, 3)`` of ``(kappa, gamma, s)`` vertices: the ``m`` polygon
        vertices at the bottom of the ``s`` range, then the same at the top. ``None`` if the
        region, under the cap, does not intersect the domain's box.

    Raises
    ------
    ValueError
        If ``region`` is not one of ``REGIONS``, or if the polygon fails the consistency
        checks of :func:`region_hull_vertices`.
    """
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
