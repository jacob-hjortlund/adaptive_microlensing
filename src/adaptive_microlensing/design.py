"""Adaptive design of one region: the failure-aware loss and the resumable build loop.

:meth:`MapBank.build` calls :func:`build_region`, which adds entries to a region with
python-``adaptive``'s ``LearnerND``. The learner triangulates the points it has been
told, gives every simplex a loss and proposes a new point in the simplex with the largest
loss. It samples inside the region's design hull (:func:`region_convex_hull`) and asks
for the hull's vertices first.

The learner's value at a point is ``[intrinsic_quantile, 1]`` for a valid entry and
``[0, 0]`` for an invalid one. The loss, :func:`failure_aware_curvature_loss`, reads the
second component as a validity flag. A simplex whose vertices are all valid gets
``adaptive``'s curvature loss of the intrinsic quantile, which is large where the
quantile is curved or the simplex is large. A simplex that straddles the validity
boundary gets a loss proportional to its size, so the edge of a failing area is refined.
A simplex whose vertices all failed gets ``invalid_multiplier`` times its size, zero by
default, so failing areas are abandoned or only weakly explored.

The learner never calls a function itself: :func:`build_region` evaluates each proposed
point with the bank, tells the learner the result and commits the point as an entry.
The learner is not saved, so a build resumes by replaying the region's committed entries
to a new learner. A build stops as soon as one of the :class:`StoppingCriteria` holds.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import adaptive
import numpy as np
from adaptive.learner.base_learner import uses_nth_neighbors
from adaptive.learner.learnerND import triangle_loss, volume
from scipy.spatial import ConvexHull, Delaunay

from .config import BankConfig, DesignSpec, StoppingCriteria
from .lensing import region_hull
from .maps import MapGenerator
from .results import BuildSummary

if TYPE_CHECKING:
    from .bank import MapBank

logger = logging.getLogger("adaptive_microlensing")
#: Tolerance of the inside-the-hull test that picks the entries a resumed build replays to its learner.
HULL_TOLERANCE = 1e-10


def failure_aware_curvature_loss(
    exploration: float = 0.05,
    boundary_multiplier: float = 2.0,
    invalid_multiplier: float = 0.0,
    adjacent_boundary_multiplier: float | None = None,
    validity_threshold: float = 0.5,
) -> Callable[..., float]:
    """Construct a LearnerND loss for values encoded as ``[scientific_value, validity_flag]``.

    ``scientific_value`` is finite and lies in [0, 1]; ``validity_flag`` is 1 for
    success and 0 for failure. In a bank the scientific value is the intrinsic quantile,
    a quantile of JS distances in base 2. The loss of a simplex depends on which of its
    vertices are valid, so the learner refines the quantile where maps can be made,
    refines the edge of failing areas and abandons, or only weakly explores, areas where
    every map failed.

    Parameters
    ----------
    exploration : float, optional
        Standard curvature-loss exploration coefficient: the weight of the volume term,
        which keeps large simplices from being neglected where the value is flat.
        Default is ``0.05``.
    boundary_multiplier : float, optional
        Priority of simplices with both valid and invalid vertices, relative to
        their characteristic size ``V**(1/d)``. Default is ``2.0``.
    invalid_multiplier : float, optional
        Weak exploration of fully invalid simplices, relative to ``V**(1/d)``; zero, the
        default, abandons them.
    adjacent_boundary_multiplier : float or None, optional
        Optional minimum loss multiplier for an all-valid simplex with an invalid
        neighbour. Usually ``None``, the default, because the adjacent mixed simplex
        already gets boundary priority.
    validity_threshold : float, optional
        Threshold used to decode the validity flag: a vertex is valid when its flag
        exceeds it. Default is ``0.5``.

    Returns
    -------
    callable
        Loss ``loss(simplex, values, value_scale, neighbors, neighbor_values)`` for the
        ``loss_per_simplex`` argument of ``adaptive.LearnerND``. It is marked with
        ``adaptive``'s ``uses_nth_neighbors(1)``, so the learner also passes it the
        vertex beyond each face of the simplex.

    Notes
    -----
    Write ``d`` for the dimension, ``V`` for the simplex's volume and ``h = V**(1/d)``
    for its characteristic size. ``LearnerND`` passes coordinates divided by the side
    lengths of the hull's bounding box, so ``V`` is measured in those units. The loss of
    a simplex is:

    - ``0`` if the simplex is degenerate, ``V <= 0``;
    - ``invalid_multiplier * h`` if every vertex is invalid;
    - ``boundary_multiplier * h`` if some vertices are valid and some are not, since the
      values at the invalid vertices are placeholders;
    - otherwise ``(C + exploration * V**((d + 2) / d))**(1 / (d + 2))``, ``adaptive``'s
      curvature loss. ``C`` is ``adaptive``'s ``triangle_loss`` of the scientific
      values: the mean volume, in the space of coordinates and value, of the simplex
      joined with each valid neighbour. Invalid neighbours are left out, and ``C`` is
      zero without valid neighbours. If ``adjacent_boundary_multiplier`` is set and a
      neighbour is invalid, the loss is at least ``adjacent_boundary_multiplier * h``.

    ``LearnerND`` multiplies every value component by ``value_scale`` before it calls
    the loss. The loss divides it out, so the scientific value keeps its own [0, 1]
    scale and the validity flag is decoded as 0 or 1.
    """

    @uses_nth_neighbors(1)
    def loss(
        simplex: Sequence[Sequence[float]],
        values: Sequence[Sequence[float]],
        value_scale: float,
        neighbors: Sequence[Any],
        neighbor_values: Sequence[Any],
    ) -> float:
        """Return the failure-aware loss of one simplex.

        Parameters
        ----------
        simplex : sequence of sequence of float
            The ``d + 1`` vertices of the simplex, in the learner's rescaled coordinates.
        values : sequence of sequence of float
            Values at the vertices, ``value_scale * [scientific_value, validity_flag]``
            each, so of shape ``(d + 1, 2)``.
        value_scale : float
            Factor by which the learner multiplied every value component.
        neighbors : sequence
            For each vertex, the vertex of the neighbouring simplex beyond the opposite
            face, in rescaled coordinates, or ``None`` where that face lies on the
            boundary of the hull. ``neighbors[i]`` is opposite ``simplex[i]``.
        neighbor_values : sequence
            Scaled values at ``neighbors``, each of shape ``(2,)``, or ``None`` where
            there is no neighbour.

        Returns
        -------
        float
            The loss of the simplex. The learner refines the simplex with the largest
            loss first.

        Raises
        ------
        ValueError
            If ``values`` is not two-dimensional with two columns, if ``value_scale`` is
            not positive and finite, or if a neighbour value does not have shape ``(2,)``.
        """
        simplex_array = np.asarray(simplex, dtype=float)
        value_array = np.asarray(values, dtype=float)
        if value_array.ndim != 2 or value_array.shape[1] != 2:
            raise ValueError("Expected output values with columns [scientific_value, validity_flag].")
        scale = float(value_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Expected a positive finite value_scale, got {scale}.")

        # LearnerND multiplies every output component by value_scale before calling the
        # loss. Undo that so the known physical [0, 1] range is used consistently.
        raw_values = value_array / scale
        scientific_values = raw_values[:, 0]
        valid = raw_values[:, 1] > validity_threshold

        dim = simplex_array.shape[1]
        simplex_volume = float(volume(simplex_array))
        if simplex_volume <= 0:
            return 0.0
        simplex_size = simplex_volume ** (1.0 / dim)

        # Every vertex failed.
        if not np.any(valid):
            return float(invalid_multiplier * simplex_size)
        # The simplex straddles the validity boundary; some values are placeholders.
        if not np.all(valid):
            return float(boundary_multiplier * simplex_size)

        # Every vertex is valid: curvature of the scientific value from valid neighbours only.
        valid_neighbor_points: list[np.ndarray | None] = []
        valid_neighbor_values: list[float | None] = []
        has_invalid_neighbor = False
        for neighbor_point, neighbor_value in zip(neighbors, neighbor_values, strict=True):
            # A None neighbour means this face lies on the outer boundary of the hull.
            if neighbor_point is None or neighbor_value is None:
                valid_neighbor_points.append(None)
                valid_neighbor_values.append(None)
                continue
            raw_neighbor_value = np.asarray(neighbor_value, dtype=float) / scale
            if raw_neighbor_value.ndim != 1 or raw_neighbor_value.shape[0] != 2:
                raise ValueError("Expected each neighbour value to have shape (2,).")
            if raw_neighbor_value[1] > validity_threshold:
                valid_neighbor_points.append(np.asarray(neighbor_point, dtype=float))
                valid_neighbor_values.append(float(raw_neighbor_value[0]))
            else:
                valid_neighbor_points.append(None)
                valid_neighbor_values.append(None)
                has_invalid_neighbor = True

        # The neighbour-based embedded-volume term of adaptive's curvature loss,
        # applied to the scientific value and valid neighbours only.
        curvature = float(
            triangle_loss(simplex_array, scientific_values, 1.0, valid_neighbor_points, valid_neighbor_values)
        )
        valid_loss = (curvature + exploration * simplex_volume ** ((dim + 2.0) / dim)) ** (1.0 / (dim + 2.0))
        if has_invalid_neighbor and adjacent_boundary_multiplier is not None:
            valid_loss = max(valid_loss, adjacent_boundary_multiplier * simplex_size)
        return float(valid_loss)

    return loss


def loss_from_spec(design: DesignSpec) -> Callable[..., float]:
    """Build the failure-aware loss with the weights of a :class:`DesignSpec`.

    :func:`build_region` calls it with the bank's ``config.design``, so a bank's loss is
    fixed by its configuration.

    Parameters
    ----------
    design : DesignSpec
        Design settings. Every field except ``n_boundary`` is passed to
        :func:`failure_aware_curvature_loss`.

    Returns
    -------
    callable
        The loss from :func:`failure_aware_curvature_loss`, with the settings of
        ``design``.
    """
    return failure_aware_curvature_loss(
        exploration=design.exploration,
        boundary_multiplier=design.boundary_multiplier,
        invalid_multiplier=design.invalid_multiplier,
        adjacent_boundary_multiplier=design.adjacent_boundary_multiplier,
        validity_threshold=design.validity_threshold,
    )


def region_convex_hull(region: str, config: BankConfig) -> ConvexHull:
    """Return the 3D design hull of a region under a bank's domain.

    The hull is the region's convex polygon in the kappa-gamma plane, inscribed in the
    region and clipped to the domain's box and magnification cap, extruded over the
    domain's ``s`` range; see :func:`~adaptive_microlensing.lensing.region_hull`.
    ``config.design.n_boundary`` sets how finely the curved edge of the region is
    sampled. :func:`build_region` samples inside this hull, starting at its vertices.

    Parameters
    ----------
    region : str
        Name of the region, one of ``REGIONS``.
    config : BankConfig
        Bank configuration, whose ``domain`` and ``design.n_boundary`` define the hull.

    Returns
    -------
    scipy.spatial.ConvexHull
        Convex hull of the prism's vertices, in ``(kappa, gamma, s)``.

    Raises
    ------
    ValueError
        If ``region`` is not one of ``REGIONS``, or if the region does not intersect the
        bank's domain.
    """
    vertices = region_hull(region, config.domain, config.design.n_boundary)
    if vertices is None:
        raise ValueError(f"Region {region!r} does not intersect the bank's domain.")
    return ConvexHull(np.asarray(vertices, dtype=float))


def _not_called(point: Any) -> Any:
    """Raise an error in place of the learner's function, which is never called.

    :func:`build_region` evaluates every point with the bank and tells the learner the
    result, so ``LearnerND`` needs a function only to be constructed.

    Parameters
    ----------
    point : Any
        Point the learner would evaluate.

    Raises
    ------
    RuntimeError
        Always.
    """
    raise RuntimeError("MapBank evaluates entries itself; the learner's function is never called.")


def _learner_value(valid: bool, quantile: float) -> np.ndarray:
    """Encode an entry as the learner's value ``[scientific_value, validity_flag]``.

    Parameters
    ----------
    valid : bool
        Whether the entry is valid.
    quantile : float
        Intrinsic quantile of the entry. It is ignored when the entry is invalid.

    Returns
    -------
    numpy.ndarray
        ``[quantile, 1.0]`` for a valid entry and ``[0.0, 0.0]`` for an invalid one.
    """
    # LearnerND scales vector values by a float, so they must be arrays, not tuples.
    return np.array([quantile, 1.0]) if valid else np.array([0.0, 0.0])


def _recent_max(residuals: list[float], count: int) -> float:
    """Return the largest of the last ``count`` residuals, ignoring NaN.

    Parameters
    ----------
    residuals : list of float
        Relative residuals of the region's build entries, oldest first.
    count : int
        Number of most recent residuals to consider, ``StoppingCriteria.min_num_residuals``.

    Returns
    -------
    float
        The largest of those residuals, which may be ``inf``, or NaN when there are none
        or they are all NaN.
    """
    recent = np.asarray(residuals[-count:], dtype=float)
    if recent.size == 0 or np.all(np.isnan(recent)):
        return math.nan
    return float(np.nanmax(recent))


def _stop_reason(
    n_valid: int,
    learner: adaptive.LearnerND,
    residuals: list[float],
    simplex_loss: float,
    max_residual: float,
    stop: StoppingCriteria,
) -> str | None:
    """Return the first stopping criterion that holds, or ``None`` to keep building.

    Only the criteria set in ``stop`` count, and they are checked in this order:

    - ``"max_valid_points"``: the region has at least ``stop.max_valid_points`` valid
      entries.
    - ``"simplex_loss_goal"``: the learner has been told every hull vertex, and its
      loss is at most ``stop.simplex_loss_goal``.
    - ``"residual_goal"``: there are at least ``stop.min_num_residuals`` residuals, and
      ``max_residual`` is finite and at most ``stop.residual_goal``.

    Parameters
    ----------
    n_valid : int
        Number of valid entries in the region, of any origin.
    learner : adaptive.LearnerND
        The build's learner, asked whether it has been told every hull vertex.
    residuals : list of float
        Relative residuals of the region's build entries so far. Only their number is
        used.
    simplex_loss : float
        The learner's current loss, the largest loss of any simplex.
    max_residual : float
        Largest of the ``stop.min_num_residuals`` most recent residuals, from
        :func:`_recent_max`.
    stop : StoppingCriteria
        The criteria to check.

    Returns
    -------
    str or None
        Name of the first criterion that holds, or ``None`` when none does.
    """
    if stop.max_valid_points is not None and n_valid >= stop.max_valid_points:
        return "max_valid_points"
    if (
        stop.simplex_loss_goal is not None
        and learner.bounds_are_done
        and simplex_loss <= stop.simplex_loss_goal
    ):
        return "simplex_loss_goal"
    if (
        stop.residual_goal is not None
        and len(residuals) >= stop.min_num_residuals
        and math.isfinite(max_residual)
        and max_residual <= stop.residual_goal
    ):
        return "residual_goal"
    return None


def build_region(
    bank: MapBank, region: str, stop: StoppingCriteria, generator: MapGenerator | None
) -> BuildSummary:
    """Add entries to one region by adaptive sampling until a stopping criterion holds.

    This is the work of :meth:`MapBank.build`. It first replays the region's committed
    entries that lie in the design hull (:func:`region_convex_hull`) to a new
    ``LearnerND``, in entry order, so the build continues from the entries of earlier
    builds, fetches and imports. It then repeats three steps until a criterion of
    ``stop`` holds: ask the learner for one point; evaluate it with the bank, which makes
    its coarse and bank maps, measures its intrinsic quantile, writes its bank map and,
    in a finalized region, computes its bank MPD; and tell the learner the result and
    commit it as a new entry with ``origin="build"``. A failure to make a usable map
    (:exc:`MapGenerationError` or :exc:`InvalidMapError`) does not stop the build: the
    point becomes an invalid entry.

    Parameters
    ----------
    bank : MapBank
        Bank that holds the region.
    region : str
        Name of the region to build, one of ``REGIONS``. It must be open for writing.
    stop : StoppingCriteria
        When to stop adding entries.
    generator : MapGenerator or None
        Generator that makes the maps. When ``None``, the generator named in the bank's
        configuration is used. It must match that configuration.

    Returns
    -------
    BuildSummary
        The region's numbers of entries and of valid entries after the build, its valid
        fraction, the criterion that stopped the build (``"max_valid_points"``,
        ``"simplex_loss_goal"`` or ``"residual_goal"``), the learner's final loss and the
        seconds spent after the replay.

    Raises
    ------
    ValueError
        If ``region`` is not one of ``REGIONS`` or does not intersect the bank's domain,
        if ``generator`` is ``None`` and no generator is registered under the bank's
        generator name, or if the generator returns a bank map of the wrong shape.
    BankReadOnlyError
        If the region is not open for writing.
    GeneratorMismatchError
        If ``generator`` differs from the generator in the bank's configuration.

    Notes
    -----
    The criteria are checked before the first new point, so a region that already meets
    one gets no new entries, and again after every committed entry. In order of
    precedence, ``max_valid_points`` counts the valid entries of every origin, inside
    the hull or not; ``simplex_loss_goal`` can hold only once every hull vertex is an
    entry; and ``residual_goal`` needs at least ``min_num_residuals`` residuals.

    Each new entry stores four diagnostics in ``entries.csv``:

    - ``simplex_loss``: the learner's loss, the largest loss of any simplex, after it
      has been told the entry; ``inf`` until the learner can triangulate its points.
    - ``abs_residual``: ``|q - q_pred|``, where ``q`` is the entry's intrinsic quantile
      and ``q_pred`` is :meth:`RegionIndex.interpolate_quantile` at its point before
      the entry is added. NaN when the entry is invalid or no fully valid tetrahedron
      contains the point.
    - ``rel_residual``: ``abs_residual / |q|``; ``inf`` when ``q`` is zero, and NaN
      when ``abs_residual`` is NaN.
    - ``max_residual``: the largest of the ``min_num_residuals`` most recent relative
      residuals of the region's build entries, including those of earlier builds; NaN
      when there are none.

    Each new entry is also logged at ``INFO`` level on the ``"adaptive_microlensing"``
    logger.
    """
    bank._require_writable(region)
    generator = bank._resolve_generator(generator)
    hull = region_convex_hull(region, bank.config)
    # The original script passed anisotropic=False, which is the default where the option exists.
    learner = adaptive.LearnerND(
        _not_called, bounds=hull, loss_per_simplex=loss_from_spec(bank.config.design)
    )

    # Resume: replay every committed entry inside the hull, in entry order.
    inside = Delaunay(hull.points[hull.vertices])
    entries = bank.entries(region)
    for row in entries.itertuples(index=False):
        point = (float(row.kappa), float(row.gamma), float(row.s))
        if inside.find_simplex(np.asarray(point), tol=HULL_TOLERANCE) >= 0:
            learner.tell(point, _learner_value(bool(row.valid), float(row.intrinsic_quantile)))
    build_rows = entries["origin"] == "build"
    residuals = [float(value) for value in entries.loc[build_rows, "rel_residual"] if not math.isnan(value)]

    state = bank._regions[region]
    started = time.perf_counter()
    added = 0
    simplex_loss = float(learner.loss())
    max_residual = _recent_max(residuals, stop.min_num_residuals)
    reason = _stop_reason(
        int(state.entries["valid"].sum()), learner, residuals, simplex_loss, max_residual, stop
    )
    while reason is None:
        points, _ = learner.ask(1, tell_pending=False)
        kappa, gamma, s = (float(value) for value in points[0])
        point = (kappa, gamma, s)
        prediction = state.index.interpolate_quantile(np.asarray(point))
        pending = bank._evaluate_entry(region, point, generator, origin="build")
        quantile = float(pending.row["intrinsic_quantile"])
        abs_residual = rel_residual = math.nan
        if pending.valid and prediction is not None:
            abs_residual = abs(quantile - prediction)
            rel_residual = abs_residual / abs(quantile) if quantile != 0.0 else math.inf
            residuals.append(rel_residual)
        learner.tell(point, _learner_value(pending.valid, quantile))
        simplex_loss = float(learner.loss())
        max_residual = _recent_max(residuals, stop.min_num_residuals)
        bank._commit(
            region,
            pending,
            {
                "simplex_loss": simplex_loss,
                "abs_residual": abs_residual,
                "rel_residual": rel_residual,
                "max_residual": max_residual,
            },
        )
        added += 1
        elapsed = time.perf_counter() - started
        logger.info(
            "%s entry %d: kappa=%.6g gamma=%.6g s=%.6g valid=%s q=%.6g simplex_loss=%.6g "
            "rel_residual=%.6g max_rel_residual=%.6g time/point=%.3gs",
            region,
            pending.row["entry_id"],
            *point,
            pending.valid,
            quantile,
            simplex_loss,
            rel_residual,
            max_residual,
            elapsed / added,
        )
        reason = _stop_reason(
            int(state.entries["valid"].sum()), learner, residuals, simplex_loss, max_residual, stop
        )

    n_entries = len(state.entries)
    n_valid = int(state.entries["valid"].sum())
    return BuildSummary(
        region=region,
        n_entries=n_entries,
        n_valid=n_valid,
        valid_fraction=n_valid / n_entries if n_entries else math.nan,
        stop_reason=reason,
        final_simplex_loss=simplex_loss,
        elapsed_seconds=time.perf_counter() - started,
    )
