"""Adaptive design of one region: the failure-aware loss and the resumable build loop."""

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
    success and 0 for failure.

    Parameters
    ----------
    exploration : float
        Standard curvature-loss exploration coefficient.
    boundary_multiplier : float
        Priority of simplices with both valid and invalid vertices, relative to
        their characteristic size ``V**(1/d)``.
    invalid_multiplier : float
        Weak exploration of fully invalid simplices; zero abandons them.
    adjacent_boundary_multiplier : float or None
        Optional minimum loss multiplier for an all-valid simplex with an invalid
        neighbour. Usually None, because the adjacent mixed simplex already gets
        boundary priority.
    validity_threshold : float
        Threshold used to decode the validity flag.
    """

    @uses_nth_neighbors(1)
    def loss(
        simplex: Sequence[Sequence[float]],
        values: Sequence[Sequence[float]],
        value_scale: float,
        neighbors: Sequence[Any],
        neighbor_values: Sequence[Any],
    ) -> float:
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
    """The failure-aware loss with the weights of a ``DesignSpec``."""
    return failure_aware_curvature_loss(
        exploration=design.exploration,
        boundary_multiplier=design.boundary_multiplier,
        invalid_multiplier=design.invalid_multiplier,
        adjacent_boundary_multiplier=design.adjacent_boundary_multiplier,
        validity_threshold=design.validity_threshold,
    )


def region_convex_hull(region: str, config: BankConfig) -> ConvexHull:
    """The 3D design hull of a region under a bank's domain."""
    vertices = region_hull(region, config.domain, config.design.n_boundary)
    if vertices is None:
        raise ValueError(f"Region {region!r} does not intersect the bank's domain.")
    return ConvexHull(np.asarray(vertices, dtype=float))


def _not_called(point: Any) -> Any:
    raise RuntimeError("MapBank evaluates entries itself; the learner's function is never called.")


def _learner_value(valid: bool, quantile: float) -> np.ndarray:
    # LearnerND scales vector values by a float, so they must be arrays, not tuples.
    return np.array([quantile, 1.0]) if valid else np.array([0.0, 0.0])


def _recent_max(residuals: list[float], count: int) -> float:
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
    """Adaptively add entries to one region until a stopping rule holds; see ``MapBank.build``."""
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
