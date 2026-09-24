"""Reference copies of the original adaptive_mpd code, used as test oracles.

``mpd_distance_quantile`` is copied unchanged from ``mpd_distance.py``. The query code is
a port of ``run_mpd_interpolator.py``: it keeps the arithmetic and control flow exactly,
but drops input validation, printing and progress bars, because the oracle only runs on
valid data. Do not "improve" this module: its numbers must match the original scripts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.distance import jensenshannon

REGIONS = ("minima", "saddle", "maxima")
PARAMETER_COLUMNS = ("kappa", "gamma", "s")
REGION_REQUIRED_COLUMNS = {"kappa", "gamma", "s", "mpd_distance", "validity"}


def mpd_distance_quantile(mag_map, *, N, M, dL, bin_edges, quantile=0.95, chunk_rows=256):  # noqa: N803
    """Point estimate from all pairs of non-overlapping MxM windows (copy of mpd_distance.py)."""
    windows_per_axis = round(N / M)
    window_pixels = round(M / dL)
    expected_pixels = round(N / dL)
    if not np.isclose(N / M, windows_per_axis):
        raise ValueError("M must divide N.")
    if not np.isclose(M / dL, window_pixels):
        raise ValueError("M/dL must be an integer.")
    if mag_map.shape != (expected_pixels, expected_pixels):
        raise ValueError(f"Expected map shape {(expected_pixels, expected_pixels)}, got {mag_map.shape}.")
    mpds = []
    n_bins = len(bin_edges) - 1
    for window_row in range(windows_per_axis):
        for window_col in range(windows_per_axis):
            row_start = window_row * window_pixels
            col_start = window_col * window_pixels
            counts = np.zeros(n_bins, dtype=np.int64)
            finite_count = 0
            for offset in range(0, window_pixels, chunk_rows):
                block = np.asarray(
                    mag_map[
                        row_start + offset : row_start + min(offset + chunk_rows, window_pixels),
                        col_start : col_start + window_pixels,
                    ]
                )
                values = block[np.isfinite(block)]
                finite_count += values.size
                counts += np.histogram(values, bins=bin_edges)[0]
            if counts.sum() != finite_count:
                raise ValueError("bin_edges do not contain every finite map value.")
            mpds.append(counts / counts.sum())
    mpds = np.asarray(mpds)
    n_windows = len(mpds)
    distances = np.fromiter(
        (jensenshannon(mpds[i], mpds[j], base=2) for i, j in combinations(range(n_windows), 2)),
        dtype=float,
        count=n_windows * (n_windows - 1) // 2,
    )
    point_estimate = float(np.quantile(distances, quantile))
    return point_estimate, distances, mpds


class InterpolationDomainError(ValueError):
    """Base class for expected failures when locating a query point."""


class OutsideConvexHullError(InterpolationDomainError):
    """Raised when a query lies outside a region's Delaunay convex hull."""


class InvalidTetrahedronError(InterpolationDomainError):
    """Raised when a containing tetrahedron has at least one invalid vertex."""


@dataclass(frozen=True)
class TetrahedronLocation:
    """A query point expressed in its containing tetrahedron."""

    simplex_index: int
    vertex_rows: np.ndarray
    vertex_points: np.ndarray
    barycentric_weights: np.ndarray


def _validate_points_and_mask(points, valid):
    points = np.asarray(points, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (n_points, 3), got {points.shape}.")
    if valid.shape != (len(points),):
        raise ValueError(f"Expected valid to have shape {(len(points),)}, got {valid.shape}.")
    if len(points) < 4:
        raise ValueError("At least four points are required for a 3D tetrahedralization.")
    if not np.all(np.isfinite(points)):
        raise ValueError("The parameter coordinates must all be finite.")
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != len(points):
        raise ValueError("The CSV contains duplicate parameter points.")
    if np.linalg.matrix_rank(points - points[0]) < 3:
        raise ValueError("The sampled points do not span three dimensions.")
    return points, valid


def locate_tetrahedron(*, points, valid, triangulation, q, tol=1e-10):
    """Locate ``q`` and calculate its barycentric tetrahedral coordinates."""
    points = np.asarray(points, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    q = np.asarray(q, dtype=float)
    if q.shape != (3,):
        raise ValueError(f"Expected q to have shape (3,), got {q.shape}.")
    simplex_index = int(triangulation.find_simplex(q, tol=tol))
    if simplex_index < 0:
        raise OutsideConvexHullError(f"Query point {q.tolist()} lies outside the convex hull.")
    vertex_rows = np.asarray(triangulation.simplices[simplex_index], dtype=int)
    if not np.all(valid[vertex_rows]):
        raise InvalidTetrahedronError("The containing tetrahedron has one or more invalid vertices.")
    ndim = points.shape[1]
    transform = triangulation.transform[simplex_index]
    first_weights = transform[:ndim] @ (q - transform[ndim])
    weights = np.concatenate([first_weights, [1.0 - first_weights.sum()]])
    if not np.all(np.isfinite(weights)):
        raise RuntimeError("Non-finite barycentric weights.")
    weights[np.abs(weights) <= tol] = 0.0
    weights[np.abs(weights - 1.0) <= tol] = 1.0
    if np.any(weights < -tol):
        raise RuntimeError(f"Computed substantially negative barycentric weights: {weights.tolist()}.")
    weights = np.maximum(weights, 0.0)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise RuntimeError("Barycentric weights have non-positive sum.")
    weights /= weight_sum
    return TetrahedronLocation(
        simplex_index=simplex_index,
        vertex_rows=vertex_rows.copy(),
        vertex_points=points[vertex_rows].copy(),
        barycentric_weights=weights,
    )


@dataclass
class RegionPairwiseDistanceInterpolator:
    """Interpolate distances from a query to its four tetrahedron vertices."""

    points: np.ndarray
    valid: np.ndarray
    triangulation: Delaunay
    distance_matrix: np.ndarray

    def __post_init__(self) -> None:
        self.points, self.valid = _validate_points_and_mask(self.points, self.valid)
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)

    def query(self, q, *, tol=1e-10):
        """Return interpolated distances from ``q`` to four local vertices."""
        location = locate_tetrahedron(
            points=self.points, valid=self.valid, triangulation=self.triangulation, q=q, tol=tol
        )
        local_distances = self.distance_matrix[np.ix_(location.vertex_rows, location.vertex_rows)]
        if not np.all(np.isfinite(local_distances)):
            raise RuntimeError("The local pairwise-distance matrix is incomplete.")
        estimated_distances = location.barycentric_weights @ local_distances
        return {
            "simplex_index": location.simplex_index,
            "vertex_rows": location.vertex_rows.copy(),
            "vertex_points": location.vertex_points.copy(),
            "barycentric_weights": location.barycentric_weights.copy(),
            "distances": estimated_distances,
        }


@dataclass
class RegionScalarInterpolator:
    """Piecewise-linear interpolation of one scalar over the same mesh."""

    points: np.ndarray
    valid: np.ndarray
    triangulation: Delaunay
    values: np.ndarray

    def __post_init__(self) -> None:
        self.points, self.valid = _validate_points_and_mask(self.points, self.valid)
        self.values = np.asarray(self.values, dtype=float)

    def interpolate_location(self, *, vertex_rows, barycentric_weights):
        """Interpolate using a location obtained from the shared mesh."""
        vertex_rows = np.asarray(vertex_rows, dtype=int)
        weights = np.asarray(barycentric_weights, dtype=float)
        local_values = self.values[vertex_rows]
        if not np.all(np.isfinite(local_values)):
            raise RuntimeError("The local scalar interpolation contains non-finite values.")
        return float(weights @ local_values)


@dataclass
class RegionInterpolationBank:
    """Pairwise-distance and intrinsic-quantile interpolation for one region."""

    region: str
    distance: RegionPairwiseDistanceInterpolator
    quantile: RegionScalarInterpolator
    bin_edges: np.ndarray

    def query(self, q, *, tol=1e-10):
        """Evaluate all local distances and the interpolated query quantile."""
        result = self.distance.query(q, tol=tol)
        vertex_rows = np.asarray(result["vertex_rows"], dtype=int)
        weights = np.asarray(result["barycentric_weights"], dtype=float)
        query_quantile = self.quantile.interpolate_location(
            vertex_rows=vertex_rows, barycentric_weights=weights
        )
        vertex_quantiles = self.quantile.values[vertex_rows].copy()
        return {**result, "query_mpd_quantile": query_quantile, "vertex_mpd_quantiles": vertex_quantiles}


def finite_magnitude_range(map_path, *, chunk_rows=256):
    """Return the minimum and maximum finite magnitudes in one ``.npy`` map."""
    mag_map = np.load(Path(map_path), mmap_mode="r")
    minimum = np.inf
    maximum = -np.inf
    number_finite = 0
    for row_start in range(0, mag_map.shape[0], chunk_rows):
        block = np.asarray(mag_map[row_start : row_start + chunk_rows])
        finite_values = block[np.isfinite(block)]
        if finite_values.size == 0:
            continue
        minimum = min(minimum, float(finite_values.min()))
        maximum = max(maximum, float(finite_values.max()))
        number_finite += int(finite_values.size)
    if number_finite == 0:
        raise ValueError(f"Map {map_path} contains no finite magnitudes.")
    return float(minimum), float(maximum)


def histogram_magnitude_map(map_path, bin_edges, *, chunk_rows=256):
    """Calculate a normalized MPD without loading the full map into RAM."""
    edges = np.asarray(bin_edges, dtype=float)
    mag_map = np.load(Path(map_path), mmap_mode="r")
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for row_start in range(0, mag_map.shape[0], chunk_rows):
        block = np.asarray(mag_map[row_start : row_start + chunk_rows])
        values = block[np.isfinite(block)]
        if values.size == 0:
            continue
        block_counts = np.histogram(values, bins=edges)[0]
        if int(block_counts.sum()) != int(values.size):
            raise ValueError("The supplied bin_edges do not contain every finite value.")
        counts += block_counts
    return counts.astype(float) / float(counts.sum())


def read_region_dataframe(csv_path):
    """Read and validate one adaptive-sampling CSV file."""
    frame = pd.read_csv(csv_path)
    missing_columns = REGION_REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}.")
    points = frame.loc[:, PARAMETER_COLUMNS].to_numpy(dtype=float)
    valid = frame["validity"].to_numpy(dtype=float) >= 1.0
    _validate_points_and_mask(points, valid)
    return frame


def map_path_for_row(map_dir, row_number):
    """Map a zero-based CSV row number to its generated map filename."""
    return Path(map_dir) / f"microlensing_map_{int(row_number):04d}.npy"


def valid_map_paths(frame, map_dir):
    """Return ``(CSV row, map path)`` pairs for all valid sampled rows."""
    valid = frame["validity"].to_numpy(dtype=float) >= 1.0
    return [(int(row), map_path_for_row(map_dir, int(row))) for row in np.flatnonzero(valid)]


def determine_common_bin_edges(region_inputs: Mapping[str, tuple[Path, Path]], *, num_bin_edges=100):
    """Choose one set of MPD bin edges shared by all requested regions."""
    global_minimum = np.inf
    global_maximum = -np.inf
    for csv_path, map_dir in region_inputs.values():
        frame = read_region_dataframe(csv_path)
        for _, map_path in valid_map_paths(frame, map_dir):
            minimum, maximum = finite_magnitude_range(map_path)
            global_minimum = min(global_minimum, minimum)
            global_maximum = max(global_maximum, maximum)
    lower = float(np.nextafter(global_minimum, -np.inf))
    upper = float(np.nextafter(global_maximum, np.inf))
    return np.linspace(lower, upper, num_bin_edges)


def build_region_interpolators(csv_path, map_dir, bin_edges):
    """Build the distance and intrinsic-quantile interpolators for a region."""
    frame = read_region_dataframe(csv_path)
    points = frame.loc[:, PARAMETER_COLUMNS].to_numpy(dtype=float)
    valid = frame["validity"].to_numpy(dtype=float) >= 1.0
    quantile_values = frame["mpd_distance"].to_numpy(dtype=float)
    triangulation = Delaunay(points)
    map_rows = valid_map_paths(frame, map_dir)
    mpds = {row: histogram_magnitude_map(path, bin_edges) for row, path in map_rows}
    number_points = len(points)
    distance_matrix = np.full((number_points, number_points), np.nan, dtype=float)
    valid_rows = np.array([row for row, _ in map_rows], dtype=int)
    distance_matrix[valid_rows, valid_rows] = 0.0
    for position, row_a in enumerate(valid_rows):
        for row_b in valid_rows[position + 1 :]:
            distance = float(jensenshannon(mpds[int(row_a)], mpds[int(row_b)], base=2.0))
            distance_matrix[row_a, row_b] = distance
            distance_matrix[row_b, row_a] = distance
    distance = RegionPairwiseDistanceInterpolator(
        points=points, valid=valid, triangulation=triangulation, distance_matrix=distance_matrix
    )
    quantile = RegionScalarInterpolator(
        points=points, valid=valid, triangulation=triangulation, values=quantile_values
    )
    return distance, quantile


def classify_image_region(kappa, gamma, *, tol=1e-12):
    """Classify a macro image as minima, saddle, maxima, or critical."""
    kappa = float(kappa)
    gamma = abs(float(gamma))
    radial_eigenvalue = 1.0 - kappa
    separation = abs(radial_eigenvalue) - gamma
    if abs(separation) <= tol:
        return "critical"
    if radial_eigenvalue > gamma:
        return "minima"
    if -radial_eigenvalue > gamma:
        return "maxima"
    return "saddle"


def _inside_closed_range(value, bounds):
    lower, upper = map(float, bounds)
    return lower <= value <= upper


def _empty_query_result():
    return {
        "is_in_bounds": False,
        "query_region": None,
        "interpolation_status": "not_evaluated",
        "is_hit": False,
        "hit_region": None,
        "simplex_index": None,
        "simplex_rows": None,
        "barycentric_weights": None,
        "vertex_distances": None,
        "matched_row": None,
        "matched_kappa": np.nan,
        "matched_gamma": np.nan,
        "matched_s": np.nan,
        "interpolated_mpd_distance": np.nan,
        "query_mpd_quantile": np.nan,
        "matched_mpd_quantile": np.nan,
        "distance_threshold": np.nan,
        "distance_margin": np.nan,
    }


def evaluate_query(query, *, banks, kappa_range, gamma_range, smooth_fraction_range, tol=1e-10):
    """Evaluate one ``(kappa, gamma, s)`` query against the map bank."""
    result = _empty_query_result()
    query = np.asarray(query, dtype=float)
    if query.shape != (3,) or not np.all(np.isfinite(query)):
        result["interpolation_status"] = "nonfinite_or_malformed_query"
        return result
    kappa, gamma, smooth_fraction = map(float, query)
    is_in_bounds = (
        _inside_closed_range(kappa, kappa_range)
        and _inside_closed_range(gamma, gamma_range)
        and _inside_closed_range(smooth_fraction, smooth_fraction_range)
    )
    result["is_in_bounds"] = is_in_bounds
    if not is_in_bounds:
        result["interpolation_status"] = "outside_parameter_range"
        return result
    query_region = classify_image_region(kappa, gamma, tol=tol)
    result["query_region"] = query_region
    if query_region == "critical":
        result["interpolation_status"] = "critical_line"
        return result
    if query_region not in banks:
        result["interpolation_status"] = "missing_region_bank"
        return result
    try:
        bank_result = banks[query_region].query(query, tol=tol)
    except OutsideConvexHullError:
        result["interpolation_status"] = "outside_convex_hull"
        return result
    except InvalidTetrahedronError:
        result["interpolation_status"] = "invalid_tetrahedron"
        return result
    distances = np.asarray(bank_result["distances"], dtype=float)
    local_match_index = int(np.argmin(distances))
    minimum_distance = float(distances[local_match_index])
    vertex_rows = np.asarray(bank_result["vertex_rows"], dtype=int)
    vertex_points = np.asarray(bank_result["vertex_points"], dtype=float)
    weights = np.asarray(bank_result["barycentric_weights"], dtype=float)
    vertex_quantiles = np.asarray(bank_result["vertex_mpd_quantiles"], dtype=float)
    matched_row = int(vertex_rows[local_match_index])
    matched_point = vertex_points[local_match_index]
    query_quantile = float(bank_result["query_mpd_quantile"])
    matched_quantile = float(vertex_quantiles[local_match_index])
    threshold = max(query_quantile, matched_quantile)
    is_hit = bool(minimum_distance <= threshold)
    result.update(
        {
            "interpolation_status": "covered",
            "is_hit": is_hit,
            "hit_region": query_region if is_hit else None,
            "simplex_index": int(bank_result["simplex_index"]),
            "simplex_rows": json.dumps(vertex_rows.tolist()),
            "barycentric_weights": json.dumps(weights.tolist()),
            "vertex_distances": json.dumps(distances.tolist()),
            "matched_row": matched_row,
            "matched_kappa": float(matched_point[0]),
            "matched_gamma": float(matched_point[1]),
            "matched_s": float(matched_point[2]),
            "interpolated_mpd_distance": minimum_distance,
            "query_mpd_quantile": query_quantile,
            "matched_mpd_quantile": matched_quantile,
            "distance_threshold": threshold,
            "distance_margin": threshold - minimum_distance,
        }
    )
    return result


def legacy_query_table(
    source: Path,
    queries: pd.DataFrame,
    kappa_range: Sequence[float] = (0.05, 2.0),
    gamma_range: Sequence[float] = (0.05, 2.0),
    smooth_fraction_range: Sequence[float] = (0.01, 0.99),
) -> pd.DataFrame:
    """Run the original query pipeline of run_mpd_interpolator.py on a legacy output directory."""
    region_inputs = {region: (source / f"{region}_data.csv", source / "maps" / region) for region in REGIONS}
    bin_edges = determine_common_bin_edges(region_inputs)
    banks = {}
    for region, (csv_path, map_dir) in region_inputs.items():
        distance, quantile = build_region_interpolators(csv_path, map_dir, bin_edges)
        banks[region] = RegionInterpolationBank(region, distance, quantile, bin_edges)
    records = [
        evaluate_query(
            query,
            banks=banks,
            kappa_range=kappa_range,
            gamma_range=gamma_range,
            smooth_fraction_range=smooth_fraction_range,
        )
        for query in queries.loc[:, PARAMETER_COLUMNS].to_numpy(dtype=float)
    ]
    return pd.DataFrame.from_records(records, index=queries.index)
