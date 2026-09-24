"""Magnification probability distributions (MPDs) and Jensen-Shannon distances."""

from __future__ import annotations

import numpy as np
from scipy.special import rel_entr

from .config import VariabilitySpec
from .errors import InvalidMapError

CHUNK_ROWS = 256


def finite_range(mag_map: np.ndarray, *, chunk_rows: int = CHUNK_ROWS) -> tuple[float, float]:
    """Smallest and largest finite value of a map, reading ``chunk_rows`` rows at a time.

    Works on memory-mapped arrays without loading them whole.
    """
    minimum = np.inf
    maximum = -np.inf
    found = False
    for start in range(0, mag_map.shape[0], chunk_rows):
        block = np.asarray(mag_map[start : start + chunk_rows])
        values = block[np.isfinite(block)]
        if values.size:
            found = True
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
    if not found:
        raise InvalidMapError("The map contains no finite pixels.")
    return float(minimum), float(maximum)


def bin_edges_from_range(minimum: float, maximum: float, n_bin_edges: int) -> np.ndarray:
    """``n_bin_edges`` evenly spaced edges covering [minimum, maximum], widened by one ulp at each end."""
    if not (np.isfinite(minimum) and np.isfinite(maximum)) or minimum >= maximum:
        raise ValueError(f"Expected finite minimum < maximum, got {minimum!r} and {maximum!r}.")
    lower = float(np.nextafter(minimum, -np.inf))
    upper = float(np.nextafter(maximum, np.inf))
    return np.linspace(lower, upper, n_bin_edges)


def _validated_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("Bin edges must be a one-dimensional array with at least two entries.")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("Bin edges must be finite and strictly increasing.")
    return edges


def histogram_with_overflow(
    mag_map: np.ndarray, edges: np.ndarray, *, chunk_rows: int = CHUNK_ROWS
) -> np.ndarray:
    """Normalised MPD of a map's finite pixels, with an underflow and an overflow bin.

    Returns ``len(edges) + 1`` fractions: pixels below ``edges[0]``, the
    ``len(edges) - 1`` histogram bins, and pixels above ``edges[-1]``.
    """
    edges = _validated_edges(edges)
    counts = np.zeros(len(edges) + 1, dtype=np.int64)
    for start in range(0, mag_map.shape[0], chunk_rows):
        block = np.asarray(mag_map[start : start + chunk_rows])
        values = block[np.isfinite(block)]
        if values.size == 0:
            continue
        below = values < edges[0]
        above = values > edges[-1]
        counts[0] += int(below.sum())
        counts[-1] += int(above.sum())
        counts[1:-1] += np.histogram(values[~(below | above)], bins=edges)[0]
    total = int(counts.sum())
    if total == 0:
        raise InvalidMapError("The map contains no finite pixels.")
    return counts / total


def js_distances(p: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Jensen-Shannon distances (base 2) from distribution ``p`` to every row of ``others``.

    Equals ``scipy.spatial.distance.jensenshannon(p, q, base=2)`` for each row ``q``.
    """
    p = np.asarray(p, dtype=float)
    q = np.atleast_2d(np.asarray(others, dtype=float))
    p = p / p.sum()
    q = q / q.sum(axis=1, keepdims=True)
    m = (p + q) / 2.0
    js = rel_entr(p, m).sum(axis=1) + rel_entr(q, m).sum(axis=1)
    js /= np.log(2.0)
    return np.sqrt(js / 2.0)


def pairwise_js_distances(mpds: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Symmetric matrix of JS distances between the valid rows of ``mpds``; NaN elsewhere."""
    mpds = np.asarray(mpds, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    distances = np.full((len(mpds), len(mpds)), np.nan)
    rows = np.flatnonzero(valid)
    for position, row in enumerate(rows):
        later = rows[position + 1 :]
        if later.size:
            values = js_distances(mpds[row], mpds[later])
            distances[row, later] = values
            distances[later, row] = values
        distances[row, row] = 0.0
    return distances


def intrinsic_quantile(mag_map: np.ndarray, spec: VariabilitySpec) -> float:
    """Quantile of the JS distances between all pairs of non-overlapping windows of a coarse map.

    The window MPDs share ``spec.n_bin_edges`` edges spanning the map's finite range.
    """
    n = spec.map.num_pixels
    if mag_map.shape != (n, n):
        raise ValueError(f"Expected a coarse map of shape {(n, n)}, got {mag_map.shape}.")
    minimum, maximum = finite_range(mag_map)
    if minimum == maximum:
        raise InvalidMapError("The coarse map is constant.")
    edges = np.linspace(minimum, maximum, spec.n_bin_edges)
    size = spec.window_pixels
    windows = []
    for row in range(spec.windows_per_axis):
        for col in range(spec.windows_per_axis):
            block = mag_map[row * size : (row + 1) * size, col * size : (col + 1) * size]
            values = block[np.isfinite(block)]
            if values.size == 0:
                raise InvalidMapError("A coarse-map window contains no finite pixels.")
            windows.append(np.histogram(values, bins=edges)[0] / values.size)
    mpds = np.asarray(windows)
    distances = np.concatenate([js_distances(mpds[i], mpds[i + 1 :]) for i in range(len(mpds) - 1)])
    return float(np.quantile(distances, spec.quantile))
