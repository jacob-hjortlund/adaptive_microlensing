"""Magnification probability distributions (MPDs) and Jensen-Shannon distances.

An MPD is a normalised histogram of a map's magnitudes. A bank MPD
(:func:`histogram_with_overflow`) bins a bank map on its region's frozen bin edges and
adds an underflow and an overflow bin, so that pixels outside the edges, as in maps made
after the edges were frozen, are still counted. :meth:`MapBank.finalize` freezes the
edges with :func:`bin_edges_from_range`, from the ranges that :func:`finite_range`
measured on the region's valid bank maps.

MPDs are compared by their Jensen-Shannon distance in base 2 (:func:`js_distances`),
which lies in [0, 1]. :class:`~adaptive_microlensing.interpolation.RegionIndex` keeps the
distances between the MPDs of all valid entries (:func:`pairwise_js_distances`) for the
hit rule.

An entry's own noise level, its intrinsic quantile, is measured on the windows of its
coarse map. :func:`intrinsic_quantile` measures it on the coarse map's own magnitude
range, the provisional value used while a region is built. :func:`window_histograms`
bins every window on a fine magnitude grid, which the bank stores with the entry, and
:func:`window_mpds` and :func:`window_quantile` turn those histograms into the quantile on
any bin edges, such as a region's frozen ones.

:func:`finite_range` and :func:`histogram_with_overflow` read a map a block of rows at a
time, so a memory-mapped bank map is never loaded whole.
"""

from __future__ import annotations

import numpy as np
from scipy.special import rel_entr

from .config import VariabilitySpec
from .errors import InvalidMapError

#: Default number of map rows that :func:`finite_range` and :func:`histogram_with_overflow` read at once.
CHUNK_ROWS = 256


def finite_range(mag_map: np.ndarray, *, chunk_rows: int = CHUNK_ROWS) -> tuple[float, float]:
    """Return the smallest and largest finite value of a map, reading ``chunk_rows`` rows at a time.

    NaN and infinite pixels, such as the ``+inf`` of pixels with zero magnification, are
    ignored. The map is read in blocks of rows, so a memory-mapped map is never loaded
    whole. The bank records the range of each bank map as the entry's ``mag_min`` and
    ``mag_max``, and :func:`intrinsic_quantile` bins a coarse map over its range.

    Parameters
    ----------
    mag_map : numpy.ndarray
        Magnitude map, possibly memory-mapped. It is read in blocks along its first axis.
    chunk_rows : int, optional
        Number of rows read at a time. Default is ``CHUNK_ROWS``.

    Returns
    -------
    minimum : float
        Smallest finite pixel value.
    maximum : float
        Largest finite pixel value.

    Raises
    ------
    InvalidMapError
        If the map has no finite pixels.
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


def bin_edges_from_range(minimum: float, maximum: float, n_bins: int) -> np.ndarray:
    """Return the edges of ``n_bins`` equal bins covering [minimum, maximum], widened by one ulp at each end.

    The first edge is the float just below ``minimum`` and the last is the float just
    above ``maximum``, so both extremes lie strictly inside the edges.
    :meth:`MapBank.finalize` calls it with the range of a region's valid bank maps to
    freeze the region's bin edges.

    Parameters
    ----------
    minimum : float
        Smallest value to cover.
    maximum : float
        Largest value to cover.
    n_bins : int
        Number of bins, which have ``n_bins + 1`` edges. The bank passes
        ``BankConfig.n_bins``.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(n_bins + 1,)`` of evenly spaced edges.

    Raises
    ------
    ValueError
        If ``minimum`` or ``maximum`` is not finite, or if ``minimum >= maximum``.
    """
    if not (np.isfinite(minimum) and np.isfinite(maximum)) or minimum >= maximum:
        raise ValueError(f"Expected finite minimum < maximum, got {minimum!r} and {maximum!r}.")
    lower = float(np.nextafter(minimum, -np.inf))
    upper = float(np.nextafter(maximum, np.inf))
    return np.linspace(lower, upper, n_bins + 1)


def _validated_edges(edges: np.ndarray) -> np.ndarray:
    """Return ``edges`` as a float array after checking that they can bin a histogram.

    Parameters
    ----------
    edges : numpy.ndarray
        Candidate bin edges.

    Returns
    -------
    numpy.ndarray
        ``edges`` as a one-dimensional float array.

    Raises
    ------
    ValueError
        If ``edges`` is not one-dimensional with at least two entries, or if the edges
        are not finite and strictly increasing.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("Bin edges must be a one-dimensional array with at least two entries.")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("Bin edges must be finite and strictly increasing.")
    return edges


def histogram_with_overflow(
    mag_map: np.ndarray, edges: np.ndarray, *, chunk_rows: int = CHUNK_ROWS
) -> np.ndarray:
    """Return the normalised MPD of a map's finite pixels, with an underflow and an overflow bin.

    This is a bank MPD: the map's magnitudes binned on a region's frozen edges, with the
    pixels that fall outside the edges counted in two extra bins rather than dropped.
    The map is read ``chunk_rows`` rows at a time, so a memory-mapped map is never
    loaded whole.

    Parameters
    ----------
    mag_map : numpy.ndarray
        Magnitude map, possibly memory-mapped. NaN and infinite pixels are ignored.
    edges : numpy.ndarray
        Bin edges: finite, strictly increasing and at least two of them.
    chunk_rows : int, optional
        Number of rows read at a time. Default is ``CHUNK_ROWS``.

    Returns
    -------
    numpy.ndarray
        Float array of ``len(edges) + 1`` fractions of the finite pixels, which sum to
        one: the pixels below ``edges[0]``, the ``len(edges) - 1`` histogram bins, and
        the pixels above ``edges[-1]``.

    Raises
    ------
    ValueError
        If ``edges`` is not one-dimensional with at least two entries, or if the edges
        are not finite and strictly increasing.
    InvalidMapError
        If the map has no finite pixels.

    Notes
    -----
    The histogram bins follow :func:`numpy.histogram`: each bin is half-open,
    ``[a, b)``, except the last, which also holds the pixels equal to ``edges[-1]``.
    With the ``BankConfig.n_bins + 1`` edges of a bank, the MPD has
    ``BankConfig.n_mpd_columns`` entries.
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
    """Return the Jensen-Shannon distances (base 2) from distribution ``p`` to every row of ``others``.

    Each distribution is first divided by its sum, so the inputs need not be normalised.
    The distance lies in [0, 1]: zero for identical distributions and one for
    distributions with disjoint support. It equals
    ``scipy.spatial.distance.jensenshannon(p, q, base=2)`` for each row ``q``, but all
    rows are computed at once.

    Parameters
    ----------
    p : numpy.ndarray
        Distribution of shape ``(m,)``, such as an MPD.
    others : numpy.ndarray
        Distributions of shape ``(k, m)``, or a single distribution of shape ``(m,)``.

    Returns
    -------
    numpy.ndarray
        Distances of shape ``(k,)``, one per row of ``others``. A distribution that
        contains NaN gives a NaN distance.

    Notes
    -----
    The distance is the square root of the Jensen-Shannon divergence in bits, which is
    half the sum of the Kullback-Leibler divergences of ``p`` and ``q`` from their mean
    ``(p + q) / 2``. The divergences are computed with :func:`scipy.special.rel_entr`,
    whose natural logarithms are divided by ``log(2)``.
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
    """Return the symmetric matrix of JS distances between the valid rows of ``mpds``.

    Each pair of valid rows is computed once, with :func:`js_distances`. Every element
    that involves an invalid row is NaN.
    :class:`~adaptive_microlensing.interpolation.RegionIndex` builds its distance matrix
    with this function.

    Parameters
    ----------
    mpds : numpy.ndarray
        MPDs of shape ``(n, m)``, one row per entry. The rows of invalid entries are
        never read, so they may hold NaN.
    valid : numpy.ndarray
        Boolean array of shape ``(n,)``, true for the rows to compare.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, n)``: the JS distance between rows ``i`` and ``j`` when both
        are valid, zero on the diagonal of a valid row, and NaN in every row and column
        of an invalid one.
    """
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
    """Return the quantile of the JS distances between all pairs of windows of a coarse map.

    This is an entry's intrinsic quantile, its own noise level: how far apart the MPDs
    of different patches of one map can be. The map is cut into
    ``spec.windows_per_axis**2`` non-overlapping square windows, ``spec.window_pixels``
    pixels a side, which tile it exactly. The bank measures it on the coarse
    (variability) map of every new entry.

    Parameters
    ----------
    mag_map : numpy.ndarray
        Coarse magnitude map of shape ``(n, n)``, with ``n = spec.map.num_pixels``. NaN
        and infinite pixels are ignored.
    spec : VariabilitySpec
        Coarse-map geometry, window size, number of bin edges and quantile.

    Returns
    -------
    float
        The ``spec.quantile`` quantile of the JS distances between the window MPDs,
        between 0 and 1.

    Raises
    ------
    ValueError
        If ``mag_map`` does not have shape ``(n, n)``.
    InvalidMapError
        If the map has no finite pixels, if all its finite pixels have the same value,
        or if a window has no finite pixels.

    Notes
    -----
    Each window's MPD is a :func:`numpy.histogram` of its finite pixels in
    ``spec.n_bins`` equal bins from the smallest to the largest finite
    pixel of the whole map, divided by the window's number of finite pixels. Unlike a
    bank MPD, it has no overflow bins, since no finite pixel lies outside these edges.
    With ``W`` windows, the quantile is :func:`numpy.quantile`, with its default linear
    method, of the ``W * (W - 1) / 2`` distances between distinct windows, computed by
    :func:`window_quantile`.
    """
    n = spec.map.num_pixels
    if mag_map.shape != (n, n):
        raise ValueError(f"Expected a coarse map of shape {(n, n)}, got {mag_map.shape}.")
    minimum, maximum = finite_range(mag_map)
    if minimum == maximum:
        raise InvalidMapError("The coarse map is constant.")
    edges = np.linspace(minimum, maximum, spec.n_bins + 1)
    size = spec.window_pixels
    windows = []
    for row in range(spec.windows_per_axis):
        for col in range(spec.windows_per_axis):
            block = mag_map[row * size : (row + 1) * size, col * size : (col + 1) * size]
            values = block[np.isfinite(block)]
            if values.size == 0:
                raise InvalidMapError("A coarse-map window contains no finite pixels.")
            windows.append(np.histogram(values, bins=edges)[0] / values.size)
    return window_quantile(np.asarray(windows), spec.quantile)


def window_histograms(mag_map: np.ndarray, spec: VariabilitySpec) -> tuple[int, np.ndarray]:
    """Histogram every window of a coarse map on the grid of ``spec.histogram_bin_width``.

    The map is cut into windows exactly as in :func:`intrinsic_quantile`. With
    ``w = spec.histogram_bin_width``, grid bin ``b`` holds the magnitudes in
    ``[b * w, (b + 1) * w)``, so a finite pixel of magnitude ``m`` falls in bin
    ``floor(m / w)``. The grid has no ends; only the bins between the map's smallest and
    largest finite pixel are returned. The bank stores the result with each entry, and
    :func:`window_mpds` turns it into window MPDs on any bin edges.

    Parameters
    ----------
    mag_map : numpy.ndarray
        Coarse magnitude map of shape ``(n, n)``, with ``n = spec.map.num_pixels``. NaN
        and infinite pixels are ignored.
    spec : VariabilitySpec
        Coarse-map geometry, window size and grid bin width.

    Returns
    -------
    start : int
        Grid bin of the map's smallest finite pixel, ``floor(min / w)``.
    counts : numpy.ndarray
        Int32 array of shape ``(spec.windows_per_axis**2, n_cols)``, windows in row-major
        order, where ``counts[k, c]`` is the number of finite pixels of window ``k`` in grid
        bin ``start + c`` and ``n_cols = floor(max / w) - start + 1``.

    Raises
    ------
    ValueError
        If ``mag_map`` does not have shape ``(n, n)``.
    InvalidMapError
        If the map has no finite pixels, if all its finite pixels have the same value,
        or if a window has no finite pixels.
    """
    n = spec.map.num_pixels
    if mag_map.shape != (n, n):
        raise ValueError(f"Expected a coarse map of shape {(n, n)}, got {mag_map.shape}.")
    minimum, maximum = finite_range(mag_map)
    if minimum == maximum:
        raise InvalidMapError("The coarse map is constant.")
    values = np.asarray(mag_map)
    finite = np.isfinite(values)
    bins = np.floor(values[finite] / spec.histogram_bin_width).astype(np.int64)
    start = int(bins.min())
    n_cols = int(bins.max()) - start + 1
    per_axis = spec.windows_per_axis
    block = np.arange(n) // spec.window_pixels
    window = (block[:, np.newaxis] * per_axis + block[np.newaxis, :])[finite]
    counts = np.bincount(window * n_cols + (bins - start), minlength=per_axis**2 * n_cols)
    counts_2d = np.asarray(counts.reshape(per_axis**2, n_cols), dtype=np.int64)
    if np.any(counts_2d.sum(axis=1) == 0):
        raise InvalidMapError("A coarse-map window contains no finite pixels.")
    return start, counts_2d.astype(np.int32)


def window_mpds(start: int, counts: np.ndarray, edges: np.ndarray, bin_width: float) -> np.ndarray:
    """Return window MPDs on ``edges``, with underflow and overflow bins, from grid histograms.

    Each edge is first moved to the nearest grid line, ``rint(edge / bin_width)``, which
    moves it by at most half a grid bin. Grid bins below the first edge go to the
    underflow bin, those at or above the last edge to the overflow bin, and the others to
    the bin between the edges that enclose them. Each row is then divided by its total. The
    layout is that of a bank MPD (:func:`histogram_with_overflow`), so on a region's frozen
    edges the result can be compared with the region's bank MPDs.

    Parameters
    ----------
    start : int
        Grid bin of column 0 of ``counts``, as returned by :func:`window_histograms`.
    counts : numpy.ndarray
        Window histograms of shape ``(n_windows, n_cols)``, as returned by
        :func:`window_histograms`.
    edges : numpy.ndarray
        Bin edges: finite, strictly increasing and at least two of them.
    bin_width : float
        Width of the grid bins, ``VariabilitySpec.histogram_bin_width``.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(n_windows, len(edges) + 1)``: per window, the fractions of
        its pixels below the first edge, in each of the ``len(edges) - 1`` bins, and at or
        above the last edge. Each row sums to one.

    Raises
    ------
    ValueError
        If ``edges`` is not one-dimensional with at least two entries, if the edges are not
        finite and strictly increasing, or if ``counts`` is not two-dimensional.
    InvalidMapError
        If a window's histogram is empty.

    Notes
    -----
    A pixel changes bin compared with exact binning only if it lies between an edge and the
    grid line the edge was moved to, so within half a grid bin of the edge. Edges that move
    to the same grid line leave an empty bin between them.
    """
    edges = _validated_edges(edges)
    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError(f"Window histograms must be two-dimensional, got shape {counts.shape}.")
    columns = np.clip(np.rint(edges / bin_width).astype(np.int64) - int(start), 0, counts.shape[1])
    cumulative = np.zeros((counts.shape[0], counts.shape[1] + 1), dtype=np.int64)
    cumulative[:, 1:] = np.cumsum(counts, axis=1, dtype=np.int64)
    totals = cumulative[:, -1]
    if np.any(totals == 0):
        raise InvalidMapError("A window histogram contains no pixels.")
    at_edges = cumulative[:, columns]
    sums = np.column_stack([at_edges[:, 0], np.diff(at_edges, axis=1), totals - at_edges[:, -1]])
    return sums / totals[:, np.newaxis]


def window_quantile(mpds: np.ndarray, quantile: float) -> float:
    """Return the ``quantile`` of the JS distances between all pairs of distinct window MPDs.

    This is the last step of an intrinsic quantile: :func:`intrinsic_quantile` calls it on
    the MPDs of a coarse map's windows on the map's own range, and the bank calls it on
    :func:`window_mpds` on a region's frozen edges.

    Parameters
    ----------
    mpds : numpy.ndarray
        Window MPDs, one per row, of shape ``(n_windows, n_columns)``.
    quantile : float
        Quantile in [0, 1], taken with :func:`numpy.quantile` and its default linear method.

    Returns
    -------
    float
        The quantile of the ``n_windows * (n_windows - 1) / 2`` JS distances, between 0 and 1.

    Raises
    ------
    ValueError
        If ``mpds`` is not two-dimensional with at least two rows.
    """
    mpds = np.asarray(mpds, dtype=float)
    if mpds.ndim != 2 or len(mpds) < 2:
        raise ValueError("The quantile needs the MPDs of at least two windows, one per row.")
    distances = np.concatenate([js_distances(mpds[i], mpds[i + 1 :]) for i in range(len(mpds) - 1)])
    return float(np.quantile(distances, quantile))
