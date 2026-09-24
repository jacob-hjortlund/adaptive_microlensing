import numpy as np
import pytest
from adaptive_microlensing import InvalidMapError, MapSpec, VariabilitySpec
from adaptive_microlensing.mpd import (
    bin_edges_from_range,
    finite_range,
    histogram_with_overflow,
    intrinsic_quantile,
    js_distances,
    pairwise_js_distances,
)
from legacy_reference import mpd_distance_quantile
from scipy.spatial.distance import jensenshannon

SPEC = VariabilitySpec(map=MapSpec(4.0, 0.1), window_half_length=1.0, n_bin_edges=12)


def test_finite_range_ignores_non_finite_pixels_and_chunks():
    """The finite range ignores non-finite pixels and does not depend on the chunk size."""
    rng = np.random.default_rng(1)
    mag_map = rng.normal(size=(50, 40))
    mag_map[3, 4] = np.inf
    mag_map[7, 1] = -np.inf
    mag_map[9, 9] = np.nan
    finite = mag_map[np.isfinite(mag_map)]
    assert finite_range(mag_map, chunk_rows=7) == (finite.min(), finite.max())
    with pytest.raises(InvalidMapError, match="no finite"):
        finite_range(np.full((4, 4), np.inf))


def test_bin_edges_from_range_widens_by_one_ulp():
    """Bin edges are widened by one ulp at each end."""
    edges = bin_edges_from_range(-1.0, 2.0, 5)
    assert len(edges) == 5
    assert edges[0] == np.nextafter(-1.0, -np.inf)
    assert edges[-1] == np.nextafter(2.0, np.inf)
    with pytest.raises(ValueError, match="minimum < maximum"):
        bin_edges_from_range(1.0, 1.0, 5)


def test_histogram_with_overflow():
    """Pixels outside the edges land in the underflow and overflow bins."""
    values = np.array([[-5.0, 0.1, 0.4], [0.6, 0.9, 7.0], [np.inf, np.nan, 0.5]])
    mpd = histogram_with_overflow(values, np.array([0.0, 0.5, 1.0]), chunk_rows=1)
    # Seven finite pixels: one below 0, two in [0, 0.5), three in [0.5, 1], one above 1.
    np.testing.assert_allclose(mpd, np.array([1, 2, 3, 1]) / 7)
    with pytest.raises(InvalidMapError):
        histogram_with_overflow(np.full((2, 2), np.nan), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="increasing"):
        histogram_with_overflow(values, np.array([1.0, 0.0]))


def test_histogram_matches_numpy_when_everything_is_in_range():
    """Within the edges, the MPD equals a normalised numpy histogram."""
    rng = np.random.default_rng(2)
    mag_map = rng.normal(size=(300, 20))
    edges = bin_edges_from_range(mag_map.min(), mag_map.max(), 10)
    mpd = histogram_with_overflow(mag_map, edges)
    expected = np.histogram(mag_map, bins=edges)[0] / mag_map.size
    assert mpd[0] == 0.0 and mpd[-1] == 0.0
    np.testing.assert_array_equal(mpd[1:-1], expected)


def test_js_distances_match_scipy():
    """Vectorised Jensen-Shannon distances match SciPy."""
    rng = np.random.default_rng(3)
    p = rng.random(15)
    others = rng.random((6, 15))
    others[2, :4] = 0.0
    expected = [jensenshannon(p, q, base=2) for q in others]
    np.testing.assert_allclose(js_distances(p, others), expected, rtol=0, atol=1e-12)
    assert js_distances(p, p[np.newaxis, :])[0] == pytest.approx(0.0, abs=1e-12)


def test_pairwise_distances_are_symmetric_with_nan_for_invalid_rows():
    """The distance matrix is symmetric, zero on the diagonal and NaN for invalid rows."""
    rng = np.random.default_rng(4)
    mpds = rng.random((5, 8))
    valid = np.array([True, True, False, True, True])
    distances = pairwise_js_distances(mpds, valid)
    assert np.all(np.isnan(distances[2]))
    assert np.all(np.isnan(distances[:, 2]))
    rows = np.flatnonzero(valid)
    block = distances[np.ix_(rows, rows)]
    np.testing.assert_array_equal(block, block.T)
    np.testing.assert_array_equal(np.diag(block), 0.0)
    assert distances[0, 4] == pytest.approx(jensenshannon(mpds[0], mpds[4], base=2), abs=1e-12)


def test_intrinsic_quantile_matches_the_original_code_on_finite_maps():
    """Intrinsic quantile matches the original code on finite maps."""
    rng = np.random.default_rng(5)
    mag_map = rng.normal(0.3, 0.4, size=(80, 80))
    expected, _, _ = mpd_distance_quantile(
        mag_map,
        N=8.0,
        M=2.0,
        dL=0.1,
        bin_edges=np.linspace(mag_map.min(), mag_map.max(), 12),
        quantile=0.95,
    )
    assert intrinsic_quantile(mag_map, SPEC) == pytest.approx(expected, rel=1e-12)


def test_intrinsic_quantile_survives_infinite_pixels():
    """The original code produced NaN bin edges here and marked the point invalid."""
    rng = np.random.default_rng(6)
    mag_map = rng.normal(size=(80, 80))
    mag_map[10, 10] = np.inf
    with pytest.raises(ValueError, match="bin_edges"), np.errstate(invalid="ignore"):
        mpd_distance_quantile(
            mag_map, N=8.0, M=2.0, dL=0.1, bin_edges=np.linspace(np.min(mag_map), np.max(mag_map), 12)
        )
    assert np.isfinite(intrinsic_quantile(mag_map, SPEC))


def test_intrinsic_quantile_rejects_unusable_maps():
    """Constant maps, empty windows and wrong shapes are rejected."""
    with pytest.raises(InvalidMapError, match="constant"):
        intrinsic_quantile(np.zeros((80, 80)), SPEC)
    mag_map = np.random.default_rng(7).normal(size=(80, 80))
    mag_map[:20, :20] = np.inf
    with pytest.raises(InvalidMapError, match="window"):
        intrinsic_quantile(mag_map, SPEC)
    with pytest.raises(ValueError, match="shape"):
        intrinsic_quantile(np.zeros((40, 40)), SPEC)
