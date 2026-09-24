import numpy as np
import pytest
from adaptive_microlensing.interpolation import RegionIndex
from adaptive_microlensing.mpd import pairwise_js_distances

UNIT = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
QUANTILES = np.array([0.10, 0.20, 0.30, 0.40])
# A hand-built distance matrix between the four unit-tetrahedron vertices.
DISTANCES = np.array(
    [
        [0.0, 0.6, 0.7, 0.8],
        [0.6, 0.0, 0.5, 0.9],
        [0.7, 0.5, 0.0, 0.4],
        [0.8, 0.9, 0.4, 0.0],
    ]
)


def _index(valid=(True, True, True, True)):
    return RegionIndex(UNIT, np.array(valid), QUANTILES, mpds=np.ones((4, 3)), distances=DISTANCES)


def test_weights_at_centroid_vertex_face_and_outside():
    """Barycentric weights are exact at the centroid, a vertex and a face; outside points have none."""
    index = _index()
    centroid = index.locate(np.array([0.25, 0.25, 0.25]))
    assert centroid is not None
    np.testing.assert_allclose(centroid.weights[np.argsort(centroid.vertices)], 0.25)
    vertex = index.locate(np.array([1.0, 0.0, 0.0]))
    assert vertex is not None
    assert vertex.weights[list(vertex.vertices).index(1)] == 1.0
    face = index.locate(np.array([0.3, 0.3, 0.0]))
    assert face is not None
    assert face.weights[list(face.vertices).index(3)] == 0.0
    assert index.locate(np.array([1.0, 1.0, 1.0])) is None


def test_hit_rule_on_a_hand_built_distance_matrix():
    """The hit rule interpolates distances and quantiles as the original script did."""
    index = _index()
    point = np.array([0.1, 0.1, 0.1])  # weights 0.7, 0.1, 0.1, 0.1 on vertices 0..3
    location = index.locate(point)
    assert location is not None
    evaluation = index.evaluate(location)
    weights = np.array([0.7, 0.1, 0.1, 0.1])
    expected = weights @ DISTANCES
    order = location.vertices
    np.testing.assert_allclose(evaluation.vertex_distances, expected[order])
    assert evaluation.matched_entry_id == 0
    assert evaluation.interpolated_distance == pytest.approx(0.21)
    assert evaluation.query_quantile == pytest.approx(weights @ QUANTILES)
    assert evaluation.matched_quantile == pytest.approx(0.10)
    assert evaluation.threshold == pytest.approx(max(weights @ QUANTILES, 0.10))
    assert evaluation.is_hit is False
    assert evaluation.margin == pytest.approx(evaluation.threshold - 0.21)
    near_vertex = index.locate(np.array([0.01, 0.01, 0.01]))
    assert near_vertex is not None and index.evaluate(near_vertex).is_hit


def test_invalid_vertices_block_interpolation():
    """A tetrahedron with an invalid vertex cannot be interpolated."""
    index = _index(valid=(True, False, True, True))
    location = index.locate(np.array([0.25, 0.25, 0.25]))
    assert location is not None
    assert not index.simplex_is_valid(location)
    assert index.interpolate_quantile(np.array([0.25, 0.25, 0.25])) is None
    assert _index().interpolate_quantile(np.array([0.25, 0.25, 0.25])) == pytest.approx(0.25)


def test_readiness_and_coincidence():
    """An index needs four points spanning 3D; coincidence uses a 1e-12 tolerance."""
    assert not RegionIndex(UNIT[:3], np.ones(3, bool), QUANTILES[:3]).ready
    flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    assert not RegionIndex(flat, np.ones(4, bool), QUANTILES).ready
    index = _index()
    assert index.ready
    assert index.coincident(np.array([0.0, 1.0, 0.0])) == 2
    assert index.coincident(np.array([0.0, 1.0, 1e-9])) is None
    empty = RegionIndex(np.empty((0, 3)), np.empty(0, bool), np.empty(0))
    assert empty.coincident(np.zeros(3)) is None and empty.locate(np.zeros(3)) is None


def test_extended_matches_a_full_rebuild():
    """Appending an entry computes the same distances as rebuilding the index."""
    rng = np.random.default_rng(0)
    points = rng.random((6, 3))
    valid = np.array([True, True, False, True, True, True])
    mpds = rng.random((6, 5))
    mpds[2] = np.nan
    base = RegionIndex(points[:5], valid[:5], rng.random(5), mpds[:5])
    grown = base.extended(points[5], True, 0.3, mpds[5])
    np.testing.assert_allclose(grown.distances, pairwise_js_distances(mpds, valid), rtol=0, atol=1e-12)
    with_invalid = base.extended(points[5], False, np.nan, None)
    assert np.all(np.isnan(with_invalid.distances[5]))
    without_mpds = RegionIndex(points[:5], valid[:5], rng.random(5)).extended(points[5], True, 0.3)
    assert without_mpds.distances is None and without_mpds.ready
