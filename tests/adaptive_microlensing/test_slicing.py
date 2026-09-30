import numpy as np
import pytest
from adaptive_microlensing import MapBank, coverage_grid, slice_region
from adaptive_microlensing.lensing import REGIONS, region_hull, region_hull_vertices
from adaptive_microlensing.slicing import AXES
from helpers import TETRAHEDRON, add_entry, small_config
from scipy.spatial import ConvexHull

# A kappa inside each region, for slices at fixed kappa.
REGION_KAPPA = {"minima": 0.4, "saddle": 1.0, "maxima": 1.6}
# A 3 x 3 x 3 lattice of minima points: its Delaunay faces lie in the lattice planes.
LATTICE = [(k, g, s) for k in (0.1, 0.3, 0.5) for g in (0.1, 0.2, 0.3) for s in (0.1, 0.5, 0.9)]


def _areas(piece):
    corners = piece.points[piece.triangles]
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    return 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])


def _lift(piece, planar):
    """The 3D point of an in-plane point."""
    point = np.empty(3)
    point[AXES.index(piece.axis)] = piece.value
    point[[AXES.index(name) for name in piece.plane_axes]] = planar
    return point


def _assert_conforming(piece):
    """No duplicate or zero-area triangles, and every point is used."""
    assert len(np.unique(np.sort(piece.triangles, axis=1), axis=0)) == len(piece.triangles)
    assert np.all(_areas(piece) > 0.0)
    np.testing.assert_array_equal(np.unique(piece.triangles), np.arange(len(piece.points)))


@pytest.fixture(scope="module")
def lattice_bank(tmp_path_factory):
    """A bank whose minima region holds the LATTICE entries."""
    bank = MapBank.create(tmp_path_factory.mktemp("lattice") / "bank", small_config())
    for point in LATTICE:
        add_entry(bank, "minima", point)
    yield bank
    bank.close()


def test_slice_of_one_tetrahedron(tetra_bank):
    """The plane s = 0.5 halves the TETRAHEDRON's edges to its apex (0.1, 0.1, 0.9)."""
    piece = slice_region(tetra_bank, "minima", s=0.5)
    assert (piece.region, piece.axis, piece.value) == ("minima", "s", 0.5)
    assert piece.plane_axes == ("kappa", "gamma")
    np.testing.assert_allclose(
        piece.points[np.lexsort(piece.points.T[::-1])], [[0.1, 0.1], [0.1, 0.3], [0.35, 0.1]], atol=1e-15
    )
    assert piece.triangles.shape == (1, 3)
    assert piece.valid.tolist() == [True]
    assert piece.simplex_index.tolist() == [0]
    index = tetra_bank._region("minima").index
    for point, quantile in zip(piece.points, piece.quantiles, strict=True):
        assert quantile == pytest.approx(index.interpolate_quantile(_lift(piece, point)), abs=1e-12)


def test_an_invalid_vertex_marks_its_cells_and_edges(failing_bank):
    """The failed vertex (0.6, 0.1, 0.1) makes the cell invalid and its edge's crossing NaN."""
    piece = slice_region(failing_bank, "minima", s=0.5)
    assert piece.valid.tolist() == [False]
    assert np.isnan(piece.quantiles).sum() == 1
    nan_point = piece.points[np.isnan(piece.quantiles)][0]
    np.testing.assert_allclose(nan_point, [0.35, 0.1], atol=1e-15)


@pytest.mark.parametrize("region", REGIONS)
def test_slice_area_equals_the_hull_cross_section(built_bank, region):
    """At every s, including the bounds, the slice covers the region's hull polygon exactly once."""
    domain = built_bank.config.domain
    index = built_bank._region(region).index
    for vertex in region_hull(region, domain, built_bank.config.design.n_boundary):
        assert index.coincident(vertex) is not None
    polygon = region_hull_vertices(
        region,
        domain.kappa_range,
        domain.gamma_range,
        domain.max_macro_magnification,
        built_bank.config.design.n_boundary,
    )
    target = ConvexHull(polygon).volume  # the enclosed area in 2D
    for s in (domain.s_range[0], 0.3, 0.5, domain.s_range[1]):
        piece = slice_region(built_bank, region, s=s)
        _assert_conforming(piece)
        assert piece.valid.all()
        assert _areas(piece).sum() == pytest.approx(target, rel=1e-12)


@pytest.mark.parametrize("region", REGIONS)
@pytest.mark.parametrize("axis", AXES)
def test_slice_interpolates_exactly_like_the_bank(built_bank, region, axis):
    """Linear interpolation on a slice triangle equals the bank's interpolation at the 3D point."""
    value = {"kappa": REGION_KAPPA[region], "gamma": 0.3, "s": 0.37}[axis]
    piece = slice_region(built_bank, region, **{axis: value})
    assert len(piece.triangles) > 0
    index = built_bank._region(region).index
    rng = np.random.default_rng(0)
    for triangle in piece.triangles[piece.valid]:
        weights = rng.dirichlet(np.ones(3))
        expected = index.interpolate_quantile(_lift(piece, weights @ piece.points[triangle]))
        assert weights @ piece.quantiles[triangle] == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    ("plane", "area"),
    [
        ({"s": 0.1}, 0.08),  # the bottom layer
        ({"s": 0.5}, 0.08),  # a layer of faces between tetrahedra
        ({"s": 0.9}, 0.08),  # the top layer
        ({"s": 0.3}, 0.08),  # between layers
        ({"kappa": 0.1}, 0.16),
        ({"kappa": 0.3}, 0.16),
        ({"kappa": 0.5}, 0.16),
        ({"gamma": 0.2}, 0.32),
    ],
)
def test_planes_through_entries_and_faces(lattice_bank, plane, area):
    """Planes through lattice entries and faces give one conforming cover of the cross-section."""
    piece = slice_region(lattice_bank, "minima", **plane)
    _assert_conforming(piece)
    assert _areas(piece).sum() == pytest.approx(area, rel=1e-12)


def test_regions_that_are_not_ready_or_not_finalized(bank, patchy_bank):
    """A region without a mesh gives an empty slice; an unfinalized region slices normally."""
    for point in TETRAHEDRON[:3]:
        add_entry(bank, "minima", point)
    for piece in (slice_region(bank, "minima", s=0.1), slice_region(patchy_bank, "maxima", s=0.5)):
        assert piece.points.shape == (0, 2)
        assert piece.triangles.shape == (0, 3)
        assert piece.quantiles.shape == piece.valid.shape == piece.simplex_index.shape == (0,)
    saddle = slice_region(patchy_bank, "saddle", s=0.5)
    assert len(saddle.triangles) > 0 and saddle.valid.all()
    assert np.all(np.isfinite(saddle.quantiles))
    minima = slice_region(patchy_bank, "minima", s=0.5)
    assert not minima.valid.all() and minima.valid.any()
    assert np.isnan(minima.quantiles).any()


@pytest.mark.parametrize(
    ("plane", "message"),
    [
        ({}, "exactly one"),
        ({"s": 0.5, "kappa": 0.3}, "exactly one"),
        ({"s": float("nan")}, "outside the domain"),
        ({"s": 1.5}, "outside the domain"),
        ({"kappa": -0.1}, "outside the domain"),
        ({"gamma": 2.5}, "outside the domain"),
    ],
)
def test_plane_arguments_are_validated(tetra_bank, plane, message):
    """Slices and grids need exactly one plane coordinate inside the domain."""
    with pytest.raises(ValueError, match=message):
        slice_region(tetra_bank, "minima", **plane)
    with pytest.raises(ValueError, match=message):
        coverage_grid(tetra_bank, **plane)


def test_unknown_region(tetra_bank):
    """An unknown region raises ValueError."""
    with pytest.raises(ValueError, match="Unknown region"):
        slice_region(tetra_bank, "ridge", s=0.5)


def test_a_plane_that_misses_a_region_gives_an_empty_slice(built_bank):
    """kappa = 0.3 lies in minima and saddle only, so the maxima slice is empty."""
    piece = slice_region(built_bank, "maxima", kappa=0.3)
    assert piece.points.shape == (0, 2) and piece.triangles.shape == (0, 3)


def test_grid_size_is_validated(tetra_bank):
    """A grid size below 2 or not an integer raises ValueError."""
    for n in (1, 2.5, True):
        with pytest.raises(ValueError, match="n must be an integer"):
            coverage_grid(tetra_bank, s=0.5, n=n)


def test_coverage_grid_agrees_with_query(patchy_bank):
    """Every grid node carries exactly what bank.query returns at that point."""
    domain = patchy_bank.config.domain
    grid = coverage_grid(patchy_bank, s=0.5, n=15)
    assert (grid.axis, grid.value, grid.plane_axes) == ("s", 0.5, ("kappa", "gamma"))
    assert grid.domain == domain
    np.testing.assert_allclose(grid.x, np.linspace(*domain.kappa_range, 15))
    np.testing.assert_allclose(grid.y, np.linspace(*domain.gamma_range, 15))
    assert grid.status.shape == grid.region.shape == grid.margin.shape == (15, 15)
    assert len(grid.table) == 225
    for iy, y in enumerate(grid.y):
        for ix, x in enumerate(grid.x):
            result = patchy_bank.query(x, y, 0.5)
            assert grid.status[iy, ix] == result.status.value
            assert grid.region[iy, ix] == result.region
            assert grid.is_hit[iy, ix] == result.is_hit
            np.testing.assert_equal(grid.margin[iy, ix], result.margin)
            np.testing.assert_equal(grid.interpolated_distance[iy, ix], result.interpolated_distance)
    assert {"hit", "region_not_ready", "outside_domain"} <= set(grid.status.ravel())


def test_coverage_grid_in_a_kappa_plane(built_bank):
    """A kappa plane spans gamma and s; nodes on a critical line have no region."""
    grid = coverage_grid(built_bank, kappa=0.5, n=40)
    assert grid.plane_axes == ("gamma", "s")
    np.testing.assert_allclose(grid.table["kappa"], 0.5)
    critical = grid.status == "critical_line"
    assert critical.any()
    assert all(region is None for region in grid.region[critical])
