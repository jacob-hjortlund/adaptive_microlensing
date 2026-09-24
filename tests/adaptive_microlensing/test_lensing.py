import numpy as np
import pytest
from adaptive_microlensing import DomainSpec
from adaptive_microlensing.lensing import (
    REGIONS,
    classify_image,
    extrude,
    in_box,
    in_domain,
    macro_magnification,
    region_hull,
    region_hull_vertices,
)


@pytest.mark.parametrize(
    ("kappa", "gamma", "expected"),
    [
        (0.4, 0.3, "minima"),
        (0.4, -0.3, "minima"),
        (1.0, 0.5, "saddle"),
        (0.2, 0.9, "saddle"),
        (1.8, 0.3, "maxima"),
        (1.8, -0.3, "maxima"),
        (0.5, 0.5, "critical"),
        (1.5, 0.5, "critical"),
    ],
)
def test_classify_image(kappa, gamma, expected):
    """Images are classified by the signs of the Jacobian eigenvalues, using abs(gamma)."""
    assert classify_image(kappa, gamma) == expected


def test_critical_tolerance():
    """Points within 1e-10 of a critical line are critical; non-finite input is rejected."""
    assert classify_image(0.5, 0.5 + 5e-11) == "critical"
    assert classify_image(0.5, 0.5 + 1e-9) == "saddle"
    with pytest.raises(ValueError, match="finite"):
        classify_image(float("nan"), 0.1)


def test_macro_magnification():
    """The signed macro magnification is infinite on a critical line."""
    assert macro_magnification(0.5, 0.0) == 4.0
    assert macro_magnification(1.0, 0.5) == -4.0
    assert macro_magnification(0.5, 0.5) == float("inf")


def test_domain_membership():
    """The box uses the raw gamma, and the domain adds the magnification cut."""
    domain = DomainSpec()
    assert in_box(0.1, 0.1, 0.5, domain)
    assert not in_box(0.1, -0.1, 0.5, domain)  # gamma is not made absolute
    assert not in_box(0.1, 0.1, 1.0, domain)
    assert in_domain(0.1, 0.1, 0.5, domain)
    # Inside the box but |mu| = 1 / |0.25 - 0.249001| > 100.
    assert in_box(0.5, 0.499, 0.5, domain)
    assert not in_domain(0.5, 0.499, 0.5, domain)


@pytest.mark.parametrize("region", REGIONS)
def test_hull_vertices_lie_in_region_box_and_mu_cut(region):
    """Every hull vertex lies in its region, in the box and inside the magnification cut."""
    domain = DomainSpec()
    vertices = region_hull_vertices(region, domain.kappa_range, domain.gamma_range, 100.0, 6)
    assert vertices is not None and len(vertices) >= 3
    for kappa, gamma in vertices:
        assert domain.kappa_range[0] - 1e-12 <= kappa <= domain.kappa_range[1] + 1e-12
        assert domain.gamma_range[0] - 1e-12 <= gamma <= domain.gamma_range[1] + 1e-12
        assert abs(macro_magnification(kappa, gamma)) <= 100.0 * (1 + 1e-6)
        assert classify_image(kappa, gamma) in (region, "critical")


def test_empty_regions_return_none():
    """Regions that miss the box have no hull; unknown regions are an error."""
    assert region_hull_vertices("minima", (1.1, 2.0), (0.05, 2.0), 100.0) is None
    assert region_hull_vertices("maxima", (0.05, 0.9), (0.05, 2.0), 100.0) is None
    with pytest.raises(ValueError, match="Unknown region"):
        region_hull_vertices("ring", (0.05, 2.0), (0.05, 2.0))


def test_extrusion_makes_a_prism():
    """Extruding a polygon along s makes a prism."""
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    prism = extrude(square, (0.2, 0.8))
    assert prism.shape == (8, 3)
    assert set(prism[:, 2]) == {0.2, 0.8}
    hull = region_hull("saddle", DomainSpec(), 6)
    assert hull is not None and hull.shape[1] == 3
