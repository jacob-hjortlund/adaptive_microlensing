"""Shared pytest fixtures."""

import pytest
from adaptive_microlensing import MapBank, StoppingCriteria, SyntheticGenerator
from adaptive_microlensing.lensing import REGIONS
from helpers import FAILURE_BOX, PATCH_BOX, TETRAHEDRON, add_entry, small_config


@pytest.fixture
def generator():
    """The default synthetic generator."""
    return SyntheticGenerator()


@pytest.fixture
def config(generator):
    """A small synthetic bank configuration."""
    return small_config(generator)


@pytest.fixture
def bank(tmp_path, config):
    """An empty bank, open for writing."""
    bank = MapBank.create(tmp_path / "bank", config)
    yield bank
    bank.close()


@pytest.fixture
def tetra_bank(bank, generator):
    """A bank whose minima region holds the four TETRAHEDRON entries and is finalized."""
    for point in TETRAHEDRON:
        add_entry(bank, "minima", point, generator)
    bank.finalize("minima")
    return bank


@pytest.fixture
def failing_generator():
    """A synthetic generator that fails inside FAILURE_BOX."""
    return SyntheticGenerator(failure_boxes=[FAILURE_BOX])


@pytest.fixture
def failing_bank(tmp_path, failing_generator):
    """Like tetra_bank, but the second tetrahedron vertex failed and is invalid."""
    bank = MapBank.create(tmp_path / "failing", small_config(failing_generator))
    for point in TETRAHEDRON:
        add_entry(bank, "minima", point, failing_generator)
    bank.finalize("minima")
    yield bank
    bank.close()


@pytest.fixture(scope="session")
def built_bank(tmp_path_factory):
    """A read-only bank whose three regions are built to 30 valid entries and finalized.

    Thirty valid entries exceed every region's design-hull vertex count (at most 18), so
    each region's mesh fills its design hull.
    """
    path = tmp_path_factory.mktemp("built") / "bank"
    with MapBank.create(path, small_config()) as bank:
        for region in REGIONS:
            bank.build(region, StoppingCriteria(max_valid_points=30))
            bank.finalize(region)
    bank = MapBank.open(path)
    yield bank
    bank.close()


@pytest.fixture(scope="session")
def patchy_bank(tmp_path_factory):
    """A read-only bank with failures in minima (finalized), an unfinalized saddle and no maxima."""
    generator = SyntheticGenerator(failure_boxes=[PATCH_BOX])
    path = tmp_path_factory.mktemp("patchy") / "bank"
    with MapBank.create(path, small_config(generator)) as bank:
        bank.build("minima", StoppingCriteria(max_valid_points=40), generator=generator)
        bank.finalize("minima")
        bank.build("saddle", StoppingCriteria(max_valid_points=30), generator=generator)
    bank = MapBank.open(path)
    yield bank
    bank.close()
