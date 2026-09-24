"""Shared pytest fixtures."""

import pytest
from adaptive_microlensing import MapBank, SyntheticGenerator
from helpers import FAILURE_BOX, TETRAHEDRON, add_entry, small_config


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
