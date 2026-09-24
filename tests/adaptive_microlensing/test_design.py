import adaptive
import numpy as np
import pandas as pd
import pytest
from adaptive.learner.learnerND import triangle_loss
from adaptive_microlensing import (
    BankReadOnlyError,
    DomainSpec,
    MapBank,
    StoppingCriteria,
    SyntheticGenerator,
)
from adaptive_microlensing.design import failure_aware_curvature_loss, region_convex_hull
from helpers import small_config

SIMPLEX = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
SIZE = (1.0 / 6.0) ** (1.0 / 3.0)  # volume ** (1 / dim)
NO_NEIGHBOURS = [None, None, None, None]


def _values(quantiles, valid):
    return np.column_stack([quantiles, np.asarray(valid, dtype=float)])


def test_loss_cases():
    """The loss treats invalid, mixed, valid and degenerate simplices as the original script did."""
    loss = failure_aware_curvature_loss(invalid_multiplier=0.1)
    invalid = loss(SIMPLEX, _values([0, 0, 0, 0], [0, 0, 0, 0]), 1.0, NO_NEIGHBOURS, NO_NEIGHBOURS)
    assert invalid == pytest.approx(0.1 * SIZE)
    mixed = loss(SIMPLEX, _values([0.3, 0, 0.3, 0.3], [1, 0, 1, 1]), 1.0, NO_NEIGHBOURS, NO_NEIGHBOURS)
    assert mixed == pytest.approx(2.0 * SIZE)
    quantiles = np.array([0.3, 0.4, 0.35, 0.2])
    valid = loss(SIMPLEX, _values(quantiles, [1, 1, 1, 1]) * 2.0, 2.0, NO_NEIGHBOURS, NO_NEIGHBOURS)
    curvature = triangle_loss(SIMPLEX, quantiles, 1.0, NO_NEIGHBOURS, NO_NEIGHBOURS)
    assert valid == pytest.approx((curvature + 0.05 * (1 / 6) ** (5 / 3)) ** (1 / 5))
    assert loss(np.zeros((4, 3)), _values(quantiles, [1, 1, 1, 1]), 1.0, NO_NEIGHBOURS, NO_NEIGHBOURS) == 0.0


def test_build_stops_at_max_valid_points(bank):
    """Build stops at max valid points."""
    summary = bank.build("minima", StoppingCriteria(max_valid_points=12))
    entries = bank.entries("minima")
    assert summary.stop_reason == "max_valid_points"
    assert summary.n_valid == 12 == entries["valid"].sum()
    assert summary.n_entries == len(entries)
    assert entries["origin"].eq("build").all()
    assert np.isfinite(entries["simplex_loss"].iloc[-1])


def test_build_starts_from_the_hull_vertices(bank, config):
    """A build evaluates every hull vertex before it can meet a loss goal."""
    hull = region_convex_hull("minima", config)
    summary = bank.build("minima", StoppingCriteria(max_valid_points=None, simplex_loss_goal=1e9))
    assert summary.stop_reason == "simplex_loss_goal"
    assert summary.n_entries == len(hull.vertices)
    np.testing.assert_allclose(
        np.sort(bank.entries("minima")[["kappa", "gamma", "s"]].to_numpy(), axis=0),
        np.sort(hull.points[hull.vertices], axis=0),
    )


def test_build_stops_on_the_residual_goal(bank):
    """Build stops on the residual goal."""
    stop = StoppingCriteria(max_valid_points=200, residual_goal=100.0, min_num_residuals=3)
    summary = bank.build("saddle", stop)
    assert summary.stop_reason == "residual_goal"
    assert bank.entries("saddle")["rel_residual"].notna().sum() == 3


def test_interrupted_build_resumes_deterministically(tmp_path, config):
    """A build of the saddle region interrupted and resumed matches one uninterrupted build.

    The saddle hull has 18 vertices, so with 24-then-40 valid entries the resumed run
    replays more than just hull vertices.
    """
    with MapBank.create(tmp_path / "straight", config) as straight:
        straight.build("saddle", StoppingCriteria(max_valid_points=40))
        expected = straight.entries("saddle")
    with MapBank.create(tmp_path / "resumed", config) as first:
        first.build("saddle", StoppingCriteria(max_valid_points=24))
    with MapBank.open(tmp_path / "resumed", writable=["saddle"]) as second:
        second.build("saddle", StoppingCriteria(max_valid_points=40))
        resumed = second.entries("saddle")
    pd.testing.assert_frame_equal(resumed.drop(columns="created_at"), expected.drop(columns="created_at"))


def test_build_replays_entries_created_by_fetch(bank, monkeypatch):
    """A resumed build tells the learner about entries created by fetch."""
    bank.build("minima", StoppingCriteria(max_valid_points=12))
    bank.finalize("minima")
    created = bank.fetch(0.3, 0.25, 0.5)
    assert created.created
    told = []
    original_tell = adaptive.LearnerND.tell

    def recording_tell(self, point, value):
        told.append(tuple(point))
        return original_tell(self, point, value)

    monkeypatch.setattr(adaptive.LearnerND, "tell", recording_tell)
    bank.build("minima", StoppingCriteria(max_valid_points=16))
    assert (0.3, 0.25, 0.5) in told
    assert bank.entries("minima")["valid"].sum() == 16


def test_failures_during_build_are_recorded(tmp_path):
    """Failed evaluations become invalid entries and do not count towards max_valid_points."""
    generator = SyntheticGenerator(failure_boxes=[[[0.0, 0.5], [0.0, 2.0], [0.0, 0.3]]])
    with MapBank.create(tmp_path / "b", small_config(generator)) as bank:
        summary = bank.build("minima", StoppingCriteria(max_valid_points=10))
        entries = bank.entries("minima")
    assert summary.n_valid == 10
    assert (~entries["valid"]).sum() > 0
    assert entries.loc[~entries["valid"], "error"].str.startswith("MapGenerationError").all()


def test_build_preconditions(tmp_path, bank, generator):
    """Building needs a region that meets the domain and is open for writing."""
    domain = DomainSpec(kappa_range=(0.05, 0.5), gamma_range=(0.05, 0.3))
    with MapBank.create(tmp_path / "narrow", small_config(generator, domain=domain)) as narrow:
        with pytest.raises(ValueError, match="does not intersect"):
            narrow.build("maxima", StoppingCriteria(max_valid_points=5))
    path = bank.path
    bank.close()
    with MapBank.open(path) as reader, pytest.raises(BankReadOnlyError):
        reader.build("minima", StoppingCriteria(max_valid_points=5))
