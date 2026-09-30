import numpy as np
import pytest
from adaptive_microlensing import (
    BankReadOnlyError,
    FetchStatus,
    GeneratorMismatchError,
    MapBank,
    QueryStatus,
    SyntheticGenerator,
)
from adaptive_microlensing.results import FETCH_COLUMNS
from helpers import CENTROID, NEAR_FIRST_VERTEX, OUTSIDE_TETRAHEDRON, TETRAHEDRON, batch_table


def test_hit_returns_the_existing_map_and_changes_nothing(tetra_bank):
    """A hit returns the existing map and leaves the bank unchanged."""
    result = tetra_bank.fetch(*NEAR_FIRST_VERTEX)
    assert result.status is FetchStatus.HIT
    assert result.entry is not None and result.entry.entry_id == 0
    assert result.created is False
    assert len(tetra_bank.entries("minima")) == 4


@pytest.mark.parametrize(
    ("point", "status"),
    [
        ((float("nan"), 0.1, 0.1), FetchStatus.MALFORMED),
        ((0.5, 0.5, 0.5), FetchStatus.CRITICAL_LINE),
        ((0.5, 0.499, 0.5), FetchStatus.OUTSIDE_DOMAIN),
        ((1.0, 0.5, 0.5), FetchStatus.REGION_NOT_READY),
    ],
)
def test_refusals_leave_the_bank_unchanged(tetra_bank, point, status):
    """Refusals leave the bank unchanged."""
    result = tetra_bank.fetch(*point)
    assert result.status is status
    assert result.entry is None and not result.created
    assert tetra_bank.summary()["entries"].sum() == 4


def test_miss_creates_and_commits_a_new_entry(tetra_bank):
    """A miss makes a map at the query point and commits it as a new entry."""
    result = tetra_bank.fetch(*CENTROID)
    assert result.query.status is QueryStatus.MISS
    assert result.status is FetchStatus.CREATED and result.created
    entry = result.entry
    assert entry is not None and entry.entry_id == 4 and entry.origin == "fetch"
    assert (entry.kappa, entry.gamma, entry.s) == CENTROID
    assert entry.map_path.is_file()
    assert entry.load().shape == (80, 80)
    state = tetra_bank._regions["minima"]
    assert np.isclose(state.mpds[4].sum(), 1.0)
    again = tetra_bank.fetch(*CENTROID)
    assert again.status is FetchStatus.HIT and again.entry == entry

    path = tetra_bank.path
    tetra_bank.close()
    with MapBank.open(path) as reader:
        assert reader.query(*CENTROID).entry == entry


def test_outside_hull_and_flagged_out_of_domain_queries_create(tetra_bank):
    """Queries outside the hull, or outside the domain with the flag, create maps."""
    assert tetra_bank.fetch(*OUTSIDE_TETRAHEDRON).status is FetchStatus.CREATED
    assert tetra_bank.fetch(0.5, 0.499, 0.5).status is FetchStatus.OUTSIDE_DOMAIN
    flagged = tetra_bank.fetch(0.5, 0.499, 0.5, allow_outside_domain=True)
    assert flagged.status is FetchStatus.CREATED
    assert tetra_bank.query(0.5, 0.499, 0.5).status is QueryStatus.OUTSIDE_DOMAIN
    assert tetra_bank.query(0.5, 0.499, 0.5, allow_outside_domain=True).status is QueryStatus.HIT


def test_a_map_fetched_for_a_negative_shear_is_made_at_its_magnitude(tetra_bank):
    """A flagged fetch at -gamma makes and stores the map at |gamma|, where later queries find it."""
    kappa, gamma, s = CENTROID
    fetched = tetra_bank.fetch(kappa, -gamma, s, allow_outside_domain=True)
    assert fetched.status is FetchStatus.CREATED
    assert (fetched.entry.kappa, fetched.entry.gamma, fetched.entry.s) == (kappa, gamma, s)
    assert tetra_bank.query(kappa, gamma, s).entry == fetched.entry


def test_invalid_simplex_creates_at_the_query_point(failing_bank, failing_generator):
    """A query in a tetrahedron with an invalid vertex creates a map at the query point."""
    result = failing_bank.fetch(*CENTROID, generator=failing_generator)
    assert result.query.status is QueryStatus.INVALID_SIMPLEX
    assert result.status is FetchStatus.CREATED


def test_known_failure_is_not_retried(failing_bank):
    """A query at an existing invalid entry is refused without retrying."""
    result = failing_bank.fetch(*TETRAHEDRON[1])
    assert result.status is FetchStatus.KNOWN_FAILURE
    assert result.error is not None and result.error.startswith("MapGenerationError")
    assert len(failing_bank.entries("minima")) == 4


def test_failed_creation_keeps_an_invalid_entry(failing_bank):
    """A failed creation still commits an invalid entry."""
    result = failing_bank.fetch(0.59, 0.11, 0.11)
    assert result.status is FetchStatus.CREATION_FAILED
    assert result.entry is None and result.created is False
    assert "Synthetic failure" in result.error
    entries = failing_bank.entries("minima")
    assert len(entries) == 5 and not entries.loc[4, "valid"] and entries.loc[4, "origin"] == "fetch"


def test_fetch_needs_write_access_and_a_matching_generator(tetra_bank):
    """Fetch needs write access and a matching generator."""
    with pytest.raises(GeneratorMismatchError):
        tetra_bank.fetch(*CENTROID, generator=SyntheticGenerator(inf_fraction=0.1))
    path = tetra_bank.path
    tetra_bank.close()
    with MapBank.open(path) as reader:
        assert reader.fetch(*NEAR_FIRST_VERTEX).status is FetchStatus.HIT
        with pytest.raises(BankReadOnlyError):
            reader.fetch(*CENTROID)


def test_fetch_many_creates_in_order_and_reuses_new_maps(tetra_bank):
    """fetch_many processes rows in order, so later rows reuse maps made for earlier ones."""
    result = tetra_bank.fetch_many(batch_table())
    assert list(result.columns) == ["kappa", "gamma", "s", "label", *FETCH_COLUMNS]
    assert result["fetch_status"].tolist() == [
        "hit",
        "created",
        "created",
        "critical_line",
        "region_not_ready",
        "hit",
    ]
    # Row 15 repeats row 11 and matches the map created for it.
    assert result.loc[15, "entry_id"] == result.loc[11, "entry_id"] == 4
    assert result.loc[11, "interpolation_status"] == "miss"  # describes the query before creation
    assert result["created"].tolist() == [False, True, True, False, False, False]
    assert result.loc[11, "map_path"].endswith("minima/maps/map_000004.npy")
    rerun = tetra_bank.fetch_many(batch_table())
    assert rerun["fetch_status"].tolist()[:3] == ["hit", "hit", "hit"]


def test_fetch_many_refuses_a_table_with_results_before_making_maps(tetra_bank):
    """A table that already has result columns is refused before any row is fetched."""
    queried = tetra_bank.query_many(batch_table())
    with pytest.raises(ValueError, match="already has result columns"):
        tetra_bank.fetch_many(queried)
    assert len(tetra_bank.entries("minima")) == len(TETRAHEDRON)
