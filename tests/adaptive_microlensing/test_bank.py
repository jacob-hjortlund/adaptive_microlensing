import json

import numpy as np
import pytest
from adaptive_microlensing import (
    BankCorruptError,
    BankReadOnlyError,
    MapBank,
    QueryStatus,
    RegionStateError,
)
from adaptive_microlensing.bank import entry_seeds
from helpers import CENTROID, NEAR_FIRST_VERTEX, OUTSIDE_TETRAHEDRON, TETRAHEDRON, add_entry, small_config


def test_create_writes_bank_json_and_empty_regions(bank, config):
    """create() writes bank.json and three empty regions."""
    meta = json.loads((bank.path / "bank.json").read_text())
    assert meta["schema_version"] == 1
    assert meta["config"] == config.to_dict()
    assert set(meta["versions"]) >= {"python", "numpy", "scipy", "pandas", "adaptive", "generator"}
    for region in ("minima", "saddle", "maxima"):
        assert (bank.path / region / "entries.csv").is_file()
        assert (bank.path / region / "maps").is_dir()
    assert bank.summary()["entries"].tolist() == [0, 0, 0]
    assert bank.config == config


def test_create_refuses_a_non_empty_directory(bank, config):
    """create() refuses a directory that is not empty."""
    with pytest.raises(FileExistsError):
        MapBank.create(bank.path, config)


def test_open_writable_argument(tetra_bank):
    """open() locks no region, one region or all regions."""
    path = tetra_bank.path
    tetra_bank.close()
    with MapBank.open(path) as reader:
        assert not any(state.writable for state in reader._regions.values())
    with MapBank.open(path, writable="saddle") as writer:
        assert [r for r, s in writer._regions.items() if s.writable] == ["saddle"]
    with pytest.raises(ValueError, match="Unknown regions"):
        MapBank.open(path, writable=["ring"])


UNUSABLE_BANK_JSON = {
    "a list": lambda meta: [1],
    "no config": lambda meta: {key: value for key, value in meta.items() if key != "config"},
    "a config without a seed": lambda meta: {
        **meta,
        "config": {key: value for key, value in meta["config"].items() if key != "seed"},
    },
    "an invalid config value": lambda meta: {**meta, "config": {**meta["config"], "n_bin_edges": 1}},
    "an unknown config field": lambda meta: {
        **meta,
        "config": {**meta["config"], "domain": {**meta["config"]["domain"], "bogus": 1}},
    },
}


@pytest.mark.parametrize("edit", UNUSABLE_BANK_JSON.values(), ids=UNUSABLE_BANK_JSON.keys())
def test_open_reports_an_unusable_bank_json_as_corrupt(bank, edit):
    """A bank.json that is not an object or holds no valid configuration raises BankCorruptError."""
    path = bank.path / "bank.json"
    bank.close()
    path.write_text(json.dumps(edit(json.loads(path.read_text()))))
    with pytest.raises(BankCorruptError, match=r"bank\.json"):
        MapBank.open(bank.path)


def test_entry_seeds_are_deterministic_positive_c_ints():
    """Entry seeds are deterministic, distinct and valid C ints for IPM."""
    seeds = {
        entry_seeds(42, region, entry) for region in ("minima", "saddle", "maxima") for entry in range(50)
    }
    assert len(seeds) == 150
    for variability_seed, map_seed in seeds:
        assert 1 <= variability_seed <= 2**31 - 1 and 1 <= map_seed <= 2**31 - 1
        assert variability_seed != map_seed
    assert entry_seeds(42, "minima", 3) == entry_seeds(42, "minima", 3)
    assert entry_seeds(42, "minima", 3) != entry_seeds(43, "minima", 3)


def test_entries_record_what_made_them(tetra_bank, generator):
    """Entries record their seeds, generator version and finite map range."""
    entries = tetra_bank.entries("minima")
    assert entries["entry_id"].tolist() == [0, 1, 2, 3]
    assert entries["valid"].all()
    assert entries["origin"].eq("build").all()
    assert entries["generator_version"].eq(generator.version()).all()
    assert entries.loc[2, "variability_seed"] == entry_seeds(42, "minima", 2)[0]
    assert (entries["mag_min"] < entries["mag_max"]).all()
    entries.loc[0, "kappa"] = 99.0  # entries() returns a copy
    assert tetra_bank.entries("minima").loc[0, "kappa"] == TETRAHEDRON[0][0]


def test_finalize_freezes_edges_and_computes_mpds(tetra_bank):
    """finalize() stores edges spanning every map and one normalised MPD per entry."""
    state = tetra_bank._regions["minima"]
    meta = json.loads((tetra_bank.path / "minima" / "region.json").read_text())
    assert meta["finalized"] is True
    assert len(meta["bin_edges"]) == 12
    entries = tetra_bank.entries("minima")
    assert meta["bin_edges"][0] < entries["mag_min"].min()
    assert meta["bin_edges"][-1] > entries["mag_max"].max()
    np.testing.assert_allclose(state.mpds.sum(axis=1), 1.0)
    assert np.all(state.mpds[:, [0, -1]] == 0.0)


def test_finalize_preconditions(tetra_bank):
    """finalize() needs a writable, unfinalized region with valid entries."""
    with pytest.raises(RegionStateError, match="already finalized"):
        tetra_bank.finalize("minima")
    with pytest.raises(RegionStateError, match="no valid entries"):
        tetra_bank.finalize("saddle")
    with pytest.raises(ValueError, match="Unknown region"):
        tetra_bank.finalize("ring")
    path = tetra_bank.path
    tetra_bank.close()
    with MapBank.open(path) as reader, pytest.raises(BankReadOnlyError):
        reader.finalize("saddle")


@pytest.mark.parametrize(
    ("point", "status", "region"),
    [
        ((float("nan"), 0.1, 0.1), QueryStatus.MALFORMED, None),
        (("abc", 0.1, 0.1), QueryStatus.MALFORMED, None),
        ((0.5, 0.5, 0.5), QueryStatus.CRITICAL_LINE, None),
        ((2.5, 0.1, 0.5), QueryStatus.OUTSIDE_DOMAIN, "maxima"),
        ((0.5, 0.499, 0.5), QueryStatus.OUTSIDE_DOMAIN, "minima"),
        ((1.0, 0.5, 0.5), QueryStatus.REGION_NOT_READY, "saddle"),
        (OUTSIDE_TETRAHEDRON, QueryStatus.OUTSIDE_HULL, "minima"),
        (CENTROID, QueryStatus.MISS, "minima"),
        (NEAR_FIRST_VERTEX, QueryStatus.HIT, "minima"),
        (TETRAHEDRON[2], QueryStatus.HIT, "minima"),
    ],
)
def test_query_statuses(tetra_bank, point, status, region):
    """Each query outcome gets its status and region."""
    result = tetra_bank.query(*point)
    assert result.status is status
    assert result.region == region
    assert result.is_hit is (status is QueryStatus.HIT)


def test_hit_and_miss_details(tetra_bank):
    """Hits and misses carry the diagnostics of the hit rule."""
    hit = tetra_bank.query(*NEAR_FIRST_VERTEX)
    assert hit.entry is not None and hit.entry.entry_id == 0
    assert hit.interpolated_distance <= hit.threshold
    assert hit.margin == pytest.approx(hit.threshold - hit.interpolated_distance)
    assert sum(hit.barycentric_weights) == pytest.approx(1.0)
    assert sorted(hit.simplex_entry_ids) == [0, 1, 2, 3]
    assert hit.region_entry_count == 4
    assert hit.is_in_bounds
    mag_map = hit.entry.load()
    assert isinstance(mag_map, np.memmap) and mag_map.shape == (80, 80)

    miss = tetra_bank.query(*CENTROID)
    assert miss.entry is None and miss.matched is not None
    assert miss.interpolated_distance > miss.threshold

    exact = tetra_bank.query(*TETRAHEDRON[2])
    assert exact.coincident_entry_id == 2 and exact.interpolated_distance == 0.0
    assert exact.entry is not None and exact.entry.entry_id == 2


def test_allow_outside_domain_skips_only_the_domain_check(tetra_bank):
    """Allow outside domain skips only the domain check."""
    assert tetra_bank.query(0.5, 0.499, 0.5).status is QueryStatus.OUTSIDE_DOMAIN
    assert tetra_bank.query(0.5, 0.499, 0.5, allow_outside_domain=True).status is QueryStatus.OUTSIDE_HULL
    assert tetra_bank.query(0.5, 0.5, 0.5, allow_outside_domain=True).status is QueryStatus.CRITICAL_LINE


def test_invalid_vertices_give_invalid_simplex(failing_bank):
    """A tetrahedron with an invalid vertex gives invalid_simplex."""
    entries = failing_bank.entries("minima")
    assert entries["valid"].tolist() == [True, False, True, True]
    assert entries.loc[1, "error"].startswith("MapGenerationError")
    assert failing_bank.query(*CENTROID).status is QueryStatus.INVALID_SIMPLEX
    exact = failing_bank.query(*TETRAHEDRON[1])
    assert exact.status is QueryStatus.INVALID_SIMPLEX and exact.coincident_entry_id == 1


def test_reopened_bank_answers_identically(tetra_bank):
    """A reopened bank answers queries exactly as before."""
    before = [tetra_bank.query(*point) for point in (CENTROID, NEAR_FIRST_VERTEX)]
    path = tetra_bank.path
    tetra_bank.close()
    with MapBank.open(path) as reader:
        after = [reader.query(*point) for point in (CENTROID, NEAR_FIRST_VERTEX)]
    assert before == after


def test_unfinalized_region_is_not_ready(tmp_path, generator):
    """A region cannot be queried before it is finalized."""
    with MapBank.create(tmp_path / "b", small_config(generator)) as bank:
        for point in TETRAHEDRON:
            add_entry(bank, "minima", point, generator)
        assert bank.query(*CENTROID).status is QueryStatus.REGION_NOT_READY
