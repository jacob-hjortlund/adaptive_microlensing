import json

import numpy as np
import pandas as pd
import pytest
from adaptive_microlensing import (
    BankCorruptError,
    BankReadOnlyError,
    MapBank,
    QueryStatus,
    RegionStateError,
)
from adaptive_microlensing.bank import entry_seeds
from adaptive_microlensing.mpd import intrinsic_quantile
from helpers import CENTROID, NEAR_FIRST_VERTEX, OUTSIDE_TETRAHEDRON, TETRAHEDRON, add_entry, small_config
from scipy.spatial.distance import jensenshannon


def test_create_writes_bank_json_and_empty_regions(bank, config):
    """create() writes bank.json and three empty regions."""
    meta = json.loads((bank.path / "bank.json").read_text())
    assert meta["schema_version"] == 2
    assert meta["config"] == config.to_dict()
    assert set(meta["versions"]) >= {"python", "numpy", "scipy", "pandas", "adaptive", "generator"}
    for region in ("minima", "saddle", "maxima"):
        assert (bank.path / region / "entries.csv").is_file()
        assert (bank.path / region / "maps").is_dir()
        assert (bank.path / region / "windows").is_dir()
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
    "an invalid config value": lambda meta: {**meta, "config": {**meta["config"], "n_bins": 0}},
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
    assert state.mpds.shape == (len(entries), 13)
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


def test_a_negative_shear_is_looked_up_at_its_magnitude(tetra_bank):
    """A negative gamma is outside the domain, but with the flag it is answered like |gamma|."""
    kappa, gamma, s = NEAR_FIRST_VERTEX
    assert tetra_bank.query(kappa, -gamma, s).status is QueryStatus.OUTSIDE_DOMAIN
    mirrored = tetra_bank.query(kappa, -gamma, s, allow_outside_domain=True)
    direct = tetra_bank.query(kappa, gamma, s)
    assert mirrored.status is QueryStatus.HIT and not mirrored.is_in_bounds
    assert mirrored.entry == direct.entry
    assert mirrored.simplex_entry_ids == direct.simplex_entry_ids
    assert mirrored.margin == direct.margin


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


def test_open_refuses_a_bank_of_an_older_layout(bank):
    """A bank written by the previous layout is refused with a way to recover."""
    path = bank.path / "bank.json"
    bank.close()
    meta = json.loads(path.read_text())
    path.write_text(json.dumps({**meta, "schema_version": 1}))
    with pytest.raises(BankCorruptError, match=r"schema version 1.*import_legacy_bank"):
        MapBank.open(bank.path)


def _coarse_map(bank, region, entry_id, generator):
    """An entry's coarse map, made again from its stored variability seed."""
    row = bank._regions[region].entries.loc[entry_id]
    spec = bank.config.variability
    return generator.generate(row["kappa"], row["gamma"], row["s"], spec.map, int(row["variability_seed"]))


def _independent_frozen_edge_quantile(bank, region, entry_id, generator):
    """An entry's quantile on its region's frozen edges, from its coarse map made again.

    The edges are moved to the nearest grid line, as the bank does, and every window is
    binned directly with numpy and compared with SciPy's Jensen-Shannon distance.
    """
    spec = bank.config.variability
    width = spec.histogram_bin_width
    edges = np.rint(bank._regions[region].bin_edges / width) * width
    coarse = _coarse_map(bank, region, entry_id, generator)
    size = spec.window_pixels
    mpds = []
    for row in range(spec.windows_per_axis):
        for col in range(spec.windows_per_axis):
            block = coarse[row * size : (row + 1) * size, col * size : (col + 1) * size]
            values = block[np.isfinite(block)]
            inner = np.histogram(values, bins=edges)[0]
            mpds.append(np.concatenate([[np.sum(values < edges[0])], inner, [np.sum(values > edges[-1])]]))
    distances = [jensenshannon(p, q, base=2) for i, p in enumerate(mpds) for q in mpds[i + 1 :]]
    return float(np.quantile(distances, spec.quantile))


def test_entries_store_window_histograms_and_a_provisional_quantile(bank, generator):
    """Before finalize, an entry's quantile in force is its coarse-range one, and its windows are stored."""
    add_entry(bank, "minima", TETRAHEDRON[0], generator)
    entries = bank.entries("minima")
    spec = bank.config.variability
    expected = intrinsic_quantile(_coarse_map(bank, "minima", 0, generator), spec)
    assert entries.loc[0, "coarse_quantile"] == pytest.approx(expected, rel=1e-12)
    assert entries.loc[0, "intrinsic_quantile"] == entries.loc[0, "coarse_quantile"]
    assert entries.loc[0, "window_file"] == "windows/windows_000000.npz"
    start, counts = bank._regions["minima"].store.read_windows(entries.loc[0, "window_file"])
    assert counts.shape[0] == spec.windows_per_axis**2 == 16
    np.testing.assert_array_equal(counts.sum(axis=1), spec.window_pixels**2)


def test_finalize_recomputes_quantiles_on_the_frozen_edges(tetra_bank, generator):
    """finalize() measures every quantile again on the frozen edges and keeps the provisional one."""
    entries = tetra_bank.entries("minima")
    spec = tetra_bank.config.variability
    for entry_id in entries["entry_id"]:
        frozen = _independent_frozen_edge_quantile(tetra_bank, "minima", entry_id, generator)
        assert entries.loc[entry_id, "intrinsic_quantile"] == pytest.approx(frozen, rel=1e-9)
        provisional = intrinsic_quantile(_coarse_map(tetra_bank, "minima", entry_id, generator), spec)
        assert entries.loc[entry_id, "coarse_quantile"] == pytest.approx(provisional, rel=1e-12)
    assert not np.allclose(entries["intrinsic_quantile"], entries["coarse_quantile"])
    index = tetra_bank._regions["minima"].index
    np.testing.assert_array_equal(index.quantiles, entries["intrinsic_quantile"].to_numpy())


def test_entries_added_after_finalize_get_frozen_edge_quantiles(tetra_bank, generator):
    """Entries made after finalize, directly or by fetch, are measured on the frozen edges at once."""
    row = add_entry(tetra_bank, "minima", CENTROID, generator)
    frozen = _independent_frozen_edge_quantile(tetra_bank, "minima", 4, generator)
    assert row["intrinsic_quantile"] == pytest.approx(frozen, rel=1e-9)
    assert row["coarse_quantile"] != row["intrinsic_quantile"]
    fetched = tetra_bank.fetch(*OUTSIDE_TETRAHEDRON)
    assert fetched.created and fetched.entry.entry_id == 5
    frozen = _independent_frozen_edge_quantile(tetra_bank, "minima", 5, generator)
    assert fetched.entry.intrinsic_quantile == pytest.approx(frozen, rel=1e-9)


def test_reopened_bank_keeps_both_quantiles(tetra_bank):
    """Both quantile columns and the window files survive closing and opening the bank."""
    before = tetra_bank.entries("minima")
    path = tetra_bank.path
    tetra_bank.close()
    with MapBank.open(path) as reader:
        pd.testing.assert_frame_equal(reader.entries("minima"), before)


def test_finalize_refuses_a_missing_window_file_and_writes_nothing(bank, generator):
    """A window file lost before finalize raises BankCorruptError and leaves the region unfinalized."""
    for point in TETRAHEDRON:
        add_entry(bank, "minima", point, generator)
    (bank.path / "minima" / "windows" / "windows_000002.npz").unlink()
    entries_before = (bank.path / "minima" / "entries.csv").read_bytes()
    with pytest.raises(BankCorruptError, match="windows_000002"):
        bank.finalize("minima")
    assert not bank._regions["minima"].finalized
    assert json.loads((bank.path / "minima" / "region.json").read_text())["finalized"] is False
    assert (bank.path / "minima" / "entries.csv").read_bytes() == entries_before


def test_finalize_refuses_a_window_file_with_the_wrong_number_of_rows_and_writes_nothing(bank, generator):
    """A loadable window file with too few rows raises BankCorruptError before any write."""
    for point in TETRAHEDRON:
        add_entry(bank, "minima", point, generator)
    np.savez(
        bank.path / "minima" / "windows" / "windows_000002.npz",
        start=np.int64(0),
        counts=np.ones((1, 5), dtype=np.int32),
    )
    directory = bank.path / "minima"
    before = {name: (directory / name).read_bytes() for name in ("entries.csv", "mpds.npy", "region.json")}
    with pytest.raises(BankCorruptError, match="windows_000002"):
        bank.finalize("minima")
    assert not bank._regions["minima"].finalized
    assert json.loads((directory / "region.json").read_text())["finalized"] is False
    for name, content in before.items():
        assert (directory / name).read_bytes() == content
