import json

import numpy as np
import pandas as pd
import pytest
from adaptive_microlensing import MapBank, MapSpec, SyntheticGenerator, import_legacy_bank, legacy
from adaptive_microlensing.legacy import LEGACY_CONFIG
from adaptive_microlensing.lensing import REGIONS, classify_image, in_domain
from adaptive_microlensing.mpd import finite_range
from legacy_reference import determine_common_bin_edges, legacy_query_table

# Column order of the original {region}_data.csv files.
LEGACY_COLUMNS = [
    "kappa",
    "gamma",
    "mpd_distance",
    "validity",
    "simplex_loss",
    "abs_residual",
    "rel_residual",
    "max_residual",
    "s",
]
INVALID_ROWS = (1, 5)
# Numeric result columns that may differ from the original only by floating-point rounding.
NUMERIC_COLUMNS = (
    "interpolated_mpd_distance",
    "query_mpd_quantile",
    "matched_mpd_quantile",
    "distance_threshold",
    "distance_margin",
)


def _every_vertex_rule(row, quantiles):
    """The hit rule on every vertex, applied to one covered row of the original script's output.

    The original script tested only the vertex with the smallest distance.
    """
    rows = json.loads(row["simplex_rows"])
    distances = np.array(json.loads(row["vertex_distances"]))
    vertex_quantiles = quantiles[row["query_region"]][rows]
    thresholds = np.maximum(row["query_mpd_quantile"], vertex_quantiles)
    passing = distances <= thresholds
    if passing.any():
        local = int(np.argmin(np.where(passing, distances, np.inf)))
    else:
        local = int(np.argmax(thresholds - distances))
    return {
        "is_hit": bool(passing.any()),
        "matched_row": rows[local],
        "interpolated_mpd_distance": distances[local],
        "matched_mpd_quantile": vertex_quantiles[local],
        "distance_threshold": thresholds[local],
        "distance_margin": thresholds[local] - distances[local],
    }


def _region_points(region, count, rng):
    points = []
    while len(points) < count:
        kappa, gamma, s = rng.uniform([0.05, 0.05, 0.01], [2.0, 2.0, 0.99])
        if classify_image(kappa, gamma) == region and in_domain(kappa, gamma, s, LEGACY_CONFIG.domain):
            points.append((kappa, gamma, s))
    return np.array(points)


def make_legacy_output(root, count=25):
    """Write a small output directory in the layout of the original scripts."""
    root.mkdir()
    rng = np.random.default_rng(2024)
    generator = SyntheticGenerator()
    for region in REGIONS:
        points = _region_points(region, count, rng)
        valid = np.ones(count)
        valid[list(INVALID_ROWS)] = 0.0
        quantiles = np.where(valid == 1.0, rng.uniform(0.05, 0.3, count), 0.0)
        frame = pd.DataFrame(
            {
                "kappa": points[:, 0],
                "gamma": points[:, 1],
                "mpd_distance": quantiles,
                "validity": valid,
                "simplex_loss": np.inf,
                "abs_residual": np.nan,
                "rel_residual": np.nan,
                "max_residual": np.nan,
                "s": points[:, 2],
            }
        )[LEGACY_COLUMNS]
        frame.to_csv(root / f"{region}_data.csv", index=False)
        map_dir = root / "maps" / region
        map_dir.mkdir(parents=True)
        for row in np.flatnonzero(valid == 1.0):
            mag_map = generator.generate(*points[row], MapSpec(2.0, 0.05), seed=42 + int(row) + 1)
            np.save(map_dir / f"microlensing_map_{row:04d}.npy", mag_map)
    return root


@pytest.fixture
def legacy_source(tmp_path):
    """A small output directory of the original scripts."""
    return make_legacy_output(tmp_path / "legacy")


@pytest.fixture
def imported(tmp_path, legacy_source):
    """The legacy_source directory imported into a new bank."""
    bank = import_legacy_bank(legacy_source, tmp_path / "imported")
    yield bank
    bank.close()


def test_import_registers_entries_and_links_maps(imported, legacy_source):
    """Import keeps the legacy row numbers, links the maps and shares one set of edges."""
    assert imported.config == LEGACY_CONFIG
    summary = imported.summary()
    assert summary["entries"].tolist() == [25, 25, 25]
    assert summary["invalid"].tolist() == [2, 2, 2]
    assert summary["finalized"].all()
    entries = imported.entries("saddle")
    assert entries["origin"].eq("legacy").all()
    assert entries.loc[0, "map_seed"] == 43 and pd.isna(entries.loc[1, "map_seed"])
    assert entries["variability_seed"].isna().all()
    link = imported.path / "saddle" / "maps" / "map_000003.npy"
    assert link.is_symlink()
    assert link.resolve() == (legacy_source / "maps" / "saddle" / "microlensing_map_0003.npy").resolve()
    edges = [json.loads((imported.path / r / "region.json").read_text())["bin_edges"] for r in REGIONS]
    assert edges[0] == edges[1] == edges[2]


def test_query_matches_the_original_query_script(imported, legacy_source):
    """Queries on the imported bank match the original run_mpd_interpolator.py code.

    The one intended difference is the hit rule, which the bank applies to every vertex.
    """
    rng = np.random.default_rng(7)
    queries = pd.DataFrame(
        {
            "kappa": rng.uniform(0.0, 2.1, 400),
            "gamma": rng.uniform(0.0, 2.1, 400),
            "s": rng.uniform(0.0, 1.0, 400),
        }
    )
    new = imported.query_many(queries)
    old = legacy_query_table(legacy_source, queries)

    covered = old["interpolation_status"] == "covered"
    assert not new.loc[~covered, "is_hit"].any()
    assert covered.sum() > 50 and new.loc[covered, "is_hit"].any() and not new.loc[covered, "is_hit"].all()
    assert new.loc[covered, "interpolation_status"].isin(["hit", "miss"]).all()
    for column in ("query_region", "simplex_rows"):
        assert (new.loc[covered, column] == old.loc[covered, column]).all(), column
    assert (
        new.loc[covered, "simplex_index"].astype(int) == old.loc[covered, "simplex_index"].astype(int)
    ).all()
    np.testing.assert_allclose(
        new.loc[covered, "query_mpd_quantile"], old.loc[covered, "query_mpd_quantile"], rtol=1e-9
    )

    # Where the original found a hit, its nearest vertex passed, so the bank matches it too.
    old_hits = covered & old["is_hit"]
    assert new.loc[old_hits, "is_hit"].all()
    assert (
        new.loc[old_hits, "matched_row"].astype(int) == old.loc[old_hits, "matched_row"].astype(int)
    ).all()
    for column in NUMERIC_COLUMNS:
        np.testing.assert_allclose(
            new.loc[old_hits, column], old.loc[old_hits, column], rtol=1e-9, err_msg=column
        )

    # Everywhere it covered, the bank agrees with the hit rule applied to every vertex.
    quantiles = {
        region: pd.read_csv(legacy_source / f"{region}_data.csv")["mpd_distance"].to_numpy()
        for region in REGIONS
    }
    expected = pd.DataFrame(
        [_every_vertex_rule(row, quantiles) for _, row in old.loc[covered].iterrows()],
        index=old.index[covered],
    )
    assert (expected["is_hit"] & ~old.loc[covered, "is_hit"]).any()  # some nearest-vertex misses become hits
    assert (new.loc[covered, "is_hit"] == expected["is_hit"]).all()
    assert (new.loc[covered, "matched_row"].astype(int) == expected["matched_row"]).all()
    for column in NUMERIC_COLUMNS:
        if column != "query_mpd_quantile":
            np.testing.assert_allclose(new.loc[covered, column], expected[column], rtol=1e-9, err_msg=column)
    for column in ("barycentric_weights", "vertex_distances"):
        new_values = np.array([json.loads(v) for v in new.loc[covered, column]])
        old_values = np.array([json.loads(v) for v in old.loc[covered, column]])
        np.testing.assert_allclose(new_values, old_values, rtol=1e-9, atol=1e-15, err_msg=column)
    mapping = {
        "outside_parameter_range": {"outside_domain", "critical_line"},
        "outside_convex_hull": {"outside_hull", "outside_domain"},
        "invalid_tetrahedron": {"invalid_simplex"},
    }
    for old_status, new_statuses in mapping.items():
        rows = old["interpolation_status"] == old_status
        assert new.loc[rows, "interpolation_status"].isin(new_statuses).all(), old_status


def test_import_refuses_bad_input_and_cleans_up_after_failures(
    tmp_path, legacy_source, imported, monkeypatch
):
    """Import refuses existing destinations and missing maps, and removes what a failed import wrote."""
    with pytest.raises(FileExistsError):
        import_legacy_bank(legacy_source, imported.path)

    calls = []

    def failing_finite_range(mag_map):
        calls.append(1)
        if len(calls) == 30:  # part-way through the saddle maps
            raise OSError("simulated read failure")
        return finite_range(mag_map)

    monkeypatch.setattr(legacy, "finite_range", failing_finite_range)
    with pytest.raises(OSError, match="simulated"):
        import_legacy_bank(legacy_source, tmp_path / "partial")
    assert not (tmp_path / "partial").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    calls.clear()
    with pytest.raises(OSError, match="simulated"):
        import_legacy_bank(legacy_source, empty)
    assert empty.is_dir() and not any(empty.iterdir())
    assert (legacy_source / "maps" / "minima" / "microlensing_map_0000.npy").is_file()
    monkeypatch.undo()

    (legacy_source / "maps" / "minima" / "microlensing_map_0000.npy").unlink()
    with pytest.raises(FileNotFoundError, match="minima row 0"):
        import_legacy_bank(legacy_source, tmp_path / "other")
    assert not (tmp_path / "other").exists()


def _invalidate(source, region):
    """Mark every row of a region's CSV invalid, as when every map of the region failed."""
    path = source / f"{region}_data.csv"
    frame = pd.read_csv(path)
    frame["validity"] = 0.0
    frame.to_csv(path, index=False)


@pytest.mark.parametrize("region", REGIONS)
def test_a_region_without_valid_rows_shares_the_edges_of_the_others(tmp_path, legacy_source, region):
    """A region whose every row failed is imported and finalized with the other regions' edges."""
    _invalidate(legacy_source, region)
    region_inputs = {r: (legacy_source / f"{r}_data.csv", legacy_source / "maps" / r) for r in REGIONS}
    expected = determine_common_bin_edges(region_inputs)
    with import_legacy_bank(legacy_source, tmp_path / "imported") as bank:
        summary = bank.summary()
        assert summary.loc[region, "valid"] == 0
        assert summary["finalized"].all()
        for name in REGIONS:
            edges = json.loads((bank.path / name / "region.json").read_text())["bin_edges"]
            np.testing.assert_array_equal(edges, expected)


def test_import_refuses_a_source_without_any_valid_row(tmp_path, legacy_source):
    """Without a single valid map there are no bin edges, so nothing is imported."""
    for region in REGIONS:
        _invalidate(legacy_source, region)
    with pytest.raises(ValueError, match="no valid rows"):
        import_legacy_bank(legacy_source, tmp_path / "imported")
    assert not (tmp_path / "imported").exists()


def test_reopened_import_is_queryable(imported):
    """An imported bank reopens read-only with every region finalized."""
    path = imported.path
    imported.close()
    with MapBank.open(path) as reader:
        assert reader.summary()["finalized"].all()
