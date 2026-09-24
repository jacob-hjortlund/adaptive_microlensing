import json

import numpy as np
import pandas as pd
import pytest
from adaptive_microlensing import MapBank, MapSpec, SyntheticGenerator, import_legacy_bank
from adaptive_microlensing.legacy import LEGACY_CONFIG
from adaptive_microlensing.lensing import REGIONS, classify_image, in_domain
from legacy_reference import legacy_query_table

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
    """Queries on the imported bank match the original run_mpd_interpolator.py code."""
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

    assert (new["is_hit"] == old["is_hit"]).all()
    covered = old["interpolation_status"] == "covered"
    assert covered.sum() > 50 and new.loc[covered, "is_hit"].any() and not new.loc[covered, "is_hit"].all()
    assert new.loc[covered, "interpolation_status"].isin(["hit", "miss"]).all()
    for column in ("query_region", "simplex_rows"):
        assert (new.loc[covered, column] == old.loc[covered, column]).all(), column
    for column in ("simplex_index", "matched_row"):
        assert (new.loc[covered, column].astype(int) == old.loc[covered, column].astype(int)).all(), column
    for column in NUMERIC_COLUMNS:
        np.testing.assert_allclose(
            new.loc[covered, column], old.loc[covered, column], rtol=1e-9, err_msg=column
        )
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


def test_import_refuses_missing_maps_and_existing_destinations(tmp_path, legacy_source, imported):
    """Import refuses missing maps and existing destinations."""
    with pytest.raises(FileExistsError):
        import_legacy_bank(legacy_source, imported.path)
    (legacy_source / "maps" / "minima" / "microlensing_map_0000.npy").unlink()
    with pytest.raises(FileNotFoundError, match="minima row 0"):
        import_legacy_bank(legacy_source, tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_reopened_import_is_queryable(imported):
    """An imported bank reopens read-only with every region finalized."""
    path = imported.path
    imported.close()
    with MapBank.open(path) as reader:
        assert reader.summary()["finalized"].all()
