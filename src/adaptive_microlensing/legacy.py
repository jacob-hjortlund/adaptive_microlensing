"""Import a bank produced by the original adaptive_mpd scripts."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .bank import MapBank
from .config import BankConfig
from .lensing import REGIONS
from .mpd import bin_edges_from_range, finite_range
from .storage import RegionStore, utc_now

logger = logging.getLogger("adaptive_microlensing")

#: Settings of the original runs (run.sh, run_map_gen.sh); they are the BankConfig defaults.
LEGACY_CONFIG = BankConfig()
#: run_map_gen.sh passed --sample-seed 42, and the script used seed = 42 + row + 1.
LEGACY_MAP_SEED_BASE = 42
REQUIRED_COLUMNS = ("kappa", "gamma", "s", "mpd_distance", "validity")
DIAGNOSTIC_COLUMNS = ("simplex_loss", "abs_residual", "rel_residual", "max_residual")


def legacy_map_path(source: Path, region: str, row: int) -> Path:
    """Where the original scripts stored the bank map of a CSV row."""
    return Path(source) / "maps" / region / f"microlensing_map_{row:04d}.npy"


def read_legacy_region(source: Path, region: str) -> pd.DataFrame:
    """Read and validate ``{region}_data.csv`` from the original design run."""
    path = Path(source) / f"{region}_data.csv"
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}.")
    points = frame.loc[:, ["kappa", "gamma", "s"]].to_numpy(dtype=float)
    valid = frame["validity"].to_numpy(dtype=float) >= 1.0
    if len(points) < 4:
        raise ValueError(f"{path} has fewer than four points.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{path} has non-finite parameter values.")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"{path} contains duplicate parameter points.")
    if np.linalg.matrix_rank(points - points[0]) < 3:
        raise ValueError(f"The points in {path} do not span three dimensions.")
    if not np.all(np.isfinite(frame.loc[valid, "mpd_distance"].to_numpy(dtype=float))):
        raise ValueError(f"{path} has non-finite mpd_distance values on valid rows.")
    return frame


def _legacy_entries(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame["validity"].to_numpy(dtype=float) >= 1.0
    created_at = utc_now()
    records = []
    for row, is_valid in enumerate(valid):
        source = frame.iloc[row]
        record = {
            "entry_id": row,
            "kappa": float(source["kappa"]),
            "gamma": float(source["gamma"]),
            "s": float(source["s"]),
            "valid": bool(is_valid),
            "error": "" if is_valid else "legacy: invalid (reason not recorded)",
            "intrinsic_quantile": float(source["mpd_distance"]) if is_valid else math.nan,
            "map_file": RegionStore.map_file(row) if is_valid else "",
            "mag_min": math.nan,
            "mag_max": math.nan,
            "variability_seed": None,
            "map_seed": LEGACY_MAP_SEED_BASE + row + 1 if is_valid else None,
            "generator_version": "",
            "origin": "legacy",
            "created_at": created_at,
        }
        for column in DIAGNOSTIC_COLUMNS:
            record[column] = float(source[column]) if column in frame.columns else math.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def import_legacy_bank(source: str | Path, destination: str | Path) -> MapBank:
    """Create a bank at ``destination`` from the outputs of the original adaptive_mpd scripts.

    ``source`` is the output directory of the 3D design run (``smooth_frac_range_output``),
    containing ``{region}_data.csv`` and ``maps/{region}/microlensing_map_NNNN.npy``.
    Maps are linked with absolute symlinks, never copied. All three regions share one
    set of bank-MPD bin edges, as in the original query script, and are finalized.
    Returns the bank opened for writing.
    """
    source = Path(source).resolve()
    frames = {region: read_legacy_region(source, region) for region in REGIONS}
    for region, frame in frames.items():
        for row in np.flatnonzero(frame["validity"].to_numpy(dtype=float) >= 1.0):
            path = legacy_map_path(source, region, int(row))
            if not path.is_file():
                raise FileNotFoundError(f"Missing map for valid {region} row {int(row)}: {path}")

    bank = MapBank.create(destination, LEGACY_CONFIG)
    try:
        minima: list[float] = []
        maxima: list[float] = []
        for region in REGIONS:
            entries = _legacy_entries(frames[region])
            store = bank._region(region).store
            rows = np.flatnonzero(entries["valid"].to_numpy(dtype=bool))
            for position, row in enumerate(rows, start=1):
                link = store.map_path(int(row))
                os.symlink(legacy_map_path(source, region, int(row)), link)
                mag_min, mag_max = finite_range(np.load(link, mmap_mode="r"))
                entries.loc[row, ["mag_min", "mag_max"]] = [mag_min, mag_max]
                if position == len(rows) or position % max(1, len(rows) // 10) == 0:
                    logger.info("%s: scanned %d/%d legacy maps", region, position, len(rows))
            bank._import_entries(region, entries)
            minima.append(float(entries.loc[rows, "mag_min"].min()))
            maxima.append(float(entries.loc[rows, "mag_max"].max()))
        edges = bin_edges_from_range(min(minima), max(maxima), LEGACY_CONFIG.n_bin_edges)
        for region in REGIONS:
            bank._finalize_with_edges(region, edges)
    except BaseException:
        bank.close()
        raise
    return bank
