"""Import a bank produced by the original adaptive_mpd scripts.

The 3D design run of the original scripts writes one output directory
(``smooth_frac_range_output``) that holds, for each region in ``REGIONS``:

- ``{region}_data.csv``, with one row per sampled point and the columns ``kappa``,
  ``gamma``, ``s``, ``mpd_distance`` (the point's intrinsic quantile) and ``validity`` (the
  point is valid when it is at least 1), plus, optionally, the diagnostics
  ``simplex_loss``, ``abs_residual``, ``rel_residual`` and ``max_residual``;
- ``maps/{region}/microlensing_map_NNNN.npy``, the bank map of each valid row, numbered by
  the zero-based CSV row (:func:`legacy_map_path`).

:func:`import_legacy_bank` turns such a directory into a bank with ``LEGACY_CONFIG``.
CSV row ``r`` of a region becomes its entry ``r``. The maps are referenced by absolute
symbolic links, not copied, so the source directory must stay where it is. All three
regions share one set of bin edges and are finalized, as in the original query script,
``run_mpd_interpolator.py``. Where that script evaluated a query, the imported bank finds
the same tetrahedron and interpolates the same quantile; the one intended difference is
the hit rule, which the bank applies to every vertex rather than only the nearest. If an
import fails, everything it wrote is removed.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
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
#: Columns that every ``{region}_data.csv`` must have.
REQUIRED_COLUMNS = ("kappa", "gamma", "s", "mpd_distance", "validity")
#: Optional build diagnostics, copied from the CSV into the entries; a missing column gives NaN.
DIAGNOSTIC_COLUMNS = ("simplex_loss", "abs_residual", "rel_residual", "max_residual")


def legacy_map_path(source: Path, region: str, row: int) -> Path:
    """Return where the original scripts stored the bank map of a CSV row.

    The map of row ``row`` of ``{region}_data.csv`` is
    ``maps/{region}/microlensing_map_NNNN.npy`` under ``source``, with the row number
    zero-padded to four digits. Only valid rows have a map.

    Parameters
    ----------
    source : pathlib.Path
        The output directory of the original design run.
    region : str
        Name of the region.
    row : int
        Zero-based row number in the CSV, not counting the header.

    Returns
    -------
    pathlib.Path
        Path of the map file. Whether it exists is not checked.
    """
    return Path(source) / "maps" / region / f"microlensing_map_{row:04d}.npy"


def read_legacy_region(source: Path, region: str) -> pd.DataFrame:
    """Read and validate ``{region}_data.csv`` from the original design run.

    The checks are those of the original query script, which needs the required columns
    and at least four distinct, finite points that span three dimensions to build its
    mesh, plus one more: every valid row (``validity`` at least 1) must have a finite
    ``mpd_distance``.

    Parameters
    ----------
    source : pathlib.Path
        The output directory of the original design run.
    region : str
        Name of the region, used only to name the file.

    Returns
    -------
    pandas.DataFrame
        The CSV as read by :func:`pandas.read_csv`, with all its rows and columns.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If a column of ``REQUIRED_COLUMNS`` is missing, if there are fewer than four rows,
        if a ``kappa``, ``gamma`` or ``s`` value is not finite, if two rows have the same
        point, if the points do not span three dimensions, or if a valid row has a
        non-finite ``mpd_distance``.
    """
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
    """Return the entries table of a legacy region, with one entry per CSV row.

    ``mag_min`` and ``mag_max`` are left NaN for :func:`import_legacy_bank` to fill in from
    the maps. The dtypes are not yet normalised; the bank does that when it commits the
    table.

    Parameters
    ----------
    frame : pandas.DataFrame
        A region's CSV, as returned by :func:`read_legacy_region`.

    Returns
    -------
    pandas.DataFrame
        One row per CSV row, in order, with the columns of an entries table.

    Notes
    -----
    Row ``r`` becomes the entry with ``entry_id = r`` and the row's ``kappa``, ``gamma``
    and ``s``. It is valid when ``validity`` is at least 1. A valid entry takes
    ``mpd_distance`` as its ``intrinsic_quantile``, ``maps/map_{r:06d}.npy`` as its
    ``map_file`` and ``LEGACY_MAP_SEED_BASE + r + 1`` as its ``map_seed``, and has an empty
    ``error``. An invalid entry has a NaN quantile, no map file, no map seed and the
    error ``"legacy: invalid (reason not recorded)"``. Every entry has no
    ``variability_seed``, since the original runs did not record it, an empty
    ``generator_version``, ``origin`` ``"legacy"`` and the same ``created_at``, the time of
    the call. The ``DIAGNOSTIC_COLUMNS`` are copied as floats, or NaN when the CSV lacks
    them.
    """
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


def _remove_partial_import(destination: Path, existed: bool) -> None:
    """Delete everything a failed import wrote; the destination was absent or empty beforehand.

    The caller closes the bank first. Symbolic links are removed as links, so the legacy
    maps that they point to are never touched.

    Parameters
    ----------
    destination : pathlib.Path
        The bank directory of the failed import.
    existed : bool
        Whether ``destination`` existed before the import. If not, it is removed with
        everything in it. If so, it was an empty directory, and only its contents are
        removed.
    """
    if not existed:
        shutil.rmtree(destination)
        return
    for child in destination.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def import_legacy_bank(source: str | Path, destination: str | Path) -> MapBank:
    """Create a bank at ``destination`` from the outputs of the original adaptive_mpd scripts.

    ``source`` is the output directory of the 3D design run (``smooth_frac_range_output``),
    containing ``{region}_data.csv`` and ``maps/{region}/microlensing_map_NNNN.npy``.
    Maps are linked with absolute symlinks, never copied. All three regions share one
    set of bank-MPD bin edges, as in the original query script, and are finalized.
    If the import fails, everything it wrote is removed again.

    Parameters
    ----------
    source : str or pathlib.Path
        The output directory of the original design run. It is resolved to an absolute
        path, and must stay in place afterwards, since the bank's maps link into it.
    destination : str or pathlib.Path
        Directory of the new bank. It must not exist, or be an empty directory.

    Returns
    -------
    MapBank
        The imported bank, with the configuration ``LEGACY_CONFIG`` and every region
        finalized, opened for writing. Close it, or use it in a ``with`` block, to release
        its region locks.

    Raises
    ------
    FileNotFoundError
        If a region's CSV is missing, or if a valid row has no map. Both are checked
        before anything is written.
    ValueError
        If a region's CSV fails the checks of :func:`read_legacy_region`.
    FileExistsError
        If ``destination`` exists and is not empty.
    InvalidMapError
        If a legacy map of a valid row has no finite pixels.

    Notes
    -----
    The import runs in four steps:

    1. All three CSVs are read and validated, and the map of every valid row is checked
       to exist.
    2. An empty bank is created at ``destination`` with ``LEGACY_CONFIG``.
    3. For each region, the map of each valid row ``r`` is linked at
       ``maps/map_{r:06d}.npy`` in the region directory and scanned, memory-mapped, for
       its finite magnitude range, which gives the entry's ``mag_min`` and ``mag_max``.
       Progress is logged at INFO level. The region's entries (one per CSV row) are then
       committed, with NaN rows in place of their MPDs.
    4. One set of ``LEGACY_CONFIG.n_bin_edges`` bin edges is made from the smallest
       ``mag_min`` and the largest ``mag_max`` over all three regions. Each region is
       finalized with these edges, which computes the MPDs of its valid maps.

    If anything fails after the bank has been created, including a
    :exc:`KeyboardInterrupt`, the bank is closed and everything written is removed: the
    whole destination if it did not exist before, or only its contents if it was an empty
    directory. The legacy files are never modified. The maps' shapes are not checked
    against ``LEGACY_CONFIG.bank_map``.
    """
    source = Path(source).resolve()
    frames = {region: read_legacy_region(source, region) for region in REGIONS}
    for region, frame in frames.items():
        for row in np.flatnonzero(frame["validity"].to_numpy(dtype=float) >= 1.0):
            path = legacy_map_path(source, region, int(row))
            if not path.is_file():
                raise FileNotFoundError(f"Missing map for valid {region} row {int(row)}: {path}")

    destination = Path(destination)
    existed = destination.exists()
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
        _remove_partial_import(destination, existed)
        raise
    return bank
