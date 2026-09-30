"""On-disk layout of a bank: atomic writes, commit order, snapshots and writer locks.

A region directory holds ``region.json``, ``entries.csv``, ``mpds.npy`` and ``maps/``.
A new entry is committed in three atomic steps: its map, then ``mpds.npy``, then
``entries.csv``. An entry exists only once its row is in ``entries.csv``.

A bank directory holds ``bank.json``, which :meth:`~adaptive_microlensing.bank.MapBank.create`
writes with the schema version, the creation time, the package and dependency versions
and the :class:`~adaptive_microlensing.config.BankConfig`, and one directory per region,
named after it (``minima``, ``saddle`` and ``maxima``). A region directory holds:

- ``region.json``: the schema version, the region's name, whether it is finalized, its
  frozen MPD bin edges (``null`` until then) and the time it was finalized.
- ``entries.csv``: the entries table, one row per entry in ``entry_id`` order.
- ``mpds.npy``: a float array of shape ``(n_entries, n_mpd_columns)`` whose row *i* is
  the bank MPD of entry *i*. A row is NaN until the region is finalized, and it stays NaN
  for an invalid entry.
- ``maps/map_XXXXXX.npy``: the bank map of each valid entry, named by its zero-padded
  entry ID (:meth:`RegionStore.map_file`). A legacy import links these files to the
  original maps instead of copying them.
- ``.lock``: the writer lock file, which records the host, PID and start time of the
  writer that holds it (:class:`RegionLock`).

Every file is replaced atomically: the new contents go to a temporary file in the same
directory, which is flushed, fsynced and then renamed over the target
(:func:`atomic_write_bytes`, :func:`atomic_write_json`, :func:`atomic_save_array`).
A crash before step 3 of a commit leaves an unused map file or an extra row in
``mpds.npy``. Both are harmless: :meth:`RegionStore.load` drops the extra rows, and the
next entry takes the same ID and overwrites both. Finalizing writes ``mpds.npy`` and then
``region.json``. Readers read the files in the opposite order to the writers
(:meth:`RegionStore.load`), so they always get a consistent snapshot.

The entries table has the columns of :data:`ENTRY_COLUMNS`, with the dtypes that
:func:`normalise_entries` sets:

- ``entry_id`` (``int64``): the row's position, ``0, 1, 2, ...``.
- ``kappa``, ``gamma``, ``s`` (``float64``): the entry's point.
- ``valid`` (``bool``): whether the entry's maps were made and checked.
- ``error`` (``str``): why an invalid entry failed, usually ``"ExceptionName: message"``;
  ``""`` for a valid entry.
- ``intrinsic_quantile`` (``float64``): the entry's noise level; NaN for an invalid entry.
- ``map_file`` (``str``): the bank map's path relative to the region directory; ``""`` for
  an invalid entry.
- ``mag_min``, ``mag_max`` (``float64``): the finite range of the bank map's magnitudes;
  NaN for an invalid entry.
- ``variability_seed``, ``map_seed`` (nullable ``Int64``): the seeds of the coarse map and
  the bank map; ``<NA>`` when not recorded. Legacy entries have no ``variability_seed``,
  and invalid legacy entries no ``map_seed`` either.
- ``generator_version`` (``str``): the generator's version; ``""`` when unknown.
- ``origin`` (``str``): how the entry was added, ``"build"``, ``"fetch"`` or ``"legacy"``.
- ``created_at`` (``str``): when the row was made, in UTC (:func:`utc_now`).
- ``simplex_loss``, ``abs_residual``, ``rel_residual``, ``max_residual`` (``float64``):
  diagnostics of the build that added the entry; NaN when not recorded.

:class:`~adaptive_microlensing.bank.MapBank` keeps one :class:`RegionStore` per region and,
for each region open for writing, one :class:`RegionLock`. This module reads, writes and
checks the consistency of the files; the maps and MPDs are computed elsewhere.
"""

from __future__ import annotations

import contextlib
import datetime
import errno
import fcntl
import json
import os
import socket
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from .errors import BankCorruptError, BankLockedError

#: Version of the file layout, written to ``bank.json`` and ``region.json``; loading a bank
#: or a region with any other version raises ``BankCorruptError``.
SCHEMA_VERSION = 1
#: Name of the bank's metadata file, in the bank directory.
BANK_FILE = "bank.json"
#: Name of a region's metadata file, in the region directory.
REGION_FILE = "region.json"
#: Name of a region's entries table, in the region directory.
ENTRIES_FILE = "entries.csv"
#: Name of a region's MPD array, one row per entry, in the region directory.
MPDS_FILE = "mpds.npy"
#: Name of the directory, inside a region directory, that holds the entries' bank maps.
MAPS_DIR = "maps"
#: Name of a region's writer lock file, in the region directory.
LOCK_FILE = ".lock"

#: Columns of the entries table stored as ``int64``.
INT_COLUMNS = ("entry_id",)
#: Columns of the entries table stored as ``bool``.
BOOL_COLUMNS = ("valid",)
#: Columns of the entries table stored as pandas' nullable ``Int64``; ``<NA>`` marks an
#: unknown value.
NULLABLE_INT_COLUMNS = ("variability_seed", "map_seed")
#: Columns of the entries table stored as ``float64``; NaN marks a missing value.
FLOAT_COLUMNS = (
    "kappa",
    "gamma",
    "s",
    "intrinsic_quantile",
    "mag_min",
    "mag_max",
    "simplex_loss",
    "abs_residual",
    "rel_residual",
    "max_residual",
)
#: Columns of the entries table stored as ``str``; a missing value becomes ``""``.
STRING_COLUMNS = ("error", "map_file", "generator_version", "origin", "created_at")
#: Every column of the entries table, in the order in which it is stored.
ENTRY_COLUMNS = (
    "entry_id",
    "kappa",
    "gamma",
    "s",
    "valid",
    "error",
    "intrinsic_quantile",
    "map_file",
    "mag_min",
    "mag_max",
    "variability_seed",
    "map_seed",
    "generator_version",
    "origin",
    "created_at",
    "simplex_loss",
    "abs_residual",
    "rel_residual",
    "max_residual",
)
_CSV_DTYPES: dict[str, Any] = {
    **{column: "int64" for column in INT_COLUMNS},
    **{column: bool for column in BOOL_COLUMNS},
    **{column: "Int64" for column in NULLABLE_INT_COLUMNS},
    **{column: "float64" for column in FLOAT_COLUMNS},
    **{column: str for column in STRING_COLUMNS},
}


def utc_now() -> str:
    """Return the current UTC time in ISO 8601, to the second.

    Used for the ``created_at`` and ``finalized_at`` fields and for the holder line of a
    lock file.

    Returns
    -------
    str
        Time with an explicit offset, such as ``"2026-09-30T15:22:33+00:00"``.
    """
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _atomic_replace(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    """Write through a temporary file in the same directory, then rename it over ``path``.

    ``write`` fills the temporary file ``.<name>.tmp`` next to ``path``. The file is then
    flushed, fsynced, so that its contents are on disk, and moved over ``path`` with
    :func:`os.replace`, which is atomic within one file system. A reader therefore sees
    either the old file or the whole new one, never a partial write.

    Parameters
    ----------
    path : pathlib.Path
        File to replace or create. Its directory must exist.
    write : callable
        Function that writes the new contents to the binary stream that it is given. Its
        return value is ignored.

    Notes
    -----
    If ``write``, the flush, the fsync or the rename raises, ``path`` is unchanged, the
    temporary file is removed and the error propagates. The temporary name is
    fixed, so two writers must never write the same ``path`` at once; within a region,
    :class:`RegionLock` ensures this. The directory is not fsynced after the rename.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` with ``data``.

    The bytes go to a temporary file in the same directory, which is fsynced and then
    renamed over ``path``, so a reader sees the old contents or the new ones, never a mix.

    Parameters
    ----------
    path : pathlib.Path
        File to replace or create. Its directory must exist.
    data : bytes
        New contents of the file.
    """
    _atomic_replace(path, lambda stream: stream.write(data))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write ``payload`` as indented JSON.

    The text has two-space indentation, sorted keys and a final newline, and is written as
    UTF-8 through :func:`atomic_write_bytes`. It is built before anything is written, so a
    payload that cannot be encoded leaves ``path`` untouched. It writes ``bank.json`` and
    ``region.json``.

    Parameters
    ----------
    path : pathlib.Path
        File to replace or create. Its directory must exist.
    payload : dict of str to Any
        JSON-serialisable mapping to write.

    Raises
    ------
    ValueError
        If ``payload`` contains NaN or an infinity, which strict JSON cannot represent.
    """
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_save_array(path: Path, array: np.ndarray) -> None:
    """Atomically write ``array`` in NumPy ``.npy`` format.

    The array is saved with :func:`numpy.save` and ``allow_pickle=False`` into a temporary
    file in the same directory, which is fsynced and then renamed over ``path``. ``path``
    is used as given: no ``.npy`` suffix is added. It writes ``mpds.npy`` and the map files.

    Parameters
    ----------
    path : pathlib.Path
        File to replace or create. Its directory must exist.
    array : numpy.ndarray
        Array to save; anything that :func:`numpy.asarray` accepts.

    Raises
    ------
    ValueError
        If the array has an object dtype, which cannot be saved without pickling. ``path``
        is then unchanged.
    """
    _atomic_replace(path, lambda stream: np.save(stream, np.asarray(array), allow_pickle=False))


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a file, raising ``BankCorruptError`` if it is missing or unreadable.

    Used for ``bank.json`` and ``region.json``. The file is decoded as UTF-8. Other errors,
    such as a permission error, propagate unchanged.

    Parameters
    ----------
    path : pathlib.Path
        File to read.

    Returns
    -------
    dict of str to Any
        The decoded JSON object.

    Raises
    ------
    BankCorruptError
        If the file does not exist, is not UTF-8, does not hold valid JSON, or holds JSON
        whose top-level value is not an object.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BankCorruptError(f"Missing file {path}.") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BankCorruptError(f"Unreadable JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BankCorruptError(f"{path} does not hold a JSON object.")
    return payload


def normalise_entries(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the entries table with exactly the standard columns, order and dtypes.

    Keeps the columns of :data:`ENTRY_COLUMNS`, in that order, and drops any others. It
    converts them to ``int64``, ``bool``, nullable ``Int64``, ``float64`` and ``str`` as
    :data:`INT_COLUMNS`, :data:`BOOL_COLUMNS`, :data:`NULLABLE_INT_COLUMNS`,
    :data:`FLOAT_COLUMNS` and :data:`STRING_COLUMNS` say, turns missing strings into ``""``
    and resets the index to ``0, 1, 2, ...``. ``frame`` is not modified. The readers and
    builders of entries tables in this module return their tables through it.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table with at least the columns of :data:`ENTRY_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        A new table in the standard form.

    Raises
    ------
    BankCorruptError
        If ``frame`` lacks any column of :data:`ENTRY_COLUMNS`, or a ``valid`` flag is
        missing. Another value that cannot be converted, such as a missing ``entry_id``,
        raises the pandas conversion error instead.
    """
    missing = sorted(set(ENTRY_COLUMNS).difference(frame.columns))
    if missing:
        raise BankCorruptError(f"The entries table is missing columns {missing}.")
    table = frame.loc[:, list(ENTRY_COLUMNS)].copy()
    for column in INT_COLUMNS:
        table[column] = table[column].astype("int64")
    for column in BOOL_COLUMNS:
        # astype(bool) would read NaN as True and None as False.
        if table[column].isna().any():
            raise BankCorruptError(f"The entries table has missing values in {column!r}.")
        table[column] = table[column].astype(bool)
    for column in NULLABLE_INT_COLUMNS:
        table[column] = table[column].astype("Int64")
    for column in FLOAT_COLUMNS:
        table[column] = table[column].astype("float64")
    for column in STRING_COLUMNS:
        table[column] = table[column].fillna("").astype(str)
    return table.reset_index(drop=True)


def empty_entries() -> pd.DataFrame:
    """Return an entries table with no rows.

    It has every column of :data:`ENTRY_COLUMNS` with its standard dtype.
    :meth:`RegionStore.initialise` writes it for a new region.

    Returns
    -------
    pandas.DataFrame
        Empty table in the standard form of :func:`normalise_entries`.
    """
    return normalise_entries(pd.DataFrame({column: [] for column in ENTRY_COLUMNS}))


def append_entry(entries: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    """Return a new entries table with ``row`` appended.

    ``row`` is normalised on its own, then concatenated with ``entries`` and normalised
    again; ``entries`` is not modified. The row's ``entry_id`` is not checked: the caller
    must give it ``len(entries)``, since :meth:`RegionStore.load` rejects a table whose
    IDs are not ``0, 1, 2, ...``.

    Parameters
    ----------
    entries : pandas.DataFrame
        Current entries table, in the standard form.
    row : dict of str to Any
        Values of the new entry, keyed by column name. It must have every column of
        :data:`ENTRY_COLUMNS`; other keys are dropped.

    Returns
    -------
    pandas.DataFrame
        A new table in the standard form, with ``row`` as its last row.

    Raises
    ------
    BankCorruptError
        If ``row`` lacks a column of :data:`ENTRY_COLUMNS`.
    """
    new = normalise_entries(pd.DataFrame([row]))
    if entries.empty:
        return new
    return normalise_entries(pd.concat([entries, new], ignore_index=True))


def read_entries(path: Path) -> pd.DataFrame:
    """Read ``entries.csv``, raising ``BankCorruptError`` if it is missing or malformed.

    Each column is parsed with its stored dtype, and floats with pandas' round-trip
    parser, so the values that :func:`write_entries` wrote come back bit for bit. The
    table is then normalised with :func:`normalise_entries`. Empty fields in float,
    nullable-integer and string columns become NaN, ``<NA>`` and ``""``.

    Parameters
    ----------
    path : pathlib.Path
        The ``entries.csv`` file.

    Returns
    -------
    pandas.DataFrame
        The entries table in the standard form.

    Raises
    ------
    BankCorruptError
        If the file does not exist, if pandas cannot parse it or convert a column to its
        dtype (an empty file included), or if it lacks a column of :data:`ENTRY_COLUMNS`.
    """
    try:
        # pandas' default float parser does not round-trip every value that to_csv writes.
        frame = pd.read_csv(path, dtype=_CSV_DTYPES, float_precision="round_trip")
    except FileNotFoundError as exc:
        raise BankCorruptError(f"Missing file {path}.") from exc
    except (ValueError, pd.errors.ParserError) as exc:
        raise BankCorruptError(f"Malformed entries table {path}: {exc}") from exc
    return normalise_entries(frame)


def write_entries(path: Path, entries: pd.DataFrame) -> None:
    """Atomically write the entries table as CSV.

    Writes ``entries.to_csv(index=False)`` as UTF-8 through :func:`atomic_write_bytes`,
    with NaN and ``<NA>`` as empty fields. In :meth:`RegionStore.commit` this is the last
    step, which makes the new rows entries of the region.

    Parameters
    ----------
    path : pathlib.Path
        The ``entries.csv`` file.
    entries : pandas.DataFrame
        Entries table in the standard form of :func:`normalise_entries`. It is written as
        given, without being normalised.
    """
    atomic_write_bytes(path, entries.to_csv(index=False).encode("utf-8"))


@dataclass(frozen=True)
class RegionSnapshot:
    """A consistent view of one region's files.

    :meth:`RegionStore.load` returns it after checking that the three parts agree, and
    :class:`~adaptive_microlensing.bank.MapBank` builds a region's in-memory state from it.

    Parameters
    ----------
    meta : dict of str to Any
        Contents of ``region.json``: ``schema_version``, ``region``, ``finalized``,
        ``bin_edges`` (``None`` until the region is finalized) and ``finalized_at``.
    entries : pandas.DataFrame
        Entries table in the standard form, with ``entry_id`` equal to ``0, 1, 2, ...``.
    mpds : numpy.ndarray
        Array of shape ``(len(entries), n_mpd_columns)`` whose row *i* is the MPD of entry
        *i*, or NaN where it has not been computed.
    """

    meta: dict[str, Any]
    entries: pd.DataFrame
    mpds: np.ndarray


class RegionStore:
    """Reader and writer of the files of one region directory.

    It knows the paths of the region's files, creates an empty region
    (:meth:`initialise`), loads a checked, consistent snapshot (:meth:`load`) and writes
    the files atomically in the commit order. It holds no table in memory and takes no
    lock: :class:`~adaptive_microlensing.bank.MapBank` keeps the region's state and holds
    its :class:`RegionLock` while it writes.

    Parameters
    ----------
    directory : pathlib.Path
        The region directory, ``<bank>/<region>``. It need not exist before
        :meth:`initialise`.
    region : str
        Name of the region, such as ``"minima"``. It is written to ``region.json`` and
        checked against it on :meth:`load`.
    """

    def __init__(self, directory: Path, region: str) -> None:
        """Store the region directory and name; no file is touched."""
        self.directory = Path(directory)
        self.region = region

    @property
    def meta_path(self) -> Path:
        """Path of ``region.json``."""
        return self.directory / REGION_FILE

    @property
    def entries_path(self) -> Path:
        """Path of ``entries.csv``."""
        return self.directory / ENTRIES_FILE

    @property
    def mpds_path(self) -> Path:
        """Path of ``mpds.npy``."""
        return self.directory / MPDS_FILE

    @property
    def lock_path(self) -> Path:
        """Path of the writer lock file, ``.lock``."""
        return self.directory / LOCK_FILE

    @staticmethod
    def map_file(entry_id: int) -> str:
        """Return the map path of an entry, relative to the region directory.

        This is the value of the ``map_file`` column: ``maps/map_{entry_id:06d}.npy``, with a
        forward slash on every platform. IDs of a million or more give more than six digits.

        Parameters
        ----------
        entry_id : int
            ID of the entry.

        Returns
        -------
        str
            Relative path such as ``"maps/map_000042.npy"``.
        """
        return f"{MAPS_DIR}/map_{entry_id:06d}.npy"

    def map_path(self, entry_id: int) -> Path:
        """Return the path of an entry's map file under this directory.

        It is ``directory / map_file(entry_id)``, so it is absolute only if ``directory``
        is. The file need not exist.

        Parameters
        ----------
        entry_id : int
            ID of the entry.

        Returns
        -------
        pathlib.Path
            Path of the entry's map file.
        """
        return self.directory / self.map_file(entry_id)

    def initialise(self, n_mpd_columns: int) -> None:
        """Create the directory and the files of an empty region.

        Creates the region directory, with any missing parents, and its ``maps``
        directory; writes ``region.json`` with ``finalized`` false and no bin edges; then
        commits an empty entries table and an empty ``mpds.npy`` of shape
        ``(0, n_mpd_columns)``. It takes no lock.
        :meth:`~adaptive_microlensing.bank.MapBank.create` calls it for each region.

        Parameters
        ----------
        n_mpd_columns : int
            Length of one MPD, the width of ``mpds.npy``; see
            :attr:`~adaptive_microlensing.config.BankConfig.n_mpd_columns`.

        Raises
        ------
        FileExistsError
            If the region's ``maps`` directory already exists. Nothing is written then.
        """
        (self.directory / MAPS_DIR).mkdir(parents=True, exist_ok=False)
        self.write_meta(
            {
                "schema_version": SCHEMA_VERSION,
                "region": self.region,
                "finalized": False,
                "bin_edges": None,
                "finalized_at": None,
            }
        )
        self.commit(empty_entries(), np.empty((0, n_mpd_columns)))

    def load(self, n_mpd_columns: int) -> RegionSnapshot:
        """Read ``region.json``, then ``entries.csv``, then ``mpds.npy``.

        Writers change these files in the opposite order, so this order always
        gives a consistent snapshot. Rows of ``mpds.npy`` beyond the length of the table,
        as left by a commit that stopped before its last step, are dropped. ``mpds.npy`` is
        read whole into memory; the map files are not opened.

        Parameters
        ----------
        n_mpd_columns : int
            Expected width of ``mpds.npy``, the length of one MPD.

        Returns
        -------
        RegionSnapshot
            The region's metadata, entries table and MPDs, with one MPD row per entry.

        Raises
        ------
        BankCorruptError
            If ``region.json``, ``entries.csv`` or ``mpds.npy`` is missing; if
            ``region.json`` is unreadable (see :func:`read_json`), does not describe schema
            :data:`SCHEMA_VERSION` of this region, lacks a boolean ``finalized`` or the
            ``bin_edges`` field, or is finalized without bin edges; if ``entries.csv`` is
            malformed; if NumPy cannot read ``mpds.npy``, or it is not two-dimensional with
            ``n_mpd_columns`` columns, or has fewer rows than the table; if the
            ``entry_id`` values are not ``0, 1, 2, ...``; or if the region is finalized and
            the MPD of a valid entry contains NaN.
        """
        meta = read_json(self.meta_path)
        if meta.get("schema_version") != SCHEMA_VERSION or meta.get("region") != self.region:
            raise BankCorruptError(
                f"{self.meta_path} does not describe schema {SCHEMA_VERSION} {self.region!r}."
            )
        if not isinstance(meta.get("finalized"), bool) or "bin_edges" not in meta:
            raise BankCorruptError(f"{self.meta_path} lacks a boolean 'finalized' or the 'bin_edges' field.")
        if meta["finalized"] and meta["bin_edges"] is None:
            raise BankCorruptError(f"{self.meta_path} is finalized but has no bin edges.")
        entries = read_entries(self.entries_path)
        try:
            mpds = np.load(self.mpds_path, allow_pickle=False)
        except FileNotFoundError as exc:
            raise BankCorruptError(f"Missing file {self.mpds_path}.") from exc
        except (ValueError, EOFError) as exc:
            raise BankCorruptError(f"Unreadable array in {self.mpds_path}: {exc}") from exc
        if mpds.ndim != 2 or mpds.shape[1] != n_mpd_columns:
            raise BankCorruptError(f"{self.mpds_path} has shape {mpds.shape}; expected (n, {n_mpd_columns}).")
        if len(mpds) < len(entries):
            raise BankCorruptError(f"{self.mpds_path} has fewer rows than {self.entries_path}.")
        mpds = mpds[: len(entries)]
        if not np.array_equal(entries["entry_id"].to_numpy(), np.arange(len(entries))):
            raise BankCorruptError(f"{self.entries_path} entry_id values are not 0, 1, 2, ...")
        if meta["finalized"] and np.isnan(mpds[entries["valid"].to_numpy(dtype=bool)]).any():
            raise BankCorruptError(f"{self.mpds_path} is missing MPDs for valid entries.")
        return RegionSnapshot(meta=meta, entries=entries, mpds=mpds)

    def write_map(self, entry_id: int, mag_map: np.ndarray) -> str:
        """Write an entry's bank map, commit step 1, and return its path relative to the region.

        The map is saved atomically to :meth:`map_path`, replacing any file already there,
        such as one left by a commit that did not finish. The entry does not exist until
        :meth:`commit` has written its row. The ``maps`` directory must exist.

        Parameters
        ----------
        entry_id : int
            ID of the new entry, which is the current number of entries.
        mag_map : numpy.ndarray
            The bank map, a square array of relative magnitudes.

        Returns
        -------
        str
            The map's path relative to the region directory, for the ``map_file`` column
            (see :meth:`map_file`).
        """
        atomic_save_array(self.map_path(entry_id), mag_map)
        return self.map_file(entry_id)

    def commit(self, entries: pd.DataFrame, mpds: np.ndarray) -> None:
        """Write ``mpds.npy``, then ``entries.csv``: commit steps 2 and 3.

        Both files are replaced whole and atomically. Writing ``entries.csv`` last is
        what commits the new rows: a crash between the two writes leaves extra MPD rows,
        which :meth:`load` drops. The two arguments are not checked against each other.

        Parameters
        ----------
        entries : pandas.DataFrame
            The complete new entries table, in the standard form.
        mpds : numpy.ndarray
            The complete new MPD array, of shape ``(len(entries), n_mpd_columns)``.
        """
        atomic_save_array(self.mpds_path, mpds)
        write_entries(self.entries_path, entries)

    def write_mpds(self, mpds: np.ndarray) -> None:
        """Replace ``mpds.npy``.

        The array is written atomically, and ``entries.csv`` is not touched. Finalizing
        writes every entry's MPD with it before it marks the region finalized in
        ``region.json``, so a reader never sees a finalized region without its MPDs.

        Parameters
        ----------
        mpds : numpy.ndarray
            MPD array of shape ``(n_entries, n_mpd_columns)``.
        """
        atomic_save_array(self.mpds_path, mpds)

    def write_meta(self, meta: dict[str, Any]) -> None:
        """Replace ``region.json``.

        ``meta`` is written atomically as indented JSON with :func:`atomic_write_json`.

        Parameters
        ----------
        meta : dict of str to Any
            Region metadata, with the keys described in :class:`RegionSnapshot`.

        Raises
        ------
        ValueError
            If ``meta`` contains NaN or an infinity, for example in ``bin_edges``.
        """
        atomic_write_json(self.meta_path, meta)


_HELD_LOCKS: set[Path] = set()


def _release_lock(fd: int, key: Path) -> None:
    """Release a region lock's OS resources: unlock, close the fd, and forget its key.

    It is the ``weakref.finalize`` callback of a :class:`RegionLock`, so it runs at most
    once per acquisition: from :meth:`RegionLock.release`, when the lock object is
    garbage-collected, or at interpreter exit, whichever comes first. An ``OSError`` from
    the unlock or the close is ignored: a descriptor that was already closed elsewhere
    released its lock when it was closed. The key is forgotten in every case, so the
    region can be locked again.

    Parameters
    ----------
    fd : int
        File descriptor of the open lock file.
    key : pathlib.Path
        Resolved path of the lock file, as recorded in the registry of locks held by this
        process.
    """
    try:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        # Closing a descriptor that was already closed fails, but that close released the lock.
        with contextlib.suppress(OSError):
            os.close(fd)
    finally:
        _HELD_LOCKS.discard(key)


class RegionLock:
    """Exclusive, non-blocking writer lock on one region directory.

    Uses ``fcntl.flock`` on the region's ``.lock`` file. On NFSv4 this is a
    whole-file lock held by the server, so it also excludes writers on other nodes.
    The lock is also released when this object is garbage-collected, even if
    ``release`` was never called.

    A module-level registry of the lock files held by this process also refuses a second
    lock on the same file from the same process, for example from a second
    :class:`~adaptive_microlensing.bank.MapBank` opened for writing on the same region. On
    NFS, where Linux emulates ``flock`` with POSIX record locks, which never conflict
    within one process, ``flock`` alone would not refuse it.

    Parameters
    ----------
    path : pathlib.Path
        Path of the lock file, usually :attr:`RegionStore.lock_path`. :meth:`acquire`
        creates the file if it does not exist, but its directory must exist.

    Notes
    -----
    While the lock is held, the file holds a single line,
    ``host=<hostname> pid=<pid> since=<UTC time>``, which a refused writer reports in its
    :exc:`~adaptive_microlensing.errors.BankLockedError`. The file, with the last holder's
    line, stays in place after release.

    :meth:`acquire` registers a ``weakref.finalize`` callback that unlocks and closes the
    file and removes it from the registry. :meth:`release` runs it early; otherwise it
    runs when this object is garbage-collected or, if the object is still alive then, at
    interpreter exit. The callback holds the file descriptor and the path, not this
    object, so it does not keep the object alive.
    """

    def __init__(self, path: Path) -> None:
        """Store the lock file's path; the lock is not taken until :meth:`acquire`."""
        self.path = Path(path)
        self._fd: int | None = None
        self._key: Path | None = None
        self._finalizer: weakref.finalize | None = None

    def acquire(self) -> None:
        """Take the lock or raise ``BankLockedError``.

        The call never waits. It refuses at once if this process already holds a lock on
        the same resolved path. Otherwise it opens the lock file, creating it with mode
        ``0o644`` if needed, and tries an exclusive, non-blocking ``flock``. On success it
        replaces the file's contents with the holder line, records the path in the
        registry of held locks and registers the finaliser that releases the lock.

        Raises
        ------
        BankLockedError
            If this process already holds a lock on the file, through this or another
            :class:`RegionLock`, or if another process holds it. For another process, the
            message names the holder recorded in the file, or "another process" if the
            file is empty.
        OSError
            If the lock file cannot be opened, if ``flock`` fails for any reason other
            than the lock being held elsewhere, or if the holder line cannot be written;
            in the last case the lock is released again.
        """
        key = self.path.resolve()
        if key in _HELD_LOCKS:
            raise BankLockedError(f"{self.path.parent} is already open for writing in this process.")
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                holder = self.path.read_text(encoding="utf-8").strip() or "another process"
                raise BankLockedError(
                    f"{self.path.parent} is already open for writing by {holder}."
                ) from None
            raise
        holder = f"host={socket.gethostname()} pid={os.getpid()} since={utc_now()}"
        try:
            os.ftruncate(fd, 0)
            os.pwrite(fd, holder.encode("utf-8"), 0)
        except BaseException:
            # Nothing tracks the lock yet, so give it back before the error propagates.
            _release_lock(fd, key)
            raise
        self._fd = fd
        self._key = key
        _HELD_LOCKS.add(key)
        self._finalizer = weakref.finalize(self, _release_lock, fd, key)

    def release(self) -> None:
        """Release the lock if it is held.

        Runs the finaliser that :meth:`acquire` registered, which unlocks and closes the
        lock file and removes it from the registry of held locks. Calling it again, or on
        a lock that was never acquired, does nothing, and the lock can be acquired again
        afterwards. The holder line stays in the file.
        """
        if self._fd is None:
            return
        if self._finalizer is not None:
            self._finalizer()
        self._fd = None
        self._key = None
