"""On-disk layout of a bank: atomic writes, commit order, snapshots and writer locks.

A region directory holds ``region.json``, ``entries.csv``, ``mpds.npy`` and ``maps/``.
A new entry is committed in three atomic steps: its map, then ``mpds.npy``, then
``entries.csv``. An entry exists only once its row is in ``entries.csv``.
"""

from __future__ import annotations

import datetime
import errno
import fcntl
import json
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from .errors import BankCorruptError, BankLockedError

SCHEMA_VERSION = 1
BANK_FILE = "bank.json"
REGION_FILE = "region.json"
ENTRIES_FILE = "entries.csv"
MPDS_FILE = "mpds.npy"
MAPS_DIR = "maps"
LOCK_FILE = ".lock"

INT_COLUMNS = ("entry_id",)
BOOL_COLUMNS = ("valid",)
NULLABLE_INT_COLUMNS = ("variability_seed", "map_seed")
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
STRING_COLUMNS = ("error", "map_file", "generator_version", "origin", "created_at")
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
    """Current UTC time in ISO 8601, to the second."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _atomic_replace(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    """Write through a temporary file in the same directory, then rename it over ``path``."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` with ``data``."""
    _atomic_replace(path, lambda stream: stream.write(data))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write ``payload`` as indented JSON."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_save_array(path: Path, array: np.ndarray) -> None:
    """Atomically write ``array`` in NumPy ``.npy`` format."""
    _atomic_replace(path, lambda stream: np.save(stream, np.asarray(array), allow_pickle=False))


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, raising ``BankCorruptError`` if it is missing or unreadable."""
    try:
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BankCorruptError(f"Missing file {path}.") from exc
    except json.JSONDecodeError as exc:
        raise BankCorruptError(f"Unreadable JSON in {path}: {exc}") from exc
    return payload


def normalise_entries(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the entries table with exactly the standard columns, order and dtypes."""
    missing = sorted(set(ENTRY_COLUMNS).difference(frame.columns))
    if missing:
        raise BankCorruptError(f"The entries table is missing columns {missing}.")
    table = frame.loc[:, list(ENTRY_COLUMNS)].copy()
    for column in INT_COLUMNS:
        table[column] = table[column].astype("int64")
    for column in BOOL_COLUMNS:
        table[column] = table[column].astype(bool)
    for column in NULLABLE_INT_COLUMNS:
        table[column] = table[column].astype("Int64")
    for column in FLOAT_COLUMNS:
        table[column] = table[column].astype("float64")
    for column in STRING_COLUMNS:
        table[column] = table[column].fillna("").astype(str)
    return table.reset_index(drop=True)


def empty_entries() -> pd.DataFrame:
    """An entries table with no rows."""
    return normalise_entries(pd.DataFrame({column: [] for column in ENTRY_COLUMNS}))


def append_entry(entries: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    """A new entries table with ``row`` appended."""
    new = normalise_entries(pd.DataFrame([row]))
    if entries.empty:
        return new
    return normalise_entries(pd.concat([entries, new], ignore_index=True))


def read_entries(path: Path) -> pd.DataFrame:
    """Read ``entries.csv``, raising ``BankCorruptError`` if it is missing or malformed."""
    try:
        # pandas' default float parser does not round-trip every value that to_csv writes.
        frame = pd.read_csv(path, dtype=_CSV_DTYPES, float_precision="round_trip")
    except FileNotFoundError as exc:
        raise BankCorruptError(f"Missing file {path}.") from exc
    except (ValueError, pd.errors.ParserError) as exc:
        raise BankCorruptError(f"Malformed entries table {path}: {exc}") from exc
    return normalise_entries(frame)


def write_entries(path: Path, entries: pd.DataFrame) -> None:
    """Atomically write the entries table as CSV."""
    atomic_write_bytes(path, entries.to_csv(index=False).encode("utf-8"))


@dataclass(frozen=True)
class RegionSnapshot:
    """A consistent view of one region's files."""

    meta: dict[str, Any]
    entries: pd.DataFrame
    mpds: np.ndarray


class RegionStore:
    """Reads and writes the files of one region directory."""

    def __init__(self, directory: Path, region: str) -> None:
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
        """Path of the writer lock file."""
        return self.directory / LOCK_FILE

    @staticmethod
    def map_file(entry_id: int) -> str:
        """Map path of an entry, relative to the region directory."""
        return f"{MAPS_DIR}/map_{entry_id:06d}.npy"

    def map_path(self, entry_id: int) -> Path:
        """Absolute-or-relative path of an entry's map file under this directory."""
        return self.directory / self.map_file(entry_id)

    def initialise(self, n_mpd_columns: int) -> None:
        """Create the directory and the files of an empty region."""
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
        gives a consistent snapshot.
        """
        meta = read_json(self.meta_path)
        if meta.get("schema_version") != SCHEMA_VERSION or meta.get("region") != self.region:
            raise BankCorruptError(
                f"{self.meta_path} does not describe schema {SCHEMA_VERSION} {self.region!r}."
            )
        entries = read_entries(self.entries_path)
        try:
            mpds = np.load(self.mpds_path, allow_pickle=False)
        except FileNotFoundError as exc:
            raise BankCorruptError(f"Missing file {self.mpds_path}.") from exc
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
        """Commit step 1: write an entry's map. Returns its path relative to the region."""
        atomic_save_array(self.map_path(entry_id), mag_map)
        return self.map_file(entry_id)

    def commit(self, entries: pd.DataFrame, mpds: np.ndarray) -> None:
        """Commit steps 2 and 3: write ``mpds.npy``, then ``entries.csv``."""
        atomic_save_array(self.mpds_path, mpds)
        write_entries(self.entries_path, entries)

    def write_mpds(self, mpds: np.ndarray) -> None:
        """Replace ``mpds.npy``."""
        atomic_save_array(self.mpds_path, mpds)

    def write_meta(self, meta: dict[str, Any]) -> None:
        """Replace ``region.json``."""
        atomic_write_json(self.meta_path, meta)


_HELD_LOCKS: set[Path] = set()


class RegionLock:
    """Exclusive, non-blocking writer lock on one region directory.

    Uses ``fcntl.flock`` on the region's ``.lock`` file. On NFSv4 this is a
    whole-file lock held by the server, so it also excludes writers on other nodes.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None
        self._key: Path | None = None

    def acquire(self) -> None:
        """Take the lock or raise ``BankLockedError``."""
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
        os.ftruncate(fd, 0)
        os.pwrite(fd, holder.encode("utf-8"), 0)
        self._fd = fd
        self._key = key
        _HELD_LOCKS.add(key)

    def release(self) -> None:
        """Release the lock if it is held."""
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None
            if self._key is not None:
                _HELD_LOCKS.discard(self._key)
                self._key = None
