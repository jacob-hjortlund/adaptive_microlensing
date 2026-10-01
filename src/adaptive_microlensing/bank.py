"""The MapBank: build, load, query and update banks of microlensing maps.

:class:`MapBank` is the package's main entry point. A bank is a directory with a
``bank.json``, which records the :class:`~adaptive_microlensing.config.BankConfig`, and
one directory for each region in :data:`~adaptive_microlensing.lensing.REGIONS`.
:meth:`MapBank.create` makes a new bank and :meth:`MapBank.open` loads one, locking the
regions that will be written. :meth:`MapBank.build` adds entries by adaptive sampling,
:meth:`MapBank.finalize` freezes a region's MPD bin edges, :meth:`MapBank.query` looks for
a matching map without writing, and :meth:`MapBank.fetch` also makes and commits a new map
on a miss. The batch methods :meth:`MapBank.query_many` and :meth:`MapBank.fetch_many` do
the same for every row of a table.

This module ties the others together. The on-disk layout, commit steps and writer locks
come from :mod:`adaptive_microlensing.storage`, the Delaunay mesh and the hit rule from
:class:`~adaptive_microlensing.interpolation.RegionIndex`, the maps from a
:class:`~adaptive_microlensing.maps.MapGenerator`, MPDs and intrinsic quantiles from
:mod:`adaptive_microlensing.mpd`, and the returned objects from
:mod:`adaptive_microlensing.results`. Each region is held in memory as a ``_Region``: its
entries table, MPDs and index, loaded once when the bank is opened and extended after each
commit. :func:`entry_seeds` derives each new entry's seeds from the bank's seed, the
region and the entry ID alone.

The private methods ``_resolve_generator``, ``_evaluate_entry`` and ``_commit`` are shared
with the build loop, :func:`~adaptive_microlensing.design.build_region`, and
``_import_entries`` and ``_finalize_with_edges`` with
:func:`~adaptive_microlensing.legacy.import_legacy_bank`.
"""

from __future__ import annotations

import importlib.metadata
import logging
import math
import platform
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd

from ._version import __version__
from .config import BankConfig, StoppingCriteria
from .errors import (
    BankCorruptError,
    BankReadOnlyError,
    GeneratorMismatchError,
    InvalidMapError,
    MapGenerationError,
    RegionStateError,
)
from .interpolation import RegionIndex
from .lensing import CRITICAL, REGIONS, classify_image, in_box, in_domain
from .maps import MapGenerator, generator_from_spec, generator_identity
from .mpd import bin_edges_from_range, finite_range, histogram_with_overflow, intrinsic_quantile
from .results import (
    FETCH_COLUMNS,
    QUERY_COLUMNS,
    BankEntry,
    BuildSummary,
    FetchResult,
    FetchStatus,
    QueryResult,
    QueryStatus,
    check_result_columns,
    fetch_row,
    query_row,
    results_table,
)
from .storage import (
    BANK_FILE,
    SCHEMA_VERSION,
    RegionLock,
    RegionSnapshot,
    RegionStore,
    append_entry,
    atomic_write_json,
    normalise_entries,
    read_json,
    utc_now,
)

logger = logging.getLogger("adaptive_microlensing")

_REFUSALS = {
    QueryStatus.MALFORMED: FetchStatus.MALFORMED,
    QueryStatus.CRITICAL_LINE: FetchStatus.CRITICAL_LINE,
    QueryStatus.OUTSIDE_DOMAIN: FetchStatus.OUTSIDE_DOMAIN,
    QueryStatus.REGION_NOT_READY: FetchStatus.REGION_NOT_READY,
}


def entry_seeds(seed: int, region: str, entry_id: int) -> tuple[int, int]:
    """Derive the seeds of an entry's coarse map and bank map from (seed, region, entry ID).

    The seeds depend on nothing else, so an entry with the same ID in the same region gets
    the same seeds in every bank with the same base seed. Both lie in [1, 2**31 - 1]: IPM
    takes a C ``int`` and treats 0 as "pick a random seed".

    Parameters
    ----------
    seed : int
        The bank's base seed, ``BankConfig.seed``.
    region : str
        Name of the entry's region, one of ``REGIONS``.
    entry_id : int
        The entry's ID, which is its row number in the region's entries table.

    Returns
    -------
    variability_seed : int
        Seed of the coarse map used to measure the intrinsic quantile.
    map_seed : int
        Seed of the bank map.

    Raises
    ------
    ValueError
        If ``region`` is not one of ``REGIONS``.

    Notes
    -----
    The seeds are the two 32-bit words that
    ``numpy.random.SeedSequence(seed, spawn_key=(REGIONS.index(region), entry_id))``
    generates, each mapped to ``value % (2**31 - 1) + 1``.
    """
    state = np.random.SeedSequence(seed, spawn_key=(REGIONS.index(region), entry_id)).generate_state(
        2, dtype=np.uint32
    )
    variability_seed, map_seed = (int(value) % (2**31 - 1) + 1 for value in state)
    return variability_seed, map_seed


def _versions(config: BankConfig) -> dict[str, str | None]:
    """Return the software versions that ``bank.json`` records when a bank is created.

    Parameters
    ----------
    config : BankConfig
        Configuration of the new bank. Its generator is built to ask for its version.

    Returns
    -------
    dict of str to str or None
        The Python version under ``"python"``, the installed versions of ``numpy``,
        ``scipy``, ``pandas`` and ``adaptive``, and the map generator's version under
        ``"generator"``. A package that is not installed gives ``None``, and so does a
        generator that is not registered or that reports no version.
    """
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "pandas", "adaptive"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        versions["generator"] = generator_from_spec(config.generator).version()
    except ValueError:
        versions["generator"] = None
    return versions


def _writable_regions(writable: bool | Iterable[str]) -> set[str]:
    """Return the regions to lock for writing, from the ``writable`` argument of ``open``.

    Parameters
    ----------
    writable : bool, str, iterable of str or None
        ``True`` for every region, ``False`` or ``None`` for none, or the name or names of
        the regions to lock.

    Returns
    -------
    set of str
        The regions to lock for writing.

    Raises
    ------
    ValueError
        If a name is not one of ``REGIONS``.
    """
    if writable is True:
        return set(REGIONS)
    if writable is False or writable is None:
        return set()
    if isinstance(writable, str):
        writable = [writable]
    regions = set(writable)
    unknown = sorted(regions.difference(REGIONS))
    if unknown:
        raise ValueError(f"Unknown regions {unknown}; expected names from {REGIONS}.")
    return regions


def _optional_int(value: Any) -> int | None:
    """Return a cell of a nullable integer column as an ``int``, or ``None`` if it is missing.

    Parameters
    ----------
    value : Any
        A cell of a nullable ``Int64`` column, such as ``map_seed``. It is ``pandas.NA``
        when the value was not recorded, as for the coarse-map seed of a legacy entry.

    Returns
    -------
    int or None
        The value as a Python ``int``, or ``None`` if ``pandas.isna`` holds for it.
    """
    return None if pd.isna(value) else int(value)


@dataclass
class PendingEntry:
    """An evaluated entry whose row has not been committed yet.

    :meth:`MapBank._evaluate_entry` returns it once the entry's maps are made and its bank
    map is written, and :meth:`MapBank._commit` adds it to the region.

    Parameters
    ----------
    row : dict of str to Any
        The entry's row for ``entries.csv``, keyed by column name. The diagnostic columns
        (``simplex_loss`` and the residuals) are NaN; the build fills them in at commit.
    mpd : numpy.ndarray
        The entry's bank MPD, of shape ``(BankConfig.n_mpd_columns,)``. It is all NaN
        unless the entry is valid and the region is finalized.
    """

    row: dict[str, Any]
    mpd: np.ndarray

    @property
    def valid(self) -> bool:
        """Whether the evaluation produced a usable entry, from the row's ``valid`` flag."""
        return bool(self.row["valid"])


class _Region:
    """In-memory state of one region: its loaded files, its index and whether it is writable.

    :meth:`MapBank.open` makes one for each region. The bank replaces ``entries``,
    ``mpds``, ``meta`` and ``index`` after each write, so they stay in step with the files
    it writes.

    Parameters
    ----------
    store : RegionStore
        Reader and writer of the region's directory.
    snapshot : RegionSnapshot
        The region's ``region.json``, entries table and MPDs, as loaded from disk.
    writable : bool
        Whether the bank holds the region's writer lock.

    Attributes
    ----------
    meta : dict of str to Any
        Contents of ``region.json``: the finalized flag, the frozen bin edges and when the
        region was finalized.
    entries : pandas.DataFrame
        The region's entries table, one row per entry, valid and invalid, in ``entry_id``
        order.
    mpds : numpy.ndarray
        Bank MPDs, of shape ``(len(entries), BankConfig.n_mpd_columns)``. Rows of invalid
        entries are NaN, and the MPDs are used only once the region is finalized.
    index : RegionIndex
        Delaunay index over every entry, valid and invalid.
    """

    def __init__(self, store: RegionStore, snapshot: RegionSnapshot, writable: bool) -> None:
        """Take over the loaded snapshot and build the region's index."""
        self.store = store
        self.meta = snapshot.meta
        self.entries = snapshot.entries
        self.mpds = snapshot.mpds
        self.writable = writable
        self.index = self.build_index()

    @property
    def finalized(self) -> bool:
        """Whether the region's bin edges are frozen and its MPDs computed, from ``region.json``."""
        return bool(self.meta["finalized"])

    @property
    def bin_edges(self) -> np.ndarray:
        """Frozen bank-MPD bin edges of the region, as a float array.

        Raises
        ------
        RegionStateError
            If the region is not finalized, so it has no bin edges yet.
        """
        if self.meta["bin_edges"] is None:
            raise RegionStateError(f"Region {self.store.region!r} is not finalized.")
        return np.asarray(self.meta["bin_edges"], dtype=float)

    def build_index(self) -> RegionIndex:
        """Build a Delaunay index over every entry of the region.

        The index gets the region's MPDs only when the region is finalized, so an
        unfinalized region can locate points and interpolate quantiles but cannot apply the
        hit rule. With the MPDs, it computes the JS distances between all pairs of valid
        entries.

        Returns
        -------
        RegionIndex
            Index of the entries' ``(kappa, gamma, s)`` points, validity flags and intrinsic
            quantiles, and of their MPDs when the region is finalized.
        """
        return RegionIndex(
            self.entries[["kappa", "gamma", "s"]].to_numpy(dtype=float),
            self.entries["valid"].to_numpy(dtype=bool),
            self.entries["intrinsic_quantile"].to_numpy(dtype=float),
            self.mpds if self.finalized else None,
        )


class MapBank:
    """A bank of microlensing magnitude maps that can be built, queried and updated.

    Create a bank with :meth:`create` or load one with :meth:`open`; do not call the
    constructor directly. A bank opened for writing holds its region locks until
    :meth:`close` is called or the ``with`` block ends.

    Each region accepts one writer at a time. Opening a region for writing takes an
    exclusive, non-blocking lock on it, which excludes other processes and other bank
    objects in this process; :meth:`open` raises :exc:`BankLockedError` if another writer
    holds it. Reading takes no lock, so any number of processes can query a bank at once.
    A bank loads each region's files once, when it is opened, and then sees only the
    entries it commits itself; open the bank again to see what other writers have added.
    The locks are also released if a bank that was never closed is garbage-collected.

    Parameters
    ----------
    path : pathlib.Path
        The bank directory.
    meta : dict of str to Any
        Contents of ``bank.json``.
    config : BankConfig
        The configuration stored in ``bank.json``.
    regions : dict of str to _Region
        In-memory state of every region, keyed by region name.
    locks : list of RegionLock
        The writer locks that the bank holds, one for each writable region.
    """

    def __init__(
        self,
        path: Path,
        meta: dict[str, Any],
        config: BankConfig,
        regions: dict[str, _Region],
        locks: list[RegionLock],
    ) -> None:
        """Hold the state of a loaded bank; use :meth:`create` or :meth:`open` instead."""
        self._path = path
        self._meta = meta
        self._config = config
        self._regions = regions
        self._locks = locks
        self._version_warned = False

    # ----------------------------------------------------------------- lifecycle

    @classmethod
    def create(cls, path: str | Path, config: BankConfig | None = None) -> MapBank:
        """Create a new, empty bank directory and return it opened for writing.

        Makes the directory, and its parents, if needed. Writes an empty region directory
        for each of ``REGIONS``, then ``bank.json`` with the schema version, the creation
        time, the configuration and the versions of the package, its dependencies and the
        map generator. The returned bank holds the writer lock of every region, so close it
        before other processes open regions for writing, for example to build them in
        parallel.

        Parameters
        ----------
        path : str or pathlib.Path
            The new bank directory. It must not exist or must be empty.
        config : BankConfig or None, optional
            How the bank's entries are made. When ``None``, the default ``BankConfig()``
            is used.

        Returns
        -------
        MapBank
            The new bank, opened with ``writable=True``.

        Raises
        ------
        FileExistsError
            If ``path`` exists and is not empty.
        """
        root = Path(path)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"{root} exists and is not empty.")
        config = BankConfig() if config is None else config
        root.mkdir(parents=True, exist_ok=True)
        for region in REGIONS:
            RegionStore(root / region, region).initialise(config.n_mpd_columns)
        atomic_write_json(
            root / BANK_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "package_version": __version__,
                "versions": _versions(config),
                "config": config.to_dict(),
            },
        )
        return cls.open(root, writable=True)

    @classmethod
    def open(cls, path: str | Path, writable: bool | Iterable[str] = False) -> MapBank:
        """Load a bank.

        Reads ``bank.json`` and checks its schema version, takes the writer lock of each
        region to be written, then loads every region's files. Regions that are not locked
        are read-only: they answer queries, but building, finalizing or making a map in one
        raises :exc:`BankReadOnlyError`. If anything fails, the locks taken so far are
        released before the error propagates.

        Parameters
        ----------
        path : str or pathlib.Path
            The bank directory.
        writable : bool or iterable of str, optional
            ``False`` (read-only), ``True`` (lock every region), or the names of the
            regions to lock for writing. A single region name is also accepted. Default
            is ``False``.

        Returns
        -------
        MapBank
            The loaded bank. Use it in a ``with`` block, or call :meth:`close`, to release
            its locks.

        Raises
        ------
        BankCorruptError
            If ``bank.json`` is missing, is not a readable JSON object, has an unknown
            schema version or holds no valid configuration, or if a region's files are
            missing, malformed or inconsistent with each other.
        ValueError
            If ``writable`` names a region that is not one of ``REGIONS``.
        BankLockedError
            If a region to be locked is already open for writing, in this process or in
            another one.
        """
        root = Path(path)
        meta = read_json(root / BANK_FILE)
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise BankCorruptError(f"{root / BANK_FILE} has an unknown schema version.")
        try:
            config = BankConfig.from_dict(meta["config"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BankCorruptError(
                f"{root / BANK_FILE} holds no valid configuration: {type(exc).__name__}: {exc}"
            ) from exc
        to_lock = _writable_regions(writable)
        locks: list[RegionLock] = []
        try:
            for region in REGIONS:
                if region in to_lock:
                    lock = RegionLock(RegionStore(root / region, region).lock_path)
                    lock.acquire()
                    locks.append(lock)
            regions = {}
            for region in REGIONS:
                store = RegionStore(root / region, region)
                regions[region] = _Region(store, store.load(config.n_mpd_columns), region in to_lock)
        except BaseException:
            for lock in locks:
                lock.release()
            raise
        return cls(root, meta, config, regions, locks)

    def close(self) -> None:
        """Release every region lock; the bank stays readable but can no longer be written.

        Queries still work on the loaded state, and every write raises
        :exc:`BankReadOnlyError`. Calling it again, or on a read-only bank, is harmless.
        """
        for lock in self._locks:
            lock.release()
        self._locks = []
        for state in self._regions.values():
            state.writable = False

    def __enter__(self) -> MapBank:
        """Return the bank itself, for use in a ``with`` block.

        Returns
        -------
        MapBank
            This bank.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the bank at the end of a ``with`` block, releasing its region locks.

        An exception raised in the block is not suppressed.

        Parameters
        ----------
        exc_type : type of BaseException or None
            Type of the exception raised in the block, or ``None`` if there was none.
        exc : BaseException or None
            The exception raised in the block, or ``None``.
        traceback : types.TracebackType or None
            Traceback of the exception, or ``None``.
        """
        self.close()

    # ---------------------------------------------------------------- inspection

    @property
    def path(self) -> Path:
        """The bank directory, as given to :meth:`create` or :meth:`open`."""
        return self._path

    @property
    def config(self) -> BankConfig:
        """The configuration stored in ``bank.json``, with which every entry is made."""
        return self._config

    def entries(self, region: str) -> pd.DataFrame:
        """Return a copy of a region's entries table.

        The table has one row per entry, valid and invalid, in ``entry_id`` order, with the
        columns of ``entries.csv``: the parameters, validity and error, intrinsic quantile,
        map file and magnitude range, seeds, provenance and build diagnostics. It shows the
        region as it was when the bank was opened, plus the entries this bank has
        committed since.

        Parameters
        ----------
        region : str
            Name of the region, one of ``REGIONS``.

        Returns
        -------
        pandas.DataFrame
            Copy of the entries table; changing it does not change the bank.

        Raises
        ------
        ValueError
            If ``region`` is not one of ``REGIONS``.
        """
        return self._region(region).entries.copy()

    def summary(self) -> pd.DataFrame:
        """Return a table of each region's entry counts and finalized state.

        The counts cover the entries loaded when the bank was opened and those it has
        committed since.

        Returns
        -------
        pandas.DataFrame
            One row per region, indexed by ``region`` in the order of ``REGIONS``, with the
            numbers of ``entries``, ``valid`` and ``invalid`` entries and the boolean
            ``finalized`` flag.
        """
        records = []
        for region, state in self._regions.items():
            valid = int(state.entries["valid"].sum())
            records.append(
                {
                    "region": region,
                    "entries": len(state.entries),
                    "valid": valid,
                    "invalid": len(state.entries) - valid,
                    "finalized": state.finalized,
                }
            )
        return pd.DataFrame.from_records(records).set_index("region")

    # -------------------------------------------------------------- build, finalize

    def build(
        self,
        region: str,
        stop: StoppingCriteria | None = None,
        *,
        generator: MapGenerator | None = None,
    ) -> BuildSummary:
        """Add entries to a region by adaptive sampling until a stopping rule holds.

        Resumes from the entries already in the region: every committed entry inside the
        region's design hull is first replayed into the learner, so an interrupted build
        can simply be run again. Each new entry is committed as soon as its maps are made,
        including an entry whose maps failed, which stays in the mesh as an invalid entry.
        If the stopping rule already holds, no entry is added. The region may already be
        finalized, in which case new entries get their MPDs on the frozen bin edges.

        Parameters
        ----------
        region : str
            Name of the region to build, one of ``REGIONS``.
        stop : StoppingCriteria or None, optional
            When to stop adding entries. When ``None``, ``StoppingCriteria()`` is used,
            which stops once the region has 500 valid entries.
        generator : MapGenerator or None, optional
            Generator that makes the maps. When ``None``, the generator named in the
            bank's configuration is used. It must match that configuration.

        Returns
        -------
        BuildSummary
            The region's entry counts, why the build stopped, the final simplex loss and
            the time spent adding entries.

        Raises
        ------
        ValueError
            If ``region`` is not one of ``REGIONS`` or does not intersect the bank's
            domain, if ``generator`` is ``None`` and no generator is registered under the
            configured name, or if the generator returns a map of the wrong shape.
        BankReadOnlyError
            If the region is not open for writing.
        GeneratorMismatchError
            If ``generator`` differs from the one recorded in the bank.

        See Also
        --------
        adaptive_microlensing.design.build_region : The build loop and its stopping checks.
        """
        from .design import build_region

        return build_region(self, region, StoppingCriteria() if stop is None else stop, generator)

    def finalize(self, region: str) -> None:
        """Freeze the region's bank-MPD bin edges and compute the MPDs of its maps.

        The ``BankConfig.n_bins + 1`` edges span the smallest ``mag_min`` to the largest
        ``mag_max`` of the region's valid entries, widened by one ulp at each end. Every
        valid entry's bank map is histogrammed on them, ``mpds.npy`` is written, and then
        ``region.json`` records the edges and marks the region finalized. A region answers
        queries only once it is finalized. The edges never change afterwards: maps added
        later get their MPDs on the same edges, with pixels outside them counted in the
        underflow and overflow bins. Progress is logged at ``INFO`` level.

        Parameters
        ----------
        region : str
            Name of the region, one of ``REGIONS``.

        Raises
        ------
        ValueError
            If ``region`` is not one of ``REGIONS``.
        BankReadOnlyError
            If the region is not open for writing.
        RegionStateError
            If the region is already finalized or has no valid entries.
        """
        state = self._region(region)
        self._require_writable(region)
        if state.finalized:
            raise RegionStateError(f"Region {region!r} is already finalized; its bin edges are frozen.")
        valid = state.entries["valid"].to_numpy(dtype=bool)
        if not valid.any():
            raise RegionStateError(f"Region {region!r} has no valid entries to finalize.")
        edges = bin_edges_from_range(
            float(state.entries.loc[valid, "mag_min"].min()),
            float(state.entries.loc[valid, "mag_max"].max()),
            self._config.n_bins,
        )
        self._finalize_with_edges(region, edges)

    # ------------------------------------------------------------- query and fetch

    def query(
        self, kappa: float, gamma: float, s: float, *, allow_outside_domain: bool = False
    ) -> QueryResult:
        """Look for an existing map indistinguishable from one at (kappa, gamma, s). Never writes.

        Classifies the point into a region, finds the tetrahedron of that region's entries
        that contains it, and applies the hit rule. It works on a read-only bank and never
        makes a map; :meth:`fetch` makes one on a miss. Bad input does not raise: it gives
        a result with status ``MALFORMED``.

        Parameters
        ----------
        kappa : float
            Total convergence of the macro model at the image.
        gamma : float
            Shear of the macro model at the image.
        s : float
            Smooth-matter fraction at the image.
        allow_outside_domain : bool, optional
            If ``True``, skip the domain check (the kappa/gamma/s box and the cap on
            ``|mu_macro|``), so the point is looked up in its region's mesh wherever it
            lies; a negative ``gamma``, which the box excludes, is looked up at
            ``|gamma|``. Points on a critical line are still refused. Default is ``False``.

        Returns
        -------
        QueryResult
            The outcome. Its ``status`` is one of these:

            - ``MALFORMED``: the input is not three finite numbers. Every other field keeps
              its default.
            - ``CRITICAL_LINE``: the point lies on a critical line, so ``region`` is
              ``None``.
            - ``OUTSIDE_DOMAIN``: the point is outside the bank's domain.
            - ``REGION_NOT_READY``: the region is not finalized, or its entries do not yet
              span a three-dimensional mesh.
            - ``OUTSIDE_HULL``: no tetrahedron of the region contains the point.
            - ``INVALID_SIMPLEX``: the point coincides with an invalid entry, whose ID is
              ``coincident_entry_id``, or its tetrahedron has an invalid vertex.
            - ``HIT``: an existing map matches. At a valid entry, that entry is the match,
              with ``interpolated_distance`` 0 and ``threshold`` and ``margin`` equal to its
              intrinsic quantile. Elsewhere, a vertex of the tetrahedron passes the hit rule.
            - ``MISS``: no vertex passes; ``matched`` is the vertex closest to passing and
              ``margin`` is negative.

            ``is_in_bounds``, whether the point lies in the domain's box, is set for every
            status but ``MALFORMED``, and ``region`` for every status after
            ``CRITICAL_LINE``. From ``REGION_NOT_READY`` on, ``region_entry_count`` is set.
            ``simplex_index``, ``simplex_entry_ids`` and ``barycentric_weights`` are set once
            the point is located in a tetrahedron, and ``vertex_distances`` only when the
            hit rule was applied there. ``entry`` is the match on a hit and ``None``
            otherwise.

        See Also
        --------
        fetch : Return a matching map, or make one on a miss.
        query_many : Query every row of a table.

        Notes
        -----
        The region is ``classify_image(kappa, gamma)``. The macro model depends only on
        ``|gamma|``, since a negative shear only turns the shear axis, so the mesh is searched
        at ``(kappa, |gamma|, s)``. A point within ``1e-12`` of an entry in every coordinate
        is treated as that entry. Otherwise vertex *i* of the
        containing tetrahedron passes when the barycentric-weighted sum of the vertices'
        JS distances to *i* is at most ``max(q_query, q_i)``, where ``q_query`` is the
        barycentric interpolation of the vertices' intrinsic quantiles. Every vertex is
        tested, and on a hit the match is the passing vertex with the smallest distance.
        """
        try:
            point = np.array([kappa, gamma, s], dtype=float)
        except (TypeError, ValueError):
            return QueryResult(QueryStatus.MALFORMED)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return QueryResult(QueryStatus.MALFORMED)
        k, g, s_value = (float(value) for value in point)
        domain = self._config.domain
        box = in_box(k, g, s_value, domain)
        region = classify_image(k, g)
        if region == CRITICAL:
            return QueryResult(QueryStatus.CRITICAL_LINE, is_in_bounds=box)
        if not allow_outside_domain and not in_domain(k, g, s_value, domain):
            return QueryResult(QueryStatus.OUTSIDE_DOMAIN, region=region, is_in_bounds=box)

        # The macro model depends only on |gamma|, as classify_image does: a negative shear
        # only turns the shear axis, so the point is looked up at |gamma|.
        lookup = np.array([k, abs(g), s_value])
        state = self._regions[region]
        index = state.index
        common: dict[str, Any] = {
            "region": region,
            "is_in_bounds": box,
            "region_entry_count": len(state.entries),
        }
        if not state.finalized or not index.ready:
            return QueryResult(QueryStatus.REGION_NOT_READY, **common)

        coincident = index.coincident(lookup)
        if coincident is not None:
            if not index.valid[coincident]:
                return QueryResult(QueryStatus.INVALID_SIMPLEX, coincident_entry_id=coincident, **common)
            quantile = float(index.quantiles[coincident])
            return QueryResult(
                QueryStatus.HIT,
                is_hit=True,
                matched=self._entry(region, coincident),
                interpolated_distance=0.0,
                query_quantile=quantile,
                matched_quantile=quantile,
                threshold=quantile,
                margin=quantile,
                coincident_entry_id=coincident,
                **common,
            )

        location = index.locate(lookup)
        if location is None:
            return QueryResult(QueryStatus.OUTSIDE_HULL, **common)
        common.update(
            simplex_index=location.simplex_index,
            simplex_entry_ids=tuple(int(vertex) for vertex in location.vertices),
            barycentric_weights=tuple(float(weight) for weight in location.weights),
        )
        if not index.simplex_is_valid(location):
            return QueryResult(QueryStatus.INVALID_SIMPLEX, **common)

        evaluation = index.evaluate(location)
        return QueryResult(
            QueryStatus.HIT if evaluation.is_hit else QueryStatus.MISS,
            is_hit=evaluation.is_hit,
            matched=self._entry(region, evaluation.matched_entry_id),
            vertex_distances=tuple(float(value) for value in evaluation.vertex_distances),
            interpolated_distance=evaluation.interpolated_distance,
            query_quantile=evaluation.query_quantile,
            matched_quantile=evaluation.matched_quantile,
            threshold=evaluation.threshold,
            margin=evaluation.margin,
            **common,
        )

    def fetch(
        self,
        kappa: float,
        gamma: float,
        s: float,
        *,
        generator: MapGenerator | None = None,
        allow_outside_domain: bool = False,
    ) -> FetchResult:
        """Return a matching map, creating one at (kappa, gamma, s) and adding it on a miss.

        Queries the point first. A hit returns the existing map and writes nothing. Points
        that :meth:`query` refuses (bad input, a critical line, outside the domain, a region
        that is not ready) are refused here too, and a point at an existing invalid entry
        gets that entry's recorded failure without a retry. Otherwise, on a miss, outside
        the region's mesh or in a tetrahedron with an invalid vertex, the region must be
        open for writing: the maps are made at the query point, with ``|gamma|`` for
        ``gamma``, and committed as a new entry with origin ``"fetch"`` before this method
        returns. Its MPD is computed on the region's frozen bin edges, and the next query
        sees it. A failed map is committed too, as an invalid entry, so the same point is
        not tried again.

        Parameters
        ----------
        kappa : float
            Total convergence of the macro model at the image.
        gamma : float
            Shear of the macro model at the image.
        s : float
            Smooth-matter fraction at the image.
        generator : MapGenerator or None, optional
            Generator that makes a new map. When ``None``, the generator named in the
            bank's configuration is used. It must match that configuration. It is used,
            and checked, only when a map has to be made.
        allow_outside_domain : bool, optional
            If ``True``, skip the domain check, as in :meth:`query`, so maps can also be
            made outside the bank's domain. For a negative ``gamma`` the map is made, and
            stored, at ``|gamma|``. Default is ``False``.

        Returns
        -------
        FetchResult
            The outcome, with the result of the first query under ``query``. Its
            ``status`` is ``HIT`` (``entry`` is the existing entry), ``CREATED`` (``entry``
            is the new entry and ``created`` is true), ``CREATION_FAILED`` (``error`` holds
            the failure), ``KNOWN_FAILURE`` (``error`` holds the failure recorded for the
            coincident entry, or ``None`` if none was recorded), or one of the refusals
            ``MALFORMED``, ``CRITICAL_LINE``, ``OUTSIDE_DOMAIN`` and ``REGION_NOT_READY``.
            ``entry`` is ``None`` for every status but ``HIT`` and ``CREATED``.

        Raises
        ------
        BankReadOnlyError
            If a map has to be made and the region is not open for writing.
        GeneratorMismatchError
            If a map has to be made and ``generator`` differs from the one recorded in the
            bank.
        ValueError
            If a map has to be made, ``generator`` is ``None`` and no generator is
            registered under the configured name, or if the generator returns a map of the
            wrong shape.

        See Also
        --------
        query : The same lookup, without writing.
        fetch_many : Fetch every row of a table.

        Notes
        -----
        A :exc:`MapGenerationError` from the generator, or an :exc:`InvalidMapError` for an
        unusable map, gives an invalid entry whose ``error`` is the exception's name and
        message, and the status ``CREATION_FAILED``. Any other exception propagates, and no
        entry is committed.
        """
        result = self.query(kappa, gamma, s, allow_outside_domain=allow_outside_domain)
        if result.status is QueryStatus.HIT:
            return FetchResult(FetchStatus.HIT, result.entry, False, result)
        if result.status in _REFUSALS:
            return FetchResult(_REFUSALS[result.status], None, False, result)
        region = result.region
        assert region is not None
        if result.coincident_entry_id is not None:
            error = str(self._regions[region].entries["error"].iloc[result.coincident_entry_id])
            return FetchResult(FetchStatus.KNOWN_FAILURE, None, False, result, error or None)
        self._require_writable(region)
        pending = self._evaluate_entry(
            region, (float(kappa), abs(float(gamma)), float(s)), self._resolve_generator(generator), "fetch"
        )
        self._commit(region, pending)
        if pending.valid:
            return FetchResult(
                FetchStatus.CREATED, self._entry(region, pending.row["entry_id"]), True, result
            )
        return FetchResult(FetchStatus.CREATION_FAILED, None, False, result, pending.row["error"])

    def query_many(self, table: pd.DataFrame, *, allow_outside_domain: bool = False) -> pd.DataFrame:
        """Query every row of a table with ``kappa``, ``gamma`` and ``s`` columns.

        Each row is passed to :meth:`query` on its own, so the results match single
        queries, and nothing is written. A row with a missing or non-finite parameter gets
        the status ``"malformed"``.

        Parameters
        ----------
        table : pandas.DataFrame
            Query points, with ``kappa``, ``gamma`` and ``s`` columns. Other columns are
            kept in the result.
        allow_outside_domain : bool, optional
            Passed to :meth:`query` for every row. Default is ``False``.

        Returns
        -------
        pandas.DataFrame
            A copy of ``table``, with the same index, followed by the result columns of
            ``results.QUERY_COLUMNS``. They use the names of the original query script, for
            example ``interpolation_status`` (the status value), ``is_hit``,
            ``matched_row`` and ``distance_margin``.

        Raises
        ------
        ValueError
            If ``table`` lacks a ``kappa``, ``gamma`` or ``s`` column, or already has one
            of the result columns.

        See Also
        --------
        adaptive_microlensing.results.hit_summary : Hit counts and rates of the returned table.
        """
        points = _parameters(table)
        check_result_columns(table, QUERY_COLUMNS)
        rows = [
            query_row(self.query(kappa, gamma, s, allow_outside_domain=allow_outside_domain))
            for kappa, gamma, s in points
        ]
        return results_table(table, rows, QUERY_COLUMNS)

    def fetch_many(
        self,
        table: pd.DataFrame,
        *,
        generator: MapGenerator | None = None,
        allow_outside_domain: bool = False,
    ) -> pd.DataFrame:
        """Fetch every row of a table in order; each new map is committed as soon as it is made.

        Rows are passed to :meth:`fetch` one at a time, so a later row can hit a map made
        for an earlier one. If a row raises, the entries made for earlier rows stay
        committed, but no table is returned.

        Parameters
        ----------
        table : pandas.DataFrame
            Query points, with ``kappa``, ``gamma`` and ``s`` columns. Other columns are
            kept in the result.
        generator : MapGenerator or None, optional
            Passed to :meth:`fetch` for every row. When ``None``, the generator named in
            the bank's configuration is used.
        allow_outside_domain : bool, optional
            Passed to :meth:`fetch` for every row. Default is ``False``.

        Returns
        -------
        pandas.DataFrame
            A copy of ``table``, with the same index, followed by the result columns of
            ``results.FETCH_COLUMNS``: the columns of :meth:`query_many`, which describe
            each row's first query, then ``fetch_status``, ``entry_id``, ``map_path``,
            ``created`` and ``fetch_error``.

        Raises
        ------
        ValueError
            If ``table`` lacks a ``kappa``, ``gamma`` or ``s`` column, or already has one
            of the result columns, or for a row as in :meth:`fetch`. The columns are
            checked before any row is fetched.
        BankReadOnlyError
            If a row needs a new map and its region is not open for writing.
        GeneratorMismatchError
            If a row needs a new map and ``generator`` differs from the one recorded in the
            bank.

        See Also
        --------
        adaptive_microlensing.results.hit_summary : Hit counts and rates of the returned table.
        """
        points = _parameters(table)
        # Refuse a table that already has results before any row makes and commits a map.
        check_result_columns(table, FETCH_COLUMNS)
        rows = [
            fetch_row(
                self.fetch(kappa, gamma, s, generator=generator, allow_outside_domain=allow_outside_domain)
            )
            for kappa, gamma, s in points
        ]
        return results_table(table, rows, FETCH_COLUMNS)

    # ------------------------------------------------------------------ internals

    def _region(self, region: str) -> _Region:
        """Return the in-memory state of a region, checking its name.

        Parameters
        ----------
        region : str
            Name of the region, one of ``REGIONS``.

        Returns
        -------
        _Region
            The region's entries, MPDs, index and write state.

        Raises
        ------
        ValueError
            If ``region`` is not one of ``REGIONS``.
        """
        try:
            return self._regions[region]
        except KeyError:
            raise ValueError(f"Unknown region {region!r}; expected one of {REGIONS}.") from None

    def _require_writable(self, region: str) -> None:
        """Check that a region is open for writing.

        Parameters
        ----------
        region : str
            Name of the region, one of ``REGIONS``.

        Raises
        ------
        ValueError
            If ``region`` is not one of ``REGIONS``.
        BankReadOnlyError
            If the region was opened read-only or the bank has been closed.
        """
        if not self._region(region).writable:
            raise BankReadOnlyError(
                f"Region {region!r} is not open for writing; use MapBank.open(path, writable=True)."
            )

    def _resolve_generator(self, generator: MapGenerator | None) -> MapGenerator:
        """Return the generator for new maps, checked against the bank's configuration.

        A generator matches when its name and its options, normalised through JSON, equal
        those of the bank's ``GeneratorSpec``. If both the generator's version and the one
        recorded in ``bank.json`` are known and they differ, a warning is logged, once per
        bank object.

        Parameters
        ----------
        generator : MapGenerator or None
            Generator supplied by the caller. When ``None``, one is built from the bank's
            ``GeneratorSpec``.

        Returns
        -------
        MapGenerator
            The supplied or newly built generator.

        Raises
        ------
        GeneratorMismatchError
            If ``generator`` has a different name or options from the bank's
            ``GeneratorSpec``.
        ValueError
            If ``generator`` is ``None`` and no generator is registered under the spec's
            name.
        """
        spec = self._config.generator
        if generator is None:
            generator = generator_from_spec(spec)
        elif generator_identity(generator) != (spec.name, spec.options):
            raise GeneratorMismatchError(
                f"The bank's maps come from {spec.name!r} with options {spec.options}; "
                f"got {generator_identity(generator)}."
            )
        recorded = self._meta.get("versions", {}).get("generator")
        current = generator.version()
        if not self._version_warned and recorded and current and current != recorded:
            logger.warning(
                "Generator version %s differs from %s recorded when the bank was created.", current, recorded
            )
            self._version_warned = True
        return generator

    def _entry(self, region: str, entry_id: int) -> BankEntry:
        """Return the :class:`BankEntry` of a valid entry, with the absolute path of its map.

        The row's fields are copied without checking that the entry is valid, so callers
        pass only valid entries.

        Parameters
        ----------
        region : str
            Name of the region, one of ``REGIONS``.
        entry_id : int
            The entry's ID, which is also its row position in the entries table.

        Returns
        -------
        BankEntry
            The entry's parameters, intrinsic quantile, map path and ``BankConfig.bank_map``
            geometry, its seeds (``None`` where not recorded), origin and generator version
            (``None`` if not recorded).
        """
        state = self._regions[region]
        row = state.entries.iloc[entry_id]
        return BankEntry(
            region=region,
            entry_id=int(row["entry_id"]),
            kappa=float(row["kappa"]),
            gamma=float(row["gamma"]),
            s=float(row["s"]),
            intrinsic_quantile=float(row["intrinsic_quantile"]),
            map_path=(state.store.directory / str(row["map_file"])).absolute(),
            map_spec=self._config.bank_map,
            variability_seed=_optional_int(row["variability_seed"]),
            map_seed=_optional_int(row["map_seed"]),
            origin=str(row["origin"]),
            generator_version=str(row["generator_version"]) or None,
        )

    def _evaluate_entry(
        self, region: str, point: tuple[float, float, float], generator: MapGenerator, origin: str
    ) -> PendingEntry:
        """Make the maps of a new entry and write its bank map; the caller commits the row.

        The entry takes the next entry ID and the seeds from :func:`entry_seeds`. The coarse
        map is made first and used only to measure the intrinsic quantile. The bank map is
        made next, checked for its shape and for finite pixels, histogrammed on the frozen
        bin edges if the region is finalized, and written to ``maps/`` (commit step 1). If
        the generator raises :exc:`MapGenerationError`, or a map is rejected with
        :exc:`InvalidMapError`, no map is written and the row is marked invalid, with
        ``"<exception name>: <message>"`` in its ``error`` column.

        Parameters
        ----------
        region : str
            Name of the region, one of ``REGIONS``.
        point : tuple of (float, float, float)
            The entry's ``(kappa, gamma, s)``.
        generator : MapGenerator
            Generator that makes the coarse map and the bank map, already checked against
            the bank's configuration.
        origin : str
            What created the entry, recorded in its ``origin`` column, such as ``"build"``
            or ``"fetch"``.

        Returns
        -------
        PendingEntry
            The entry's row and MPD, ready for :meth:`_commit`.

        Raises
        ------
        ValueError
            If ``region`` is not one of ``REGIONS``, or if the generator returns a map of
            the wrong shape.
        BankReadOnlyError
            If the region is not open for writing.

        Notes
        -----
        Nothing is added to the region until :meth:`_commit` runs. Until then the bank map
        written here is ignored, and the next entry evaluated with the same ID overwrites
        it.
        """
        self._require_writable(region)
        state = self._regions[region]
        entry_id = len(state.entries)
        kappa, gamma, s = (float(value) for value in point)
        variability_seed, map_seed = entry_seeds(self._config.seed, region, entry_id)
        row: dict[str, Any] = {
            "entry_id": entry_id,
            "kappa": kappa,
            "gamma": gamma,
            "s": s,
            "valid": False,
            "error": "",
            "intrinsic_quantile": math.nan,
            "map_file": "",
            "mag_min": math.nan,
            "mag_max": math.nan,
            "variability_seed": variability_seed,
            "map_seed": map_seed,
            "generator_version": generator.version() or "",
            "origin": origin,
            "created_at": utc_now(),
            "simplex_loss": math.nan,
            "abs_residual": math.nan,
            "rel_residual": math.nan,
            "max_residual": math.nan,
        }
        mpd = np.full(self._config.n_mpd_columns, np.nan)
        try:
            coarse = generator.generate(kappa, gamma, s, self._config.variability.map, variability_seed)
            quantile = intrinsic_quantile(coarse, self._config.variability)
            del coarse
            bank_map = generator.generate(kappa, gamma, s, self._config.bank_map, map_seed)
            n = self._config.bank_map.num_pixels
            if bank_map.shape != (n, n):
                raise ValueError(
                    f"The generator returned a bank map of shape {bank_map.shape}; expected {(n, n)}."
                )
            mag_min, mag_max = finite_range(bank_map)
            if state.finalized:
                mpd = histogram_with_overflow(bank_map, state.bin_edges)
            row["map_file"] = state.store.write_map(entry_id, bank_map)
        except (MapGenerationError, InvalidMapError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.info("%s entry %d at %s is invalid: %s", region, entry_id, point, row["error"])
        else:
            row.update(valid=True, intrinsic_quantile=quantile, mag_min=mag_min, mag_max=mag_max)
        return PendingEntry(row=row, mpd=mpd)

    def _commit(
        self, region: str, pending: PendingEntry, diagnostics: Mapping[str, float] | None = None
    ) -> None:
        """Commit an evaluated entry: ``mpds.npy``, then ``entries.csv``, then the in-memory index.

        :meth:`_evaluate_entry` has already written the entry's bank map (commit step 1).
        This writes ``mpds.npy`` with the new MPD row appended (step 2), then
        ``entries.csv`` with the new row appended (step 3), each by an atomic replace. The
        entry exists on disk only once step 3 is done: after a crash before it, the extra
        map and MPD row are ignored when the bank is loaded and overwritten by the next
        entry. Last, the in-memory entries, MPDs and index are extended, so the next query
        sees the entry; in a finalized region only the new entry's JS distances are
        computed.

        Parameters
        ----------
        region : str
            Name of the region, one of ``REGIONS``.
        pending : PendingEntry
            The evaluated entry. Its ``entry_id`` must be the next one in the region.
        diagnostics : mapping of str to float or None, optional
            Column values that override those in the row, such as the build's
            ``simplex_loss`` and residuals. When ``None``, the row is committed as it is.

        Raises
        ------
        ValueError
            If ``region`` is not one of ``REGIONS``.
        BankReadOnlyError
            If the region is not open for writing.
        RuntimeError
            If another entry was committed after ``pending`` was evaluated, so its entry ID
            is stale.
        """
        self._require_writable(region)
        state = self._regions[region]
        row = {**pending.row, **(diagnostics or {})}
        if row["entry_id"] != len(state.entries):
            raise RuntimeError("The pending entry is stale: another entry was committed first.")
        entries = append_entry(state.entries, row)
        mpds = np.vstack([state.mpds, pending.mpd[np.newaxis, :]])
        state.store.commit(entries, mpds)
        state.entries = entries
        state.mpds = mpds
        state.index = state.index.extended(
            np.array([row["kappa"], row["gamma"], row["s"]], dtype=float),
            bool(row["valid"]),
            float(row["intrinsic_quantile"]),
            pending.mpd if state.finalized else None,
        )

    def _import_entries(self, region: str, entries: pd.DataFrame) -> None:
        """Commit a whole entries table into an empty region (used by the legacy import).

        The table is normalised to the standard columns, order and dtypes and written with
        an all-NaN MPD array, then the region's index is rebuilt. Nothing checks that the
        entries' map files exist, and the region is left unfinalized; the legacy import
        links the maps first and calls :meth:`_finalize_with_edges` afterwards.

        Parameters
        ----------
        region : str
            Name of the region, one of ``REGIONS``.
        entries : pandas.DataFrame
            The entries table, with every column of ``entries.csv`` and ``entry_id`` values
            0, 1, 2, ... in row order, as :meth:`open` requires.

        Raises
        ------
        ValueError
            If ``region`` is not one of ``REGIONS``.
        BankReadOnlyError
            If the region is not open for writing.
        RegionStateError
            If the region already has entries.
        BankCorruptError
            If ``entries`` lacks one of the standard columns.
        """
        state = self._region(region)
        self._require_writable(region)
        if len(state.entries):
            raise RegionStateError(f"Region {region!r} is not empty.")
        table = normalise_entries(entries)
        mpds = np.full((len(table), self._config.n_mpd_columns), np.nan)
        state.store.commit(table, mpds)
        state.entries = table
        state.mpds = mpds
        state.index = state.build_index()

    def _finalize_with_edges(self, region: str, edges: np.ndarray) -> None:
        """Compute every valid entry's MPD against ``edges`` and mark the region finalized.

        Each valid entry's bank map is read memory-mapped and histogrammed with underflow
        and overflow bins, with progress logged at ``INFO`` level. ``mpds.npy`` is written
        first and ``region.json`` second, so a crash in between leaves the region
        unfinalized. The in-memory state is then updated and the index rebuilt with the
        MPDs, which computes the JS distances between all pairs of valid entries. Unlike
        :meth:`finalize`, this does not check whether the region is already finalized; the
        legacy import uses it to give every region the same edges.

        Parameters
        ----------
        region : str
            Name of the region, one of ``REGIONS``.
        edges : numpy.ndarray
            The bin edges to freeze: finite, strictly increasing and at least two of them.

        Raises
        ------
        ValueError
            If ``region`` is not one of ``REGIONS``, or if the region has a valid entry and
            ``edges`` are not finite, strictly increasing and at least two.
        BankReadOnlyError
            If the region is not open for writing.
        InvalidMapError
            If a valid entry's bank map has no finite pixels.
        """
        state = self._region(region)
        self._require_writable(region)
        mpds = np.full((len(state.entries), self._config.n_mpd_columns), np.nan)
        rows = np.flatnonzero(state.entries["valid"].to_numpy(dtype=bool))
        for position, row in enumerate(rows, start=1):
            path = state.store.directory / str(state.entries["map_file"].iloc[row])
            mpds[row] = histogram_with_overflow(np.load(path, mmap_mode="r"), edges)
            if position == len(rows) or position % max(1, len(rows) // 10) == 0:
                logger.info("%s: MPDs %d/%d", region, position, len(rows))
        state.store.write_mpds(mpds)
        meta = {
            **state.meta,
            "finalized": True,
            "bin_edges": [float(edge) for edge in edges],
            "finalized_at": utc_now(),
        }
        state.store.write_meta(meta)
        state.mpds = mpds
        state.meta = meta
        state.index = state.build_index()


def _parameters(table: pd.DataFrame) -> np.ndarray:
    """Return the ``(kappa, gamma, s)`` columns of a query table as a float array.

    Parameters
    ----------
    table : pandas.DataFrame
        Query points, with ``kappa``, ``gamma`` and ``s`` columns.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(table), 3)`` of ``(kappa, gamma, s)`` points, in row order.

    Raises
    ------
    ValueError
        If ``table`` lacks any of the ``kappa``, ``gamma`` and ``s`` columns.
    """
    missing = sorted({"kappa", "gamma", "s"}.difference(table.columns))
    if missing:
        raise ValueError(f"The table is missing columns {missing}.")
    return table.loc[:, ["kappa", "gamma", "s"]].to_numpy(dtype=float)
