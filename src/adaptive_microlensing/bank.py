"""The MapBank: build, load, query and update banks of microlensing maps."""

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
from .config import BankConfig
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
    QUERY_COLUMNS,
    BankEntry,
    QueryResult,
    QueryStatus,
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


def entry_seeds(seed: int, region: str, entry_id: int) -> tuple[int, int]:
    """Seeds of an entry's coarse map and bank map, derived only from (seed, region, entry ID).

    Both lie in [1, 2**31 - 1]: IPM takes a C ``int`` and treats 0 as "pick a random seed".
    """
    state = np.random.SeedSequence(seed, spawn_key=(REGIONS.index(region), entry_id)).generate_state(
        2, dtype=np.uint32
    )
    variability_seed, map_seed = (int(value) % (2**31 - 1) + 1 for value in state)
    return variability_seed, map_seed


def _versions(config: BankConfig) -> dict[str, str | None]:
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
    return None if pd.isna(value) else int(value)


@dataclass
class PendingEntry:
    """An evaluated entry whose row has not been committed yet."""

    row: dict[str, Any]
    mpd: np.ndarray

    @property
    def valid(self) -> bool:
        """Whether the evaluation produced a usable entry."""
        return bool(self.row["valid"])


class _Region:
    """In-memory state of one region."""

    def __init__(self, store: RegionStore, snapshot: RegionSnapshot, writable: bool) -> None:
        self.store = store
        self.meta = snapshot.meta
        self.entries = snapshot.entries
        self.mpds = snapshot.mpds
        self.writable = writable
        self.index = self.build_index()

    @property
    def finalized(self) -> bool:
        return bool(self.meta["finalized"])

    @property
    def bin_edges(self) -> np.ndarray:
        if self.meta["bin_edges"] is None:
            raise RegionStateError(f"Region {self.store.region!r} is not finalized.")
        return np.asarray(self.meta["bin_edges"], dtype=float)

    def build_index(self) -> RegionIndex:
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
    """

    def __init__(
        self,
        path: Path,
        meta: dict[str, Any],
        config: BankConfig,
        regions: dict[str, _Region],
        locks: list[RegionLock],
    ) -> None:
        self._path = path
        self._meta = meta
        self._config = config
        self._regions = regions
        self._locks = locks
        self._version_warned = False

    # ----------------------------------------------------------------- lifecycle

    @classmethod
    def create(cls, path: str | Path, config: BankConfig | None = None) -> MapBank:
        """Create a new, empty bank directory and return it opened for writing."""
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

        Parameters
        ----------
        path : str or Path
            The bank directory.
        writable : bool or iterable of str, optional
            ``False`` (read-only), ``True`` (lock every region), or the names of the
            regions to lock for writing.
        """
        root = Path(path)
        meta = read_json(root / BANK_FILE)
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise BankCorruptError(f"{root / BANK_FILE} has an unknown schema version.")
        config = BankConfig.from_dict(meta["config"])
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
        """Release every region lock; the bank stays readable but can no longer be written."""
        for lock in self._locks:
            lock.release()
        self._locks = []
        for state in self._regions.values():
            state.writable = False

    def __enter__(self) -> MapBank:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ---------------------------------------------------------------- inspection

    @property
    def path(self) -> Path:
        """The bank directory."""
        return self._path

    @property
    def config(self) -> BankConfig:
        """The configuration stored in ``bank.json``."""
        return self._config

    def entries(self, region: str) -> pd.DataFrame:
        """A copy of a region's entries table."""
        return self._region(region).entries.copy()

    def summary(self) -> pd.DataFrame:
        """One row per region: numbers of entries, valid and invalid entries, and finalized state."""
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

    def finalize(self, region: str) -> None:
        """Freeze the region's bank-MPD bin edges and compute the MPDs of its maps."""
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
            self._config.n_bin_edges,
        )
        self._finalize_with_edges(region, edges)

    # ------------------------------------------------------------- query and fetch

    def query(
        self, kappa: float, gamma: float, s: float, *, allow_outside_domain: bool = False
    ) -> QueryResult:
        """Look for an existing map indistinguishable from one at (kappa, gamma, s). Never writes."""
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

        state = self._regions[region]
        index = state.index
        common: dict[str, Any] = {
            "region": region,
            "is_in_bounds": box,
            "region_entry_count": len(state.entries),
        }
        if not state.finalized or not index.ready:
            return QueryResult(QueryStatus.REGION_NOT_READY, **common)

        coincident = index.coincident(point)
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

        location = index.locate(point)
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

    def query_many(self, table: pd.DataFrame, *, allow_outside_domain: bool = False) -> pd.DataFrame:
        """Query every row of a table with ``kappa``, ``gamma`` and ``s`` columns."""
        rows = [
            query_row(self.query(kappa, gamma, s, allow_outside_domain=allow_outside_domain))
            for kappa, gamma, s in _parameters(table)
        ]
        return results_table(table, rows, QUERY_COLUMNS)

    # ------------------------------------------------------------------ internals

    def _region(self, region: str) -> _Region:
        try:
            return self._regions[region]
        except KeyError:
            raise ValueError(f"Unknown region {region!r}; expected one of {REGIONS}.") from None

    def _require_writable(self, region: str) -> None:
        if not self._region(region).writable:
            raise BankReadOnlyError(
                f"Region {region!r} is not open for writing; use MapBank.open(path, writable=True)."
            )

    def _resolve_generator(self, generator: MapGenerator | None) -> MapGenerator:
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
        """Make the maps of a new entry and write its bank map; the caller commits the row."""
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
        """Commit an evaluated entry: ``mpds.npy``, then ``entries.csv``, then the in-memory index."""
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
        """Commit a whole entries table into an empty region (used by the legacy import)."""
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
        """Compute every valid entry's MPD against ``edges`` and mark the region finalized."""
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
    missing = sorted({"kappa", "gamma", "s"}.difference(table.columns))
    if missing:
        raise ValueError(f"The table is missing columns {missing}.")
    return table.loc[:, ["kappa", "gamma", "s"]].to_numpy(dtype=float)
