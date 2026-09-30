"""Result types returned by MapBank, their table form, and hit summaries."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MapSpec
from .lensing import REGIONS


class QueryStatus(StrEnum):
    """Outcome of :meth:`MapBank.query`."""

    MALFORMED = "malformed"
    CRITICAL_LINE = "critical_line"
    OUTSIDE_DOMAIN = "outside_domain"
    REGION_NOT_READY = "region_not_ready"
    OUTSIDE_HULL = "outside_hull"
    INVALID_SIMPLEX = "invalid_simplex"
    HIT = "hit"
    MISS = "miss"


class FetchStatus(StrEnum):
    """Outcome of :meth:`MapBank.fetch`."""

    HIT = "hit"
    CREATED = "created"
    CREATION_FAILED = "creation_failed"
    KNOWN_FAILURE = "known_failure"
    MALFORMED = "malformed"
    CRITICAL_LINE = "critical_line"
    OUTSIDE_DOMAIN = "outside_domain"
    REGION_NOT_READY = "region_not_ready"


@dataclass(frozen=True)
class BankEntry:
    """One valid map in a bank."""

    region: str
    entry_id: int
    kappa: float
    gamma: float
    s: float
    intrinsic_quantile: float
    map_path: Path
    map_spec: MapSpec
    variability_seed: int | None
    map_seed: int | None
    origin: str
    generator_version: str | None

    def load(self, mmap: bool = True) -> np.ndarray:
        """Load the map; memory-mapped read-only by default."""
        return np.load(self.map_path, mmap_mode="r" if mmap else None)


@dataclass(frozen=True)
class QueryResult:
    """Outcome of one query.

    ``matched`` is the vertex the hit rule settled on, for hits and misses alike: the
    passing vertex nearest in MPD distance on a hit, the vertex closest to passing on a
    miss. ``entry`` is that entry only on a hit.
    """

    status: QueryStatus
    region: str | None = None
    is_hit: bool = False
    matched: BankEntry | None = None
    is_in_bounds: bool = False
    simplex_index: int | None = None
    simplex_entry_ids: tuple[int, ...] | None = None
    barycentric_weights: tuple[float, ...] | None = None
    vertex_distances: tuple[float, ...] | None = None
    interpolated_distance: float = math.nan
    query_quantile: float = math.nan
    matched_quantile: float = math.nan
    threshold: float = math.nan
    margin: float = math.nan
    region_entry_count: int | None = None
    coincident_entry_id: int | None = None

    @property
    def entry(self) -> BankEntry | None:
        """The matched entry on a hit, otherwise None."""
        return self.matched if self.is_hit else None


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one fetch."""

    status: FetchStatus
    entry: BankEntry | None
    created: bool
    query: QueryResult
    error: str | None = None


@dataclass(frozen=True)
class BuildSummary:
    """Outcome of :meth:`MapBank.build` for one region."""

    region: str
    n_entries: int
    n_valid: int
    valid_fraction: float
    stop_reason: str
    final_simplex_loss: float
    elapsed_seconds: float


QUERY_COLUMNS = (
    "is_in_bounds",
    "query_region",
    "interpolation_status",
    "is_hit",
    "hit_region",
    "simplex_index",
    "simplex_rows",
    "barycentric_weights",
    "vertex_distances",
    "matched_row",
    "matched_kappa",
    "matched_gamma",
    "matched_s",
    "interpolated_mpd_distance",
    "query_mpd_quantile",
    "matched_mpd_quantile",
    "distance_threshold",
    "distance_margin",
    "region_entry_count",
)
FETCH_COLUMNS = (*QUERY_COLUMNS, "fetch_status", "entry_id", "map_path", "created", "fetch_error")
_INTEGER_COLUMNS = ("simplex_index", "matched_row", "region_entry_count", "entry_id")
_BOOLEAN_COLUMNS = ("is_in_bounds", "is_hit", "created")


def _json_list(values: Sequence[float] | None) -> str | None:
    return None if values is None else json.dumps(list(values))


def query_row(result: QueryResult) -> dict[str, Any]:
    """Table row for a query result, using the column names of the original query script."""
    matched = result.matched
    return {
        "is_in_bounds": result.is_in_bounds,
        "query_region": result.region,
        "interpolation_status": result.status.value,
        "is_hit": result.is_hit,
        "hit_region": result.region if result.is_hit else None,
        "simplex_index": result.simplex_index,
        "simplex_rows": _json_list(result.simplex_entry_ids),
        "barycentric_weights": _json_list(result.barycentric_weights),
        "vertex_distances": _json_list(result.vertex_distances),
        "matched_row": None if matched is None else matched.entry_id,
        "matched_kappa": math.nan if matched is None else matched.kappa,
        "matched_gamma": math.nan if matched is None else matched.gamma,
        "matched_s": math.nan if matched is None else matched.s,
        "interpolated_mpd_distance": result.interpolated_distance,
        "query_mpd_quantile": result.query_quantile,
        "matched_mpd_quantile": result.matched_quantile,
        "distance_threshold": result.threshold,
        "distance_margin": result.margin,
        "region_entry_count": result.region_entry_count,
    }


def fetch_row(result: FetchResult) -> dict[str, Any]:
    """Table row for a fetch result: the query columns plus what was returned."""
    entry = result.entry
    return {
        **query_row(result.query),
        "fetch_status": result.status.value,
        "entry_id": None if entry is None else entry.entry_id,
        "map_path": None if entry is None else str(entry.map_path),
        "created": result.created,
        "fetch_error": result.error,
    }


def results_table(table: pd.DataFrame, rows: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    """``table`` with one result row per input row appended as columns."""
    overlap = sorted(set(columns) & set(table.columns))
    if overlap:
        raise ValueError(
            f"The table already has result columns {overlap}; drop them before running it again."
        )
    diagnostics = pd.DataFrame.from_records(rows, index=table.index, columns=list(columns))
    for column in _INTEGER_COLUMNS:
        if column in diagnostics:
            diagnostics[column] = diagnostics[column].astype("Int64")
    for column in _BOOLEAN_COLUMNS:
        if column in diagnostics:
            diagnostics[column] = diagnostics[column].astype(bool)
    return pd.concat([table.copy(), diagnostics], axis=1)


def hit_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Hit counts and rates per region and overall, from ``query_many`` or ``fetch_many`` output."""
    covered = table["interpolation_status"].isin([QueryStatus.HIT.value, QueryStatus.MISS.value])
    hits = table["is_hit"].astype(bool)
    masks = [(region, table["query_region"].eq(region)) for region in REGIONS]
    masks.append(("all", pd.Series(True, index=table.index)))
    records = []
    for name, mask in masks:
        queries = int(mask.sum())
        n_covered = int((mask & covered).sum())
        n_hits = int((mask & hits).sum())
        records.append(
            {
                "region": name,
                "queries": queries,
                "covered": n_covered,
                "hits": n_hits,
                "hit_rate": n_hits / queries if queries else math.nan,
                "hit_rate_covered": n_hits / n_covered if n_covered else math.nan,
            }
        )
    return pd.DataFrame.from_records(records).set_index("region")
