"""Result types returned by MapBank, their table form, and hit summaries.

:meth:`MapBank.query` returns a :class:`QueryResult`, whose :class:`QueryStatus` says how
far the query got: refused (malformed, on a critical line or outside the domain), not
answerable (region not ready, outside the mesh, or in a tetrahedron with an invalid
vertex) or answered by the hit rule (hit or miss). :meth:`MapBank.fetch` wraps its query
in a :class:`FetchResult`, whose :class:`FetchStatus` also says whether a map was made.
Both refer to bank maps through :class:`BankEntry`, and :meth:`MapBank.build` reports on
a region with a :class:`BuildSummary`.

For many points at once, :meth:`MapBank.query_many` and :meth:`MapBank.fetch_many` turn
each result into a row with :func:`query_row` or :func:`fetch_row` and append the rows to
the input table with :func:`results_table`. The columns, :data:`QUERY_COLUMNS` and
:data:`FETCH_COLUMNS`, carry the names of the original query script. :func:`hit_summary`
counts hits per region in such a table, and :mod:`adaptive_microlensing.plotting` and
:mod:`adaptive_microlensing.slicing` read the same columns.

Apart from :meth:`BankEntry.load`, which reads one map file, nothing here touches a bank
on disk.
"""

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
    """Outcome of :meth:`MapBank.query`.

    ``query`` checks the point step by step and stops at the first check that fails; the
    status names that check. Only ``HIT`` and ``MISS`` mean that the hit rule was applied.
    Each member is a string equal to its value, which is what the ``interpolation_status``
    column of :meth:`MapBank.query_many` holds.

    Attributes
    ----------
    MALFORMED : str
        The coordinates cannot be read as three floats, or one of them is not finite;
        ``"malformed"``. No other field of the result is set.
    CRITICAL_LINE : str
        The point lies on a critical line, where ``|1 - kappa|`` equals ``|gamma|`` within
        ``1e-10`` and the macro magnification diverges; ``"critical_line"``. This is checked
        before the domain, so it is returned even with ``allow_outside_domain=True``.
    OUTSIDE_DOMAIN : str
        The point lies outside the domain box or above its ``|mu_macro|`` cap, and
        ``allow_outside_domain`` is ``False``; ``"outside_domain"``.
    REGION_NOT_READY : str
        The point's region is not finalized, or it has no mesh yet because its entries are
        fewer than four or do not span three dimensions; ``"region_not_ready"``.
    OUTSIDE_HULL : str
        No tetrahedron of the region's mesh contains the point; ``"outside_hull"``.
    INVALID_SIMPLEX : str
        The tetrahedron that contains the point has an invalid vertex, or the point
        coincides with an invalid entry, whose ID is then in ``coincident_entry_id``;
        ``"invalid_simplex"``.
    HIT : str
        A vertex of the fully valid tetrahedron that contains the point passes the hit rule,
        or the point coincides with a valid entry; ``"hit"``.
    MISS : str
        The tetrahedron that contains the point is fully valid, but none of its vertices
        passes the hit rule; ``"miss"``.
    """

    MALFORMED = "malformed"
    CRITICAL_LINE = "critical_line"
    OUTSIDE_DOMAIN = "outside_domain"
    REGION_NOT_READY = "region_not_ready"
    OUTSIDE_HULL = "outside_hull"
    INVALID_SIMPLEX = "invalid_simplex"
    HIT = "hit"
    MISS = "miss"


class FetchStatus(StrEnum):
    """Outcome of :meth:`MapBank.fetch`.

    ``fetch`` first runs :meth:`MapBank.query`. A hit is returned as it is, and a refused
    query is passed on under the same name. After a miss, a point outside the hull or a
    tetrahedron with an invalid vertex, a map is made at the query point and committed as a
    new entry, unless the point coincides with an entry that already failed; if the region
    is not open for writing, :exc:`BankReadOnlyError` is raised instead. Each member is a
    string equal to its value, which is what the ``fetch_status`` column of
    :meth:`MapBank.fetch_many` holds.

    Attributes
    ----------
    HIT : str
        The query was a hit; ``entry`` is the matched entry and nothing is written;
        ``"hit"``.
    CREATED : str
        A new map was made at the query point and committed as a valid entry, which is
        ``entry``; ``"created"``.
    CREATION_FAILED : str
        Making the new map failed. The failure is still committed as an invalid entry, so a
        later fetch at the same point returns ``KNOWN_FAILURE``, and ``error`` holds the
        message; ``"creation_failed"``.
    KNOWN_FAILURE : str
        The query point coincides with an existing invalid entry, so no map is made;
        ``error`` holds that entry's recorded error; ``"known_failure"``.
    MALFORMED : str
        The query was :attr:`QueryStatus.MALFORMED`; ``"malformed"``.
    CRITICAL_LINE : str
        The query was :attr:`QueryStatus.CRITICAL_LINE`; ``"critical_line"``.
    OUTSIDE_DOMAIN : str
        The query was :attr:`QueryStatus.OUTSIDE_DOMAIN`; ``"outside_domain"``.
    REGION_NOT_READY : str
        The query was :attr:`QueryStatus.REGION_NOT_READY`, so no map is made in a region
        that is not finalized or has no mesh; ``"region_not_ready"``.
    """

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
    """One valid map in a bank.

    The results of :meth:`MapBank.query` and :meth:`MapBank.fetch` hold entries, which point
    at maps without loading them; :meth:`load` reads the map. The fields come from the
    entry's row in the region's ``entries.csv``.

    Parameters
    ----------
    region : str
        Region the entry belongs to, one of ``REGIONS``.
    entry_id : int
        Position of the entry in its region, counting from 0. It is the entry's row in
        ``entries.csv`` and ``mpds.npy``, and its vertex index in the region's mesh.
    kappa : float
        Total convergence of the entry.
    gamma : float
        Shear of the entry.
    s : float
        Smooth-matter fraction of the entry.
    intrinsic_quantile : float
        The entry's noise level in the hit rule: a quantile of the Jensen-Shannon distances
        between windows of its coarse map. For a legacy entry it is the original
        ``mpd_distance``.
    map_path : pathlib.Path
        Absolute path of the bank map file, ``maps/map_XXXXXX.npy`` in the region directory.
        For a legacy entry it is a symbolic link to the original map.
    map_spec : MapSpec
        Geometry of the bank map, the bank's :attr:`BankConfig.bank_map`.
    variability_seed : int or None
        Seed of the coarse map, or ``None`` when it was not recorded, as for legacy entries.
    map_seed : int or None
        Seed of the bank map, or ``None`` when it was not recorded.
    origin : str
        How the entry was made: ``"build"`` (:meth:`MapBank.build`), ``"fetch"``
        (:meth:`MapBank.fetch`) or ``"legacy"`` (:func:`import_legacy_bank`).
    generator_version : str or None
        Version of the map code that made the entry, or ``None`` when none was recorded, as
        for legacy entries and generators without a version.
    """

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
        """Load the map; memory-mapped read-only by default.

        Memory-mapping reads pixels from disk only when they are used, which suits large
        bank maps.

        Parameters
        ----------
        mmap : bool, optional
            ``True`` to memory-map the file read-only, ``False`` to read the whole map into
            memory. Default is ``True``.

        Returns
        -------
        numpy.ndarray
            Square array of magnitudes relative to the macro magnification,
            ``-2.5 log10(|mu_pixel| / |mu_macro|)``; pixels with zero magnification may be
            ``+inf``. Maps made by this package have shape
            ``(map_spec.num_pixels, map_spec.num_pixels)``. With ``mmap``, it is a read-only
            ``numpy.memmap``.

        Raises
        ------
        FileNotFoundError
            If the map file does not exist, as for a legacy map whose original has moved.
        """
        return np.load(self.map_path, mmap_mode="r" if mmap else None)


@dataclass(frozen=True)
class QueryResult:
    """Outcome of one query.

    ``matched`` is the vertex the hit rule settled on, for hits and misses alike: the
    passing vertex with the smallest interpolated distance on a hit, and the vertex closest
    to passing on a miss. ``entry`` is that entry only on a hit. Which other fields are set
    depends on how far the query got (see :class:`QueryStatus`); a field that does not
    apply keeps its default of ``None``, ``False`` or NaN.

    Parameters
    ----------
    status : QueryStatus
        How the query ended.
    region : str or None, optional
        Region that the point's image type falls in. Set for every status except
        ``MALFORMED`` and ``CRITICAL_LINE``. Default is ``None``.
    is_hit : bool, optional
        Whether the query is a hit, ``True`` exactly when ``status`` is ``HIT``. Default is
        ``False``.
    matched : BankEntry or None, optional
        On a hit or a miss, the entry the hit rule settled on, or the entry at the query
        point on a hit at an entry's own point. ``None`` for every other status. Default is
        ``None``.
    is_in_bounds : bool, optional
        Whether the point lies in the domain's closed kappa/gamma/s box, ignoring the
        ``|mu_macro|`` cap. Set for every status except ``MALFORMED``. Default is ``False``.
    simplex_index : int or None, optional
        Index of the tetrahedron that contains the point in the region's Delaunay mesh,
        which changes as entries are added. Set on ``HIT``, ``MISS`` and ``INVALID_SIMPLEX``
        when the point was located in the mesh, and ``None`` when it coincides with an
        entry. Default is ``None``.
    simplex_entry_ids : tuple of int or None, optional
        Entry IDs of the tetrahedron's four vertices, in the mesh's order. Set together
        with ``simplex_index``. Default is ``None``.
    barycentric_weights : tuple of float or None, optional
        Barycentric weights of the point in the tetrahedron, one per vertex in the order of
        ``simplex_entry_ids``, non-negative and summing to 1. Set together with
        ``simplex_index``. Default is ``None``.
    vertex_distances : tuple of float or None, optional
        Interpolated JS distance from the point to each vertex, in the order of
        ``simplex_entry_ids``. Set only when the hit rule ran in a tetrahedron: on ``MISS``,
        and on ``HIT`` unless the point coincides with an entry. Default is ``None``.
    interpolated_distance : float, optional
        Interpolated JS distance from the point to ``matched``, and ``0.0`` on a hit at an
        entry's own point. NaN unless ``status`` is ``HIT`` or ``MISS``. Default is NaN.
    query_quantile : float, optional
        Intrinsic quantile at the point, interpolated with the barycentric weights from the
        vertices' quantiles; on a hit at an entry's own point, that entry's quantile. NaN
        unless ``status`` is ``HIT`` or ``MISS``. Default is NaN.
    matched_quantile : float, optional
        Intrinsic quantile of ``matched``. NaN unless ``status`` is ``HIT`` or ``MISS``.
        Default is NaN.
    threshold : float, optional
        Largest distance at which ``matched`` passes, ``max(query_quantile,
        matched_quantile)``. NaN unless ``status`` is ``HIT`` or ``MISS``. Default is NaN.
    margin : float, optional
        ``threshold - interpolated_distance``: at least 0 on a hit and negative on a miss.
        On a hit at an entry's own point it equals that entry's quantile. NaN unless
        ``status`` is ``HIT`` or ``MISS``. Default is NaN.
    region_entry_count : int or None, optional
        Number of entries, valid and invalid, in the region when it was queried. Set for
        every status except ``MALFORMED``, ``CRITICAL_LINE`` and ``OUTSIDE_DOMAIN``.
        Default is ``None``.
    coincident_entry_id : int or None, optional
        ID of the entry at the query point (every coordinate within ``1e-12``): a valid one
        on ``HIT``, an invalid one on ``INVALID_SIMPLEX``. ``None`` when the point coincides
        with no entry. Default is ``None``.

    Notes
    -----
    In a tetrahedron with barycentric weights ``w``, the interpolated distance from the
    point to vertex ``i`` is ``sum_j w_j d(v_j, v_i)``, where ``d`` is the JS distance between
    the vertices' bank MPDs. Vertex ``i`` passes when that distance is at most
    ``max(query_quantile, q_i)``, with ``q_i`` its intrinsic quantile. Every vertex is tested;
    on a miss the vertex closest to passing is the one with the largest
    ``threshold - distance``.
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
        """The matched entry on a hit, otherwise ``None``.

        This is the entry whose map can stand in for one at the query point; on a miss,
        ``matched`` is still set, for diagnostics.
        """
        return self.matched if self.is_hit else None


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one fetch.

    ``query`` describes the bank as it was before the fetch, so after ``CREATED`` it still
    shows the miss, or other status, that led to the new map.

    Parameters
    ----------
    status : FetchStatus
        How the fetch ended.
    entry : BankEntry or None
        Entry to use: the matched entry on ``HIT``, the new entry on ``CREATED``, and
        ``None`` for every other status.
    created : bool
        Whether a new valid entry was made, ``True`` only on ``CREATED``. A fetch that ends
        in ``CREATION_FAILED`` also commits an entry, an invalid one, but ``created`` is
        ``False``.
    query : QueryResult
        Result of the query that the fetch ran first, before any new entry was committed.
    error : str or None, optional
        On ``CREATION_FAILED``, the failure as ``"ExceptionName: message"``; on
        ``KNOWN_FAILURE``, the error recorded with the coincident invalid entry, or ``None``
        if none was recorded. ``None`` for every other status. Default is ``None``.
    """

    status: FetchStatus
    entry: BankEntry | None
    created: bool
    query: QueryResult
    error: str | None = None


@dataclass(frozen=True)
class BuildSummary:
    """Outcome of :meth:`MapBank.build` for one region.

    The counts describe the whole region when the build stopped, including entries made
    before this call or by :meth:`MapBank.fetch`, not only the entries this call added.

    Parameters
    ----------
    region : str
        Region that was built.
    n_entries : int
        Number of entries, valid and invalid, in the region.
    n_valid : int
        Number of valid entries in the region.
    valid_fraction : float
        ``n_valid / n_entries``, or NaN if the region has no entries.
    stop_reason : str
        The :class:`StoppingCriteria` criterion that stopped the build:
        ``"max_valid_points"``, ``"simplex_loss_goal"`` or ``"residual_goal"``. When several
        hold, the first in that order.
    final_simplex_loss : float
        Largest simplex loss of the learner when the build stopped, or ``inf`` if the learner
        had no simplices.
    elapsed_seconds : float
        Wall-clock time of this call's sampling loop, in seconds, measured from after the
        region's existing entries were replayed into the learner.
    """

    region: str
    n_entries: int
    n_valid: int
    valid_fraction: float
    stop_reason: str
    final_simplex_loss: float
    elapsed_seconds: float


#: Result columns that :meth:`MapBank.query_many` appends, in order; see :func:`query_row`.
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
#: Result columns that :meth:`MapBank.fetch_many` appends: :data:`QUERY_COLUMNS`, then what
#: each fetch returned; see :func:`fetch_row`.
FETCH_COLUMNS = (*QUERY_COLUMNS, "fetch_status", "entry_id", "map_path", "created", "fetch_error")
_INTEGER_COLUMNS = ("simplex_index", "matched_row", "region_entry_count", "entry_id")
_BOOLEAN_COLUMNS = ("is_in_bounds", "is_hit", "created")


def _json_list(values: Sequence[float] | None) -> str | None:
    """Return ``values`` as a JSON array, or ``None`` when ``values`` is ``None``.

    Parameters
    ----------
    values : sequence of float or None
        Numbers to encode, such as entry IDs or barycentric weights.

    Returns
    -------
    str or None
        JSON text such as ``"[3, 2, 1, 0]"``, or ``None``.
    """
    return None if values is None else json.dumps(list(values))


def query_row(result: QueryResult) -> dict[str, Any]:
    """Return the table row of a query result, using the column names of the original query script.

    :meth:`MapBank.query_many` makes one such row per query and appends the rows to its
    table with :func:`results_table`. Unset fields become ``None`` or NaN, and tuples become
    JSON arrays so that each value fits in a CSV cell. ``coincident_entry_id`` has no column.

    Parameters
    ----------
    result : QueryResult
        Result of :meth:`MapBank.query`.

    Returns
    -------
    dict of str to Any
        Values keyed by the names in :data:`QUERY_COLUMNS`, in that order:

        - ``is_in_bounds`` (bool): :attr:`QueryResult.is_in_bounds`.
        - ``query_region`` (str or None): :attr:`QueryResult.region`.
        - ``interpolation_status`` (str): value of :attr:`QueryResult.status`, such as
          ``"hit"``.
        - ``is_hit`` (bool): :attr:`QueryResult.is_hit`.
        - ``hit_region`` (str or None): the region on a hit, otherwise ``None``.
        - ``simplex_index`` (int or None): :attr:`QueryResult.simplex_index`.
        - ``simplex_rows``, ``barycentric_weights`` and ``vertex_distances`` (str or None):
          ``simplex_entry_ids``, ``barycentric_weights`` and ``vertex_distances`` as JSON
          arrays.
        - ``matched_row`` (int or None): entry ID of ``matched``.
        - ``matched_kappa``, ``matched_gamma`` and ``matched_s`` (float): coordinates of
          ``matched``, NaN without one.
        - ``interpolated_mpd_distance``, ``query_mpd_quantile``, ``matched_mpd_quantile``,
          ``distance_threshold`` and ``distance_margin`` (float): ``interpolated_distance``,
          ``query_quantile``, ``matched_quantile``, ``threshold`` and ``margin``.
        - ``region_entry_count`` (int or None): :attr:`QueryResult.region_entry_count`.
    """
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
    """Return the table row of a fetch result: the query columns plus what was returned.

    :meth:`MapBank.fetch_many` makes one such row per fetch. The query columns describe the
    query that the fetch ran first, before any new entry was committed, so a created map
    appears there as a miss (or other status) and its own ID is in ``entry_id``.

    Parameters
    ----------
    result : FetchResult
        Result of :meth:`MapBank.fetch`.

    Returns
    -------
    dict of str to Any
        Values keyed by the names in :data:`FETCH_COLUMNS`, in that order: the
        :func:`query_row` of ``result.query``, followed by

        - ``fetch_status`` (str): value of :attr:`FetchResult.status`, such as
          ``"created"``.
        - ``entry_id`` (int or None): ID of the returned entry, ``None`` without one.
        - ``map_path`` (str or None): absolute path of the returned entry's map file.
        - ``created`` (bool): :attr:`FetchResult.created`.
        - ``fetch_error`` (str or None): :attr:`FetchResult.error`.
    """
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
    """Return ``table`` with one result row per input row appended as columns.

    The rows are matched to ``table`` by position and take its index. Integer columns
    become pandas' nullable ``Int64``, so a missing ID is ``<NA>``, and flag columns become
    ``bool``. ``table`` itself is not modified.

    Parameters
    ----------
    table : pandas.DataFrame
        Input table, one row per query.
    rows : list of dict of str to Any
        One result row per row of ``table``, in the same order, as made by
        :func:`query_row` or :func:`fetch_row`.
    columns : sequence of str
        Names and order of the result columns, :data:`QUERY_COLUMNS` or
        :data:`FETCH_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        A copy of ``table`` followed by the result columns. ``simplex_index``,
        ``matched_row``, ``region_entry_count`` and ``entry_id`` have dtype ``Int64``;
        ``is_in_bounds``, ``is_hit`` and ``created`` have dtype ``bool``; the matched
        coordinates, distances, quantiles, threshold and margin are ``float64``, NaN where
        unset. The text columns keep the dtype that pandas infers (``object``, or ``str``
        from pandas 3), with missing values where unset.

    Raises
    ------
    ValueError
        If ``table`` already has one of ``columns``, as when a result table is passed in
        again.
    """
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
    """Return hit counts and rates per region and overall, from ``query_many`` or ``fetch_many`` output.

    A query is covered when the hit rule ran on it, that is when its
    ``interpolation_status`` is ``"hit"`` or ``"miss"``. For ``fetch_many`` output the
    counts describe the queries before any map was made, so a created map counts as a
    miss. ``is_hit`` is cast with ``astype(bool)``, so a NaN flag, as a left merge leaves on
    rows that were never queried, would count as a hit; the flags in tables straight from
    ``query_many`` and ``fetch_many`` are never missing.

    Parameters
    ----------
    table : pandas.DataFrame
        Table with the ``query_region``, ``interpolation_status`` and ``is_hit`` columns of
        :meth:`MapBank.query_many` or :meth:`MapBank.fetch_many`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``region``: one row per region in ``REGIONS``, in that order, then
        ``"all"`` for every row of ``table``, including rows with no region such as
        malformed or critical-line queries. Columns:

        - ``queries`` (int64): number of rows.
        - ``covered`` (int64): number of covered rows.
        - ``hits`` (int64): number of rows whose ``is_hit`` is true.
        - ``hit_rate`` (float64): ``hits / queries``, NaN when there are no queries.
        - ``hit_rate_covered`` (float64): ``hits / covered``, NaN when no query is covered.

    Raises
    ------
    KeyError
        If ``table`` lacks one of the three columns it reads.
    """
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
