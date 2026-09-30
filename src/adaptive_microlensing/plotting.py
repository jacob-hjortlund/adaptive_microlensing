"""Figures of a bank: mesh slices, entries, coverage, queries and build convergence.

This module needs the ``plot`` extra (matplotlib and seaborn). The package does not
import it, so it is imported explicitly::

    from adaptive_microlensing import plotting

Importing it without matplotlib or seaborn raises an :exc:`ImportError` that names the
extra.

The figures read a :class:`MapBank`, the geometry that :mod:`adaptive_microlensing.slicing`
computes from it, or the tables it returns. None of them writes to the bank.

- :func:`plot_slice` shades the interpolated intrinsic quantile on a plane through each
  region's Delaunay mesh, cut with :func:`slice_region`, and marks the entries near it.
- :func:`plot_coverage` draws the hit-rule status or distance margin at every node of a
  grid of queries in a plane, from :func:`coverage_grid`.
- :func:`plot_entries` draws a corner pair plot of the entries, grouped by validity,
  origin, region or error cause, or coloured by intrinsic quantile.
- :func:`plot_queries` scatters a :meth:`MapBank.query_many` or :meth:`MapBank.fetch_many`
  table by status or margin, and :func:`plot_hit_rate` bins its hit rate with Wilson
  intervals, like a binned :func:`hit_summary`. Both use the column names of
  :mod:`adaptive_microlensing.results`.
- :func:`plot_build` draws a region's build convergence against the goals of a
  :class:`StoppingCriteria`.

The functions with an ``ax`` argument draw on those axes, or on a new figure when it is
``None``, and return the axes, so several figures compose into one. :func:`plot_entries`
returns a seaborn ``PairGrid`` with its own figure, and :func:`plot_build` returns the
figure of its three panels.

No function changes matplotlib's global state: ``rcParams``, styles and seaborn's theme
are left alone. Palettes and colour maps are passed to each call, so the user's own style
applies. Categorical colours come from seaborn's colourblind palette, ``PALETTE``, apart
from the light grey of queries outside the domain, and every category of a ``*_STYLE``
constant also has its own marker, so identity never depends on colour alone.
"""

from __future__ import annotations

import math
import re
import textwrap
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import norm as normal_distribution

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import colors as mcolors
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    from matplotlib.tri import Triangulation
except ImportError as exc:
    raise ImportError(
        "adaptive_microlensing.plotting needs matplotlib and seaborn; install them with "
        "pip install 'adaptive_microlensing[plot]'."
    ) from exc

from .config import DomainSpec, StoppingCriteria
from .lensing import REGIONS
from .slicing import AXES, CoverageGrid, axis_range, coverage_grid, slice_region

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .bank import MapBank

#: A ``(colour, marker)`` pair: a matplotlib colour and a matplotlib marker code.
Style = tuple[str, str]

#: seaborn's colourblind palette as hex strings; every categorical colour below is one of
#: its entries, except ``LIGHT_GREY``.
PALETTE: tuple[str, ...] = tuple(sns.color_palette("colorblind").as_hex())
#: The palette's grey, ``PALETTE[7]``: invalid mesh cells, invalid entries and histograms
#: of quantile pair plots, pooled "Other" errors and the ``outside_hull`` and
#: ``region_not_ready`` statuses.
GREY = PALETTE[7]
#: Light grey level of queries outside the domain, and of the queries that are neither
#: hits nor misses in a margin scatter plot.
LIGHT_GREY = "0.8"
#: Dark grey level of the hatching of regions that are not ready, the "all" hit-rate line
#: and the rolling-maximum residual line.
DARK_GREY = "0.2"

#: (colour, marker) of each region. In this and the other ``*_STYLE`` constants, colours
#: come from PALETTE in a fixed order (with ``LIGHT_GREY`` for queries outside the domain),
#: and every category has its own marker, so identity never depends on colour alone.
REGION_STYLE: dict[str, Style] = {
    "minima": (PALETTE[0], "o"),
    "saddle": (PALETTE[1], "s"),
    "maxima": (PALETTE[2], "^"),
}
#: (colour, marker) of valid entries and of invalid entries, whose maps could not be made
#: or were rejected.
VALIDITY_STYLE: dict[str, Style] = {"valid": (PALETTE[0], "o"), "invalid": (PALETTE[3], "X")}
#: (colour, marker) of entries by origin: added by a build, by a fetch, or imported from
#: a legacy bank.
ORIGIN_STYLE: dict[str, Style] = {
    "build": (PALETTE[0], "o"),
    "fetch": (PALETTE[1], "^"),
    "legacy": (PALETTE[2], "s"),
}
#: (colour, marker) of each query status, in legend order. Critical-line and malformed
#: queries are drawn as ``outside_domain``, which coverage maps leave blank.
#: ``region_not_ready`` shares the grey of ``outside_hull`` and is told apart by its
#: marker, or by hatching in coverage maps.
STATUS_STYLE: dict[str, Style] = {
    "hit": (PALETTE[0], "o"),
    "miss": (PALETTE[3], "X"),
    "invalid_simplex": (PALETTE[4], "D"),
    "outside_hull": (GREY, "s"),
    "region_not_ready": (GREY, "v"),
    "outside_domain": (LIGHT_GREY, "."),
}
#: Styles of the most common error messages in ``plot_entries(hue="error")``, most common
#: first; the rest are "Other".
ERROR_STYLE: tuple[Style, ...] = ((PALETTE[3], "X"), (PALETTE[4], "D"), (PALETTE[2], "P"))
#: Style of the error messages pooled as "Other" in ``plot_entries(hue="error")``.
OTHER_STYLE: Style = (GREY, ".")
#: Colour map of the intrinsic quantile: the default of :func:`plot_slice`, and the map of
#: ``plot_entries(hue="quantile")``.
QUANTILE_CMAP = "viridis"
#: Diverging colour map of distance margins, centred on zero: misses (negative margins)
#: are red and hits (positive margins) blue.
MARGIN_CMAP = "vlag_r"
#: Colour-bar label of distance margins.
MARGIN_LABEL = "margin (threshold − interpolated distance)"

_TEX = {"kappa": r"\kappa", "gamma": r"\gamma", "s": "s"}
#: Query statuses drawn together as "outside domain".
_OUTSIDE = ("outside_domain", "critical_line", "malformed")
#: Hatch pattern of coverage-map nodes whose region is not ready.
_HATCH = "///"
#: A number in an error message: an optionally signed integer or decimal, with an optional
#: exponent.
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
#: The failing point of an error message once its numbers are replaced by ``#``.
_AT_POINT = re.compile(r"\s*at kappa=#, gamma=#(?:, s=#)?")


def _error_kind(message: str) -> str:
    """Return an error message without its numbers and failing point, so that failures group by cause.

    Every number becomes ``#``, and the failing point, ``at kappa=#, gamma=#`` with an
    optional ``, s=#`` and the whitespace before it, is removed. Runs of whitespace are
    then collapsed, and a result longer than 60 characters loses words from its end,
    which are replaced by "…".

    Parameters
    ----------
    message : str
        Error message of an invalid entry, as in the ``error`` column of
        :meth:`MapBank.entries`.

    Returns
    -------
    str
        The cause of the failure, at most 60 characters long.
    """
    return textwrap.shorten(_AT_POINT.sub("", _NUMBER.sub("#", message)), width=60, placeholder="…")


def _share(count: int, total: int) -> str:
    """Return ``count / total`` as a whole percentage; a nonzero share that rounds to zero reads "<1%".

    Parameters
    ----------
    count : int
        Number of items in the group.
    total : int
        Number of items in all groups; it must be positive.

    Returns
    -------
    str
        The share, such as ``"75%"``. It is ``"<1%"`` when ``count`` is positive but the
        share rounds to ``"0%"``, and ``"0%"`` when ``count`` is zero.
    """
    text = f"{count / total:.0%}"
    return "<1%" if count and text == "0%" else text


def _label(name: str) -> str:
    """Return the axis label of a column: its TeX symbol for a coordinate, its name otherwise.

    Parameters
    ----------
    name : str
        Column name.

    Returns
    -------
    str
        A math-mode TeX label (κ, γ or s) for ``"kappa"``, ``"gamma"`` and ``"s"``, and
        ``name`` unchanged for any other column.
    """
    return f"${_TEX[name]}$" if name in _TEX else name


def _axes(ax: Axes | None) -> Axes:
    """Return the axes to draw on, making a new figure when none are given.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axes to draw on. When ``None``, a new figure with one axes is made with
        :func:`matplotlib.pyplot.subplots`, so pyplot manages it like any figure it makes.

    Returns
    -------
    matplotlib.axes.Axes
        ``ax`` itself, or the axes of the new figure.
    """
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _region_names(regions: str | Iterable[str]) -> list[str]:
    """Return the names of the regions to draw, checked against ``REGIONS``.

    Parameters
    ----------
    regions : str or iterable of str
        One region name, or several. Their order and any repeats are kept.

    Returns
    -------
    list of str
        The region names, as a list.

    Raises
    ------
    ValueError
        If no name is given, or if a name is not one of ``REGIONS``.
    """
    names = [regions] if isinstance(regions, str) else list(regions)
    unknown = sorted(set(names).difference(REGIONS))
    if not names or unknown:
        raise ValueError(f"regions must name at least one of {REGIONS}; got {names!r}.")
    return names


def _require(table: pd.DataFrame, columns: Sequence[str]) -> None:
    """Check that a table has the columns a figure needs.

    Parameters
    ----------
    table : pandas.DataFrame
        Table to check.
    columns : sequence of str
        Names of the columns the figure needs.

    Raises
    ------
    ValueError
        If any of ``columns`` is missing from ``table``; the message lists the missing
        columns.
    """
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"The table is missing columns {missing}.")


def _status_groups(status: Any) -> np.ndarray:
    """Return query statuses, with critical-line and malformed queries counted as outside the domain.

    Parameters
    ----------
    status : array_like of str
        ``interpolation_status`` values, as in a :meth:`MapBank.query_many` table or a
        :class:`CoverageGrid`.

    Returns
    -------
    numpy.ndarray
        String array of the same shape, with ``"critical_line"`` and ``"malformed"``
        replaced by ``"outside_domain"``. Every value is converted with ``str``, so a
        missing status becomes ``"nan"`` or ``"None"`` and belongs to no status group.
    """
    values = np.asarray(status, dtype=str)
    return np.where(np.isin(values, _OUTSIDE), "outside_domain", values)


def _status_label(name: str) -> str:
    """Return a status name as legend text.

    Parameters
    ----------
    name : str
        Status name, such as ``"region_not_ready"``.

    Returns
    -------
    str
        The name with spaces for underscores, such as ``"region not ready"``.
    """
    return name.replace("_", " ")


def _margin_norm(margins: np.ndarray, vmin: float | None, vmax: float | None) -> mcolors.TwoSlopeNorm:
    """Return a diverging colour norm of distance margins, centred on zero.

    A margin of zero, where a query just passes the hit rule, sits at the centre of the
    colour map, so hits and misses fall on opposite sides of it.

    Parameters
    ----------
    margins : numpy.ndarray
        Margins of the hits and misses drawn; non-finite values are ignored.
    vmin : float or None
        Margin at the low end of the colour map; it must be negative. When ``None``, minus
        the largest absolute finite margin.
    vmax : float or None
        Margin at the high end of the colour map; it must be positive. When ``None``, the
        largest absolute finite margin.

    Returns
    -------
    matplotlib.colors.TwoSlopeNorm
        Norm with ``vcenter=0``. When no margin is finite, or all finite margins are zero,
        the default limits are ``-1`` and ``1``.
    """
    finite = margins[np.isfinite(margins)]
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    limit = limit if limit > 0.0 else 1.0
    return mcolors.TwoSlopeNorm(
        vcenter=0.0, vmin=-limit if vmin is None else vmin, vmax=limit if vmax is None else vmax
    )


def _legend(ax: Axes, extra: Sequence[Any] = ()) -> None:
    """Draw a legend of the labelled artists plus ``extra`` handles, one entry per label.

    Artists without a label, or with one that starts with ``_``, are left out, and a label
    shared by several artists is listed once. When there is nothing to list, no legend is
    drawn. The legend is placed at the "best" location in a small font.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose artists are listed and on which the legend is drawn.
    extra : sequence of Any, optional
        Further legend handles, such as :class:`matplotlib.patches.Patch` objects, listed
        under their own labels unless an artist already has that label. Default is none.
    """
    handles, labels = ax.get_legend_handles_labels()
    entries = dict(zip(labels, handles, strict=True))
    for handle in extra:
        entries.setdefault(handle.get_label(), handle)
    if entries:
        ax.legend(list(entries.values()), list(entries.keys()), loc="best", fontsize="small")


def _vertical_lines(ax: Axes, positions: Iterable[float], label: str, style: dict[str, Any]) -> None:
    """Draw vertical lines at the given positions, labelling only the first for the legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    positions : iterable of float
        Positions of the lines along the x axis, in data coordinates.
    label : str
        Legend label of the first line. The other lines have none, so the legend lists
        the set once.
    style : dict of str to Any
        Keyword arguments of :meth:`matplotlib.axes.Axes.axvline`, such as colour and
        line style.
    """
    for number, position in enumerate(positions):
        ax.axvline(position, label=label if number == 0 else None, **style)


def _critical_curves(ax: Axes, domain: DomainSpec, axis: str, value: float | None) -> None:
    """Draw the critical lines of infinite magnification (dashed) and the mu_max curves (solid) in a plane.

    The macro magnification ``1 / ((1 - kappa)**2 - gamma**2)`` does not depend on ``s``.
    Dashed mid-grey lines mark ``|mu| = inf``, where ``gamma = |1 - kappa|``, and solid
    near-black lines mark ``|mu| = max_macro_magnification``, the domain's cap, where
    ``|(1 - kappa)**2 - gamma**2| = 1 / max_macro_magnification``. Each kind is labelled
    once for the legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    domain : DomainSpec
        Domain that gives ``max_macro_magnification`` and the coordinate ranges.
    axis : str
        Fixed axis of the plane, one of ``"kappa"``, ``"gamma"`` or ``"s"``. For ``"s"``,
        the curves are drawn in the kappa-gamma plane over the domain's kappa and gamma
        ranges. In a plane of constant kappa (or gamma), each condition holds at a few
        values of gamma (or kappa), whatever ``s``; those inside the domain's range are
        drawn as vertical lines.
    value : float or None
        Value of ``axis`` on the plane. It is needed for ``"kappa"`` and ``"gamma"``, and
        ignored, so it may be ``None``, for ``"s"``.
    """
    eps = 1.0 / domain.max_macro_magnification
    infinite: dict[str, Any] = {"color": "0.45", "lw": 1.0, "ls": "--", "zorder": 3}
    finite: dict[str, Any] = {"color": "0.15", "lw": 1.2, "ls": "-", "zorder": 3}
    infinite_label, finite_label = r"$|\mu| = \infty$", r"$|\mu| = \mu_{\max}$"
    if axis == "s":
        kappa = np.linspace(*domain.kappa_range, 512)
        gamma = np.linspace(*domain.gamma_range, 512)
        ax.plot(kappa, np.abs(1.0 - kappa), label=infinite_label, **infinite)
        offset = np.sqrt(gamma**2 + eps)
        ax.plot(1.0 - offset, gamma, label=finite_label, **finite)
        ax.plot(1.0 + offset, gamma, **finite)
        ax.plot(kappa, np.sqrt((1.0 - kappa) ** 2 + eps), **finite)
        return
    assert value is not None
    if axis == "kappa":
        radial = abs(1.0 - value)
        infinite_at = [radial]
        finite_at = [math.sqrt(radial**2 + eps)]
        if radial**2 > eps:
            finite_at.append(math.sqrt(radial**2 - eps))
        lower, upper = domain.gamma_range
    else:
        infinite_at = [1.0 - value, 1.0 + value]
        root = math.sqrt(value**2 + eps)
        finite_at = [1.0 - root, 1.0 + root]
        if value**2 > eps:
            inner = math.sqrt(value**2 - eps)
            finite_at += [1.0 - inner, 1.0 + inner]
        lower, upper = domain.kappa_range
    _vertical_lines(ax, [x for x in infinite_at if lower <= x <= upper], infinite_label, infinite)
    _vertical_lines(ax, [x for x in finite_at if lower <= x <= upper], finite_label, finite)


def _frame(ax: Axes, domain: DomainSpec, axis: str, value: float, plane_axes: tuple[str, str]) -> None:
    """Set the limits, labels, title and aspect of a plane through the domain.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the plane.
    domain : DomainSpec
        Domain whose ranges along ``plane_axes`` become the axis limits.
    axis : str
        Fixed axis of the plane, one of ``"kappa"``, ``"gamma"`` or ``"s"``. A plane of
        constant ``s``, which shows kappa against gamma, gets an equal aspect ratio.
    value : float
        Value of ``axis`` on the plane, shown in the title, such as ``"$s = 0.5$"``.
    plane_axes : tuple of (str, str)
        Coordinates along the x and y axes, labelled with their TeX symbols.
    """
    ax.set_xlim(*axis_range(domain, plane_axes[0]))
    ax.set_ylim(*axis_range(domain, plane_axes[1]))
    ax.set_xlabel(_label(plane_axes[0]))
    ax.set_ylabel(_label(plane_axes[1]))
    ax.set_title(f"${_TEX[axis]} = {value:g}$")
    if axis == "s":
        ax.set_aspect("equal", adjustable="box")


def plot_slice(
    bank: MapBank,
    *,
    s: float | None = None,
    kappa: float | None = None,
    gamma: float | None = None,
    regions: str | Iterable[str] = REGIONS,
    ax: Axes | None = None,
    cmap: str = QUANTILE_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    mesh: bool = True,
    entries: bool = True,
    entry_tolerance: float | None = None,
    curves: bool = True,
    colorbar: bool = True,
    legend: bool = True,
) -> Axes:
    """Draw the interpolated MPD-distance quantile of each region on a plane through the bank's mesh.

    Each region's Delaunay mesh of entries is cut by the plane with :func:`slice_region`.
    The cut triangles of tetrahedra whose vertices are all valid are shaded by the
    intrinsic quantile with Gouraud shading, which is linear inside each triangle like the
    bank's barycentric interpolation. Cells of tetrahedra with an invalid vertex, where
    queries of a finalized region end as ``"invalid_simplex"``, are grey. Entries within
    ``entry_tolerance`` of the plane (2 % of that axis's domain range by default) are
    drawn as markers.

    Parameters
    ----------
    bank : MapBank
        Bank to draw.
    s : float or None, optional
        Value of ``s`` on a plane of constant ``s``, which shows kappa along x and gamma
        along y. Give exactly one of ``s``, ``kappa`` and ``gamma``, inside the domain's
        range on that axis.
    kappa : float or None, optional
        Value of ``kappa`` on a plane of constant kappa, which shows gamma along x and
        ``s`` along y. See ``s``.
    gamma : float or None, optional
        Value of ``gamma`` on a plane of constant gamma, which shows kappa along x and
        ``s`` along y. See ``s``.
    regions : str or iterable of str, optional
        Region, or regions, to draw, from ``REGIONS``. Default is all three. They share the
        axes and one colour scale. A region whose entries do not form a mesh yet, or whose
        mesh misses the plane, adds no cells.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. When ``None``, a new figure with one axes is made with
        :func:`matplotlib.pyplot.subplots`.
    cmap : str, optional
        Name of the matplotlib colour map of the quantile. Default is ``QUANTILE_CMAP``,
        ``"viridis"``.
    vmin : float or None, optional
        Quantile at the low end of the colour map. When ``None``, the smallest finite
        quantile at the corners of the valid cells of all the regions drawn, or ``0`` when
        there are none.
    vmax : float or None, optional
        Quantile at the high end of the colour map. When ``None``, the largest such
        quantile, or ``1`` when there are none. When the two limits (nearly) coincide,
        both are moved out by 1 % of ``max(|vmin|, 1)``.
    mesh : bool, optional
        Draw the edges of every cut triangle, valid or not, as thin translucent black
        lines. Default is ``True``.
    entries : bool, optional
        Draw the entries of ``regions`` within ``entry_tolerance`` of the plane at their
        in-plane coordinates: valid entries as circles and invalid ones as crosses, in the
        colours of ``VALIDITY_STYLE``, labelled "valid entry" and "invalid entry". Default
        is ``True``.
    entry_tolerance : float or None, optional
        Largest distance from the plane, along its fixed axis, of the entries drawn. When
        ``None``, 2 % of the domain's range on that axis.
    curves : bool, optional
        Draw the critical lines, where ``|mu|`` is infinite (dashed), and the curves where
        ``|mu|`` equals the domain's ``max_macro_magnification`` (solid). Default is
        ``True``.
    colorbar : bool, optional
        Add a colour bar labelled with the bank's quantile level, such as "95% quantile of
        the MPD distance". It takes its space from ``ax``. Default is ``True``.
    legend : bool, optional
        Draw a legend of the entry markers and the curves. Default is ``True``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on: ``ax``, or the axes of the new figure.

    Raises
    ------
    ValueError
        If ``regions`` is empty or names a region not in ``REGIONS``, if not exactly one of
        ``s``, ``kappa`` and ``gamma`` is given or its value lies outside the domain's
        range, or if the lower colour limit exceeds the upper one.

    Notes
    -----
    The axis limits are the domain box along the two in-plane axes, the title names the
    plane, such as ``"$s = 0.5$"``, and a plane of constant ``s`` has an equal aspect
    ratio. The shaded valid cells are rasterised, so vector output stays small.
    """
    slices = [slice_region(bank, region, s=s, kappa=kappa, gamma=gamma) for region in _region_names(regions)]
    domain = bank.config.domain
    axis, value, plane_axes = slices[0].axis, slices[0].value, slices[0].plane_axes
    shown = [piece.quantiles[np.unique(piece.triangles[piece.valid])] for piece in slices]
    values = np.concatenate(shown)
    values = values[np.isfinite(values)]
    low = vmin if vmin is not None else (float(values.min()) if values.size else 0.0)
    high = vmax if vmax is not None else (float(values.max()) if values.size else 1.0)
    if low > high:
        raise ValueError(f"Expected vmin <= vmax, got {low} and {high}.")
    if math.isclose(low, high):
        delta = max(1e-12, 0.01 * max(abs(low), 1.0))
        low, high = low - delta, high + delta
    norm = mcolors.Normalize(low, high)

    ax = _axes(ax)
    for piece in slices:
        if len(piece.triangles) == 0:
            continue
        x, y = piece.points[:, 0], piece.points[:, 1]
        if piece.valid.any():
            # Gouraud shading is linear inside each triangle, like the bank's interpolation.
            # Points used only by invalid cells carry NaN; give them a finite placeholder.
            quantiles = np.where(np.isfinite(piece.quantiles), piece.quantiles, low)
            ax.tripcolor(
                Triangulation(x, y, piece.triangles[piece.valid]),
                quantiles,
                shading="gouraud",
                cmap=cmap,
                norm=norm,
                rasterized=True,
                zorder=1,
            )
        if not piece.valid.all():
            ax.add_collection(
                PolyCollection(
                    piece.points[piece.triangles[~piece.valid]],
                    facecolors=mcolors.to_rgba(GREY, 0.35),
                    edgecolors=GREY,
                    linewidths=0.5,
                    zorder=1,
                )
            )
        if mesh:
            ax.triplot(Triangulation(x, y, piece.triangles), color="k", lw=0.3, alpha=0.4, zorder=2)

    if entries:
        lower, upper = axis_range(domain, axis)
        tolerance = 0.02 * (upper - lower) if entry_tolerance is None else float(entry_tolerance)
        for piece in slices:
            table = bank.entries(piece.region)
            near = table[(table[axis] - value).abs() <= tolerance]
            for name, (color, marker) in VALIDITY_STYLE.items():
                rows = near[near["valid"] == (name == "valid")]
                if len(rows):
                    ax.scatter(
                        rows[plane_axes[0]],
                        rows[plane_axes[1]],
                        color=color,
                        marker=marker,
                        s=18,
                        edgecolors="white",
                        linewidths=0.5,
                        zorder=4,
                        label=f"{name} entry",
                    )
    if curves:
        _critical_curves(ax, domain, axis, value)
    _frame(ax, domain, axis, value, plane_axes)
    if colorbar:
        ax.figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            pad=0.02,
            label=f"{bank.config.variability.quantile:.0%} quantile of the MPD distance",
        )
    if legend:
        _legend(ax)
    return ax


def plot_coverage(
    bank: MapBank | None = None,
    *,
    s: float | None = None,
    kappa: float | None = None,
    gamma: float | None = None,
    n: int = 200,
    grid: CoverageGrid | None = None,
    color: str = "status",
    ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    curves: bool = True,
    legend: bool = True,
    colorbar: bool = True,
) -> Axes:
    """Draw where the bank answers queries on a plane: the hit-rule status or margin of every grid node.

    Give either ``bank`` with a plane, and the grid is computed with :func:`coverage_grid`,
    or a ``grid`` that :func:`coverage_grid` made. Each node is drawn as a cell centred on
    it, coloured by its status as in ``STATUS_STYLE``: hits blue, misses vermilion,
    invalid simplices pink, and nodes outside the hull or in a region that is not ready
    grey. Nodes of a region that is not ready are also hatched with ``///``.
    Nodes outside the domain, on a critical line or malformed are left blank.

    Parameters
    ----------
    bank : MapBank or None, optional
        Bank to query. Its grid is made with :func:`coverage_grid` from the plane and
        ``n``, which runs :meth:`MapBank.query_many` on ``n * n`` nodes and writes nothing.
        When ``None``, ``grid`` must be given.
    s : float or None, optional
        Value of ``s`` on a plane of constant ``s``, which shows kappa along x and gamma
        along y. Only with ``bank``, which needs exactly one of ``s``, ``kappa`` and
        ``gamma``, inside the domain's range on that axis.
    kappa : float or None, optional
        Value of ``kappa`` on a plane of constant kappa, which shows gamma along x and
        ``s`` along y. See ``s``.
    gamma : float or None, optional
        Value of ``gamma`` on a plane of constant gamma, which shows kappa along x and
        ``s`` along y. See ``s``.
    n : int, optional
        Number of grid nodes along each in-plane axis, at least 2. It is used only with
        ``bank``. Default is ``200``.
    grid : CoverageGrid or None, optional
        Grid made by :func:`coverage_grid`, drawn instead of querying a bank. It fixes the
        plane. When ``None``, ``bank`` must be given.
    color : str, optional
        ``"status"`` (default) colours every node by its status. ``"margin"`` colours the
        hits and misses by their distance margin, ``threshold - interpolated distance``,
        on ``MARGIN_CMAP`` centred on zero, so hits (positive margins) are blue and misses
        (negative margins) red; the other nodes keep their status colours.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. When ``None``, a new figure with one axes is made with
        :func:`matplotlib.pyplot.subplots`.
    vmin : float or None, optional
        Margin at the low end of the colour map, used only with ``color="margin"``. It must
        be negative, or matplotlib raises :exc:`ValueError`. When ``None``, minus the
        largest absolute margin of the hits and misses, or ``-1`` when that is zero or
        there are none.
    vmax : float or None, optional
        Margin at the high end of the colour map, used only with ``color="margin"``. It
        must be positive. When ``None``, the largest absolute margin of the hits and
        misses, or ``1`` when that is zero or there are none.
    curves : bool, optional
        Draw the critical lines, where ``|mu|`` is infinite (dashed), and the curves where
        ``|mu|`` equals the domain's ``max_macro_magnification`` (solid). Default is
        ``True``.
    legend : bool, optional
        Draw a legend with a patch for each status present, labelled with its share of the
        nodes that are not left blank, such as "hit (42%)", plus the curves. With
        ``color="margin"``, hits and misses are left out, since the colour bar shows them.
        Default is ``True``.
    colorbar : bool, optional
        With ``color="margin"``, add a colour bar labelled ``MARGIN_LABEL``; it takes its
        space from ``ax``. It is ignored with ``color="status"``. Default is ``True``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on: ``ax``, or the axes of the new figure.

    Raises
    ------
    ValueError
        If neither or both of ``bank`` and ``grid`` are given, if ``color`` is not
        ``"status"`` or ``"margin"``, or if ``grid`` is given with ``s``, ``kappa`` or
        ``gamma``. With ``bank``, also if not exactly one of ``s``, ``kappa`` and ``gamma``
        is given, if its value lies outside the domain's range, or if ``n`` is not an
        integer of at least 2.

    Notes
    -----
    The axis limits are the domain box along the two in-plane axes, the title names the
    plane, and a plane of constant ``s`` has an equal aspect ratio. The coloured cells are
    rasterised, so vector output stays small.
    """
    if (bank is None) == (grid is None):
        raise ValueError("Give either bank (with s, kappa or gamma) or a precomputed grid.")
    if color not in ("status", "margin"):
        raise ValueError(f"color must be 'status' or 'margin', got {color!r}.")
    if grid is None:
        assert bank is not None
        grid = coverage_grid(bank, s=s, kappa=kappa, gamma=gamma, n=n)
    elif any(value is not None for value in (s, kappa, gamma)):
        raise ValueError("A precomputed grid already fixes the plane; drop s, kappa and gamma.")

    ax = _axes(ax)
    status = _status_groups(grid.status)
    covered = np.isin(status, ("hit", "miss"))
    names = [name for name in STATUS_STYLE if name != "outside_domain"]
    codes = np.full(status.shape, np.nan)
    for code, name in enumerate(names):
        codes[status == name] = code
    if color == "margin":
        codes[covered] = np.nan
    ax.pcolormesh(
        grid.x,
        grid.y,
        np.ma.masked_invalid(codes),
        cmap=mcolors.ListedColormap([STATUS_STYLE[name][0] for name in names]),
        norm=mcolors.BoundaryNorm(np.arange(len(names) + 1) - 0.5, len(names)),
        shading="nearest",
        rasterized=True,
        zorder=1,
    )
    not_ready = status == "region_not_ready"
    if not_ready.any():
        hatching = ax.contourf(
            grid.x,
            grid.y,
            not_ready.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=[_HATCH],
            zorder=2,
        )
        hatching.set_edgecolor(DARK_GREY)
        hatching.set_linewidth(0.0)
    if color == "margin":
        margins = np.where(covered, grid.margin, np.nan)
        mesh = ax.pcolormesh(
            grid.x,
            grid.y,
            np.ma.masked_invalid(margins),
            cmap=MARGIN_CMAP,
            norm=_margin_norm(margins[covered], vmin, vmax),
            shading="nearest",
            rasterized=True,
            zorder=1,
        )
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, pad=0.02, label=MARGIN_LABEL)

    if curves:
        _critical_curves(ax, grid.domain, grid.axis, grid.value)
    _frame(ax, grid.domain, grid.axis, grid.value, grid.plane_axes)
    if legend:
        inside = int((status != "outside_domain").sum())
        patches = []
        for name in names:
            count = int((status == name).sum())
            if count and not (color == "margin" and name in ("hit", "miss")):
                patches.append(
                    Patch(
                        facecolor=STATUS_STYLE[name][0],
                        hatch=_HATCH if name == "region_not_ready" else None,
                        label=f"{_status_label(name)} ({_share(count, inside)})",
                    )
                )
        _legend(ax, patches)
    return ax


def _entry_groups(table: pd.DataFrame, hue: str) -> tuple[pd.Categorical, dict[str, Style]]:
    """Return the group of every entry and the style of each group present, in legend order.

    For ``"error"``, valid entries form the group ``"valid"``, and invalid entries are
    grouped by the cause of their failure, their error message without numbers or the
    failing point. The ``len(ERROR_STYLE)`` most common causes, ties broken
    alphabetically, get the styles of ``ERROR_STYLE`` in that order, and the other causes
    are pooled as ``"Other"``.

    Parameters
    ----------
    table : pandas.DataFrame
        Entries of one or more regions, with a ``valid`` column and the ``origin``,
        ``region`` or ``error`` column that ``hue`` groups by.
    hue : str
        ``"validity"``, ``"origin"``, ``"region"`` or ``"error"``. Any other value is
        treated as ``"error"``.

    Returns
    -------
    groups : pandas.Categorical
        Group of each entry, with the groups present as categories, in legend order. A
        label without a style, such as an origin not in ``ORIGIN_STYLE``, is not a
        category.
    styles : dict of str to tuple of (str, str)
        ``(colour, marker)`` of each group present, in legend order: the order of
        ``VALIDITY_STYLE``, ``ORIGIN_STYLE`` or ``REGION_STYLE``, or for ``"error"``,
        ``"valid"``, the common causes from most to least common, then ``"Other"``.
    """
    valid = table["valid"].to_numpy(dtype=bool)
    if hue == "validity":
        labels = np.where(valid, "valid", "invalid")
        styles = VALIDITY_STYLE
    elif hue == "origin":
        labels = table["origin"].to_numpy(dtype=str)
        styles = ORIGIN_STYLE
    elif hue == "region":
        labels = table["region"].to_numpy(dtype=str)
        styles = REGION_STYLE
    else:
        errors = np.array([_error_kind(message) for message in table["error"].astype(str)], dtype=object)
        counts = pd.Series(errors[~valid]).value_counts()
        ranked = sorted(counts.index, key=lambda message: (-counts[message], message))
        common = ranked[: len(ERROR_STYLE)]
        labels = np.where(valid, "valid", np.where(np.isin(errors, common), errors, "Other"))
        styles = {
            "valid": VALIDITY_STYLE["valid"],
            **dict(zip(common, ERROR_STYLE, strict=False)),
            "Other": OTHER_STYLE,
        }
    present = {name: style for name, style in styles.items() if name in set(labels)}
    # A label without a style becomes missing, which leaves its entries out of the plot.
    known = pd.Series(labels, dtype=object)
    known = known.where(known.isin(list(present)))
    return pd.Categorical(known, categories=list(present)), present


def _relabel(grid: Any) -> None:
    """Replace the coordinate names in a pair grid's axis labels by their TeX symbols.

    Parameters
    ----------
    grid : seaborn.PairGrid
        Pair grid relabelled in place. The empty cells of a corner grid are skipped, and
        labels other than ``"kappa"``, ``"gamma"`` and ``"s"`` are kept.
    """
    for ax in grid.axes.flat:
        if ax is not None:
            ax.set_xlabel(_label(ax.get_xlabel()))
            ax.set_ylabel(_label(ax.get_ylabel()))


def plot_entries(bank: MapBank, *, regions: str | Iterable[str] = REGIONS, hue: str = "validity") -> Any:
    """Draw a corner pair plot of the entries in the kappa-gamma, kappa-s and gamma-s projections.

    The entries of ``regions`` are pooled and drawn with seaborn on a new figure. Each
    panel below the diagonal is a scatter plot of one pair of coordinates, and each
    diagonal panel a histogram of one coordinate. ``hue`` chooses what the colours and
    markers show. For every choice but ``"quantile"``, the histograms of the groups are
    stacked, and the legend lists the groups present in the order of their style
    constant, or for ``"error"``, valid entries first and "Other" last.

    Parameters
    ----------
    bank : MapBank
        Bank whose entries are drawn.
    regions : str or iterable of str, optional
        Region, or regions, whose entries are drawn, from ``REGIONS``. Default is all
        three.
    hue : str, optional
        What the colours and markers show:

        - ``"validity"`` (default): valid and invalid entries, styled by
          ``VALIDITY_STYLE``.
        - ``"origin"``: entries added by a build, by a fetch or imported from a legacy
          bank, styled by ``ORIGIN_STYLE``.
        - ``"region"``: the region of each entry, styled by ``REGION_STYLE``.
        - ``"error"``: valid entries, then invalid entries grouped by the cause of their
          failure, their error message without numbers or the failing point. The three
          most common causes, ties broken alphabetically, get ``ERROR_STYLE`` in that
          order, and the rest are pooled as "Other" in ``OTHER_STYLE``.
        - ``"quantile"``: valid entries coloured by intrinsic quantile on
          ``QUANTILE_CMAP``, from the smallest to the largest valid quantile, with a
          colour bar labelled with the bank's quantile level. Invalid entries are grey
          crosses, labelled "invalid" in the kappa-gamma panel, and the diagonal holds
          grey histograms of all entries.

    Returns
    -------
    seaborn.PairGrid
        The pair plot, which owns its figure, ``grid.figure``.

    Raises
    ------
    ValueError
        If ``hue`` is not one of the five choices, if ``regions`` is empty or names a
        region not in ``REGIONS``, if the regions have no entries, or, with
        ``hue="quantile"``, if none of their entries is valid.
    """
    if hue not in ("validity", "origin", "region", "error", "quantile"):
        raise ValueError(f"hue must be validity, origin, region, error or quantile; got {hue!r}.")
    names = _region_names(regions)
    table = pd.concat([bank.entries(region).assign(region=region) for region in names], ignore_index=True)
    if table.empty:
        raise ValueError(f"Regions {names} have no entries.")
    columns = list(AXES)
    if hue == "quantile":
        return _quantile_pairs(table, bank.config.variability.quantile)
    groups, styles = _entry_groups(table, hue)
    data = table[columns].assign(**{hue: groups})
    grid = sns.pairplot(
        data,
        vars=columns,
        hue=hue,
        hue_order=list(styles),
        palette={name: color for name, (color, _) in styles.items()},
        markers={name: marker for name, (_, marker) in styles.items()},
        corner=True,
        diag_kind="hist",
        plot_kws={"s": 14, "edgecolor": "white", "linewidth": 0.3},
        diag_kws={"multiple": "stack", "element": "step"},
    )
    _relabel(grid)
    return grid


def _quantile_pairs(table: pd.DataFrame, quantile: float) -> Any:
    """Draw a corner pair plot of the entries, coloured by intrinsic quantile.

    Valid entries are coloured by ``intrinsic_quantile`` on ``QUANTILE_CMAP``, scaled from
    the smallest to the largest valid quantile. Invalid entries are grey crosses, with a
    legend entry "invalid" in the kappa-gamma panel. The diagonal holds grey histograms of
    all entries, and one colour bar serves every panel.

    Parameters
    ----------
    table : pandas.DataFrame
        Entries of one or more regions, with ``kappa``, ``gamma``, ``s``, ``valid`` and
        ``intrinsic_quantile`` columns.
    quantile : float
        The bank's quantile level, ``VariabilitySpec.quantile``, used only in the colour
        bar's label, such as "95% quantile of the MPD distance".

    Returns
    -------
    seaborn.PairGrid
        The pair plot, which owns its new figure.

    Raises
    ------
    ValueError
        If no entry is valid.
    """
    valid = table["valid"].to_numpy(dtype=bool)
    if not valid.any():
        raise ValueError("There are no valid entries to colour by quantile.")
    values = table["intrinsic_quantile"].to_numpy(dtype=float)
    norm = mcolors.Normalize(float(values[valid].min()), float(values[valid].max()))
    grid = sns.PairGrid(table, vars=list(AXES), corner=True)
    grid.map_diag(sns.histplot, color=GREY, element="step")
    for row in range(3):
        for column in range(row):
            ax = grid.axes[row, column]
            x, y = table[AXES[column]].to_numpy(), table[AXES[row]].to_numpy()
            ax.scatter(
                x[valid],
                y[valid],
                c=values[valid],
                cmap=QUANTILE_CMAP,
                norm=norm,
                s=14,
                edgecolors="white",
                linewidths=0.3,
            )
            if not valid.all():
                ax.scatter(
                    x[~valid],
                    y[~valid],
                    color=GREY,
                    marker="X",
                    s=14,
                    edgecolors="white",
                    linewidths=0.3,
                    label="invalid",
                )
    if not valid.all():
        grid.axes[1, 0].legend(loc="best", fontsize="small")
    grid.figure.colorbar(
        ScalarMappable(norm=norm, cmap=QUANTILE_CMAP),
        ax=[ax for ax in grid.axes.flat if ax is not None],
        label=f"{quantile:.0%} quantile of the MPD distance",
    )
    _relabel(grid)
    return grid


def plot_queries(
    table: pd.DataFrame,
    *,
    x: str = "kappa",
    y: str = "gamma",
    color: str = "status",
    domain: DomainSpec | None = None,
    ax: Axes | None = None,
    size: float = 6.0,
    vmin: float | None = None,
    vmax: float | None = None,
    legend: bool = True,
    colorbar: bool = True,
) -> Axes:
    """Scatter the queries of a ``query_many`` or ``fetch_many`` table, by status or margin.

    Each query is drawn at its ``x`` and ``y`` coordinates, so the figure is a projection
    along the third coordinate. The markers have no edges and are rasterised. With
    ``domain``, the limits are the domain box and the kappa-gamma projection shows the
    critical curves.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :meth:`MapBank.query_many` or :meth:`MapBank.fetch_many`, or any table
        with ``kappa``, ``gamma``, ``s`` and ``interpolation_status`` columns, and a
        ``distance_margin`` column for ``color="margin"``.
    x : str, optional
        Coordinate along the x axis, one of ``AXES``: ``"kappa"``, ``"gamma"`` or ``"s"``.
        Default is ``"kappa"``.
    y : str, optional
        Coordinate along the y axis, one of ``AXES`` other than ``x``. Default is
        ``"gamma"``.
    color : str, optional
        ``"status"`` (default) draws each status group in its colour and marker from
        ``STATUS_STYLE``, labelled with its number of rows, such as "hit (120)".
        Critical-line and malformed queries are counted as "outside domain", and rows with
        any other status, such as a missing one, are not drawn. ``"margin"`` colours the
        hits and misses by ``distance_margin`` on ``MARGIN_CMAP`` centred on zero, so hits
        (positive margins) are blue and misses (negative margins) red, and draws every
        other row as a light grey dot labelled "not covered".
    domain : DomainSpec or None, optional
        Domain whose ranges along ``x`` and ``y`` become the axis limits. With
        ``x="kappa"`` and ``y="gamma"``, its critical lines (dashed) and
        ``max_macro_magnification`` curves (solid) are drawn too. When ``None``,
        matplotlib chooses the limits and no curves are drawn.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. When ``None``, a new figure with one axes is made with
        :func:`matplotlib.pyplot.subplots`.
    size : float, optional
        Marker area in points squared, matplotlib's ``s``. Default is ``6.0``.
    vmin : float or None, optional
        Margin at the low end of the colour map, used only with ``color="margin"``. It must
        be negative, or matplotlib raises :exc:`ValueError`. When ``None``, minus the
        largest absolute margin of the hits and misses, or ``-1`` when that is zero or
        there are none.
    vmax : float or None, optional
        Margin at the high end of the colour map, used only with ``color="margin"``. It
        must be positive. When ``None``, the largest absolute margin of the hits and
        misses, or ``1`` when that is zero or there are none.
    legend : bool, optional
        Draw a legend of the status groups, or of the "not covered" dots with
        ``color="margin"``, plus the curves. Default is ``True``.
    colorbar : bool, optional
        With ``color="margin"``, add a colour bar labelled ``MARGIN_LABEL``; it takes its
        space from ``ax``. It is ignored with ``color="status"``. Default is ``True``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on: ``ax``, or the axes of the new figure.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` are not two different names from ``AXES``, if ``color`` is not
        ``"status"`` or ``"margin"``, or if ``table`` lacks a column the figure needs.

    Notes
    -----
    A malformed query has a coordinate that is not finite. When that coordinate is ``x``
    or ``y``, matplotlib does not draw the query, although ``color="status"`` still counts
    it in the "outside domain" label.
    """
    if x not in AXES or y not in AXES or x == y:
        raise ValueError(f"x and y must be two different names from {AXES}; got {x!r} and {y!r}.")
    if color not in ("status", "margin"):
        raise ValueError(f"color must be 'status' or 'margin', got {color!r}.")
    required = ["kappa", "gamma", "s", "interpolation_status"]
    _require(table, required + (["distance_margin"] if color == "margin" else []))
    ax = _axes(ax)
    status = _status_groups(table["interpolation_status"])
    xs, ys = table[x].to_numpy(dtype=float), table[y].to_numpy(dtype=float)
    common: dict[str, Any] = {"s": size, "linewidths": 0.0, "rasterized": True}
    if color == "status":
        for name, (colour, marker) in STATUS_STYLE.items():
            rows = status == name
            if rows.any():
                ax.scatter(
                    xs[rows],
                    ys[rows],
                    color=colour,
                    marker=marker,
                    label=f"{_status_label(name)} ({int(rows.sum())})",
                    **common,
                )
    else:
        covered = np.isin(status, ("hit", "miss"))
        if (~covered).any():
            ax.scatter(
                xs[~covered], ys[~covered], color=LIGHT_GREY, marker=".", label="not covered", **common
            )
        margins = table["distance_margin"].to_numpy(dtype=float)
        points = ax.scatter(
            xs[covered],
            ys[covered],
            c=margins[covered],
            cmap=MARGIN_CMAP,
            norm=_margin_norm(margins[covered], vmin, vmax),
            **common,
        )
        if colorbar:
            ax.figure.colorbar(points, ax=ax, pad=0.02, label=MARGIN_LABEL)
    if domain is not None:
        ax.set_xlim(*axis_range(domain, x))
        ax.set_ylim(*axis_range(domain, y))
        if (x, y) == ("kappa", "gamma"):
            _critical_curves(ax, domain, "s", None)
    ax.set_xlabel(_label(x))
    ax.set_ylabel(_label(y))
    if legend:
        _legend(ax)
    return ax


def _wilson(hits: np.ndarray, counts: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the Wilson score interval of the hit rate ``hits / counts``; NaN where counts is zero.

    Parameters
    ----------
    hits : numpy.ndarray
        Number of hits in each bin.
    counts : numpy.ndarray
        Number of queries counted in each bin, of the same shape as ``hits``.
    z : float
        Half-width of the interval in standard deviations of the normal distribution,
        such as ``1.0`` for about 68 %.

    Returns
    -------
    low : numpy.ndarray
        Lower end of the interval in each bin.
    high : numpy.ndarray
        Upper end of the interval in each bin.

    Notes
    -----
    With rate ``p = hits / counts`` and ``n = counts``, the interval is
    ``(p + z**2 / (2 n) ± z sqrt(p (1 - p) / n + z**2 / (4 n**2))) / (1 + z**2 / n)``.
    Unlike the normal approximation, it stays inside ``[0, 1]`` and keeps a nonzero width
    when ``p`` is 0 or 1.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = hits / counts
        denominator = 1.0 + z**2 / counts
        centre = (rate + z**2 / (2.0 * counts)) / denominator
        half = z * np.sqrt(rate * (1.0 - rate) / counts + z**2 / (4.0 * counts**2)) / denominator
    return centre - half, centre + half


def _bin_edges(values: np.ndarray, bins: int | Sequence[float]) -> np.ndarray:
    """Return the bin edges of a numeric column.

    Parameters
    ----------
    values : numpy.ndarray
        Values to bin. Non-finite values are ignored when the edges are chosen.
    bins : int or sequence of float
        Number of equal-width bins from the smallest to the largest finite value, or the
        edges themselves. When all finite values are equal, the bins span a width of 1
        centred on that value. A bool is not taken as a number of bins.

    Returns
    -------
    numpy.ndarray
        Strictly increasing edges as floats, ``bins + 1`` of them for an integer ``bins``.

    Raises
    ------
    ValueError
        If an integer ``bins`` is below 1 or ``values`` has no finite value, or if the
        edges given are not a one-dimensional, strictly increasing sequence of at least
        two numbers.
    """
    if isinstance(bins, int | np.integer) and not isinstance(bins, bool):
        finite = values[np.isfinite(values)]
        if bins < 1 or finite.size == 0:
            raise ValueError(f"Need bins >= 1 and at least one finite value; got bins={bins}.")
        low, high = float(finite.min()), float(finite.max())
        if low == high:
            low, high = low - 0.5, high + 0.5
        return np.linspace(low, high, int(bins) + 1)
    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("bins must be a positive integer or an increasing sequence of edges.")
    return edges


def plot_hit_rate(
    table: pd.DataFrame,
    *,
    by: str = "s",
    bins: int | Sequence[float] = 10,
    among: str = "all",
    interval: float = 0.68,
    ax: Axes | None = None,
    legend: bool = True,
) -> Axes:
    """Draw the hit rate per bin of a numeric column, per region and overall, with Wilson intervals.

    The rows are binned by ``by``, and in each bin the hit rate is the number of hits over
    the number of rows counted. Lines join the rates at the bin centres: one per region in
    ``REGIONS``, split by ``query_region`` and styled by ``REGION_STYLE``, and a dark grey
    line with circles, labelled "all", of every row counted, rows without a region
    included. An empty bin leaves a gap in its line, and a region with no rows counted
    gets no line. A translucent band of the line's colour spans each non-empty bin from
    edge to edge and shows its Wilson score interval. These are the binned counterparts of
    the rates of :func:`hit_summary`.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :meth:`MapBank.query_many` or :meth:`MapBank.fetch_many`, possibly with
        extra columns, with ``by``, ``query_region``, ``is_hit`` and
        ``interpolation_status`` columns. Only rows whose ``is_hit`` is ``True`` count as
        hits; ``False`` and missing values, as a left merge leaves on rows never queried,
        do not.
    by : str, optional
        Numeric column to bin by, such as ``"s"`` (default), another coordinate or a
        column added by the caller. Rows whose value is not finite or lies outside the
        edges are left out.
    bins : int or sequence of float, optional
        Number of equal-width bins from the smallest to the largest finite value of
        ``by``, or strictly increasing bin edges. Each bin holds its lower edge, and the
        last also its upper edge. Default is ``10``.
    among : str, optional
        Which rows the rates count: ``"all"`` (default) counts every row, and
        ``"covered"`` only the hits and misses, the rows whose ``interpolation_status`` is
        ``"hit"`` or ``"miss"``.
    interval : float, optional
        Probability content of the Wilson score interval, in ``(0, 1)``. Default is
        ``0.68``, about one standard deviation.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. When ``None``, a new figure with one axes is made with
        :func:`matplotlib.pyplot.subplots`.
    legend : bool, optional
        Draw a legend of the lines. Default is ``True``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on: ``ax``, or the axes of the new figure. The y axis spans
        ``[0, 1]`` and reads "hit rate", or "hit rate (covered)" with
        ``among="covered"``.

    Raises
    ------
    ValueError
        If ``among`` is not ``"all"`` or ``"covered"``, if ``interval`` is not strictly
        between 0 and 1, if ``table`` lacks a column the figure needs, or if ``bins`` is
        not usable: an integer below 1, an integer while ``by`` has no finite value, or
        edges that are not a strictly increasing sequence of at least two numbers.
    """
    if among not in ("all", "covered"):
        raise ValueError(f"among must be 'all' or 'covered', got {among!r}.")
    if not 0.0 < interval < 1.0:
        raise ValueError(f"interval must lie in (0, 1), got {interval!r}.")
    _require(table, [by, "query_region", "is_hit", "interpolation_status"])
    values = table[by].to_numpy(dtype=float)
    edges = _bin_edges(values, bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centres)
    inside = np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    which = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)
    # Only a true flag is a hit: NaN, as a merge leaves on rows never queried, would cast to True.
    hits = table["is_hit"].eq(True).fillna(False).to_numpy(dtype=bool)
    counted = inside.copy()
    if among == "covered":
        counted &= np.isin(table["interpolation_status"].to_numpy(dtype=str), ("hit", "miss"))
    z = float(normal_distribution.ppf(0.5 + interval / 2.0))
    regions = table["query_region"].to_numpy(dtype=object)
    series: list[tuple[str, Style, np.ndarray]] = [
        (region, REGION_STYLE[region], regions == region) for region in REGIONS
    ]
    series.append(("all", (DARK_GREY, "o"), np.ones(len(table), dtype=bool)))

    ax = _axes(ax)
    for name, (colour, marker), mask in series:
        use = mask & counted
        if not use.any():
            continue
        counts = np.bincount(which[use], minlength=n_bins).astype(float)
        hit_counts = np.bincount(which[use], weights=hits[use].astype(float), minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = hit_counts / counts
        low, high = _wilson(hit_counts, counts, z)
        ax.plot(centres, rate, color=colour, marker=marker, lw=1.5, label=name)
        # Each bin's interval spans the bin's edges, so a bin between empty bins keeps its band.
        ax.fill_between(
            np.repeat(edges, 2)[1:-1],
            np.repeat(low, 2),
            np.repeat(high, 2),
            color=colour,
            alpha=0.2,
            linewidth=0.0,
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(_label(by))
    ax.set_ylabel("hit rate" if among == "all" else "hit rate (covered)")
    if legend:
        _legend(ax)
    return ax


def plot_build(
    bank: MapBank, region: str, *, stop: StoppingCriteria | None = None, axes: Sequence[Axes] | None = None
) -> Figure:
    """Draw a region's build convergence: simplex loss, relative residuals and valid fraction per entry.

    The build and legacy entries of ``region`` are drawn in entry order on three panels,
    against their ``entry_id``; fetched entries are left out. Legacy entries carry the
    diagnostics of the original run where it recorded them. The panels are, from top to
    bottom:

    - the adaptive learner's loss, its largest simplex loss, after each entry
      (``simplex_loss``), as a line on a log scale, titled with the region;
    - the relative residual of each entry (``rel_residual``), the absolute difference
      between its intrinsic quantile and the mesh's prediction there before it was added,
      over the quantile, as dots, and the build's ``max_residual``, the largest of the
      last ``min_num_residuals`` relative residuals, as a dark grey line labelled "rolling
      maximum", both on a log scale;
    - the fraction of valid entries among the entries drawn up to each one.

    The log-scale panels leave out values that are missing, zero or negative, such as the
    residuals of invalid entries.

    Parameters
    ----------
    bank : MapBank
        Bank to draw.
    region : str
        Region of the build, one of ``REGIONS``.
    stop : StoppingCriteria or None, optional
        Criteria whose goals are drawn as dotted near-black lines: ``simplex_loss_goal`` as
        a horizontal line on the loss panel and ``residual_goal`` as one on the residual
        panel, both labelled "goal", and ``max_valid_points`` as a vertical line labelled
        "max valid points" on every panel. That line marks the entry with which the
        region's valid entries, fetched ones included, first number ``max_valid_points``,
        where a build with these criteria would stop; it is left out when the region has
        fewer valid entries. Goals set to ``None`` are not drawn. When ``stop`` is ``None``, no
        goals are drawn.
    axes : sequence of matplotlib.axes.Axes or None, optional
        Three axes for the loss, residual and valid-fraction panels, in that order. When
        ``None``, a new 6.4 by 7.2 inch figure is made with three stacked panels that
        share the x axis.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn on: the new one, or the figure of the first of ``axes``.

    Raises
    ------
    ValueError
        If ``region`` is not one of the bank's regions, if it has no build or legacy
        entries, or if ``axes`` does not hold exactly three axes.
    """
    table = bank.entries(region)
    rows = table[table["origin"].isin(["build", "legacy"])]
    if rows.empty:
        raise ValueError(f"Region {region!r} has no build or legacy entries.")
    if axes is None:
        figure, created = plt.subplots(3, 1, sharex=True, figsize=(6.4, 7.2))
        panels = list(created)
    else:
        panels = list(axes)
        if len(panels) != 3:
            raise ValueError(f"axes must hold exactly three axes, got {len(panels)}.")
        figure = panels[0].figure
    loss_ax, residual_ax, valid_ax = panels
    entry = rows["entry_id"].to_numpy()

    def positive(column: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the entry numbers and values of the drawn rows where a column is positive.

        The loss and residual panels have log scales, so values that are not finite or
        not positive are left out.

        Parameters
        ----------
        column : str
            Column of the entries table, such as ``"simplex_loss"``.

        Returns
        -------
        entry_ids : numpy.ndarray
            ``entry_id`` of each build or legacy row kept.
        values : numpy.ndarray
            Value of ``column`` in each row kept, as floats.
        """
        values = rows[column].to_numpy(dtype=float)
        keep = np.isfinite(values) & (values > 0.0)
        return entry[keep], values[keep]

    loss_ax.plot(*positive("simplex_loss"), color=PALETTE[0], lw=1.2)
    loss_ax.set_yscale("log")
    loss_ax.set_ylabel("simplex loss")
    loss_ax.set_title(region)
    residual_ax.scatter(
        *positive("rel_residual"), color=PALETTE[0], s=6, linewidths=0.0, label="relative residual"
    )
    residual_ax.plot(*positive("max_residual"), color=DARK_GREY, lw=1.2, label="rolling maximum")
    residual_ax.set_yscale("log")
    residual_ax.set_ylabel("relative residual")
    valid = rows["valid"].to_numpy(dtype=bool)
    valid_ax.plot(entry, np.cumsum(valid) / np.arange(1, len(valid) + 1), color=PALETTE[0], lw=1.2)
    valid_ax.set_ylim(0.0, 1.02)
    valid_ax.set_ylabel("valid fraction")
    valid_ax.set_xlabel("entry")

    if stop is not None:
        goal: dict[str, Any] = {"color": "0.15", "ls": ":", "lw": 1.2, "label": "goal"}
        if stop.simplex_loss_goal is not None:
            loss_ax.axhline(stop.simplex_loss_goal, **goal)
        if stop.residual_goal is not None:
            residual_ax.axhline(stop.residual_goal, **goal)
        if stop.max_valid_points is not None:
            # A build stops on the valid entries of the whole region, fetched ones included.
            reached = np.flatnonzero(np.cumsum(table["valid"].to_numpy(dtype=bool)) >= stop.max_valid_points)
            if reached.size:
                stop_entry = table["entry_id"].iloc[reached[0]]
                for panel in panels:
                    panel.axvline(stop_entry, **{**goal, "label": "max valid points"})
    for panel in panels:
        if panel.get_legend_handles_labels()[1]:
            _legend(panel)
    return figure
