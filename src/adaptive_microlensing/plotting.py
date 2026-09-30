"""Figures of a bank: mesh slices, entries, coverage, queries and build convergence.

This module needs the ``plot`` extra and is imported explicitly::

    from adaptive_microlensing import plotting

No function changes matplotlib's global state. Palettes and colour maps are passed to
each call, so the user's own style applies.
"""

from __future__ import annotations

import math
import re
import textwrap
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

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

from .config import DomainSpec
from .lensing import REGIONS
from .slicing import AXES, CoverageGrid, axis_range, coverage_grid, slice_region

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .bank import MapBank

Style = tuple[str, str]

#: seaborn's colourblind palette; every categorical colour below is one of its entries.
PALETTE: tuple[str, ...] = tuple(sns.color_palette("colorblind").as_hex())
GREY = PALETTE[7]
LIGHT_GREY = "0.8"
DARK_GREY = "0.2"

#: (colour, marker) of each category. Colours come from PALETTE in a fixed order, and
#: every category has its own marker, so identity never depends on colour alone.
REGION_STYLE: dict[str, Style] = {
    "minima": (PALETTE[0], "o"),
    "saddle": (PALETTE[1], "s"),
    "maxima": (PALETTE[2], "^"),
}
VALIDITY_STYLE: dict[str, Style] = {"valid": (PALETTE[0], "o"), "invalid": (PALETTE[3], "X")}
ORIGIN_STYLE: dict[str, Style] = {
    "build": (PALETTE[0], "o"),
    "fetch": (PALETTE[1], "^"),
    "legacy": (PALETTE[2], "s"),
}
STATUS_STYLE: dict[str, Style] = {
    "hit": (PALETTE[0], "o"),
    "miss": (PALETTE[3], "X"),
    "invalid_simplex": (PALETTE[4], "D"),
    "outside_hull": (GREY, "s"),
    "region_not_ready": (GREY, "v"),
    "outside_domain": (LIGHT_GREY, "."),
}
#: Styles of the most common error messages in ``plot_entries(hue="error")``; the rest are "Other".
ERROR_STYLE: tuple[Style, ...] = ((PALETTE[3], "X"), (PALETTE[4], "D"), (PALETTE[2], "P"))
OTHER_STYLE: Style = (GREY, ".")
QUANTILE_CMAP = "viridis"
MARGIN_CMAP = "vlag_r"
MARGIN_LABEL = "margin (threshold − interpolated distance)"

_TEX = {"kappa": r"\kappa", "gamma": r"\gamma", "s": "s"}
#: Query statuses drawn together as "outside domain".
_OUTSIDE = ("outside_domain", "critical_line", "malformed")
_HATCH = "///"
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_AT_POINT = re.compile(r"\s*at kappa=#, gamma=#(?:, s=#)?")


def _error_kind(message: str) -> str:
    """An error message without its numbers and failing point, so that failures group by cause."""
    return textwrap.shorten(_AT_POINT.sub("", _NUMBER.sub("#", message)), width=60, placeholder="…")


def _share(count: int, total: int) -> str:
    """``count / total`` as a whole percentage; a share that rounds to zero is shown as "<1%"."""
    text = f"{count / total:.0%}"
    return "<1%" if count and text == "0%" else text


def _label(name: str) -> str:
    return f"${_TEX[name]}$" if name in _TEX else name


def _axes(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _region_names(regions: str | Iterable[str]) -> list[str]:
    names = [regions] if isinstance(regions, str) else list(regions)
    unknown = sorted(set(names).difference(REGIONS))
    if not names or unknown:
        raise ValueError(f"regions must name at least one of {REGIONS}; got {names!r}.")
    return names


def _require(table: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"The table is missing columns {missing}.")


def _status_groups(status: Any) -> np.ndarray:
    """Query statuses, with critical-line and malformed queries counted as outside the domain."""
    values = np.asarray(status, dtype=str)
    return np.where(np.isin(values, _OUTSIDE), "outside_domain", values)


def _status_label(name: str) -> str:
    return name.replace("_", " ")


def _margin_norm(margins: np.ndarray, vmin: float | None, vmax: float | None) -> mcolors.TwoSlopeNorm:
    finite = margins[np.isfinite(margins)]
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    limit = limit if limit > 0.0 else 1.0
    return mcolors.TwoSlopeNorm(
        vcenter=0.0, vmin=-limit if vmin is None else vmin, vmax=limit if vmax is None else vmax
    )


def _legend(ax: Axes, extra: Sequence[Any] = ()) -> None:
    """A legend of the labelled artists plus ``extra`` handles, one entry per label."""
    handles, labels = ax.get_legend_handles_labels()
    entries = dict(zip(labels, handles, strict=True))
    for handle in extra:
        entries.setdefault(handle.get_label(), handle)
    if entries:
        ax.legend(list(entries.values()), list(entries.keys()), loc="best", fontsize="small")


def _vertical_lines(ax: Axes, positions: Iterable[float], label: str, style: dict[str, Any]) -> None:
    for number, position in enumerate(positions):
        ax.axvline(position, label=label if number == 0 else None, **style)


def _critical_curves(ax: Axes, domain: DomainSpec, axis: str, value: float | None) -> None:
    """The critical lines of infinite magnification (dashed) and the mu_max curves (solid) in a plane."""
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
    """Limits, labels, title and aspect of a plane through the domain."""
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

    Cells of tetrahedra with an invalid vertex are grey. Entries within ``entry_tolerance``
    of the plane (2 % of that axis's domain range by default) are drawn as markers.
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

    Give either ``bank`` with a plane, or a ``grid`` from ``coverage_grid``.
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
    """The group of every entry and the style of each group present, in legend order."""
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
    return pd.Categorical(labels, categories=list(present)), present


def _relabel(grid: Any) -> None:
    for ax in grid.axes.flat:
        if ax is not None:
            ax.set_xlabel(_label(ax.get_xlabel()))
            ax.set_ylabel(_label(ax.get_ylabel()))


def plot_entries(bank: MapBank, *, regions: str | Iterable[str] = REGIONS, hue: str = "validity") -> Any:
    """Corner pair plot of the entries in the kappa-gamma, kappa-s and gamma-s projections.

    ``hue`` is ``"validity"``, ``"origin"``, ``"region"``, ``"error"`` or ``"quantile"``.
    Returns the seaborn ``PairGrid``, which owns its figure.
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
