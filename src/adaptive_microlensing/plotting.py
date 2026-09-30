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
except ImportError as exc:
    raise ImportError(
        "adaptive_microlensing.plotting needs matplotlib and seaborn; install them with "
        "pip install 'adaptive_microlensing[plot]'."
    ) from exc

from .config import DomainSpec
from .lensing import REGIONS
from .slicing import axis_range

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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
