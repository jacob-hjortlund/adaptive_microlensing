import importlib
import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
sns = pytest.importorskip("seaborn")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")
plotting = importlib.import_module("adaptive_microlensing.plotting")

QUANTILE_LABEL = "95% quantile of the MPD distance"
INFINITE, FINITE = r"$|\mu| = \infty$", r"$|\mu| = \mu_{\max}$"


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test made."""
    yield
    plt.close("all")


def test_styles_come_from_the_colorblind_palette():
    """Every categorical colour is the documented entry of seaborn's colourblind palette."""
    palette = tuple(sns.color_palette("colorblind").as_hex())
    assert palette == plotting.PALETTE
    assert [color for color, _ in plotting.REGION_STYLE.values()] == [palette[0], palette[1], palette[2]]
    assert [color for color, _ in plotting.VALIDITY_STYLE.values()] == [palette[0], palette[3]]
    assert [color for color, _ in plotting.ORIGIN_STYLE.values()] == [palette[0], palette[1], palette[2]]
    status = {name: color for name, (color, _) in plotting.STATUS_STYLE.items()}
    assert (status["hit"], status["miss"], status["invalid_simplex"]) == (palette[0], palette[3], palette[4])
    assert status["outside_hull"] == status["region_not_ready"] == palette[7]
    assert [color for color, _ in plotting.ERROR_STYLE] == [palette[3], palette[4], palette[2]]
    for styles in (
        plotting.REGION_STYLE,
        plotting.VALIDITY_STYLE,
        plotting.ORIGIN_STYLE,
        plotting.STATUS_STYLE,
    ):
        markers = [marker for _, marker in styles.values()]
        assert len(set(markers)) == len(markers)


def test_import_without_matplotlib_names_the_extra(monkeypatch):
    """Importing plotting without matplotlib raises an ImportError with the install hint."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "adaptive_microlensing.plotting")
    with pytest.raises(ImportError, match=r"pip install 'adaptive_microlensing\[plot\]'"):
        importlib.import_module("adaptive_microlensing.plotting")


def _curve_points(line, axis, value):
    """The (kappa, gamma) points of a drawn curve."""
    x = np.asarray(line.get_xdata(), dtype=float)
    y = np.asarray(line.get_ydata(), dtype=float)
    if axis == "s":
        return x, y
    if axis == "kappa":
        return np.full_like(x, value), x
    return x, np.full_like(x, value)


@pytest.mark.parametrize(
    ("axis", "value", "n_infinite", "n_finite"),
    [("s", 0.5, 1, 3), ("kappa", 0.3, 1, 2), ("kappa", 1.0, 0, 1), ("gamma", 0.3, 2, 4)],
)
def test_critical_curves_satisfy_their_equations(patchy_bank, axis, value, n_infinite, n_finite):
    """Dashed curves have |mu| = infinity and solid ones |mu| = mu_max, in every orientation."""
    domain = patchy_bank.config.domain
    eps = 1.0 / domain.max_macro_magnification
    _, ax = plt.subplots()
    plotting._critical_curves(ax, domain, axis, value)
    dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
    solid = [line for line in ax.lines if line.get_linestyle() == "-"]
    assert (len(dashed), len(solid)) == (n_infinite, n_finite)
    for lines, target in ((dashed, 0.0), (solid, eps)):
        for line in lines:
            kappa, gamma = _curve_points(line, axis, value)
            np.testing.assert_allclose(np.abs((1.0 - kappa) ** 2 - gamma**2), target, atol=1e-12)
    labels = {line.get_label() for line in ax.lines}
    assert (INFINITE in labels) == (n_infinite > 0) and FINITE in labels


def test_share_labels():
    """Shares are whole percentages, and a nonzero share that rounds to zero reads <1%."""
    assert plotting._share(3, 4) == "75%"
    assert plotting._share(1, 400) == "<1%"
    assert plotting._share(2, 400) == "<1%"
    assert plotting._share(0, 4) == "0%"


def test_error_kinds_group_by_cause():
    """Numbers and the failing point are removed, so one cause is one group."""
    cases = {
        "MapGenerationError: Synthetic failure at kappa=0.21, gamma=0.3, s=0.47.": (
            "MapGenerationError: Synthetic failure."
        ),
        "MapGenerationError: IPM failed at kappa=0.3, gamma=0.2, s=0.5: taylor_smooth must be <= 101": (
            "MapGenerationError: IPM failed: taylor_smooth must be <= #"
        ),
        "InvalidMapError: The coarse map is constant.": "InvalidMapError: The coarse map is constant.",
    }
    for message, kind in cases.items():
        assert plotting._error_kind(message) == kind
    assert len(plotting._error_kind("x" * 200)) <= 60


def _legend_labels(ax):
    legend = ax.get_legend()
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


def test_plot_slice_draws_every_layer(patchy_bank):
    """Shaded valid cells, grey invalid cells, mesh, entries, curves, colourbar and legend."""
    ax = plt.figure().add_subplot()
    assert plotting.plot_slice(patchy_bank, s=0.5, ax=ax, entry_tolerance=1.0) is ax
    kinds = {type(artist).__name__ for artist in ax.collections}
    assert {"TriMesh", "PolyCollection", "PathCollection"} <= kinds
    assert ax.get_title() == "$s = 0.5$"
    assert (ax.get_xlabel(), ax.get_ylabel()) == (r"$\kappa$", r"$\gamma$")
    assert ax.get_xlim() == patchy_bank.config.domain.kappa_range
    assert ax.get_aspect() == 1.0
    assert set(_legend_labels(ax)) == {"valid entry", "invalid entry", INFINITE, FINITE}
    assert ax.figure.axes[-1].get_ylabel() == QUANTILE_LABEL


def test_plot_slice_options_and_orientations(built_bank):
    """Layers can be switched off; kappa slices put gamma and s on the axes."""
    ax = plotting.plot_slice(
        built_bank, kappa=0.4, mesh=False, entries=False, curves=False, colorbar=False, legend=False
    )
    assert ax.get_title() == r"$\kappa = 0.4$"
    assert (ax.get_xlabel(), ax.get_ylabel()) == (r"$\gamma$", "$s$")
    assert ax.get_aspect() == "auto"
    assert ax.get_legend() is None and len(ax.lines) == 0 and len(ax.figure.axes) == 1
    plotting.plot_slice(built_bank, gamma=0.3, regions="saddle", vmin=0.0, vmax=1.0)
    with pytest.raises(ValueError, match="vmin <= vmax"):
        plotting.plot_slice(built_bank, s=0.5, vmin=1.0, vmax=0.0)
    with pytest.raises(ValueError, match="regions"):
        plotting.plot_slice(built_bank, s=0.5, regions=[])
    with pytest.raises(ValueError, match="regions"):
        plotting.plot_slice(built_bank, s=0.5, regions=["ridge"])
