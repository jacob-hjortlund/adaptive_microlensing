import importlib
import sys

import numpy as np
import pandas as pd
import pytest
from adaptive_microlensing import StoppingCriteria, coverage_grid
from helpers import TETRAHEDRON, add_entry, batch_table

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


def test_plot_coverage_by_status(patchy_bank):
    """Statuses are listed with their shares, and unfinalized regions are hatched."""
    grid = coverage_grid(patchy_bank, s=0.5, n=30)
    ax = plotting.plot_coverage(grid=grid)
    labels = _legend_labels(ax)
    assert any(label.startswith("hit (") for label in labels)
    assert any(label.startswith("region not ready (") for label in labels)
    assert all(label.endswith("%)") for label in labels if label not in (INFINITE, FINITE))
    hatches = [list(getattr(collection, "hatches", None) or []) for collection in ax.collections]
    assert ["///"] in hatches
    patches = [
        handle for handle in ax.get_legend().legend_handles if handle.get_label().startswith("region not")
    ]
    assert [patch.get_hatch() for patch in patches] == ["///"]
    assert ax.get_title() == "$s = 0.5$"


def test_plot_coverage_by_margin_and_from_a_bank(built_bank, patchy_bank):
    """Margin mode colours hits and misses with a centred colourbar; a bank computes its own grid."""
    grid = coverage_grid(patchy_bank, s=0.5, n=20)
    ax = plotting.plot_coverage(grid=grid, color="margin")
    assert ax.figure.axes[-1].get_ylabel() == plotting.MARGIN_LABEL
    labels = _legend_labels(ax)
    assert not any(label.startswith(("hit (", "miss (")) for label in labels)
    ax = plotting.plot_coverage(built_bank, kappa=0.4, n=10)
    assert (ax.get_xlabel(), ax.get_ylabel()) == (r"$\gamma$", "$s$")


def test_figures_compose_into_one_paper_figure(patchy_bank):
    """A slice and a coverage map share one figure, each with its own colourbar."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    plotting.plot_slice(patchy_bank, s=0.5, ax=left)
    plotting.plot_coverage(patchy_bank, s=0.5, n=10, color="margin", ax=right)
    assert len(figure.axes) == 4
    assert figure.axes[2].get_ylabel() == QUANTILE_LABEL
    assert figure.axes[3].get_ylabel() == plotting.MARGIN_LABEL


def test_plot_coverage_arguments(built_bank):
    """Exactly one of bank and grid; a grid fixes the plane; color is status or margin."""
    grid = coverage_grid(built_bank, s=0.5, n=5)
    with pytest.raises(ValueError, match="either bank"):
        plotting.plot_coverage()
    with pytest.raises(ValueError, match="either bank"):
        plotting.plot_coverage(built_bank, grid=grid)
    with pytest.raises(ValueError, match="already fixes the plane"):
        plotting.plot_coverage(grid=grid, s=0.5)
    with pytest.raises(ValueError, match="color"):
        plotting.plot_coverage(grid=grid, color="quantile")


@pytest.mark.parametrize(
    ("hue", "labels"),
    [
        ("validity", ["valid", "invalid"]),
        ("error", ["valid", "MapGenerationError: Synthetic failure."]),
        ("region", ["minima", "saddle"]),
        ("origin", ["build"]),
    ],
)
def test_plot_entries_groups(patchy_bank, hue, labels):
    """The pair plot's legend lists the groups present, in the fixed order."""
    grid = plotting.plot_entries(patchy_bank, hue=hue)
    assert isinstance(grid, sns.PairGrid)
    assert [text.get_text() for text in grid.legend.get_texts()] == labels
    assert grid.axes[2, 0].get_xlabel() == r"$\kappa$"
    assert grid.axes[2, 0].get_ylabel() == "$s$"


def test_plot_entries_by_quantile_and_arguments(built_bank, patchy_bank):
    """Quantile hue adds a colourbar; unknown hues and regions without entries raise."""
    grid = plotting.plot_entries(patchy_bank, hue="quantile")
    assert grid.figure.axes[-1].get_ylabel() == QUANTILE_LABEL
    assert "invalid" in _legend_labels(grid.axes[1, 0])
    grid = plotting.plot_entries(built_bank, hue="error")
    assert [text.get_text() for text in grid.legend.get_texts()] == ["valid"]
    with pytest.raises(ValueError, match="hue"):
        plotting.plot_entries(built_bank, hue="seed")
    with pytest.raises(ValueError, match="no entries"):
        plotting.plot_entries(patchy_bank, regions=["maxima"])


@pytest.fixture(scope="module")
def query_table(patchy_bank):
    """query_many output for 400 random points over the domain box."""
    rng = np.random.default_rng(2)
    points = pd.DataFrame(
        {
            "kappa": rng.uniform(0.05, 2.0, 400),
            "gamma": rng.uniform(0.05, 2.0, 400),
            "s": rng.uniform(0.01, 0.99, 400),
        }
    )
    return patchy_bank.query_many(points)


def test_plot_queries_by_status(query_table, patchy_bank):
    """Each status group is labelled with its count; kappa-gamma with a domain gets curves."""
    ax = plotting.plot_queries(query_table, domain=patchy_bank.config.domain)
    groups = plotting._status_groups(query_table["interpolation_status"])
    expected = {
        f"{name.replace('_', ' ')} ({int((groups == name).sum())})"
        for name in plotting.STATUS_STYLE
        if (groups == name).any()
    }
    assert set(_legend_labels(ax)) == expected | {INFINITE, FINITE}
    assert ax.get_xlim() == patchy_bank.config.domain.kappa_range
    ax = plotting.plot_queries(query_table, x="kappa", y="s", domain=patchy_bank.config.domain)
    assert len(ax.lines) == 0 and (ax.get_xlabel(), ax.get_ylabel()) == (r"$\kappa$", "$s$")


def test_plot_queries_by_margin_and_arguments(query_table):
    """Margin mode adds a colourbar; bad axes, colours and missing columns raise."""
    ax = plotting.plot_queries(query_table, color="margin")
    assert ax.figure.axes[-1].get_ylabel() == plotting.MARGIN_LABEL
    assert "not covered" in _legend_labels(ax)
    with pytest.raises(ValueError, match="two different names"):
        plotting.plot_queries(query_table, x="kappa", y="kappa")
    with pytest.raises(ValueError, match="two different names"):
        plotting.plot_queries(query_table, x="mu")
    with pytest.raises(ValueError, match="color"):
        plotting.plot_queries(query_table, color="region")
    with pytest.raises(ValueError, match="interpolation_status"):
        plotting.plot_queries(query_table.drop(columns="interpolation_status"))
    with pytest.raises(ValueError, match="distance_margin"):
        plotting.plot_queries(query_table.drop(columns="distance_margin"), color="margin")


HIT_TABLE = pd.DataFrame(
    {
        "s": [0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.6, 0.9],
        "redshift": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
        "query_region": ["minima", "minima", "minima", "saddle", "minima", "minima", "maxima", None],
        "is_hit": [True, True, False, True, False, False, True, False],
        "interpolation_status": ["hit", "hit", "miss", "hit", "miss", "outside_hull", "hit", "critical_line"],
    }
)


@pytest.mark.parametrize(
    ("among", "expected", "ylabel"),
    [
        ("all", [[2 / 3, 0.0], [1.0, np.nan], [np.nan, 1.0], [0.75, 0.25]], "hit rate"),
        ("covered", [[2 / 3, 0.0], [1.0, np.nan], [np.nan, 1.0], [0.75, 0.5]], "hit rate (covered)"),
    ],
)
def test_plot_hit_rate_counts_by_hand(among, expected, ylabel):
    """Rates per bin match a hand count for minima, saddle, maxima and all queries."""
    ax = plotting.plot_hit_rate(HIT_TABLE, bins=[0.0, 0.5, 1.0], among=among)
    assert [line.get_label() for line in ax.lines] == ["minima", "saddle", "maxima", "all"]
    for line, rates in zip(ax.lines, expected, strict=True):
        np.testing.assert_allclose(line.get_xdata(), [0.25, 0.75])
        np.testing.assert_allclose(line.get_ydata(), rates)
    assert ax.get_ylabel() == ylabel and ax.get_ylim() == (0.0, 1.0)
    ax = plotting.plot_hit_rate(HIT_TABLE, bins=2)
    np.testing.assert_allclose(ax.lines[-1].get_xdata(), [0.3, 0.7])
    ax = plotting.plot_hit_rate(HIT_TABLE, by="redshift", bins=2)
    assert ax.get_xlabel() == "redshift"


def test_rows_without_a_value_are_left_out_of_hit_rates():
    """A row with NaN in the binned column, such as a malformed query, counts in no bin."""
    malformed = {"s": [np.nan], "redshift": [np.nan], "query_region": [None], "is_hit": [False]}
    table = pd.concat(
        [HIT_TABLE, pd.DataFrame({**malformed, "interpolation_status": ["malformed"]})], ignore_index=True
    )
    ax = plotting.plot_hit_rate(table, bins=[0.0, 0.5, 1.0])
    np.testing.assert_allclose(ax.lines[-1].get_ydata(), [0.75, 0.25])
    ax = plotting.plot_hit_rate(table, bins=2)
    np.testing.assert_allclose(ax.lines[-1].get_xdata(), [0.3, 0.7])


def test_query_figures_accept_fetch_many_tables(tetra_bank):
    """fetch_many output, with its extra columns and refused rows, draws like query_many output."""
    table = tetra_bank.fetch_many(batch_table())
    ax = plotting.plot_queries(table, domain=tetra_bank.config.domain)
    assert "outside domain (1)" in _legend_labels(ax)
    ax = plotting.plot_hit_rate(table, bins=2)
    assert [line.get_label() for line in ax.lines] == ["minima", "saddle", "all"]


def test_wilson_interval_and_hit_rate_arguments():
    """The Wilson interval matches a hand calculation; bad arguments raise."""
    low, high = plotting._wilson(np.array([5.0, 0.0]), np.array([10.0, 0.0]), 1.0)
    np.testing.assert_allclose([low[0], high[0]], [0.5 - 0.0275**0.5 / 1.1, 0.5 + 0.0275**0.5 / 1.1])
    assert np.isnan(low[1]) and np.isnan(high[1])
    for kwargs, message in (
        ({"among": "hits"}, "among"),
        ({"interval": 1.0}, "interval"),
        ({"bins": 0}, "bins"),
        ({"bins": [1.0, 0.0]}, "bins"),
        ({"by": "kappa"}, "kappa"),
    ):
        with pytest.raises(ValueError, match=message):
            plotting.plot_hit_rate(HIT_TABLE, **kwargs)


def test_plot_build_panels_and_goals(built_bank):
    """Three panels; goals are horizontal lines, the max-valid-points entry a vertical one."""
    stop = StoppingCriteria(max_valid_points=20, simplex_loss_goal=0.2, residual_goal=0.1)
    figure = plotting.plot_build(built_bank, "minima", stop=stop)
    loss_ax, residual_ax, valid_ax = figure.axes
    assert (loss_ax.get_yscale(), residual_ax.get_yscale()) == ("log", "log")
    assert loss_ax.get_title() == "minima"
    assert [ax.get_ylabel() for ax in figure.axes] == ["simplex loss", "relative residual", "valid fraction"]

    def lines(ax, label):
        return [line for line in ax.lines if line.get_label() == label]

    assert lines(loss_ax, "goal")[0].get_ydata()[0] == 0.2
    assert lines(residual_ax, "goal")[0].get_ydata()[0] == 0.1
    for ax in figure.axes:
        assert lines(ax, "max valid points")[0].get_xdata()[0] == 19


def test_the_residual_maximum_is_labelled_as_a_rolling_window(built_bank):
    """max_residual is the maximum of the last min_num_residuals residuals, not of every one so far."""
    figure = plotting.plot_build(built_bank, "minima")
    residual_ax = figure.axes[1]
    assert [line.get_label() for line in residual_ax.lines] == ["rolling maximum"]


def test_plot_build_rows_and_arguments(bank, built_bank, patchy_bank):
    """Build and legacy rows are plotted and fetch rows are not; bad axes and empty regions raise."""
    for point in TETRAHEDRON:
        add_entry(bank, "minima", point, origin="legacy")
    add_entry(bank, "minima", (0.2, 0.2, 0.5), origin="fetch")
    _, axes = plt.subplots(3)
    figure = plotting.plot_build(bank, "minima", axes=axes)
    assert figure is axes[0].figure
    np.testing.assert_array_equal(axes[2].lines[0].get_xdata(), [0, 1, 2, 3])
    with pytest.raises(ValueError, match="exactly three"):
        plotting.plot_build(built_bank, "minima", axes=axes[:2])
    with pytest.raises(ValueError, match="no build or legacy"):
        plotting.plot_build(patchy_bank, "maxima")


def test_figures_leave_rcparams_unchanged(built_bank, patchy_bank, query_table):
    """No function changes matplotlib's global settings."""
    before = dict(matplotlib.rcParams)
    plotting.plot_slice(patchy_bank, s=0.5)
    plotting.plot_coverage(patchy_bank, s=0.5, n=10, color="margin")
    plotting.plot_entries(patchy_bank, hue="error")
    plotting.plot_entries(built_bank, hue="quantile")
    plotting.plot_queries(query_table, color="margin")
    plotting.plot_hit_rate(query_table)
    plotting.plot_build(built_bank, "saddle")
    assert dict(matplotlib.rcParams) == before


def test_every_non_empty_bin_keeps_a_visible_interval():
    """A bin between empty bins still gets its Wilson band, spanning the bin."""
    table = pd.DataFrame(
        {
            "s": [0.05, 0.15, 0.55, 0.56, 0.95, 0.96],
            "query_region": ["minima", "minima", "maxima", "maxima", "minima", "minima"],
            "is_hit": [True, False, True, False, True, False],
            "interpolation_status": ["hit", "miss", "hit", "miss", "hit", "miss"],
        }
    )
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax = plotting.plot_hit_rate(table, bins=edges)
    for line, band in zip(ax.lines, ax.collections, strict=True):
        spans = [path.get_extents() for path in band.get_paths()]
        for i in np.flatnonzero(np.isfinite(np.asarray(line.get_ydata(), dtype=float))):
            assert any(
                span.x0 <= edges[i] + 1e-12 and span.x1 >= edges[i + 1] - 1e-12 and span.height > 0
                for span in spans
            ), (line.get_label(), i)


def test_the_stop_line_counts_every_valid_entry_like_the_build(bank):
    """After fetches extend a region, the max-valid-points line sits where a build would stop."""
    for point in TETRAHEDRON:
        add_entry(bank, "minima", point)
    for point in ((0.2, 0.2, 0.5), (0.3, 0.1, 0.4), (0.15, 0.25, 0.6)):
        add_entry(bank, "minima", point, origin="fetch")
    for point in ((0.25, 0.15, 0.3), (0.2, 0.3, 0.7)):
        add_entry(bank, "minima", point)
    figure = plotting.plot_build(bank, "minima", stop=StoppingCriteria(max_valid_points=8))
    for ax in figure.axes:
        marks = [line for line in ax.lines if line.get_label() == "max valid points"]
        assert [line.get_xdata()[0] for line in marks] == [7]


def test_rows_without_a_hit_flag_are_not_hits():
    """A NaN is_hit, as a left merge leaves on rows that were never queried, is not a hit."""
    unqueried = {"s": [0.6], "redshift": [1.8], "query_region": [None], "is_hit": [np.nan]}
    table = pd.concat(
        [HIT_TABLE, pd.DataFrame({**unqueried, "interpolation_status": [np.nan]})], ignore_index=True
    )
    ax = plotting.plot_hit_rate(table, bins=[0.0, 0.5, 1.0])
    np.testing.assert_allclose(ax.lines[-1].get_ydata(), [0.75, 0.2])
