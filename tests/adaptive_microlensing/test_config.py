import json

import pytest
from adaptive_microlensing import (
    BankConfig,
    DesignSpec,
    DomainSpec,
    GeneratorSpec,
    MapSpec,
    StoppingCriteria,
    VariabilitySpec,
)


def test_defaults_reproduce_the_original_runs():
    """The defaults are the settings of run.sh and run_map_gen.sh."""
    config = BankConfig()
    assert config.domain == DomainSpec((0.05, 2.0), (0.05, 2.0), (0.01, 0.99), 100.0)
    assert config.variability.map == MapSpec(80.0, 0.1)
    assert config.variability.map.num_pixels == 1600
    assert config.variability.window_half_length == 10.0
    assert config.variability.windows_per_axis == 8
    assert config.variability.window_pixels == 200
    assert config.variability.quantile == 0.95
    assert config.variability.n_bin_edges == 100
    assert config.bank_map == MapSpec(20.0, 0.01)
    assert config.bank_map.num_pixels == 4000
    assert config.n_bin_edges == 100
    assert config.n_mpd_columns == 101
    assert config.seed == 42
    assert config.design == DesignSpec(6, 0.05, 2.0, 0.0, None, 0.5)
    assert config.generator == GeneratorSpec("ipm", {"rectangular": True})
    assert StoppingCriteria() == StoppingCriteria(500, None, None, 100)


def test_config_round_trips_through_json():
    """A config survives a JSON round trip."""
    config = BankConfig(seed=7, generator=GeneratorSpec("synthetic", {"inf_fraction": 0.1}))
    restored = BankConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert isinstance(restored.domain.kappa_range, tuple)


@pytest.mark.parametrize(
    ("half_length", "pixel_scale"),
    [(0.0, 0.1), (-1.0, 0.1), (1.0, 0.0), (float("inf"), 0.1), (1.0, 0.3)],
)
def test_map_spec_rejects_bad_geometry(half_length, pixel_scale):
    """MapSpec rejects non-positive, infinite and non-integral geometry."""
    with pytest.raises(ValueError, match="MapSpec"):
        MapSpec(half_length, pixel_scale)


def test_variability_spec_rejects_bad_windows():
    """VariabilitySpec rejects windows that do not tile the coarse map."""
    with pytest.raises(ValueError, match="window width in pixels"):
        VariabilitySpec(map=MapSpec(4.0, 0.1), window_half_length=1.025)
    with pytest.raises(ValueError, match="at least 2 windows"):
        VariabilitySpec(map=MapSpec(4.0, 0.1), window_half_length=4.0)
    with pytest.raises(ValueError, match="map width / window width"):
        VariabilitySpec(map=MapSpec(4.0, 0.1), window_half_length=1.5)
    with pytest.raises(ValueError, match="quantile"):
        VariabilitySpec(quantile=1.5)
    with pytest.raises(ValueError, match="n_bin_edges"):
        VariabilitySpec(n_bin_edges=1)
    with pytest.raises(TypeError, match="MapSpec"):
        VariabilitySpec(map={"half_length": 80.0, "pixel_scale": 0.1})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kappa_range": (2.0, 1.0)},
        {"kappa_range": (-0.1, 1.0)},
        {"gamma_range": (0.0, float("nan"))},
        {"s_range": (0.0, 1.5)},
        {"s_range": (0.5,)},
        {"max_macro_magnification": 0.0},
    ],
)
def test_domain_spec_rejects_bad_values(kwargs):
    """DomainSpec rejects empty, negative and out-of-range intervals."""
    with pytest.raises(ValueError, match="DomainSpec"):
        DomainSpec(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_boundary": 1},
        {"exploration": -1.0},
        {"adjacent_boundary_multiplier": -2.0},
        {"validity_threshold": 1.0},
    ],
)
def test_design_spec_rejects_bad_values(kwargs):
    """DesignSpec rejects invalid loss settings."""
    with pytest.raises(ValueError, match="DesignSpec"):
        DesignSpec(**kwargs)


def test_generator_spec_normalises_options():
    """GeneratorSpec normalises options through JSON and rejects bad names and options."""
    spec = GeneratorSpec("synthetic", {"failure_boxes": ((0, 1), (0, 1), (0, 1))})
    assert spec.options == {"failure_boxes": [[0, 1], [0, 1], [0, 1]]}
    with pytest.raises(ValueError, match="JSON"):
        GeneratorSpec("synthetic", {"bad": object()})
    with pytest.raises(ValueError, match="name"):
        GeneratorSpec("", {})


def test_bank_config_rejects_bad_values():
    """BankConfig rejects bad edge counts, seeds and field types."""
    with pytest.raises(ValueError, match="n_bin_edges"):
        BankConfig(n_bin_edges=1)
    with pytest.raises(ValueError, match="seed"):
        BankConfig(seed=-1)
    with pytest.raises(TypeError, match="bank_map"):
        BankConfig(bank_map=(20.0, 0.01))


def test_stopping_criteria_validation():
    """StoppingCriteria needs at least one valid criterion."""
    with pytest.raises(ValueError, match="at least one"):
        StoppingCriteria(max_valid_points=None)
    with pytest.raises(ValueError, match="max_valid_points"):
        StoppingCriteria(max_valid_points=0)
    with pytest.raises(ValueError, match="residual_goal"):
        StoppingCriteria(residual_goal=-1.0)
    with pytest.raises(ValueError, match="min_num_residuals"):
        StoppingCriteria(min_num_residuals=0)
    assert StoppingCriteria(max_valid_points=None, simplex_loss_goal=0.1).simplex_loss_goal == 0.1
