import subprocess
import sys
import types

import numpy as np
import pytest
from adaptive_microlensing import (
    GeneratorSpec,
    IPMGenerator,
    MapGenerationError,
    MapGenerator,
    MapSpec,
    SyntheticGenerator,
    generator_from_spec,
    register_generator,
)
from adaptive_microlensing.maps import generator_identity, relative_magnitudes

SPEC = MapSpec(2.0, 0.05)


def test_importing_the_package_does_not_import_microlensing():
    """Importing the package does not import microlensing."""
    code = "import sys, adaptive_microlensing; print('microlensing' in sys.modules)"
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert output.stdout.strip() == "False"


def test_relative_magnitudes():
    """Magnifications become magnitudes relative to the macro magnification."""
    # kappa=0.5, gamma=0 gives mu_macro = 4.
    magnitudes = relative_magnitudes(np.array([4.0, 8.0, 0.0]), 0.5, 0.0)
    np.testing.assert_allclose(magnitudes[:2], [0.0, -2.5 * np.log10(2.0)])
    assert magnitudes[2] == np.inf
    with pytest.raises(MapGenerationError, match="singular"):
        relative_magnitudes(np.ones(3), 0.5, 0.5)


def test_registry_builds_generators_from_specs():
    """generator_from_spec builds registered generators and rejects unknown names."""
    synthetic = generator_from_spec(GeneratorSpec("synthetic", {"inf_fraction": 0.25}))
    assert isinstance(synthetic, SyntheticGenerator)
    assert synthetic.inf_fraction == 0.25
    ipm = generator_from_spec(GeneratorSpec())
    assert isinstance(ipm, IPMGenerator)
    assert ipm.options == {"rectangular": True}
    ipm.options["rectangular"] = False
    assert ipm.options == {"rectangular": True}
    assert isinstance(ipm, MapGenerator)
    with pytest.raises(ValueError, match="No map generator"):
        generator_from_spec(GeneratorSpec("nonexistent", {}))


def test_register_generator():
    """register_generator adds a name to the registry."""
    register_generator("constant-test", lambda options: SyntheticGenerator(**options))
    assert isinstance(generator_from_spec(GeneratorSpec("constant-test", {})), SyntheticGenerator)


def test_generator_identity_normalises_options():
    """Generator identity normalises options."""
    generator = SyntheticGenerator(failure_boxes=[((0, 1), (0, 1), (0, 1))])
    name, options = generator_identity(generator)
    assert name == "synthetic"
    assert options["failure_boxes"] == [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]


def test_synthetic_generator_is_deterministic_and_configurable():
    """Synthetic generator is deterministic and configurable."""
    generator = SyntheticGenerator(
        failure_boxes=[[[0.0, 0.2], [0.0, 0.2], [0.0, 1.0]]],
        inf_fraction=0.1,
        constant_boxes=[[[0.5, 0.6], [0.0, 0.2], [0.0, 1.0]]],
    )
    first = generator.generate(0.3, 0.1, 0.5, SPEC, seed=11)
    second = generator.generate(0.3, 0.1, 0.5, SPEC, seed=11)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (80, 80)
    assert np.isinf(first).sum() == 640
    assert not np.array_equal(first, generator.generate(0.3, 0.1, 0.5, SPEC, seed=12))
    with pytest.raises(MapGenerationError, match="Synthetic failure"):
        generator.generate(0.1, 0.1, 0.5, SPEC, seed=1)
    np.testing.assert_array_equal(generator.generate(0.55, 0.1, 0.5, SPEC, seed=1), 0.0)
    assert SyntheticGenerator(**generator.options).options == generator.options
    with pytest.raises(ValueError, match="three"):
        SyntheticGenerator(failure_boxes=[[[0.0, 1.0]]])
    with pytest.raises(ValueError, match="inf_fraction"):
        SyntheticGenerator(inf_fraction=1.0)


class _FakeIPM:
    """Stands in for microlensing.IPM.ipm.IPM; records its arguments."""

    calls: list[dict] = []
    fail = False

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)
        self.kwargs = kwargs

    def run(self):
        if type(self).fail:
            raise Exception("Error running IPM")
        n = self.kwargs["num_pixels_y1"]
        self.magnifications = np.full((n, n), 8.0, dtype=np.float32)


@pytest.fixture
def fake_ipm(monkeypatch):
    """Install a fake microlensing.IPM.ipm module for the duration of a test."""
    module = types.ModuleType("microlensing.IPM.ipm")
    module.IPM = _FakeIPM
    monkeypatch.setitem(sys.modules, "microlensing", types.ModuleType("microlensing"))
    monkeypatch.setitem(sys.modules, "microlensing.IPM", types.ModuleType("microlensing.IPM"))
    monkeypatch.setitem(sys.modules, "microlensing.IPM.ipm", module)
    _FakeIPM.calls = []
    _FakeIPM.fail = False
    return _FakeIPM


def test_ipm_generator_calls_ipm_like_the_original_scripts(fake_ipm):
    """IPMGenerator calls IPM with the arguments the original scripts used."""
    generator = IPMGenerator({"rectangular": True}, verbose=1)
    mag_map = generator.generate(0.5, 0.0, 0.25, SPEC, seed=99)
    (call,) = fake_ipm.calls
    assert call == {
        "kappa_tot": 0.5,
        "shear": 0.0,
        "kappa_star": 0.375,
        "half_length_y1": 2.0,
        "half_length_y2": 2.0,
        "num_pixels_y1": 80,
        "num_pixels_y2": 80,
        "random_seed": 99,
        "verbose": 1,
        "rectangular": True,
    }
    assert mag_map.dtype == np.float64
    np.testing.assert_allclose(mag_map, -2.5 * np.log10(2.0))


def test_ipm_generator_wraps_ipm_failures(fake_ipm):
    """IPM failures and singular points become MapGenerationError."""
    fake_ipm.fail = True
    with pytest.raises(MapGenerationError, match="Error running IPM"):
        IPMGenerator().generate(0.5, 0.1, 0.25, SPEC, seed=1)
    with pytest.raises(MapGenerationError, match="singular"):
        IPMGenerator().generate(0.5, 0.5, 0.25, SPEC, seed=1)


def test_ipm_generator_rejects_reserved_options():
    """IPMGenerator rejects options it sets itself."""
    with pytest.raises(ValueError, match="random_seed"):
        IPMGenerator({"random_seed": 3})
    version = IPMGenerator().version()
    assert version is None or isinstance(version, str)
