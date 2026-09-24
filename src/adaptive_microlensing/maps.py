"""Map generators: the interface between a bank and the code that makes maps."""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .config import GeneratorSpec, MapSpec
from .errors import MapGenerationError
from .synthetic import SyntheticGenerator

# IPM keyword arguments that IPMGenerator sets itself.
_RESERVED_IPM_OPTIONS = frozenset(
    {
        "kappa_tot",
        "shear",
        "smooth_fraction",
        "kappa_star",
        "half_length_y1",
        "half_length_y2",
        "num_pixels_y1",
        "num_pixels_y2",
        "random_seed",
        "verbose",
    }
)


@runtime_checkable
class MapGenerator(Protocol):
    """Anything that makes magnitude maps for a bank.

    ``generate`` returns magnitudes relative to the macro magnification,
    ``-2.5 * log10(|mu_pixel| / |mu_macro|)`` with ``mu_macro = 1 / ((1 - kappa)**2 - gamma**2)``,
    as a float array of shape ``(spec.num_pixels, spec.num_pixels)``. Pixels with zero
    magnification may be ``+inf``. Failures of the underlying code are raised as
    ``MapGenerationError``.
    """

    @property
    def name(self) -> str:
        """Registered name, recorded in ``bank.json``."""

    @property
    def options(self) -> Mapping[str, Any]:
        """Options that change the maps, recorded in ``bank.json``."""

    def version(self) -> str | None:
        """Version of the underlying code, recorded with every entry."""

    def generate(self, kappa: float, gamma: float, s: float, spec: MapSpec, seed: int) -> np.ndarray:
        """Make one magnitude map."""


def generator_identity(generator: MapGenerator) -> tuple[str, dict[str, Any]]:
    """The (name, options) pair that a bank compares, with options normalised through JSON."""
    return generator.name, json.loads(json.dumps(dict(generator.options)))


def relative_magnitudes(magnifications: np.ndarray, kappa: float, gamma: float) -> np.ndarray:
    """Convert pixel magnifications to magnitudes relative to the macro magnification."""
    denominator = (1.0 - kappa) ** 2 - gamma**2
    if denominator == 0.0:
        raise MapGenerationError(f"Macro magnification is singular at kappa={kappa}, gamma={gamma}.")
    mu_macro = abs(1.0 / denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(np.abs(np.asarray(magnifications, dtype=np.float64)) / mu_macro)


class IPMGenerator:
    """Make maps with the GPU inverse-polygon-mapping code of the ``microlensing`` package.

    Parameters
    ----------
    options : mapping, optional
        Extra keyword arguments for ``microlensing.IPM.ipm.IPM``, such as
        ``{"rectangular": True}``. They are part of the generator's identity.
    verbose : int, optional
        IPM's verbosity level. It is not part of the identity.
    """

    name = "ipm"

    def __init__(self, options: Mapping[str, Any] | None = None, verbose: int = 0) -> None:
        self._options = dict(options or {})
        reserved = sorted(_RESERVED_IPM_OPTIONS.intersection(self._options))
        if reserved:
            raise ValueError(f"IPMGenerator sets {reserved} itself; remove them from options.")
        self.verbose = int(verbose)

    @property
    def options(self) -> dict[str, Any]:
        """The extra IPM keyword arguments, as a copy; they are part of the generator's identity."""
        return dict(self._options)

    def version(self) -> str | None:
        """Installed version of ``microlensing``, or None if it is not installed."""
        try:
            return importlib.metadata.version("microlensing")
        except importlib.metadata.PackageNotFoundError:
            return None

    def generate(self, kappa: float, gamma: float, s: float, spec: MapSpec, seed: int) -> np.ndarray:
        """Run IPM once and return the relative-magnitude map."""
        if (1.0 - kappa) ** 2 - gamma**2 == 0.0:
            raise MapGenerationError(f"Macro magnification is singular at kappa={kappa}, gamma={gamma}.")
        try:
            from microlensing.IPM.ipm import IPM
        except ImportError as exc:
            raise ImportError(
                "IPMGenerator needs the optional 'microlensing' package: "
                "pip install 'adaptive_microlensing[ipm]'."
            ) from exc
        n = spec.num_pixels
        try:
            ipm = IPM(
                kappa_tot=kappa,
                shear=gamma,
                kappa_star=(1.0 - s) * kappa,
                half_length_y1=spec.half_length,
                half_length_y2=spec.half_length,
                num_pixels_y1=n,
                num_pixels_y2=n,
                random_seed=seed,
                verbose=self.verbose,
                **self._options,
            )
            ipm.run()
            magnifications = np.array(ipm.magnifications, dtype=np.float64)
        except Exception as exc:
            raise MapGenerationError(f"IPM failed at kappa={kappa}, gamma={gamma}, s={s}: {exc}") from exc
        mag_map = relative_magnitudes(magnifications, kappa, gamma)
        if mag_map.shape != (n, n):
            raise ValueError(f"IPM returned a map of shape {mag_map.shape}; expected {(n, n)}.")
        return mag_map


GeneratorFactory = Callable[[Mapping[str, Any]], MapGenerator]
_GENERATORS: dict[str, GeneratorFactory] = {}


def register_generator(name: str, factory: GeneratorFactory) -> None:
    """Let :func:`generator_from_spec` build generators called ``name``.

    ``factory`` receives the options stored in the bank and returns a generator.
    """
    _GENERATORS[name] = factory


def generator_from_spec(spec: GeneratorSpec) -> MapGenerator:
    """Build the generator that a bank's ``GeneratorSpec`` describes."""
    try:
        factory = _GENERATORS[spec.name]
    except KeyError:
        raise ValueError(
            f"No map generator is registered as {spec.name!r}; registered: {sorted(_GENERATORS)}. "
            "Use register_generator() to add one."
        ) from None
    return factory(spec.options)


register_generator("ipm", IPMGenerator)
register_generator("synthetic", lambda options: SyntheticGenerator(**options))
