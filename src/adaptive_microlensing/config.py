"""Configuration dataclasses for map banks.

Every class is a frozen dataclass that validates itself when it is constructed.
``BankConfig`` round-trips through JSON-compatible dictionaries.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


def _positive_integer(value: float, name: str) -> int:
    """Return ``value`` as an int if it is a positive integer within a relative 1e-9."""
    rounded = round(value)
    if rounded < 1 or not math.isclose(value, rounded, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(rounded)


def _finite_positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    return number


def _finite_non_negative(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}.")
    return number


def _integer_at_least(value: Any, minimum: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def _range(
    value: Sequence[float],
    name: str,
    lowest: float | None = None,
    highest: float | None = None,
) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value!r}.")
    lower, upper = (float(v) for v in value)
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
        raise ValueError(f"{name} must be finite with lower < upper, got {value!r}.")
    if lowest is not None and lower < lowest:
        raise ValueError(f"{name} must not extend below {lowest}, got {value!r}.")
    if highest is not None and upper > highest:
        raise ValueError(f"{name} must not extend above {highest}, got {value!r}.")
    return (lower, upper)


@dataclass(frozen=True)
class MapSpec:
    """Geometry of a square magnitude map, in Einstein radii.

    Parameters
    ----------
    half_length : float
        Half of the map's side length.
    pixel_scale : float
        Side length of one pixel.
    """

    half_length: float
    pixel_scale: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "half_length", _finite_positive(self.half_length, "MapSpec.half_length"))
        object.__setattr__(self, "pixel_scale", _finite_positive(self.pixel_scale, "MapSpec.pixel_scale"))
        _positive_integer(self.width / self.pixel_scale, "MapSpec width / pixel_scale")

    @property
    def width(self) -> float:
        """Side length of the map."""
        return 2.0 * self.half_length

    @property
    def num_pixels(self) -> int:
        """Number of pixels along each side."""
        return _positive_integer(self.width / self.pixel_scale, "MapSpec width / pixel_scale")


@dataclass(frozen=True)
class VariabilitySpec:
    """How the intrinsic MPD-distance quantile is measured on a coarse map.

    The coarse map is split into non-overlapping square windows of side
    ``2 * window_half_length``. The quantile is taken over the Jensen-Shannon
    distances between the MPDs of all window pairs.
    """

    map: MapSpec = field(default_factory=lambda: MapSpec(80.0, 0.1))
    window_half_length: float = 10.0
    quantile: float = 0.95
    n_bin_edges: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.map, MapSpec):
            raise TypeError(f"VariabilitySpec.map must be a MapSpec, got {type(self.map).__name__}.")
        window = _finite_positive(self.window_half_length, "VariabilitySpec.window_half_length")
        object.__setattr__(self, "window_half_length", window)
        _positive_integer(2.0 * window / self.map.pixel_scale, "VariabilitySpec window width in pixels")
        if self.windows_per_axis < 2:
            raise ValueError("VariabilitySpec needs at least 2 windows along each axis of the coarse map.")
        quantile = float(self.quantile)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"VariabilitySpec.quantile must lie in [0, 1], got {self.quantile!r}.")
        object.__setattr__(self, "quantile", quantile)
        object.__setattr__(
            self, "n_bin_edges", _integer_at_least(self.n_bin_edges, 2, "VariabilitySpec.n_bin_edges")
        )

    @property
    def window_pixels(self) -> int:
        """Side length of one window, in pixels."""
        return _positive_integer(
            2.0 * self.window_half_length / self.map.pixel_scale, "VariabilitySpec window width in pixels"
        )

    @property
    def windows_per_axis(self) -> int:
        """Number of windows along each axis of the coarse map."""
        return _positive_integer(
            self.map.width / (2.0 * self.window_half_length), "VariabilitySpec map width / window width"
        )


@dataclass(frozen=True)
class DomainSpec:
    """The parameter region a bank is designed to cover.

    A point is in the domain when it lies in the closed kappa/gamma/s box and its
    macro magnification satisfies ``|mu| <= max_macro_magnification``.
    """

    kappa_range: tuple[float, float] = (0.05, 2.0)
    gamma_range: tuple[float, float] = (0.05, 2.0)
    s_range: tuple[float, float] = (0.01, 0.99)
    max_macro_magnification: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kappa_range", _range(self.kappa_range, "DomainSpec.kappa_range", lowest=0.0)
        )
        object.__setattr__(
            self, "gamma_range", _range(self.gamma_range, "DomainSpec.gamma_range", lowest=0.0)
        )
        object.__setattr__(
            self, "s_range", _range(self.s_range, "DomainSpec.s_range", lowest=0.0, highest=1.0)
        )
        object.__setattr__(
            self,
            "max_macro_magnification",
            _finite_positive(self.max_macro_magnification, "DomainSpec.max_macro_magnification"),
        )


@dataclass(frozen=True)
class DesignSpec:
    """Settings of the adaptive design used by ``MapBank.build``."""

    n_boundary: int = 6
    exploration: float = 0.05
    boundary_multiplier: float = 2.0
    invalid_multiplier: float = 0.0
    adjacent_boundary_multiplier: float | None = None
    validity_threshold: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_boundary", _integer_at_least(self.n_boundary, 2, "DesignSpec.n_boundary"))
        for name in ("exploration", "boundary_multiplier", "invalid_multiplier"):
            object.__setattr__(self, name, _finite_non_negative(getattr(self, name), f"DesignSpec.{name}"))
        if self.adjacent_boundary_multiplier is not None:
            object.__setattr__(
                self,
                "adjacent_boundary_multiplier",
                _finite_non_negative(
                    self.adjacent_boundary_multiplier, "DesignSpec.adjacent_boundary_multiplier"
                ),
            )
        threshold = float(self.validity_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"DesignSpec.validity_threshold must lie in (0, 1), got {self.validity_threshold!r}."
            )
        object.__setattr__(self, "validity_threshold", threshold)


@dataclass(frozen=True)
class GeneratorSpec:
    """Which map code makes a bank's maps, and with which options.

    ``options`` are normalised through JSON, so tuples become lists.
    """

    name: str = "ipm"
    options: Mapping[str, Any] = field(default_factory=lambda: {"rectangular": True})

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"GeneratorSpec.name must be a non-empty string, got {self.name!r}.")
        try:
            options = json.loads(json.dumps(dict(self.options), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"GeneratorSpec.options must be JSON-serialisable: {exc}") from exc
        object.__setattr__(self, "options", options)

    @classmethod
    def from_generator(cls, generator: Any) -> GeneratorSpec:
        """The spec that identifies ``generator``: its name and options."""
        return cls(name=generator.name, options=dict(generator.options))


@dataclass(frozen=True)
class BankConfig:
    """Complete description of how a bank's entries are made.

    The defaults reproduce the settings of the original adaptive_mpd runs.
    """

    domain: DomainSpec = field(default_factory=DomainSpec)
    variability: VariabilitySpec = field(default_factory=VariabilitySpec)
    bank_map: MapSpec = field(default_factory=lambda: MapSpec(20.0, 0.01))
    n_bin_edges: int = 100
    seed: int = 42
    design: DesignSpec = field(default_factory=DesignSpec)
    generator: GeneratorSpec = field(default_factory=GeneratorSpec)

    def __post_init__(self) -> None:
        for name, kind in (
            ("domain", DomainSpec),
            ("variability", VariabilitySpec),
            ("bank_map", MapSpec),
            ("design", DesignSpec),
            ("generator", GeneratorSpec),
        ):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"BankConfig.{name} must be a {kind.__name__}.")
        object.__setattr__(
            self, "n_bin_edges", _integer_at_least(self.n_bin_edges, 2, "BankConfig.n_bin_edges")
        )
        object.__setattr__(self, "seed", _integer_at_least(self.seed, 0, "BankConfig.seed"))

    @property
    def n_mpd_columns(self) -> int:
        """Length of a bank MPD: ``n_bin_edges - 1`` bins plus two overflow bins."""
        return self.n_bin_edges + 1

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dictionary; tuples become lists."""
        result: dict[str, Any] = json.loads(json.dumps(asdict(self), allow_nan=False))
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BankConfig:
        """Rebuild a config written by :meth:`to_dict`."""
        variability = dict(data["variability"])
        return cls(
            domain=DomainSpec(**data["domain"]),
            variability=VariabilitySpec(map=MapSpec(**variability.pop("map")), **variability),
            bank_map=MapSpec(**data["bank_map"]),
            n_bin_edges=data["n_bin_edges"],
            seed=data["seed"],
            design=DesignSpec(**data["design"]),
            generator=GeneratorSpec(**data["generator"]),
        )


@dataclass(frozen=True)
class StoppingCriteria:
    """When ``MapBank.build`` stops adding entries to a region.

    The build stops as soon as any criterion that is set holds.
    """

    max_valid_points: int | None = 500
    simplex_loss_goal: float | None = None
    residual_goal: float | None = None
    min_num_residuals: int = 100

    def __post_init__(self) -> None:
        if self.max_valid_points is None and self.simplex_loss_goal is None and self.residual_goal is None:
            raise ValueError(
                "StoppingCriteria needs at least one of max_valid_points, simplex_loss_goal or residual_goal."
            )
        if self.max_valid_points is not None:
            object.__setattr__(
                self,
                "max_valid_points",
                _integer_at_least(self.max_valid_points, 1, "StoppingCriteria.max_valid_points"),
            )
        for name in ("simplex_loss_goal", "residual_goal"):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self, name, _finite_positive(getattr(self, name), f"StoppingCriteria.{name}")
                )
        object.__setattr__(
            self,
            "min_num_residuals",
            _integer_at_least(self.min_num_residuals, 1, "StoppingCriteria.min_num_residuals"),
        )
