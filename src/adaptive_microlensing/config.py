"""Configuration dataclasses for map banks.

Every class is a frozen dataclass that validates itself when it is constructed and
converts its fields to canonical types (floats, ints, tuples and JSON-normalised
dicts), so a bad setting fails at once with a :exc:`ValueError` or :exc:`TypeError`
rather than part-way through a build.

:class:`BankConfig` is the complete description of how a bank's entries are made. It
combines the parameter domain (:class:`DomainSpec`), the coarse map on which each
entry's variability is measured (:class:`VariabilitySpec`), the geometry of the bank
maps (:class:`MapSpec`), the adaptive design (:class:`DesignSpec`) and the map code
(:class:`GeneratorSpec`). ``BankConfig`` round-trips through JSON-compatible
dictionaries: :meth:`MapBank.create` stores :meth:`BankConfig.to_dict` in
``bank.json``, and :meth:`MapBank.open` reads it back with :meth:`BankConfig.from_dict`.
The defaults are the settings of the original adaptive_mpd runs, except that the MPDs have
100 bins where the original runs had 100 bin edges; :data:`legacy.LEGACY_CONFIG` holds the
original settings exactly.

:class:`StoppingCriteria` is not stored with the bank; it is passed to
:meth:`MapBank.build` to say when a region has enough entries.

The module depends only on the standard library. :mod:`adaptive_microlensing.lensing`,
:mod:`adaptive_microlensing.mpd`, :mod:`adaptive_microlensing.maps`,
:mod:`adaptive_microlensing.design` and :mod:`adaptive_microlensing.bank` read these
settings.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


def _positive_integer(value: float, name: str) -> int:
    """Return ``value`` as an int if it is a positive integer within a relative 1e-9.

    Used to check that a length holds a whole number of pixels or windows, allowing for
    rounding in the floating-point division that produced ``value``.

    Parameters
    ----------
    value : float
        Number to check, such as a map width divided by a pixel scale.
    name : str
        Name of the quantity, used in the error message.

    Returns
    -------
    int
        ``value`` rounded to the nearest integer.

    Raises
    ------
    ValueError
        If ``value`` rounds to less than 1, or differs from the nearest integer by more
        than a relative 1e-9.
    """
    rounded = round(value)
    if rounded < 1 or not math.isclose(value, rounded, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(rounded)


def _finite_positive(value: Any, name: str) -> float:
    """Return ``value`` as a float after checking that it is positive and finite.

    Parameters
    ----------
    value : Any
        Number to check; anything that ``float`` accepts.
    name : str
        Qualified field name, used in the error message.

    Returns
    -------
    float
        ``value`` converted to a float.

    Raises
    ------
    ValueError
        If ``value`` is zero, negative, infinite or NaN.
    """
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    return number


def _finite_non_negative(value: Any, name: str) -> float:
    """Return ``value`` as a float after checking that it is non-negative and finite.

    Parameters
    ----------
    value : Any
        Number to check; anything that ``float`` accepts.
    name : str
        Qualified field name, used in the error message.

    Returns
    -------
    float
        ``value`` converted to a float.

    Raises
    ------
    ValueError
        If ``value`` is negative, infinite or NaN.
    """
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}.")
    return number


def _integer_at_least(value: Any, minimum: int, name: str) -> int:
    """Return ``value`` as an int after checking that it is an integer of at least ``minimum``.

    Integral floats such as ``6.0`` are accepted and converted; booleans are rejected.

    Parameters
    ----------
    value : Any
        Number to check.
    minimum : int
        Smallest allowed value.
    name : str
        Qualified field name, used in the error message.

    Returns
    -------
    int
        ``value`` converted to an int.

    Raises
    ------
    ValueError
        If ``value`` is a bool, is not equal to an integer, or is less than ``minimum``.
    """
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def _hashable(value: Any) -> Any:
    """Return a JSON-normalised value as nested tuples that compare and hash like it.

    Dicts become tuples of ``(key, value)`` pairs sorted by key, and lists become tuples.
    Numbers are kept as they are, so values that compare equal, such as ``1``, ``1.0`` and
    ``True``, also hash equally, which JSON text would not do.

    Parameters
    ----------
    value : Any
        A value made of dicts with string keys, lists, strings, numbers, booleans and
        ``None``, as JSON normalisation gives.

    Returns
    -------
    Any
        The hashable equivalent of ``value``.
    """
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    return value


def _range(
    value: Sequence[float],
    name: str,
    lowest: float | None = None,
    highest: float | None = None,
) -> tuple[float, float]:
    """Return a closed interval as a tuple of two floats after checking it.

    Parameters
    ----------
    value : sequence of float
        The lower and the upper end of the interval, in any two-element sequence such as
        a tuple or a JSON list.
    name : str
        Qualified field name, used in the error messages.
    lowest : float or None, optional
        Smallest allowed lower end. When ``None``, the lower end is not bounded.
    highest : float or None, optional
        Largest allowed upper end. When ``None``, the upper end is not bounded.

    Returns
    -------
    lower : float
        Lower end of the interval.
    upper : float
        Upper end of the interval.

    Raises
    ------
    ValueError
        If ``value`` does not hold exactly two values, if either end is not finite, if
        ``lower >= upper``, if ``lower < lowest`` or if ``upper > highest``.
    """
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

    The map is ``2 * half_length`` wide along both axes, and its side must hold a whole
    number of pixels. The same class describes the coarse map used to measure variability
    (:attr:`VariabilitySpec.map`) and the bank map that is stored and returned
    (:attr:`BankConfig.bank_map`).

    Parameters
    ----------
    half_length : float
        Half of the map's side length, in Einstein radii. Must be positive and finite.
    pixel_scale : float
        Side length of one pixel, in Einstein radii. Must be positive and finite, and must
        divide the side length ``2 * half_length`` a whole number of times.
    """

    half_length: float
    pixel_scale: float

    def __post_init__(self) -> None:
        """Validate the fields and convert them to floats.

        Raises
        ------
        ValueError
            If ``half_length`` or ``pixel_scale`` is not positive and finite, or if
            ``2 * half_length / pixel_scale`` is not a positive integer within a relative 1e-9.
        """
        object.__setattr__(self, "half_length", _finite_positive(self.half_length, "MapSpec.half_length"))
        object.__setattr__(self, "pixel_scale", _finite_positive(self.pixel_scale, "MapSpec.pixel_scale"))
        _positive_integer(self.width / self.pixel_scale, "MapSpec width / pixel_scale")

    @property
    def width(self) -> float:
        """Side length of the map, ``2 * half_length``, in Einstein radii."""
        return 2.0 * self.half_length

    @property
    def num_pixels(self) -> int:
        """Number of pixels along each side, ``width / pixel_scale``."""
        return _positive_integer(self.width / self.pixel_scale, "MapSpec width / pixel_scale")


@dataclass(frozen=True)
class VariabilitySpec:
    """How the intrinsic MPD-distance quantile is measured on a coarse map.

    Every entry gets a coarse map as well as its bank map. The coarse map is split into
    non-overlapping square windows of side ``2 * window_half_length``, and each window's
    magnitudes are histogrammed in ``n_bins`` equal bins spanning the map's finite range.
    The quantile is taken over the Jensen-Shannon distances between the MPDs of all window
    pairs (:func:`mpd.intrinsic_quantile`). It is the entry's own noise level in the hit
    rule. The coarse map is not stored.

    Parameters
    ----------
    map : MapSpec, optional
        Geometry of the coarse map. Default is ``MapSpec(80.0, 0.1)``: 160 Einstein radii
        across, in 1600 pixels.
    window_half_length : float, optional
        Half the side length of one window, in Einstein radii. The window side must be a
        whole number of coarse-map pixels, and the map side a whole number of windows, at
        least 2. Default is ``10.0``, which gives 8 x 8 windows of 200 pixels on the default
        map.
    quantile : float, optional
        Quantile, in [0, 1], of the window-pair distances. Default is ``0.95``.
    n_bins : int, optional
        Number of bins of the window MPDs, at least 1. Their ``n_bins + 1`` edges are evenly
        spaced over the coarse map's finite magnitude range. Default is ``100``.
    """

    map: MapSpec = field(default_factory=lambda: MapSpec(80.0, 0.1))
    window_half_length: float = 10.0
    quantile: float = 0.95
    n_bins: int = 100

    def __post_init__(self) -> None:
        """Validate the fields and convert them to their canonical types.

        ``window_half_length`` and ``quantile`` become floats and ``n_bins`` an int.

        Raises
        ------
        TypeError
            If ``map`` is not a :class:`MapSpec`.
        ValueError
            If ``window_half_length`` is not positive and finite; if the window side in pixels,
            ``2 * window_half_length / map.pixel_scale``, or the number of windows per axis,
            ``map.width / (2 * window_half_length)``, is not a positive integer within a
            relative 1e-9; if there are fewer than 2 windows per axis; if ``quantile`` does not
            lie in [0, 1]; or if ``n_bins`` is not an integer of at least 1.
        """
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
        object.__setattr__(self, "n_bins", _integer_at_least(self.n_bins, 1, "VariabilitySpec.n_bins"))

    @property
    def window_pixels(self) -> int:
        """Side length of one window, in coarse-map pixels."""
        return _positive_integer(
            2.0 * self.window_half_length / self.map.pixel_scale, "VariabilitySpec window width in pixels"
        )

    @property
    def windows_per_axis(self) -> int:
        """Number of windows along each axis of the coarse map; the map holds its square."""
        return _positive_integer(
            self.map.width / (2.0 * self.window_half_length), "VariabilitySpec map width / window width"
        )


@dataclass(frozen=True)
class DomainSpec:
    """The parameter region a bank is designed to cover.

    A point is in the domain when it lies in the closed kappa/gamma/s box and its
    macro magnification satisfies ``|mu| <= max_macro_magnification``
    (:func:`lensing.in_domain`). The build designs each region's hull inside the domain
    (:func:`lensing.region_hull`), and :meth:`MapBank.query` reports points outside it as
    ``OUTSIDE_DOMAIN`` unless ``allow_outside_domain`` is set.

    Parameters
    ----------
    kappa_range : tuple of (float, float), optional
        Closed ``(lower, upper)`` range of the total convergence kappa, with
        ``0 <= lower < upper``. Default is ``(0.05, 2.0)``.
    gamma_range : tuple of (float, float), optional
        Closed range of the shear gamma, with ``0 <= lower < upper``. The box test compares
        gamma as given, without taking its absolute value. Default is ``(0.05, 2.0)``.
    s_range : tuple of (float, float), optional
        Closed range of the smooth-matter fraction s, with ``0 <= lower < upper <= 1``.
        Default is ``(0.01, 0.99)``.
    max_macro_magnification : float, optional
        Largest allowed ``|mu_macro|``, where ``mu_macro = 1 / ((1 - kappa)**2 - gamma**2)``.
        It keeps the domain away from the critical lines, where ``mu_macro`` diverges. Must be
        positive and finite. Default is ``100.0``.
    """

    kappa_range: tuple[float, float] = (0.05, 2.0)
    gamma_range: tuple[float, float] = (0.05, 2.0)
    s_range: tuple[float, float] = (0.01, 0.99)
    max_macro_magnification: float = 100.0

    def __post_init__(self) -> None:
        """Validate the fields and convert them to their canonical types.

        Each range, which may be any sequence of two numbers such as a JSON list, becomes a
        tuple of two floats, and ``max_macro_magnification`` becomes a float.

        Raises
        ------
        ValueError
            If a range does not hold exactly two values, if either end is not finite, if its
            lower end is not below its upper end, if the lower end of ``kappa_range``,
            ``gamma_range`` or ``s_range`` is negative, if the upper end of ``s_range``
            exceeds 1, or if ``max_macro_magnification`` is not positive and finite.
        """
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
    """Settings of the adaptive design used by :meth:`MapBank.build`.

    ``n_boundary`` shapes each region's design hull. The other fields weight the
    failure-aware curvature loss (:func:`design.failure_aware_curvature_loss`) with which
    python-``adaptive``'s ``LearnerND`` chooses the next point. The learner sees the value
    ``[intrinsic_quantile, 1]`` for a valid entry and ``[0, 0]`` for an invalid one, so the
    loss can tell simplices whose vertices all worked from those that straddle or lie in an
    area where map generation fails.

    Parameters
    ----------
    n_boundary : int, optional
        Number of samples along each region's curved edge, where
        ``|mu_macro| = max_macro_magnification``, when its convex kappa-gamma polygon is built
        (:func:`lensing.region_hull_vertices`). The polygon is extruded along s into the
        design hull, and every hull vertex becomes an entry before adaptive sampling starts,
        so more samples follow the curve more closely but cost more maps. At least 2.
        Default is ``6``.
    exploration : float, optional
        Weight of the volume term in the loss of a simplex whose vertices are all valid,
        ``(curvature + exploration * V**((d + 2) / d)) ** (1 / (d + 2))``, where ``V`` is the
        simplex volume and ``d = 3``. Larger values favour large simplices over curved ones
        and so spread the points more evenly. Non-negative. Default is ``0.05``.
    boundary_multiplier : float, optional
        Loss of a simplex with both valid and invalid vertices, in units of its size
        ``V**(1 / d)``. It concentrates points on the edge of areas where map generation
        fails. Non-negative. Default is ``2.0``.
    invalid_multiplier : float, optional
        Loss of a simplex whose vertices are all invalid, in units of ``V**(1 / d)``. Zero
        abandons such simplices. Non-negative. Default is ``0.0``.
    adjacent_boundary_multiplier : float or None, optional
        When set, the smallest loss, in units of ``V**(1 / d)``, of a simplex whose vertices
        are all valid but which has an invalid neighbour. ``None`` sets no such floor, which
        is usually enough because the neighbouring mixed simplex already gets
        ``boundary_multiplier``. Non-negative when set. Default is ``None``.
    validity_threshold : float, optional
        Threshold that decodes the validity flag of a learner value: a vertex is valid when
        its flag exceeds it. The flags are 1 or 0, so any value well inside (0, 1) gives the
        same design. Must lie strictly between 0 and 1. Default is ``0.5``.

    Notes
    -----
    ``LearnerND`` rescales each axis so that the bounding box of the design hull has sides
    of length 1 before it computes a loss, so simplex volumes and sizes are in those scaled
    units.
    """

    n_boundary: int = 6
    exploration: float = 0.05
    boundary_multiplier: float = 2.0
    invalid_multiplier: float = 0.0
    adjacent_boundary_multiplier: float | None = None
    validity_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate the fields and convert them to their canonical types.

        ``n_boundary`` becomes an int, and the other fields, when set, become floats.

        Raises
        ------
        ValueError
            If ``n_boundary`` is not an integer of at least 2; if ``exploration``,
            ``boundary_multiplier``, ``invalid_multiplier`` or a set
            ``adjacent_boundary_multiplier`` is negative or not finite; or if
            ``validity_threshold`` does not lie strictly between 0 and 1.
        """
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

    The spec is stored in ``bank.json``. :func:`maps.generator_from_spec` builds the
    generator it names, and a generator passed to :meth:`MapBank.build` or
    :meth:`MapBank.fetch` must have the same name and options, or
    :exc:`GeneratorMismatchError` is raised. ``options`` are normalised through JSON, so
    tuples become lists and keys become strings.

    Parameters
    ----------
    name : str, optional
        Name under which the generator is registered with :func:`maps.register_generator`,
        such as ``"ipm"`` (:class:`IPMGenerator`) or ``"synthetic"``
        (:class:`SyntheticGenerator`). Default is ``"ipm"``.
    options : mapping of str to Any, optional
        Options that change the maps, passed to the generator's factory. They must be
        JSON-serialisable without NaN or infinite values, and are stored as a plain dict.
        Default is ``{"rectangular": True}``, a keyword argument of IPM.
    """

    name: str = "ipm"
    options: Mapping[str, Any] = field(default_factory=lambda: {"rectangular": True})

    def __post_init__(self) -> None:
        """Validate the fields and normalise ``options`` through JSON.

        ``options`` is replaced by the plain dict that ``json.loads(json.dumps(...))`` gives
        back, so tuples become lists and keys become strings.

        Raises
        ------
        ValueError
            If ``name`` is not a non-empty string, or if ``options`` cannot be converted to a
            dict or is not JSON-serialisable (NaN and infinite values included).
        """
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"GeneratorSpec.name must be a non-empty string, got {self.name!r}.")
        try:
            options = json.loads(json.dumps(dict(self.options), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"GeneratorSpec.options must be JSON-serialisable: {exc}") from exc
        object.__setattr__(self, "options", options)

    def __hash__(self) -> int:
        """Hash the name and the options, since a dict is not hashable.

        Specs that compare equal hash equally, even when their options differ only in
        numeric type, such as ``1``, ``1.0`` and ``True``. This also makes
        :class:`BankConfig`, whose generated hash includes this spec, hashable.

        Returns
        -------
        int
            Hash of ``name`` and of ``options`` converted by :func:`_hashable`.
        """
        return hash((self.name, _hashable(self.options)))

    @classmethod
    def from_generator(cls, generator: Any) -> GeneratorSpec:
        """Return the spec that identifies ``generator``: its name and options.

        Use it to build a :class:`BankConfig` whose generator matches an existing generator
        instance, which the bank then accepts in :meth:`MapBank.build` and
        :meth:`MapBank.fetch`.

        Parameters
        ----------
        generator : MapGenerator
            Generator whose ``name`` and ``options`` attributes are read.

        Returns
        -------
        GeneratorSpec
            Spec with the generator's name and a JSON-normalised copy of its options.

        Raises
        ------
        ValueError
            If the generator's name is not a non-empty string, or its options are not
            JSON-serialisable.
        """
        return cls(name=generator.name, options=dict(generator.options))


@dataclass(frozen=True)
class BankConfig:
    """Complete description of how a bank's entries are made.

    The config is written to ``bank.json`` when the bank is created and read back every
    time it is opened, so all entries of a bank are made with the same settings. The
    defaults are the settings of the original adaptive_mpd runs apart from the numbers of
    bins, which the original runs set as 100 bin edges (:data:`legacy.LEGACY_CONFIG`).

    Parameters
    ----------
    domain : DomainSpec, optional
        Parameter domain the bank covers. Default is ``DomainSpec()``.
    variability : VariabilitySpec, optional
        How each entry's intrinsic quantile is measured on its coarse map. Default is
        ``VariabilitySpec()``.
    bank_map : MapSpec, optional
        Geometry of the bank map that is stored and returned for each entry. Default is
        ``MapSpec(20.0, 0.01)``: 40 Einstein radii across, in 4000 pixels.
    n_bins : int, optional
        Number of bins of the bank MPDs between their underflow and overflow bins, at least 1.
        :meth:`MapBank.finalize` spaces their ``n_bins + 1`` edges evenly over the finite
        magnitude range of the region's valid bank maps and freezes them. Default is ``100``.
    seed : int, optional
        Non-negative base seed. The seeds of each entry's coarse map and bank map are
        derived from it, the region and the entry ID (:func:`bank.entry_seeds`). Default
        is ``42``.
    design : DesignSpec, optional
        Settings of the adaptive design. Default is ``DesignSpec()``.
    generator : GeneratorSpec, optional
        Map code that makes the maps, and its options. Default is ``GeneratorSpec()``: IPM
        with ``{"rectangular": True}``.
    """

    domain: DomainSpec = field(default_factory=DomainSpec)
    variability: VariabilitySpec = field(default_factory=VariabilitySpec)
    bank_map: MapSpec = field(default_factory=lambda: MapSpec(20.0, 0.01))
    n_bins: int = 100
    seed: int = 42
    design: DesignSpec = field(default_factory=DesignSpec)
    generator: GeneratorSpec = field(default_factory=GeneratorSpec)

    def __post_init__(self) -> None:
        """Validate the fields and convert ``n_bins`` and ``seed`` to ints.

        The nested specs validate themselves when they are constructed, so only their types
        are checked here.

        Raises
        ------
        TypeError
            If ``domain``, ``variability``, ``bank_map``, ``design`` or ``generator`` is not a
            :class:`DomainSpec`, :class:`VariabilitySpec`, :class:`MapSpec`,
            :class:`DesignSpec` or :class:`GeneratorSpec` respectively.
        ValueError
            If ``n_bins`` is not an integer of at least 1, or if ``seed`` is not a
            non-negative integer.
        """
        for name, kind in (
            ("domain", DomainSpec),
            ("variability", VariabilitySpec),
            ("bank_map", MapSpec),
            ("design", DesignSpec),
            ("generator", GeneratorSpec),
        ):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"BankConfig.{name} must be a {kind.__name__}.")
        object.__setattr__(self, "n_bins", _integer_at_least(self.n_bins, 1, "BankConfig.n_bins"))
        object.__setattr__(self, "seed", _integer_at_least(self.seed, 0, "BankConfig.seed"))

    @property
    def n_mpd_columns(self) -> int:
        """Length of a bank MPD: ``n_bins`` bins plus the underflow and overflow bins.

        It is also the number of columns of each region's ``mpds.npy``.
        """
        return self.n_bins + 2

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a JSON-compatible dictionary; tuples become lists.

        :meth:`MapBank.create` stores the result in ``bank.json``, and :meth:`from_dict`
        rebuilds an equal config from it.

        Returns
        -------
        dict of str to Any
            The fields ``domain``, ``variability``, ``bank_map``, ``n_bins``, ``seed``,
            ``design`` and ``generator``. Each spec becomes a dictionary of its own fields,
            with the coarse map nested under ``variability["map"]``.
        """
        result: dict[str, Any] = json.loads(json.dumps(asdict(self), allow_nan=False))
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BankConfig:
        """Rebuild a config written by :meth:`to_dict`.

        Each section goes through its spec's constructor, so every value is validated
        again. Keys missing from the ``domain``, ``design`` and ``generator`` sections, and
        from ``variability`` apart from ``map``, take their defaults. ``data`` is not
        modified.

        Parameters
        ----------
        data : mapping of str to Any
            Dictionary laid out like the output of :meth:`to_dict`, such as the ``config``
            section of ``bank.json``.

        Returns
        -------
        BankConfig
            The rebuilt config.

        Raises
        ------
        KeyError
            If one of ``domain``, ``variability``, ``variability["map"]``, ``bank_map``,
            ``n_bins``, ``seed``, ``design`` or ``generator`` is missing.
        TypeError
            If a section has a key its spec does not accept, or a map section lacks
            ``half_length`` or ``pixel_scale``.
        ValueError
            If a value fails the validation of its spec.
        """
        variability = dict(data["variability"])
        return cls(
            domain=DomainSpec(**data["domain"]),
            variability=VariabilitySpec(map=MapSpec(**variability.pop("map")), **variability),
            bank_map=MapSpec(**data["bank_map"]),
            n_bins=data["n_bins"],
            seed=data["seed"],
            design=DesignSpec(**data["design"]),
            generator=GeneratorSpec(**data["generator"]),
        )


@dataclass(frozen=True)
class StoppingCriteria:
    """When :meth:`MapBank.build` stops adding entries to a region.

    The build stops as soon as any criterion that is set holds. The criteria are checked
    before every new point, in the order ``max_valid_points``, ``simplex_loss_goal``,
    ``residual_goal``, and :attr:`BuildSummary.stop_reason` names the first that holds, so a
    build that resumes a region which already meets a criterion adds no entries. At least
    one of the three must be set.

    Parameters
    ----------
    max_valid_points : int or None, optional
        Stop once the region holds at least this many valid entries, counting every valid
        entry of the region, whatever its origin (build, fetch or legacy import). At least 1
        when set; ``None`` disables the criterion. Default is ``500``.
    simplex_loss_goal : float or None, optional
        Stop once every vertex of the design hull has been evaluated and the largest simplex
        loss of the learner (see :class:`DesignSpec`) is at most this value. Positive and
        finite when set; ``None`` disables the criterion. Default is ``None``.
    residual_goal : float or None, optional
        Stop once at least ``min_num_residuals`` relative residuals have been recorded and the
        largest of the most recent ``min_num_residuals`` is finite and at most this value. A
        new valid build entry has the residual ``|q - q_pred| / |q|``, where ``q`` is its
        measured intrinsic quantile and ``q_pred`` the quantile interpolated from the region's
        entries before it was added; an entry that no fully valid tetrahedron contains has no
        residual. Residuals recorded by earlier builds of the region count too. Positive and
        finite when set; ``None`` disables the criterion. Default is ``None``.
    min_num_residuals : int, optional
        Number of most recent residuals that ``residual_goal`` looks at, and the number that
        must exist before it can stop the build. It also sets the window of the
        ``max_residual`` diagnostic recorded with each build entry. At least 1. Default is
        ``100``.
    """

    max_valid_points: int | None = 500
    simplex_loss_goal: float | None = None
    residual_goal: float | None = None
    min_num_residuals: int = 100

    def __post_init__(self) -> None:
        """Validate the fields and convert them to their canonical types.

        ``max_valid_points`` and ``min_num_residuals`` become ints and the set goals floats.

        Raises
        ------
        ValueError
            If ``max_valid_points``, ``simplex_loss_goal`` and ``residual_goal`` are all
            ``None``; if a set ``max_valid_points`` is not an integer of at least 1; if a set
            ``simplex_loss_goal`` or ``residual_goal`` is not positive and finite; or if
            ``min_num_residuals`` is not an integer of at least 1.
        """
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
