"""Map generators: the interface between a bank and the code that makes maps.

A map generator turns a point ``(kappa, gamma, s)``, a map geometry
(:class:`~adaptive_microlensing.config.MapSpec`) and a seed into a square map of
magnitudes relative to the macro magnification. :class:`MapGenerator` is the protocol
that every generator meets. :class:`~adaptive_microlensing.bank.MapBank` calls it twice
for each new entry, once for the coarse (variability) map and once for the bank map, and
records a :exc:`~adaptive_microlensing.errors.MapGenerationError` as an invalid entry.

Two generators come with the package. :class:`IPMGenerator` runs the GPU
inverse-polygon-mapping code of the optional ``microlensing`` package, and
:class:`~adaptive_microlensing.synthetic.SyntheticGenerator` is a fast CPU stand-in for
tests. A bank stores only a generator's name and options, as the
:class:`~adaptive_microlensing.config.GeneratorSpec` in ``bank.json``.
:func:`generator_from_spec` rebuilds the generator from them through a registry of
factories, which holds ``"ipm"`` and ``"synthetic"`` when this module is imported and
which :func:`register_generator` extends. :func:`generator_identity` gives the
``(name, options)`` pair that a bank compares with its spec, and
:func:`relative_magnitudes` converts pixel magnifications to relative magnitudes.

Importing this module does not import ``microlensing``; :meth:`IPMGenerator.generate`
imports it only when it makes a map.
"""

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

    Implementations do not inherit from this class; they only need the four members
    below. The protocol is runtime-checkable, so ``isinstance(obj, MapGenerator)`` tests
    only that the members exist, not their signatures or behaviour. A protocol cannot be
    instantiated; implementations define their own constructors.

    Notes
    -----
    The bank uses the members as follows.

    - ``name`` and ``options`` are the generator's identity, which must match the bank's
      :class:`~adaptive_microlensing.config.GeneratorSpec`, stored in ``bank.json``.
      ``build``, when it starts, and ``fetch``, when it has to make a map, raise
      :exc:`~adaptive_microlensing.errors.GeneratorMismatchError` if the generator passed
      to them has a different :func:`generator_identity`. When no generator is passed,
      the bank calls the factory registered under the spec's name with the spec's
      options (:func:`generator_from_spec`). ``options`` must therefore be
      JSON-serialisable and hold every setting that changes the maps.
    - ``version()`` is recorded in ``bank.json`` when the bank is created and in the
      ``generator_version`` column of every entry. The bank logs a warning, once per
      opened bank, when a generator reports a different version from the one recorded at
      creation.
    - ``generate`` must raise :exc:`~adaptive_microlensing.errors.MapGenerationError`
      when the map code fails. The bank then records the entry as invalid, with the
      error message, and carries on. Any other exception, including the ``ValueError``
      raised for a map of the wrong shape, propagates to the caller of ``build`` or
      ``fetch``.
    - The bank derives each ``seed`` from its own seed, the region and the entry ID, so
      a generator that is deterministic for a given ``seed`` makes the bank reproducible.
    """

    @property
    def name(self) -> str:
        """Registered name, recorded in ``bank.json``.

        It is the key under which :func:`register_generator` stores the generator's
        factory, so that :func:`generator_from_spec` can rebuild the generator from a bank's
        spec. Implementations usually define it as a class attribute.
        """

    @property
    def options(self) -> Mapping[str, Any]:
        """Options that change the maps, recorded in ``bank.json``.

        A JSON-serialisable mapping from option name to value. Together with ``name``, it
        is the generator's identity, and the factory registered under ``name`` must rebuild
        an equivalent generator from it. Settings that do not change the maps, such as a
        verbosity level, do not belong in it.
        """

    def version(self) -> str | None:
        """Return the version of the underlying code, recorded with every entry.

        The bank stores it in the ``generator_version`` column of each new entry, and in
        ``bank.json`` when the bank is created.

        Returns
        -------
        str or None
            Version string, or ``None`` if it is unknown. The bank stores ``None`` as an
            empty string.
        """

    def generate(self, kappa: float, gamma: float, s: float, spec: MapSpec, seed: int) -> np.ndarray:
        """Make one magnitude map.

        The bank calls it twice for each new entry: with the coarse map's geometry and the
        variability seed, then with the bank map's geometry and the map seed.

        Parameters
        ----------
        kappa : float
            Total convergence of the macro model.
        gamma : float
            Shear of the macro model.
        s : float
            Smooth-matter fraction: the share of ``kappa`` in smooth matter rather than
            in stars.
        spec : MapSpec
            Geometry of the map: its half side length and pixel scale, in Einstein radii.
        seed : int
            Seed of the map's random realisation, such as the star field. The bank passes
            seeds in ``[1, 2**31 - 1]``.

        Returns
        -------
        numpy.ndarray
            Float array of shape ``(spec.num_pixels, spec.num_pixels)`` of magnitudes
            relative to the macro magnification. Pixels with zero magnification may be
            ``+inf``; MPDs and magnitude ranges use only the finite pixels.

        Raises
        ------
        MapGenerationError
            If the underlying code fails to make the map.
        """


def generator_identity(generator: MapGenerator) -> tuple[str, dict[str, Any]]:
    """Return the (name, options) pair that a bank compares, with options normalised through JSON.

    When ``build`` or ``fetch`` is given a generator, the bank compares this pair with its
    :class:`~adaptive_microlensing.config.GeneratorSpec`, whose options are normalised the
    same way, and raises :exc:`~adaptive_microlensing.errors.GeneratorMismatchError` if they
    differ. The JSON round trip turns tuples into lists, so options compare equal whatever
    sequence types the generator uses.

    Parameters
    ----------
    generator : MapGenerator
        The generator to identify. Its ``options`` must be JSON-serialisable.

    Returns
    -------
    name : str
        The generator's ``name``.
    options : dict of str to Any
        A new dict holding the generator's ``options`` after a JSON round trip.
    """
    return generator.name, json.loads(json.dumps(dict(generator.options)))


def relative_magnitudes(magnifications: np.ndarray, kappa: float, gamma: float) -> np.ndarray:
    """Convert pixel magnifications to magnitudes relative to the macro magnification.

    Computes ``-2.5 * log10(|mu_pixel| / |mu_macro|)`` with
    ``mu_macro = 1 / ((1 - kappa)**2 - gamma**2)``. A pixel with zero magnification becomes
    ``+inf`` and a NaN pixel stays NaN; NumPy's divide and invalid-value warnings are
    silenced. :class:`IPMGenerator` applies it to IPM's output, and other generators can
    use it too.

    Parameters
    ----------
    magnifications : numpy.ndarray
        Pixel magnifications, of any shape. Their sign is ignored.
    kappa : float
        Total convergence of the macro model.
    gamma : float
        Shear of the macro model.

    Returns
    -------
    numpy.ndarray
        Float64 array of relative magnitudes, of the same shape as ``magnifications``.

    Raises
    ------
    MapGenerationError
        If ``(1 - kappa)**2 - gamma**2`` is exactly zero, where the macro magnification is
        singular.
    """
    denominator = (1.0 - kappa) ** 2 - gamma**2
    if denominator == 0.0:
        raise MapGenerationError(f"Macro magnification is singular at kappa={kappa}, gamma={gamma}.")
    mu_macro = abs(1.0 / denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(np.abs(np.asarray(magnifications, dtype=np.float64)) / mu_macro)


class IPMGenerator:
    """Map generator that runs the GPU inverse-polygon-mapping code of the ``microlensing`` package.

    It is the default generator of a bank, since the default
    :class:`~adaptive_microlensing.config.GeneratorSpec` names ``"ipm"`` with
    ``{"rectangular": True}``, and it is registered as ``"ipm"``. ``microlensing`` is an
    optional dependency (the ``ipm`` extra) that is imported only when a map is made, so a
    generator can be created, compared and recorded without it.

    Parameters
    ----------
    options : mapping of str to Any or None, optional
        Extra keyword arguments for ``microlensing.IPM.ipm.IPM``, such as
        ``{"rectangular": True}``. They are part of the generator's identity. They must
        not include the arguments that the generator sets itself: ``kappa_tot``,
        ``shear``, ``smooth_fraction``, ``kappa_star``, ``half_length_y1``,
        ``half_length_y2``, ``num_pixels_y1``, ``num_pixels_y2``, ``random_seed`` and
        ``verbose``. ``smooth_fraction`` is reserved because the generator sets the
        smooth-matter fraction through ``kappa_star``. When ``None``, no extra arguments
        are passed.
    verbose : int, optional
        IPM's verbosity level. It is not part of the identity. Default is ``0``.

    Raises
    ------
    ValueError
        If ``options`` contains one of the reserved arguments.
    """

    name = "ipm"

    def __init__(self, options: Mapping[str, Any] | None = None, verbose: int = 0) -> None:
        """Check the extra IPM options against the reserved arguments and store a copy of them."""
        self._options = dict(options or {})
        reserved = sorted(_RESERVED_IPM_OPTIONS.intersection(self._options))
        if reserved:
            raise ValueError(f"IPMGenerator sets {reserved} itself; remove them from options.")
        self.verbose = int(verbose)

    @property
    def options(self) -> dict[str, Any]:
        """The extra IPM keyword arguments, as a copy; they are part of the generator's identity.

        Each access returns a new dict, so changing it does not change the generator.
        ``verbose`` is not included.
        """
        return dict(self._options)

    def version(self) -> str | None:
        """Return the installed version of ``microlensing``, or None if it is not installed.

        It is read from the package metadata, without importing ``microlensing``. The bank
        records it in ``bank.json`` and in each entry's ``generator_version`` column.

        Returns
        -------
        str or None
            Version string, or ``None`` if ``microlensing`` is not installed.
        """
        try:
            return importlib.metadata.version("microlensing")
        except importlib.metadata.PackageNotFoundError:
            return None

    def generate(self, kappa: float, gamma: float, s: float, spec: MapSpec, seed: int) -> np.ndarray:
        """Run IPM once and return the relative-magnitude map.

        The call first checks that the macro magnification is not singular, then imports
        ``microlensing.IPM.ipm.IPM`` and builds it with ``kappa_tot=kappa``,
        ``shear=gamma``, ``kappa_star=(1 - s) * kappa``, ``spec.half_length`` as
        ``half_length_y1`` and ``half_length_y2``, ``spec.num_pixels`` as
        ``num_pixels_y1`` and ``num_pixels_y2``, ``random_seed=seed``, ``verbose`` and the
        extra options. It calls ``run()`` and converts the resulting ``magnifications``
        with :func:`relative_magnitudes`.

        Parameters
        ----------
        kappa : float
            Total convergence of the macro model.
        gamma : float
            Shear of the macro model.
        s : float
            Smooth-matter fraction; the stars carry ``(1 - s) * kappa``.
        spec : MapSpec
            Geometry of the map: its half side length and pixel scale, in Einstein radii.
        seed : int
            IPM's random seed, which sets the star field.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(spec.num_pixels, spec.num_pixels)`` of magnitudes
            relative to the macro magnification. Pixels with zero magnification are
            ``+inf``.

        Raises
        ------
        MapGenerationError
            If ``(1 - kappa)**2 - gamma**2`` is exactly zero, where the macro magnification
            is singular, or if building IPM, running it or reading its magnifications as a
            float64 array raises any :class:`Exception`. The original exception is chained
            as the cause.
        ImportError
            If ``microlensing`` is not installed. It is not wrapped in
            ``MapGenerationError``, so the bank does not record the entry as invalid.
        ValueError
            If IPM's magnifications do not have shape ``(spec.num_pixels, spec.num_pixels)``.
        """
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


#: Callable that builds a generator from the options stored in a bank's ``GeneratorSpec``.
GeneratorFactory = Callable[[Mapping[str, Any]], MapGenerator]
_GENERATORS: dict[str, GeneratorFactory] = {}


def register_generator(name: str, factory: GeneratorFactory) -> None:
    """Let :func:`generator_from_spec` build generators called ``name``.

    ``factory`` receives the options stored in the bank and returns a generator.
    The registry is kept in memory only, so register a custom generator in every process
    that lets a bank build it, that is, that calls ``build`` or ``fetch`` without a
    ``generator``. A later registration under the same name replaces the earlier one.
    ``"ipm"`` (:class:`IPMGenerator`, called with the options mapping) and
    ``"synthetic"`` (:class:`~adaptive_microlensing.synthetic.SyntheticGenerator`, called
    with the options as keyword arguments) are registered when this module is imported.

    Parameters
    ----------
    name : str
        Name of the generator, as its ``name`` member and the bank's
        :class:`~adaptive_microlensing.config.GeneratorSpec` spell it.
    factory : GeneratorFactory
        Callable that takes the options mapping and returns a :class:`MapGenerator`. The
        generator should report the same ``name`` and ``options``, since the bank does not
        check the identity of a generator that it builds itself.
    """
    _GENERATORS[name] = factory


def generator_from_spec(spec: GeneratorSpec) -> MapGenerator:
    """Build the generator that a bank's ``GeneratorSpec`` describes.

    Calls the factory registered under ``spec.name`` with ``spec.options``. The bank uses
    it when ``build`` or ``fetch`` is given no generator, and when a bank is created, to
    record the generator's version (as ``None`` if the name is not registered). Errors
    raised by the factory, such as the ``ValueError`` of :class:`IPMGenerator` for a
    reserved option, propagate.

    Parameters
    ----------
    spec : GeneratorSpec
        Name and options of the generator, as stored in ``bank.json``.

    Returns
    -------
    MapGenerator
        A new generator from the registered factory.

    Raises
    ------
    ValueError
        If no generator is registered as ``spec.name``. The message lists the registered
        names.
    """
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
