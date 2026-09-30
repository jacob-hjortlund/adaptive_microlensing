"""A fast, deterministic stand-in for a real map code.

:class:`SyntheticGenerator` meets the :class:`~adaptive_microlensing.maps.MapGenerator`
protocol with NumPy alone, on the CPU, so the tests, the documentation and dry runs can
create, build, finalize and query whole banks without a GPU or the ``microlensing``
package. Its maps are not physical: every pixel is an independent normal draw whose mean
and spread change smoothly with ``(kappa, gamma, s)``, so nearby points give similar MPDs
and distant points do not. The same point, number of pixels and seed always give the
same map.

The generator can also imitate the ways a real map code goes wrong. Inside a failure box,
``generate`` raises :exc:`~adaptive_microlensing.errors.MapGenerationError`; inside a
constant box it returns a constant map, which the bank rejects as invalid; and
``inf_fraction`` sets a share of the pixels to ``+inf``, like pixels that no ray reaches.
:mod:`adaptive_microlensing.maps` registers it as ``"synthetic"``, so a bank whose
:class:`~adaptive_microlensing.config.GeneratorSpec` names it rebuilds it from the stored
options.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._version import __version__
from .config import MapSpec
from .errors import MapGenerationError

#: A box in parameter space, ``[[kappa_lo, kappa_hi], [gamma_lo, gamma_hi], [s_lo, s_hi]]``;
#: its bounds belong to it.
Box = list[list[float]]


def _boxes(value: Sequence[Sequence[Sequence[float]]], name: str) -> list[Box]:
    """Convert box specifications to lists of float bounds, checking each box.

    Parameters
    ----------
    value : sequence of boxes
        Boxes, each a sequence of three ``(low, high)`` pairs, for kappa, gamma and s.
    name : str
        Name of the constructor argument, used in the error message.

    Returns
    -------
    list of Box
        The boxes as lists of ``[low, high]`` float pairs, which are JSON-compatible.

    Raises
    ------
    ValueError
        If a box does not have exactly three pairs, or a pair has ``low > high``. A pair
        that does not hold exactly two values raises ``ValueError`` too, when it is unpacked.
        ``low == high`` is allowed.
    """
    boxes = []
    for box in value:
        bounds = [[float(lo), float(hi)] for lo, hi in box]
        if len(bounds) != 3 or any(lo > hi for lo, hi in bounds):
            raise ValueError(f"Each of {name} must be three [low, high] pairs for kappa, gamma and s.")
        boxes.append(bounds)
    return boxes


def _inside(point: tuple[float, float, float], box: Box) -> bool:
    """Return whether a point lies in a box, bounds included.

    Parameters
    ----------
    point : tuple of (float, float, float)
        The ``(kappa, gamma, s)`` point.
    box : Box
        Three ``[low, high]`` pairs, for kappa, gamma and s.

    Returns
    -------
    bool
        ``True`` if ``low <= x <= high`` holds on all three axes.
    """
    return all(lo <= x <= hi for x, (lo, hi) in zip(point, box, strict=True))


class SyntheticGenerator:
    """Deterministic fake magnitude maps for tests, documentation and dry runs.

    Pixels are independent normal draws whose mean (``kappa - gamma``) and spread
    (``0.2 + 0.5 s + 0.1 gamma``) change smoothly with the parameters, so nearby
    points give similar MPDs and distant points do not. A map depends only on the point,
    ``spec.num_pixels`` and the seed; it ignores ``spec.half_length`` and does not model
    the macro magnification. The class meets the
    :class:`~adaptive_microlensing.maps.MapGenerator` protocol and is registered as
    ``"synthetic"``.

    Parameters
    ----------
    failure_boxes : sequence of boxes, optional
        ``[[kappa_lo, kappa_hi], [gamma_lo, gamma_hi], [s_lo, s_hi]]`` boxes inside
        which ``generate`` raises ``MapGenerationError``. Bounds are part of the box.
        They are stored, as lists of float pairs, in the attribute of the same name.
        Default is no boxes.
    inf_fraction : float, optional
        Fraction of pixels set to ``+inf``, like pixels that no ray reaches. It must lie
        in ``[0, 1)``. Default is ``0.0``.
    constant_boxes : sequence of boxes, optional
        Boxes inside which the map is constant, which makes it unusable: the bank records
        the entry as invalid because the coarse map is constant. Same format and storage
        as ``failure_boxes``. Default is no boxes.

    Raises
    ------
    ValueError
        If a box is not three ``[low, high]`` pairs with ``low <= high``, or if
        ``inf_fraction`` is outside ``[0, 1)``.
    """

    name = "synthetic"

    def __init__(
        self,
        failure_boxes: Sequence[Sequence[Sequence[float]]] = (),
        inf_fraction: float = 0.0,
        constant_boxes: Sequence[Sequence[Sequence[float]]] = (),
    ) -> None:
        """Check and store the failure boxes, the ``+inf`` fraction and the constant boxes."""
        self.failure_boxes = _boxes(failure_boxes, "failure_boxes")
        self.constant_boxes = _boxes(constant_boxes, "constant_boxes")
        fraction = float(inf_fraction)
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"inf_fraction must lie in [0, 1), got {inf_fraction!r}.")
        self.inf_fraction = fraction

    @property
    def options(self) -> dict[str, Any]:
        """The constructor arguments as JSON-compatible values.

        They are the generator's identity, recorded in ``bank.json``, and
        ``SyntheticGenerator(**options)`` rebuilds an equivalent generator, as the
        ``"synthetic"`` registry entry does. The dict is new on every access, but its box
        lists are the generator's own lists, not copies.
        """
        return {
            "failure_boxes": self.failure_boxes,
            "inf_fraction": self.inf_fraction,
            "constant_boxes": self.constant_boxes,
        }

    def version(self) -> str:
        """Return the adaptive_microlensing version, since this generator ships with it.

        The bank records it in ``bank.json`` and in each entry's ``generator_version``
        column.

        Returns
        -------
        str
            The installed ``adaptive_microlensing`` version.
        """
        return __version__

    def generate(self, kappa: float, gamma: float, s: float, spec: MapSpec, seed: int) -> np.ndarray:
        """Make one synthetic magnitude map.

        The point is tested in this order. Inside any failure box, the call raises
        :exc:`MapGenerationError`. Otherwise, inside any constant box, the map is all
        zeros, with no ``+inf`` pixels and without using ``seed``. Otherwise the pixels are
        drawn from ``numpy.random.default_rng(seed)``, so the same arguments always give
        the same map.

        Parameters
        ----------
        kappa : float
            Total convergence of the macro model.
        gamma : float
            Shear of the macro model.
        s : float
            Smooth-matter fraction.
        spec : MapSpec
            Geometry of the map. Only ``spec.num_pixels`` is used.
        seed : int
            Seed of the random generator.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(n, n)``, with ``n = spec.num_pixels``. Its pixels are
            normal draws with mean ``kappa - gamma`` and standard deviation
            ``0.2 + 0.5 s + 0.1 gamma``. When ``inf_fraction`` is positive,
            ``int(round(inf_fraction * n * n))`` distinct pixels, chosen at random with the
            same generator after the draws, are then set to ``+inf``.

        Raises
        ------
        MapGenerationError
            If ``(kappa, gamma, s)`` lies inside one of ``failure_boxes``, bounds included.
        """
        point = (float(kappa), float(gamma), float(s))
        if any(_inside(point, box) for box in self.failure_boxes):
            raise MapGenerationError(f"Synthetic failure at kappa={kappa}, gamma={gamma}, s={s}.")
        n = spec.num_pixels
        if any(_inside(point, box) for box in self.constant_boxes):
            return np.zeros((n, n))
        rng = np.random.default_rng(seed)
        mean = point[0] - point[1]
        spread = 0.2 + 0.5 * point[2] + 0.1 * point[1]
        mag_map = rng.normal(mean, spread, size=(n, n))
        if self.inf_fraction > 0.0:
            count = int(round(self.inf_fraction * n * n))
            mag_map.flat[rng.choice(n * n, size=count, replace=False)] = np.inf
        return mag_map
