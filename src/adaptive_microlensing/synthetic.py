"""A fast, deterministic stand-in for a real map code."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._version import __version__
from .config import MapSpec
from .errors import MapGenerationError

Box = list[list[float]]


def _boxes(value: Sequence[Sequence[Sequence[float]]], name: str) -> list[Box]:
    boxes = []
    for box in value:
        bounds = [[float(lo), float(hi)] for lo, hi in box]
        if len(bounds) != 3 or any(lo > hi for lo, hi in bounds):
            raise ValueError(f"Each of {name} must be three [low, high] pairs for kappa, gamma and s.")
        boxes.append(bounds)
    return boxes


def _inside(point: tuple[float, float, float], box: Box) -> bool:
    return all(lo <= x <= hi for x, (lo, hi) in zip(point, box, strict=True))


class SyntheticGenerator:
    """Deterministic fake magnitude maps for tests, documentation and dry runs.

    Pixels are independent normal draws whose mean (``kappa - gamma``) and spread
    (``0.2 + 0.5 s + 0.1 gamma``) change smoothly with the parameters, so nearby
    points give similar MPDs and distant points do not.

    Parameters
    ----------
    failure_boxes : sequence of boxes, optional
        ``[[kappa_lo, kappa_hi], [gamma_lo, gamma_hi], [s_lo, s_hi]]`` boxes inside
        which ``generate`` raises ``MapGenerationError``.
    inf_fraction : float, optional
        Fraction of pixels set to ``+inf``, like pixels that no ray reaches.
    constant_boxes : sequence of boxes, optional
        Boxes inside which the map is constant, which makes it unusable.
    """

    name = "synthetic"

    def __init__(
        self,
        failure_boxes: Sequence[Sequence[Sequence[float]]] = (),
        inf_fraction: float = 0.0,
        constant_boxes: Sequence[Sequence[Sequence[float]]] = (),
    ) -> None:
        self.failure_boxes = _boxes(failure_boxes, "failure_boxes")
        self.constant_boxes = _boxes(constant_boxes, "constant_boxes")
        fraction = float(inf_fraction)
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"inf_fraction must lie in [0, 1), got {inf_fraction!r}.")
        self.inf_fraction = fraction

    @property
    def options(self) -> dict[str, Any]:
        """The constructor arguments as JSON-compatible values."""
        return {
            "failure_boxes": self.failure_boxes,
            "inf_fraction": self.inf_fraction,
            "constant_boxes": self.constant_boxes,
        }

    def version(self) -> str:
        """The adaptive_microlensing version, since this generator ships with it."""
        return __version__

    def generate(self, kappa: float, gamma: float, s: float, spec: MapSpec, seed: int) -> np.ndarray:
        """Make one synthetic magnitude map."""
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
