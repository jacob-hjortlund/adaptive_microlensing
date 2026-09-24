"""Helpers shared by the tests: a small synthetic configuration and direct entry creation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from adaptive_microlensing import (
    BankConfig,
    GeneratorSpec,
    MapBank,
    MapGenerator,
    MapSpec,
    SyntheticGenerator,
    VariabilitySpec,
)

# Four minima points spanning a tetrahedron. All lie in the default domain.
TETRAHEDRON = ((0.10, 0.10, 0.10), (0.60, 0.10, 0.10), (0.10, 0.50, 0.10), (0.10, 0.10, 0.90))
CENTROID = (0.225, 0.2, 0.3)
NEAR_FIRST_VERTEX = (0.101, 0.101, 0.101)
OUTSIDE_TETRAHEDRON = (0.3, 0.3, 0.8)
# A failure box around the second tetrahedron vertex, (0.6, 0.1, 0.1).
FAILURE_BOX = [[0.55, 0.65], [0.05, 0.15], [0.05, 0.15]]


def small_config(generator: SyntheticGenerator | None = None, **overrides: Any) -> BankConfig:
    """A bank configuration with 80 x 80 synthetic maps and 12 bin edges, fast enough for tests."""
    generator = SyntheticGenerator() if generator is None else generator
    settings: dict[str, Any] = {
        "variability": VariabilitySpec(map=MapSpec(4.0, 0.1), window_half_length=1.0, n_bin_edges=12),
        "bank_map": MapSpec(2.0, 0.05),
        "n_bin_edges": 12,
        "generator": GeneratorSpec.from_generator(generator),
    }
    settings.update(overrides)
    return BankConfig(**settings)


def add_entry(
    bank: MapBank,
    region: str,
    point: tuple[float, float, float],
    generator: MapGenerator | None = None,
    origin: str = "build",
) -> dict[str, Any]:
    """Evaluate and commit one entry at ``point`` without adaptive sampling; returns its row."""
    pending = bank._evaluate_entry(region, point, bank._resolve_generator(generator), origin)
    bank._commit(region, pending)
    return pending.row


# Query points for the batch tests: hit, miss, outside the hull, critical line,
# a region that is not ready, and a repeat of the miss.
BATCH_POINTS = [NEAR_FIRST_VERTEX, CENTROID, OUTSIDE_TETRAHEDRON, (0.5, 0.5, 0.5), (1.0, 0.5, 0.5), CENTROID]


def batch_table() -> pd.DataFrame:
    """BATCH_POINTS as a table with a non-default index and an extra column."""
    table = pd.DataFrame(BATCH_POINTS, columns=["kappa", "gamma", "s"], index=[10, 11, 12, 13, 14, 15])
    return table.assign(label=list("abcdef"))
