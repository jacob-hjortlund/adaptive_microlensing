"""Benchmarks of bank queries, run with airspeed velocity (asv).

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from adaptive_microlensing import (
    BankConfig,
    GeneratorSpec,
    MapBank,
    MapSpec,
    StoppingCriteria,
    SyntheticGenerator,
    VariabilitySpec,
)
from adaptive_microlensing.mpd import pairwise_js_distances


def _config():
    return BankConfig(
        variability=VariabilitySpec(map=MapSpec(4.0, 0.1), window_half_length=1.0, n_bins=11),
        bank_map=MapSpec(2.0, 0.05),
        n_bins=11,
        generator=GeneratorSpec.from_generator(SyntheticGenerator()),
    )


class QuerySuite:
    """Query throughput on a small synthetic bank."""

    def setup_cache(self):
        """Build the bank once; asv passes its path to every benchmark."""
        path = Path(tempfile.mkdtemp()) / "bank"
        with MapBank.create(path, _config()) as bank:
            for region in ("minima", "saddle", "maxima"):
                bank.build(region, StoppingCriteria(max_valid_points=60))
                bank.finalize(region)
        return str(path)

    def setup(self, path):
        """Open the bank and draw the queries."""
        self.bank = MapBank.open(path)
        rng = np.random.default_rng(0)
        self.queries = pd.DataFrame(
            {
                "kappa": rng.uniform(0.05, 2.0, 1000),
                "gamma": rng.uniform(0.05, 2.0, 1000),
                "s": rng.uniform(0.01, 0.99, 1000),
            }
        )

    def time_query_many(self, path):
        """Time 1000 queries."""
        self.bank.query_many(self.queries)


def time_pairwise_distances():
    """Time the distance matrix of a 500-entry region."""
    mpds = np.random.default_rng(0).random((500, 101))
    pairwise_js_distances(mpds, np.ones(500, dtype=bool))
