import json

import numpy as np
import pandas as pd
import pytest
from adaptive_microlensing import hit_summary
from adaptive_microlensing.results import QUERY_COLUMNS
from helpers import NEAR_FIRST_VERTEX, batch_table


def test_query_many_matches_single_queries(tetra_bank):
    """query_many appends one result row per input row, matching single queries."""
    table = batch_table()
    result = tetra_bank.query_many(table)
    assert list(result.columns) == ["kappa", "gamma", "s", "label", *QUERY_COLUMNS]
    assert list(result.index) == list(table.index)
    assert result["interpolation_status"].tolist() == [
        "hit",
        "miss",
        "outside_hull",
        "critical_line",
        "region_not_ready",
        "miss",
    ]
    single = tetra_bank.query(*NEAR_FIRST_VERTEX)
    first = result.iloc[0]
    assert first["matched_row"] == single.entry.entry_id
    assert json.loads(first["simplex_rows"]) == list(single.simplex_entry_ids)
    assert first["distance_margin"] == single.margin
    assert str(result["matched_row"].dtype) == "Int64"
    assert result["is_hit"].dtype == bool
    assert pd.isna(result.loc[13, "query_region"])


def test_batch_methods_need_parameter_columns(tetra_bank):
    """Batch methods need kappa, gamma and s columns, and refuse tables that already have results."""
    with pytest.raises(ValueError, match="missing columns"):
        tetra_bank.query_many(pd.DataFrame({"kappa": [0.1], "gamma": [0.1]}))
    with pytest.raises(ValueError, match="already has result columns"):
        tetra_bank.query_many(tetra_bank.query_many(batch_table()))


def test_hit_summary(tetra_bank):
    """hit_summary counts queries, covered queries and hits per region."""
    summary = hit_summary(tetra_bank.query_many(batch_table()))
    assert summary.loc["all", "queries"] == 6
    assert summary.loc["all", "covered"] == 3
    assert summary.loc["all", "hits"] == 1
    assert summary.loc["minima", "hit_rate_covered"] == pytest.approx(1 / 3)
    assert np.isnan(summary.loc["maxima", "hit_rate"])


def test_summary(tetra_bank):
    """summary() counts entries per region."""
    summary = tetra_bank.summary()
    assert summary.loc["minima"].to_dict() == {"entries": 4, "valid": 4, "invalid": 0, "finalized": True}
    assert summary.loc["saddle", "entries"] == 0
