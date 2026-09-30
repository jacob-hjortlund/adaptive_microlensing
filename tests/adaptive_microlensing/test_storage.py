import math
import os

import numpy as np
import pandas as pd
import pytest
from adaptive_microlensing import BankCorruptError, BankLockedError
from adaptive_microlensing.storage import (
    ENTRY_COLUMNS,
    RegionLock,
    RegionStore,
    append_entry,
    atomic_write_json,
    empty_entries,
    normalise_entries,
    read_entries,
    read_json,
    write_entries,
)


def _row(entry_id, **overrides):
    row = {
        "entry_id": entry_id,
        "kappa": 0.1 + entry_id,
        "gamma": 0.2,
        "s": 1 / 3,
        "valid": True,
        "error": "",
        "intrinsic_quantile": 0.123456789012345,
        "map_file": RegionStore.map_file(entry_id),
        "mag_min": -1.5,
        "mag_max": 2.5,
        "variability_seed": 12345,
        "map_seed": None,
        "generator_version": "",
        "origin": "build",
        "created_at": "2026-09-23T10:00:00+00:00",
        "simplex_loss": math.nan,
        "abs_residual": math.nan,
        "rel_residual": math.inf,
        "max_residual": 0.5,
    }
    row.update(overrides)
    return row


def test_entries_round_trip_through_csv(tmp_path):
    """The entries table survives a CSV round trip with its dtypes."""
    entries = append_entry(empty_entries(), _row(0))
    entries = append_entry(entries, _row(1, valid=False, error="MapGenerationError: boom", map_file=""))
    path = tmp_path / "entries.csv"
    write_entries(path, entries)
    restored = read_entries(path)
    pd.testing.assert_frame_equal(restored, entries)
    assert list(restored.columns) == list(ENTRY_COLUMNS)
    assert restored["map_seed"].isna().all()
    assert restored.loc[1, "error"] == "MapGenerationError: boom"
    assert restored.loc[0, "s"] == 1 / 3


def test_float_columns_round_trip_exactly(tmp_path):
    """Float columns survive the CSV round trip bit for bit."""
    values = np.random.default_rng(0).random(200)
    rows = [
        _row(entry_id, kappa=value, intrinsic_quantile=value / 7) for entry_id, value in enumerate(values)
    ]
    entries = normalise_entries(pd.DataFrame(rows))
    path = tmp_path / "entries.csv"
    write_entries(path, entries)
    restored = read_entries(path)
    np.testing.assert_array_equal(restored["kappa"].to_numpy(), values)
    np.testing.assert_array_equal(restored["intrinsic_quantile"].to_numpy(), values / 7)


def test_empty_entries_round_trip(tmp_path):
    """An empty entries table survives a CSV round trip."""
    path = tmp_path / "entries.csv"
    write_entries(path, empty_entries())
    restored = read_entries(path)
    assert restored.empty
    pd.testing.assert_frame_equal(restored, empty_entries())


def test_atomic_writes_leave_no_temporary_files(tmp_path):
    """Atomic writes leave no temporary files."""
    path = tmp_path / "payload.json"
    atomic_write_json(path, {"a": [1, 2]})
    atomic_write_json(path, {"a": [3]})
    assert read_json(path) == {"a": [3]}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["payload.json"]
    with pytest.raises(BankCorruptError, match="Missing"):
        read_json(tmp_path / "absent.json")


def test_region_store_snapshot_rules(tmp_path):
    """Loading drops extra MPD rows and rejects inconsistent files."""
    store = RegionStore(tmp_path / "minima", "minima")
    store.initialise(n_mpd_columns=4)
    snapshot = store.load(4)
    assert snapshot.entries.empty and snapshot.mpds.shape == (0, 4)
    assert snapshot.meta["finalized"] is False

    entries = append_entry(empty_entries(), _row(0))
    store.commit(entries, np.ones((1, 4)))
    # An extra MPD row, as left by a crash between commit steps 2 and 3, is dropped.
    np.save(store.mpds_path, np.ones((3, 4)))
    assert store.load(4).mpds.shape == (1, 4)
    # Fewer MPD rows than entries, or the wrong width, is corruption.
    np.save(store.mpds_path, np.ones((0, 4)))
    with pytest.raises(BankCorruptError, match="fewer rows"):
        store.load(4)
    np.save(store.mpds_path, np.ones((1, 5)))
    with pytest.raises(BankCorruptError, match="shape"):
        store.load(4)
    np.save(store.mpds_path, np.ones((1, 4)))
    write_entries(store.entries_path, append_entry(empty_entries(), _row(3)))
    with pytest.raises(BankCorruptError, match="entry_id"):
        store.load(4)


def test_region_lock_release_is_idempotent(tmp_path):
    """A held lock excludes a second one; after a double release it can be taken again."""
    lock = RegionLock(tmp_path / ".lock")
    lock.acquire()
    with pytest.raises(BankLockedError, match="this process"):
        RegionLock(tmp_path / ".lock").acquire()
    lock.release()
    lock.release()
    other = RegionLock(tmp_path / ".lock")
    other.acquire()
    assert "pid=" in (tmp_path / ".lock").read_text()
    other.release()


def test_a_lock_whose_descriptor_was_closed_elsewhere_can_be_taken_again(tmp_path):
    """Releasing a lock whose file descriptor is already closed still frees it for this process."""
    lock = RegionLock(tmp_path / ".lock")
    lock.acquire()
    os.close(lock._fd)  # as os.closerange or a daemonising library would
    lock.release()
    other = RegionLock(tmp_path / ".lock")
    other.acquire()
    other.release()
