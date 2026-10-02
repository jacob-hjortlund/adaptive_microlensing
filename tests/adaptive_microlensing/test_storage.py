import errno
import io
import math
import os
import zipfile

import numpy as np
import pandas as pd
import pytest
from adaptive_microlensing import BankCorruptError, BankLockedError, storage
from adaptive_microlensing.storage import (
    ENTRY_COLUMNS,
    WINDOWS_DIR,
    RegionLock,
    RegionStore,
    append_entry,
    atomic_save_array,
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
        "coarse_quantile": 0.123456789012345,
        "map_file": RegionStore.map_file(entry_id),
        "window_file": "",
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


@pytest.mark.parametrize("missing", [math.nan, None])
def test_a_missing_validity_flag_is_refused(missing):
    """A missing valid flag is neither taken as valid nor as invalid."""
    with pytest.raises(BankCorruptError, match="valid"):
        normalise_entries(pd.DataFrame([_row(0, valid=missing)]))


def test_atomic_writes_leave_no_temporary_files(tmp_path):
    """Atomic writes leave no temporary files."""
    path = tmp_path / "payload.json"
    atomic_write_json(path, {"a": [1, 2]})
    atomic_write_json(path, {"a": [3]})
    assert read_json(path) == {"a": [3]}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["payload.json"]
    with pytest.raises(BankCorruptError, match="Missing"):
        read_json(tmp_path / "absent.json")


def test_a_failed_atomic_write_leaves_the_file_and_no_temporary_file(tmp_path):
    """A write that fails part-way keeps the old file and removes its temporary file."""
    path = tmp_path / "array.npy"
    atomic_save_array(path, np.arange(3.0))
    with pytest.raises(ValueError, match="pickle"):
        atomic_save_array(path, np.array([{"a": 1}], dtype=object))
    np.testing.assert_array_equal(np.load(path), [0.0, 1.0, 2.0])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["array.npy"]


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


def _finalized_store(tmp_path):
    """A finalized region store with one valid entry and 4-column MPDs."""
    store = RegionStore(tmp_path / "minima", "minima")
    store.initialise(n_mpd_columns=4)
    store.commit(append_entry(empty_entries(), _row(0)), np.ones((1, 4)))
    store.write_meta({**read_json(store.meta_path), "finalized": True, "bin_edges": [0.0, 1.0, 2.0]})
    return store


def _truncate_mpds(length):
    """Corrupt mpds.npy by keeping only its first ``length`` bytes."""
    return lambda store: store.mpds_path.write_bytes(store.mpds_path.read_bytes()[:length])


def _edit_meta(edit):
    """Corrupt region.json by rewriting its contents with ``edit``."""
    return lambda store: store.write_meta(edit(read_json(store.meta_path)))


def _without(key):
    """Drop ``key`` from a mapping."""
    return lambda meta: {name: value for name, value in meta.items() if name != key}


CORRUPT_REGION_FILES = {
    "mpds.npy of garbage": lambda store: store.mpds_path.write_bytes(b"not an npy file"),
    "mpds.npy cut in its header": _truncate_mpds(100),
    "mpds.npy cut in its data": _truncate_mpds(-8),
    "empty mpds.npy": _truncate_mpds(0),
    "region.json without finalized": _edit_meta(_without("finalized")),
    "region.json without bin_edges": _edit_meta(_without("bin_edges")),
    "finalized region.json without edges": _edit_meta(lambda meta: {**meta, "bin_edges": None}),
    "region.json holding a list": lambda store: store.meta_path.write_text("[1]"),
    "region.json that is not UTF-8": lambda store: store.meta_path.write_bytes(b'{"region": "\xff"}'),
}


@pytest.mark.parametrize("corrupt", CORRUPT_REGION_FILES.values(), ids=CORRUPT_REGION_FILES.keys())
def test_unusable_region_files_are_reported_as_corrupt(tmp_path, corrupt):
    """Region files that cannot be read or lack required fields raise BankCorruptError naming the file."""
    store = _finalized_store(tmp_path)
    assert store.load(4).meta["finalized"] is True
    corrupt(store)
    with pytest.raises(BankCorruptError, match=r"(mpds\.npy|region\.json)"):
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


def test_a_lock_that_cannot_record_its_holder_is_released_again(tmp_path, monkeypatch):
    """If writing the holder line fails after the lock is taken, acquire gives the lock back."""

    def failing_pwrite(fd, data, offset):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(storage.os, "pwrite", failing_pwrite)
    with pytest.raises(OSError, match="No space"):
        RegionLock(tmp_path / ".lock").acquire()
    monkeypatch.undo()
    other = RegionLock(tmp_path / ".lock")
    other.acquire()
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


def test_entries_have_coarse_quantile_and_window_file_columns():
    """The provisional quantile sits next to the quantile in force, the window file next to the map."""
    columns = list(ENTRY_COLUMNS)
    assert columns.index("coarse_quantile") == columns.index("intrinsic_quantile") + 1
    assert columns.index("window_file") == columns.index("map_file") + 1


def test_window_histograms_round_trip(tmp_path):
    """Window histograms are written atomically to windows/ and read back unchanged."""
    store = RegionStore(tmp_path / "minima", "minima")
    store.initialise(n_mpd_columns=4)
    assert (store.directory / WINDOWS_DIR).is_dir()
    counts = np.arange(48, dtype=np.int32).reshape(16, 3)
    relative = store.write_windows(7, -1234, counts)
    assert relative == "windows/windows_000007.npz" == RegionStore.window_file(7)
    start, restored = store.read_windows(relative)
    assert start == -1234
    assert restored.dtype == np.int32
    np.testing.assert_array_equal(restored, counts)
    assert sorted(p.name for p in (store.directory / WINDOWS_DIR).iterdir()) == ["windows_000007.npz"]


def _npy_bytes(array):
    """The bytes of ``array`` saved as a plain .npy file."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _corrupt_deflate_stream(path):
    """Overwrite the first bytes of the compressed ``counts.npy`` member with 0xFF."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("counts.npy")
    data = bytearray(path.read_bytes())
    header = info.header_offset
    name_length = int.from_bytes(data[header + 26 : header + 28], "little")
    extra_length = int.from_bytes(data[header + 28 : header + 30], "little")
    first = header + 30 + name_length + extra_length
    data[first : first + 4] = b"\xff" * 4
    path.write_bytes(bytes(data))


CORRUPT_WINDOW_FILES = {
    "missing": lambda path: path.unlink(),
    "garbage": lambda path: path.write_bytes(b"not an npz file"),
    "truncated": lambda path: path.write_bytes(path.read_bytes()[:40]),
    "a plain npy": lambda path: path.write_bytes(_npy_bytes(np.zeros(3))),
    "without counts": lambda path: np.savez(path, start=np.int64(0)),
    "corrupt deflate stream": _corrupt_deflate_stream,
    "float counts": lambda path: np.savez(path, start=np.int64(0), counts=np.zeros((2, 2))),
    "negative counts": lambda path: np.savez(
        path, start=np.int64(0), counts=np.array([[1, -1], [2, 2]], dtype=np.int32)
    ),
    "an all-zero row": lambda path: np.savez(
        path, start=np.int64(0), counts=np.array([[1, 1], [0, 0]], dtype=np.int32)
    ),
}


@pytest.mark.parametrize("corrupt", CORRUPT_WINDOW_FILES.values(), ids=CORRUPT_WINDOW_FILES.keys())
def test_unusable_window_files_are_reported_as_corrupt(tmp_path, corrupt):
    """A window file that is missing or unreadable raises BankCorruptError naming it."""
    store = RegionStore(tmp_path / "minima", "minima")
    store.initialise(n_mpd_columns=4)
    relative = store.write_windows(0, 0, np.ones((4, 2), dtype=np.int32))
    corrupt(store.directory / relative)
    with pytest.raises(BankCorruptError, match="windows_000000"):
        store.read_windows(relative)


def test_a_schema_1_region_is_refused(tmp_path):
    """A region written by the previous layout is refused with both schema versions named."""
    store = RegionStore(tmp_path / "minima", "minima")
    store.initialise(n_mpd_columns=4)
    store.write_meta({**read_json(store.meta_path), "schema_version": 1})
    with pytest.raises(BankCorruptError, match=r"schema version 1.*schema 2"):
        store.load(4)


def test_a_schema_from_a_newer_package_advises_an_upgrade(tmp_path):
    """A region of a newer layout is not to be rebuilt: the message says to upgrade the package."""
    store = RegionStore(tmp_path / "minima", "minima")
    store.initialise(n_mpd_columns=4)
    store.write_meta({**read_json(store.meta_path), "schema_version": 3})
    with pytest.raises(BankCorruptError, match=r"schema version 3") as caught:
        store.load(4)
    assert "pgrade" in str(caught.value)
    assert "Rebuild" not in str(caught.value)


def test_an_unfinalized_region_loads_its_provisional_quantiles(tmp_path):
    """A finalize interrupted before region.json may leave frozen-edge quantiles; loading undoes it."""
    store = RegionStore(tmp_path / "minima", "minima")
    store.initialise(n_mpd_columns=4)
    row = _row(0, intrinsic_quantile=0.05, coarse_quantile=0.08)
    store.commit(append_entry(empty_entries(), row), np.full((1, 4), np.nan))
    assert read_entries(store.entries_path).loc[0, "intrinsic_quantile"] == 0.05
    assert store.load(4).entries.loc[0, "intrinsic_quantile"] == 0.08


def test_a_finalized_region_needs_a_quantile_for_every_valid_entry(tmp_path):
    """A finalized region whose valid entry has no quantile in force is corrupt."""
    store = _finalized_store(tmp_path)
    write_entries(store.entries_path, append_entry(empty_entries(), _row(0, intrinsic_quantile=math.nan)))
    with pytest.raises(BankCorruptError, match="intrinsic quantile"):
        store.load(4)


def test_write_finalized_writes_mpds_then_entries_then_region_json(tmp_path, monkeypatch):
    """region.json is written last, so it commits a finalize."""
    store = RegionStore(tmp_path / "minima", "minima")
    store.initialise(n_mpd_columns=4)
    written = []
    for name in ("atomic_save_array", "write_entries", "atomic_write_json"):
        original = getattr(storage, name)

        def recording(path, payload, _original=original):
            written.append(path.name)
            return _original(path, payload)

        monkeypatch.setattr(storage, name, recording)
    meta = {**read_json(store.meta_path), "finalized": True, "bin_edges": [0.0, 1.0, 2.0]}
    store.write_finalized(append_entry(empty_entries(), _row(0)), np.ones((1, 4)), meta)
    assert written == ["mpds.npy", "entries.csv", "region.json"]
    assert store.load(4).meta["finalized"] is True
