import subprocess
import sys

import numpy as np
import pytest
from adaptive_microlensing import BankLockedError, MapBank, storage
from adaptive_microlensing.storage import RegionStore
from helpers import CENTROID, add_entry


def test_crash_after_writing_the_map_leaves_the_bank_unchanged(tetra_bank, monkeypatch, generator):
    """Crash after writing the map leaves the bank unchanged."""

    def crash(*args, **kwargs):
        raise OSError("simulated crash")

    monkeypatch.setattr(RegionStore, "commit", crash)
    with pytest.raises(OSError, match="simulated"):
        add_entry(tetra_bank, "minima", CENTROID, generator)
    monkeypatch.undo()
    assert (tetra_bank.path / "minima" / "maps" / "map_000004.npy").exists()
    path = tetra_bank.path
    tetra_bank.close()
    with MapBank.open(path, writable=True) as reopened:
        assert len(reopened.entries("minima")) == 4
        row = add_entry(reopened, "minima", CENTROID, generator)
        assert row["entry_id"] == 4
        assert len(reopened.entries("minima")) == 5


def test_crash_between_mpds_and_entries_is_recovered(tetra_bank, monkeypatch, generator):
    """Crash between mpds and entries is recovered."""

    def crash(*args, **kwargs):
        raise OSError("simulated crash")

    monkeypatch.setattr(storage, "write_entries", crash)
    with pytest.raises(OSError, match="simulated"):
        add_entry(tetra_bank, "minima", CENTROID, generator)
    monkeypatch.undo()
    assert np.load(tetra_bank.path / "minima" / "mpds.npy").shape[0] == 5
    path = tetra_bank.path
    tetra_bank.close()
    with MapBank.open(path, writable=True) as reopened:
        assert len(reopened.entries("minima")) == 4
        add_entry(reopened, "minima", CENTROID, generator)
        assert len(reopened.entries("minima")) == 5
        assert np.all(np.isfinite(reopened._regions["minima"].mpds[4]))


def test_a_second_writer_in_the_same_process_is_refused(bank):
    """A second writer in the same process is refused."""
    with pytest.raises(BankLockedError, match="this process"):
        MapBank.open(bank.path, writable=["saddle"])
    with MapBank.open(bank.path) as reader:
        assert reader.summary()["entries"].sum() == 0


def test_a_second_writer_in_another_process_is_refused(bank):
    """A second writer in another process is refused."""
    code = (
        "import sys\n"
        "from adaptive_microlensing import BankLockedError, MapBank\n"
        "try:\n"
        "    MapBank.open(sys.argv[1], writable=['maxima'])\n"
        "except BankLockedError as error:\n"
        "    print(error)\n"
        "    sys.exit(3)\n"
    )
    result = subprocess.run([sys.executable, "-c", code, str(bank.path)], capture_output=True, text=True)
    assert result.returncode == 3, result.stderr
    assert "pid=" in result.stdout


def test_locks_are_released_on_close(bank):
    """Locks are released on close."""
    path = bank.path
    bank.close()
    with MapBank.open(path, writable=True) as reopened:
        assert "pid=" in (path / "minima" / ".lock").read_text()
        assert reopened._regions["minima"].writable
