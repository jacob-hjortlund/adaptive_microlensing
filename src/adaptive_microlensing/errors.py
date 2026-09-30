"""Exceptions raised by adaptive_microlensing.

Every exception that the package defines derives from :exc:`BankError`, so
``except BankError`` catches them all. They cover a bank's files and locks
(:exc:`BankLockedError`, :exc:`BankReadOnlyError`, :exc:`BankCorruptError`), the state of
its regions and its generator (:exc:`RegionStateError`, :exc:`GeneratorMismatchError`), and
failed maps (:exc:`MapGenerationError`, :exc:`InvalidMapError`).
:class:`~adaptive_microlensing.bank.MapBank` catches the last two while it makes a new
entry, and records the entry as invalid instead of raising. Invalid arguments and
configurations raise built-in exceptions such as :exc:`ValueError` and :exc:`TypeError`.
"""


class BankError(Exception):
    """Base class of every exception that this package defines.

    It is never raised itself. Catch it to handle any of the package's own errors; it does
    not catch the built-in exceptions raised for invalid arguments.
    """


class BankLockedError(BankError):
    """A region is already open for writing by another writer.

    Raised by :meth:`~adaptive_microlensing.storage.RegionLock.acquire`, and so by
    :meth:`~adaptive_microlensing.bank.MapBank.open`, when a region to be locked for writing
    is already locked by another bank object in this process or by another process. The
    message names the holder recorded in the lock file, when there is one. Opening a bank
    read-only never raises it.
    """


class BankReadOnlyError(BankError):
    """A write needs a region that is not open for writing.

    Raised by the :class:`~adaptive_microlensing.bank.MapBank` methods that change a region,
    ``build`` and ``finalize``, and by ``fetch`` and ``fetch_many`` when they have to make a
    map, if the region was opened read-only or the bank has been closed. Open the bank with
    ``writable=True``, or with the region's name, to write to it.
    """


class BankCorruptError(BankError):
    """Bank files are missing, unreadable, or disagree with each other.

    Raised while a bank is loaded, by :meth:`~adaptive_microlensing.bank.MapBank.open` and
    the readers in :mod:`adaptive_microlensing.storage`: when ``bank.json`` or
    ``region.json`` is missing, is not valid JSON or has an unknown schema version, when
    ``region.json`` names another region, when ``entries.csv`` is missing or malformed, when
    ``mpds.npy`` is missing, has the wrong number of columns or has fewer rows than
    ``entries.csv``, when the entry IDs are not 0, 1, 2, ..., or when a finalized region lacks
    the MPD of a valid entry. It is also raised when an entries table lacks one of the
    standard columns.
    """


class GeneratorMismatchError(BankError):
    """A supplied map generator differs from the one recorded in the bank.

    Raised by :meth:`~adaptive_microlensing.bank.MapBank.build`,
    :meth:`~adaptive_microlensing.bank.MapBank.fetch` and
    :meth:`~adaptive_microlensing.bank.MapBank.fetch_many` when the ``generator`` passed to
    them has a different name or options from the bank's
    :class:`~adaptive_microlensing.config.GeneratorSpec`. ``fetch`` checks the generator only
    when it has to make a map. A different generator version is only logged as a warning.
    """


class RegionStateError(BankError):
    """A region is in the wrong state for the requested operation.

    Raised by :meth:`~adaptive_microlensing.bank.MapBank.finalize` when the region is
    already finalized or has no valid entries, when the frozen bin edges of a region that is
    not finalized are requested, and when a whole entries table is imported into a region
    that already has entries.
    """


class MapGenerationError(BankError):
    """A map generator failed to produce a map.

    Generators raise it when the underlying map code fails:
    :class:`~adaptive_microlensing.maps.IPMGenerator` raises it where the macro
    magnification is singular and wraps any exception from IPM in it, and
    :class:`~adaptive_microlensing.synthetic.SyntheticGenerator` raises it inside its failure
    boxes. Other :class:`~adaptive_microlensing.maps.MapGenerator` implementations should do
    the same. :class:`~adaptive_microlensing.bank.MapBank` catches it while making a new
    entry, and records the entry as invalid, with the error message, instead of raising.
    """


class InvalidMapError(BankError):
    """A generated map cannot be used.

    Raised by the helpers in :mod:`adaptive_microlensing.mpd` when a map has no finite
    pixels, when a coarse map is constant, or when a window of a coarse map has no finite
    pixels. Like :exc:`MapGenerationError`, :class:`~adaptive_microlensing.bank.MapBank`
    catches it while making a new entry and records the entry as invalid. The legacy import
    lets it propagate, after removing what it wrote.
    """
