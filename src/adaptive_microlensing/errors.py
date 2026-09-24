"""Exceptions raised by adaptive_microlensing."""


class BankError(Exception):
    """Base class for every error raised by this package."""


class BankLockedError(BankError):
    """A region is already open for writing by another writer."""


class BankReadOnlyError(BankError):
    """A write needs a region that is not open for writing."""


class BankCorruptError(BankError):
    """Bank files are missing, unreadable, or disagree with each other."""


class GeneratorMismatchError(BankError):
    """A supplied map generator differs from the one recorded in the bank."""


class RegionStateError(BankError):
    """A region is in the wrong state for the requested operation."""


class MapGenerationError(BankError):
    """A map generator failed to produce a map."""


class InvalidMapError(BankError):
    """A generated map cannot be used."""
