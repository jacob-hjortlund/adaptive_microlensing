"""Adaptive banks of microlensing magnitude maps that can be built, queried and updated."""

from ._version import __version__
from .config import (
    BankConfig,
    DesignSpec,
    DomainSpec,
    GeneratorSpec,
    MapSpec,
    StoppingCriteria,
    VariabilitySpec,
)
from .errors import (
    BankCorruptError,
    BankError,
    BankLockedError,
    BankReadOnlyError,
    GeneratorMismatchError,
    InvalidMapError,
    MapGenerationError,
    RegionStateError,
)

__all__ = [
    "BankConfig",
    "BankCorruptError",
    "BankError",
    "BankLockedError",
    "BankReadOnlyError",
    "DesignSpec",
    "DomainSpec",
    "GeneratorMismatchError",
    "GeneratorSpec",
    "InvalidMapError",
    "MapGenerationError",
    "MapSpec",
    "RegionStateError",
    "StoppingCriteria",
    "VariabilitySpec",
    "__version__",
]
