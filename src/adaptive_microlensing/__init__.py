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
from .maps import (
    IPMGenerator,
    MapGenerator,
    generator_from_spec,
    register_generator,
)
from .synthetic import SyntheticGenerator

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
    "IPMGenerator",
    "InvalidMapError",
    "MapGenerationError",
    "MapGenerator",
    "MapSpec",
    "RegionStateError",
    "StoppingCriteria",
    "SyntheticGenerator",
    "VariabilitySpec",
    "__version__",
    "generator_from_spec",
    "register_generator",
]
