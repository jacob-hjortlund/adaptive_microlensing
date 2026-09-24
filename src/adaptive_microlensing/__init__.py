"""Adaptive banks of microlensing magnitude maps that can be built, queried and updated."""

from ._version import __version__
from .bank import MapBank
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
from .legacy import import_legacy_bank
from .maps import (
    IPMGenerator,
    MapGenerator,
    generator_from_spec,
    register_generator,
)
from .results import (
    BankEntry,
    BuildSummary,
    FetchResult,
    FetchStatus,
    QueryResult,
    QueryStatus,
    hit_summary,
)
from .synthetic import SyntheticGenerator

__all__ = [
    "BankConfig",
    "BankCorruptError",
    "BankEntry",
    "BankError",
    "BankLockedError",
    "BankReadOnlyError",
    "BuildSummary",
    "DesignSpec",
    "DomainSpec",
    "FetchResult",
    "FetchStatus",
    "GeneratorMismatchError",
    "GeneratorSpec",
    "IPMGenerator",
    "InvalidMapError",
    "MapBank",
    "MapGenerationError",
    "MapGenerator",
    "MapSpec",
    "QueryResult",
    "QueryStatus",
    "RegionStateError",
    "StoppingCriteria",
    "SyntheticGenerator",
    "VariabilitySpec",
    "__version__",
    "generator_from_spec",
    "hit_summary",
    "import_legacy_bank",
    "register_generator",
]
