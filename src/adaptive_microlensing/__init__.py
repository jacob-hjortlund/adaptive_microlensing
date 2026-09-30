"""Adaptive banks of microlensing magnitude maps that can be built, queried and updated.

A bank holds magnitude maps, relative to the macro magnification, at points of total
convergence ``kappa``, shear ``gamma`` and smooth-matter fraction ``s``, with one region
for each macro-image type (``"minima"``, ``"saddle"`` and ``"maxima"``). A query finds the
tetrahedron of bank entries around a point and returns an existing map when the
interpolated Jensen-Shannon distance between magnification probability distributions
(MPDs) is within the maps' own window-to-window variability. The package has a Python API
only; there is no command-line interface.

:class:`MapBank` is the entry point for all work on a bank. :meth:`MapBank.create` makes a
new bank and :meth:`MapBank.open` loads one, read-only or with writer locks on chosen
regions; close it, or use it as a context manager, to release them. :meth:`MapBank.build`
samples a region adaptively and :meth:`MapBank.finalize` freezes its MPD bin edges.
After that, :meth:`MapBank.query` and :meth:`MapBank.query_many` look points up without
writing, and :meth:`MapBank.fetch` and :meth:`MapBank.fetch_many` also make, commit and
return a new map on a miss. :func:`import_legacy_bank` imports a bank made by the original
``adaptive_mpd`` scripts without copying its maps.

A bank's configuration is a :class:`BankConfig` made of frozen, self-validating parts:
:class:`DomainSpec` (the parameter domain), :class:`MapSpec` (the geometry of a map),
:class:`VariabilitySpec` (how the intrinsic variability is measured), :class:`DesignSpec`
(the adaptive design) and :class:`GeneratorSpec` (which map code to use).
:class:`StoppingCriteria` says when a build stops. Queries and fetches return a
:class:`QueryResult` or :class:`FetchResult`, with a :class:`QueryStatus` or
:class:`FetchStatus`; a matched map is a :class:`BankEntry`, whose ``load`` returns the
map, memory-mapped by default. A build returns a :class:`BuildSummary`, and
:func:`hit_summary` counts the hits in a table from the batch methods. The package's own
exceptions derive from :exc:`BankError`.

Maps are made by a :class:`MapGenerator`. :class:`IPMGenerator` runs the GPU code of the
``microlensing`` package, from the ``ipm`` extra, and :class:`SyntheticGenerator` is a
fast, deterministic stand-in for tests and dry runs. :func:`register_generator` adds other
generators, and :func:`generator_from_spec` builds the one that a bank records.
:func:`slice_region` cuts a region's mesh with a plane, giving a :class:`RegionSlice`, and
:func:`coverage_grid` applies the hit rule on a grid in a plane, giving a
:class:`CoverageGrid`. Figures of a bank come from :mod:`adaptive_microlensing.plotting`,
which needs the ``plot`` extra and is imported explicitly with
``from adaptive_microlensing import plotting``; importing the package alone does not load
matplotlib. Progress is reported through the ``"adaptive_microlensing"`` logger.
"""

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
from .slicing import CoverageGrid, RegionSlice, coverage_grid, slice_region
from .synthetic import SyntheticGenerator

__all__ = [
    "BankConfig",
    "BankCorruptError",
    "BankEntry",
    "BankError",
    "BankLockedError",
    "BankReadOnlyError",
    "BuildSummary",
    "CoverageGrid",
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
    "RegionSlice",
    "RegionStateError",
    "StoppingCriteria",
    "SyntheticGenerator",
    "VariabilitySpec",
    "__version__",
    "coverage_grid",
    "generator_from_spec",
    "hit_summary",
    "import_legacy_bank",
    "register_generator",
    "slice_region",
]
