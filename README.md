
# adaptive_microlensing

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/adaptive_microlensing?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/adaptive_microlensing/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/my-organization/adaptive_microlensing/smoke-test.yml)](https://github.com/my-organization/adaptive_microlensing/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/my-organization/adaptive_microlensing/branch/main/graph/badge.svg)](https://codecov.io/gh/my-organization/adaptive_microlensing)
[![Read The Docs](https://img.shields.io/readthedocs/adaptive-microlensing)](https://adaptive-microlensing.readthedocs.io/)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/my-organization/adaptive_microlensing/asv-main.yml?label=benchmarks)](https://my-organization.github.io/adaptive_microlensing/)

Banks of microlensing magnitude maps that are built adaptively, queried for statistically
equivalent maps, and extended on demand.

A bank covers total convergence κ, shear γ and smooth-matter fraction s, with a separate
region for each macro-image type (minima, saddles and maxima). A query finds the tetrahedron
of bank entries around a point and asks whether one of their maps is indistinguishable from
a map made at that point: the interpolated Jensen–Shannon distance between magnification
probability distributions (MPDs) must not exceed the maps' own window-to-window variability.
On a hit the existing map is returned; on a miss, `fetch` makes a new map at the query point,
adds it to the bank and returns it.

## Installation

From a clone of this repository:

```
>> pip install .           # build, load and query banks
>> pip install '.[ipm]'    # also make maps on a GPU with the microlensing package
```

## Usage

```python
import pandas as pd

from adaptive_microlensing import MapBank, BankConfig, StoppingCriteria, hit_summary

# Build: create the bank once, then build each region in its own GPU job.
MapBank.create("bank/", BankConfig()).close()
with MapBank.open("bank/", writable=["minima"]) as bank:
    bank.build("minima", StoppingCriteria(max_valid_points=500))
    bank.finalize("minima")

# Load and query (read-only, no GPU needed).
bank = MapBank.open("bank/")
result = bank.query(0.41, 0.50, 0.25)
if result.is_hit:
    magnitudes = result.entry.load()  # memory-mapped array

# Update: return a matching map, or make one and add it to the bank.
with MapBank.open("bank/", writable=True) as bank:
    fetched = bank.fetch(0.41, 0.50, 0.25)
    table = bank.fetch_many(pd.read_csv("image_params.csv"))  # needs kappa, gamma, s columns
    print(hit_summary(table))
```

Banks made by the original `adaptive_mpd` scripts can be imported without copying their maps:

```python
from adaptive_microlensing import import_legacy_bank

bank = import_legacy_bank("adaptive_mpd/smooth_frac_range_output", "legacy_bank/")
```

Progress is reported through the `adaptive_microlensing` logger; call
`logging.basicConfig(level=logging.INFO)` to see it.

## Bank layout

```
bank/
  bank.json                  # configuration and package versions, written once
  minima/ saddle/ maxima/
    region.json              # finalized flag and frozen MPD bin edges
    entries.csv              # one row per entry: parameters, validity, seeds, provenance
    mpds.npy                 # one MPD per entry
    maps/map_000123.npy      # the bank maps
```

Any number of processes may read a bank at once; each region accepts one writer at a time.

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create -n <env_name> python=3.11
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> ./.setup_dev.sh
>> conda install pandoc
```

Notes:
1. `./.setup_dev.sh` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
2. Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)
