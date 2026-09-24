
.. adaptive_microlensing documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to adaptive_microlensing's documentation!
========================================================================================

``adaptive_microlensing`` keeps banks of microlensing magnitude maps that are built
adaptively, queried for statistically equivalent maps, and extended on demand.

A bank covers total convergence :math:`\kappa`, shear :math:`\gamma` and smooth-matter
fraction :math:`s`, with a separate region for each macro-image type (minima, saddles and
maxima). A query finds the tetrahedron of bank entries around a point and asks whether one
of their maps is indistinguishable from a map made at that point: the interpolated
Jensen–Shannon distance between magnification probability distributions (MPDs) must not
exceed the maps' own window-to-window variability. On a hit the existing map is returned;
on a miss, ``fetch`` makes a new map at the query point, adds it to the bank and returns it.


Usage
-----

.. code-block:: python

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
       table = bank.fetch_many(pd.read_csv("image_params.csv"))
       print(hit_summary(table))

Banks made by the original ``adaptive_mpd`` scripts can be imported without copying their
maps with :func:`adaptive_microlensing.import_legacy_bank`. The quickstart notebook runs
the whole workflow with :class:`adaptive_microlensing.SyntheticGenerator`, which needs no GPU.


Dev Guide - Getting Started
---------------------------

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

.. code-block:: console

   >> conda create env -n <env_name> python=3.11
   >> conda activate <env_name>


Once you have created a new environment, you can install this project for local
development using the following commands:

.. code-block:: console

   >> pip install -e .'[dev]'
   >> pre-commit install
   >> conda install pandoc


Notes:

1) The single quotes around ``'[dev]'`` may not be required for your operating system.
2) ``pre-commit install`` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on
   `pre-commit <https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html>`_.
3) Installing ``pandoc`` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   `Sphinx and Python Notebooks <https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks>`_.


.. toctree::
   :hidden:

   Home page <self>
   API Reference <autoapi/index>
   Notebooks <notebooks>
