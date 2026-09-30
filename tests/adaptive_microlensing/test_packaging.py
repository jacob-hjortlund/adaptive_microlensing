import subprocess
import sys

import adaptive_microlensing


def test_version():
    """Check to see that we can get the package version"""
    assert adaptive_microlensing.__version__ is not None


def test_the_base_import_leaves_the_plot_extra_unloaded():
    """Importing the package loads neither matplotlib nor seaborn; they belong to the plot extra."""
    code = "import sys, adaptive_microlensing; print(sorted({'matplotlib', 'seaborn'} & set(sys.modules)))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
