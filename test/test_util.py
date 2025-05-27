# 2025 Juan Felipe Giraldo. Licensed under the MIT License.
# test_util.py
import os
import shutil
import warnings

#def clear_dijitso_cache():
#.. only a function that clear the cashe in the folder.

def filter_mpi_warnings():
    """Suppress RuntimeWarnings related to mpi4py mismatches."""
    warnings.filterwarnings(
        "ignore",
        message=r".*mpi4py\.MPI\.Session size changed, may indicate binary incompatibility.*",
        category=RuntimeWarning,
    )

def prepare_test_environment():
    """Call common preparation routines before running tests."""
    #clear_dijitso_cache()
    filter_mpi_warnings()
