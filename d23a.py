"""
d23a:
    Functions that support the analysis contained in the d23a-fusion repository.

Author:
    Benjamin S. Grandey, 2023.
"""

from functools import cache
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import warnings
from watermark import watermark
import xarray as xr


# Matplotlib settings
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['axes.titleweight'] = 'bold'  # titles for subplots
plt.rcParams['figure.titleweight'] = 'bold'  # suptitle
plt.rcParams['figure.titlesize'] = 'x-large'  # suptitle


def get_watermark():
    """Return watermark string, including versions of dependencies."""
    packages = ('matplotlib,numpy,pandas,seaborn,xarray')
    return watermark(machine=True, conda=True, python=True, packages=packages)
