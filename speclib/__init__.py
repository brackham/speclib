# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *  # noqa

import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

# ----------------------------------------------------------------------------

__all__ = []

from .main import *  # noqa
from .photometry import *  # noqa
from .utils import *  # noqa
