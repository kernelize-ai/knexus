"""isort:skip_file"""
__version__ = '0.2.12'

# ---------------------------------------
# Note: import order is significant here.

import os
_KNEXUS_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ['KNEXUS_HOME'] = _KNEXUS_PACKAGE_DIR
os.environ['KNEXUS_RUNTIME_PATH'] = os.path.join(_KNEXUS_PACKAGE_DIR, 'runtime_libs')
os.environ['KNEXUS_DEVICE_PATH'] = os.path.join(_KNEXUS_PACKAGE_DIR, 'device_lib')

from ._C.libknexus import *

# Import utility functions
from . import utils
from .utils import (
    version_info,
    format_device_info,
    get_data_type,
)

# ``knexus.dump(obj=None)``: the object tree (the System by default) as a dict.
# Last: the module imports knexus and reads ``knexus.property``. The name is the
# function, not the module; ``python -m knexus.dump`` still runs the module.
from .dump import dump

__all__ = [
    'version_info',
    'format_device_info',
    'get_data_type',
    'dump',
]


