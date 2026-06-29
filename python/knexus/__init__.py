"""isort:skip_file"""
__version__ = '0.2.2'

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

__all__ = [
    'version_info',
    'format_device_info',
    'get_data_type',
]


