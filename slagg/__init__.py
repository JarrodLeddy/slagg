try:
    from importlib.metadata import version, PackageNotFoundError
except (ModuleNotFoundError, ImportError):
    from importlib_metadata import version, PackageNotFoundError
try:
    __version__ = version("slagg")
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

from slagg.grid import Grid
from slagg.decomp import Decomp
from slagg.geometry import Geometry

__all__ = ["__version__"]
