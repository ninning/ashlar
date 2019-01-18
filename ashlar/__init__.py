# Versioneer boilerplate.
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .geometry import Vector, Rectangle
from .metadata import Tile, TileSet
from .align import TileAlignment, register_tiles
