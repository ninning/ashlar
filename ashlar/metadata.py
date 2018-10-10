from __future__ import division, print_function
from abc import abstractmethod
import attr
import numpy as np
import scipy.spatial.distance
import networkx as nx
from . import util, geometry, plot


class ImageReader(util.ABC):

    @abstractmethod
    def read(self, image_number, channel):
        """Return one image plane from a multi-channel image series."""
        pass


@attr.s(frozen=True)
class TileSet(object):
    """Physical layout of a set of image tiles and access to the pixels.

    Tile positions and shapes are always in microns, not pixels!

    """
    tile_shape = attr.ib(converter=util.array_copy_immutable)
    positions = attr.ib(converter=util.array_copy_immutable)
    _reader = attr.ib(validator=attr.validators.instance_of(ImageReader))

    @property
    def grid_shape(self):
        """Return shape of tile grid, if tile positions do form a grid."""
        pos = self.positions
        shape = np.array([len(set(pos[:, d])) for d in range(2)])
        if np.prod(shape) != len(self):
            raise ValueError("Series positions do not form a grid")
        return shape

    @property
    def centers(self):
        """Return array of Y, X tile centers."""
        return self.positions + self.tile_shape / 2

    @property
    def origin(self):
        """Return array of minimum Y, X coordinates."""
        return geometry.Vector.from_ndarray(np.min(self.positions, axis=0))

    @property
    def rectangles(self):
        """Return list of Rectangles representing tiles."""
        ts = geometry.Vector.from_ndarray(self.tile_shape)
        rectangles = [
            geometry.Rectangle.from_shape(geometry.Vector.from_ndarray(p), ts)
            for p in self.positions
        ]
        return rectangles

    @property
    def plot(self):
        """Return plotter utility object (see plot.TileSetPlotter)."""
        return plot.TileSetPlotter(self)

    def build_neighbors_graph(self, bias=0):
        """Return graph of neighboring tiles.

        Tiles are considered neighboring if their bounding rectangles overlap.
        The `bias` parameter will expand or contract the rectangles for a more
        or less inclusive test. By default (`bias`=0) only strictly overlapping
        tiles are counted as neighbors.

        """
        recs = [r.inflate(bias) for r in self.rectangles]
        overlaps = [[r1.intersection(r2).area for r2 in recs] for r1 in recs]
        graph = nx.from_edgelist(
            (t1, t2) for t1, t2 in zip(*np.nonzero(overlaps)) if t1 < t2
        )
        return graph

    def get_tile(self, tile_number, channel):
        img = self._reader.read(tile_number, channel)
        bounds = self.rectangles[tile_number]
        tile = Tile(tile_number, channel, img, bounds)
        return tile

    def __len__(self):
        return len(self.positions)


@attr.s(frozen=True)
class Tile(object):
    tile_number = attr.ib()
    channel = attr.ib()
    img = attr.ib(validator=attr.validators.instance_of(np.ndarray))
    bounds = attr.ib(validator=attr.validators.instance_of(geometry.Rectangle))


@attr.s(frozen=True)
class TilePair(object):
    tile1 = attr.ib(validator=attr.validators.instance_of(Tile))
    tile2 = attr.ib(validator=attr.validators.instance_of(Tile))
    padding = attr.ib()
