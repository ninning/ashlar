from __future__ import division, print_function
import attr
import numpy as np
import scipy.spatial.distance
import networkx as nx
from .util import array_copy_immutable
from .plot import TileSetMetadataPlotter


@attr.s(frozen=True)
class TileSetMetadata(object):
    """Tile set metadata.

    Tile position coordinates are always kept in microns, not pixels!
    """
    pixel_dtype = attr.ib()
    pixel_size = attr.ib()
    num_channels = attr.ib()
    tile_shape = attr.ib(converter=array_copy_immutable)
    positions = attr.ib(converter=array_copy_immutable)

    @property
    def tile_shape_microns(self):
        return self.tile_shape * self.pixel_size

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
        return np.min(self.positions, axis=0)

    @property
    def plot(self):
        """Return plotter utility object (see MetadataPlotter)."""
        return TileSetMetadataPlotter(self)

    def build_neighbors_graph(self, bias=0):
        """Return graph of neighboring (overlapping or abutting) tiles.

        Tiles are considered neighbors if the 'city block' distance between them
        is less than the largest tile dimension plus `bias`.

        By default (`bias`=0) only strictly overlapping tiles are counted as
        neighbors. Increasing `bias` will include tiles that just touch or are
        slightly separated. Decreasing `bias` will exclude tiles that only
        slightly overlap.

        """
        # FIXME: This should properly test for overlap, possibly via
        # intersection of bounding rectangles.
        pdist = scipy.spatial.distance.pdist(self.positions, metric='cityblock')
        sp = scipy.spatial.distance.squareform(pdist)
        max_distance = np.max(self.tile_shape_microns) + bias
        edges = zip(*np.nonzero((sp > 0) & (sp < max_distance)))
        graph = nx.from_edgelist(edges)
        return graph

    def __len__(self):
        return len(self.positions)
