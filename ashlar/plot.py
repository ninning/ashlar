import attr
import numpy as np
import networkx as nx


@attr.s(frozen=True)
class TileSetMetadataPlotter(object):
    """TileSetMetadata plotting helper

    Call one of the plot methods explicitly, or call the plotter itself for
    the default plot, `scatter`.
    """
    metadata = attr.ib()

    def __call__(self, **kwargs):
        self.scatter(**kwargs)

    @property
    def plt(self):
        import matplotlib.pyplot as plt
        return plt

    def scatter(self, ax=None, **kwargs):
        """Create a scatter plot of the tile positions."""
        if ax is None:
            ax = self.plt.gca()
        y, x = self.metadata.positions.T
        ax.scatter(x, y, **kwargs)
        self._aspect_equal(ax)

    def neighbors_graph(self, ax=None, **kwargs):
        """Draw the neighbors graph using the tile positions for layout."""
        pos = np.fliplr(self.metadata.positions)
        g = self.metadata.neighbors_graph
        if ax is None:
            ax = self.plt.gca()
        nx.draw(g, ax=ax, pos=pos)
        self._aspect_equal(ax)

    def _aspect_equal(self, ax):
        """Set ax to equal aspect ratio mode."""
        ax.set_aspect('equal')
