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
        """Draw the neighbors graph using the tile centers for layout."""
        defaults = dict(
            font_size=6, node_size=100, node_color='orange'
        )
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        ng_kwargs = {}
        try:
            ng_kwargs['bias'] = kwargs.pop('bias')
        except KeyError:
            pass
        g = self.metadata.build_neighbors_graph(**ng_kwargs)
        pos = np.fliplr(self.metadata.centers)
        if ax is None:
            ax = self.plt.gca()
        nx.draw(g, ax=ax, pos=pos, with_labels=True, **kwargs)
        self._aspect_equal(ax)

    def rectangles(self, ax=None, **kwargs):
        """Draw a rectangle representing each tile's position and size."""
        defaults = dict(color='black', fill=False, lw=0.5)
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        if ax is None:
            ax = self.plt.gca()
        for r in self.metadata.rectangles:
            xy = (r.vector1.x, r.vector1.y)
            w = r.shape.x
            h = r.shape.y
            mrect = self.plt.Rectangle(xy, w, h, **kwargs)
            ax.add_patch(mrect)
        ax.autoscale_view()
        self._aspect_equal(ax)

    def _aspect_equal(self, ax):
        """Set ax to equal aspect ratio mode."""
        ax.set_aspect('equal')
