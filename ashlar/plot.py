import warnings
import attr
import numpy as np
import networkx as nx

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    # FIXME Figure out what to do here.
    warnings.warn("plotting not available; please install matplotlib")
    pass


@attr.s(frozen=True)
class TileSetMetadataPlotter(object):
    """TileSetMetadata plotting helper

    Call one of the plot methods explicitly, or call the plotter itself for
    the default plot, `scatter`.
    """
    metadata = attr.ib()

    def __call__(self, **kwargs):
        self.scatter(**kwargs)

    def scatter(self, ax=None, **kwargs):
        """Create a scatter plot of the tile positions."""
        if ax is None:
            ax = plt.gca()
        y, x = self.metadata.positions.T
        ax.scatter(x, y, **kwargs)
        ax.set_aspect('equal')

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
            ax = plt.gca()
        nx.draw(g, ax=ax, pos=pos, with_labels=True, **kwargs)
        ax.set_aspect('equal')

    def rectangles(self, ax=None, **kwargs):
        """Draw a rectangle representing each tile's position and size."""
        for r in self.metadata.rectangles:
            draw_rectangle(r, ax, **kwargs)


def draw_rectangle(rect, ax=None, **kwargs):
    defaults = dict(color='black', fill=False, lw=0.5)
    for k, v in defaults.items():
        kwargs.setdefault(k, v)
    if ax is None:
        ax = plt.gca()
    xy = (rect.vector1.x, rect.vector1.y)
    w = rect.shape.x
    h = rect.shape.y
    mrect = mpatches.Rectangle(xy, w, h, **kwargs)
    ax.add_patch(mrect)
    ax.autoscale_view()
    ax.set_aspect('equal')
    return mrect
