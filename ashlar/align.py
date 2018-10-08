import numbers
import attr
import numpy as np
import .geometry as geometry


@attr.s(frozen=True)
class Tile(object):
    img = attr.ib()
    position = attr.ib()


@attr.s(frozen=True)
class TilePair(object):
    tile1 = attr.ib()
    tile2 = attr.ib()
    intersection_shape = attr.ib()
    intersection_padding = attr.ib()
