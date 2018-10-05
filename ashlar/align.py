import attr
import numpy as np


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


@attr.s(frozen=True)
class Point(object):
    x = attr.ib(converter=float)
    y = attr.ib(converter=float)

    def rmin(self, other):
        """Return Point at "rectangle minimum" of self and other."""
        return Point(x=min(self.x, other.x), y=min(self.y, other.y))

    def rmax(self, other):
        """Return Point at "rectangle maximum" of self and other."""
        return Point(x=max(self.x, other.x), y=max(self.y, other.y))

    def __add__(self, other):
        if not isinstance(other, Point):
            raise NotImplementedError
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        if not isinstance(other, Point):
            raise NotImplementedError
        return Point(self.x - other.x, self.y - other.y)

    def __array__(self):
        return np.array([self.y, self.x])


@attr.s(frozen=True)
class Rectangle(object):
    point1 = attr.ib()
    point2 = attr.ib()

    def __attrs_post_init__(self):
        # Enforce that p1 is lower corner and p2 is upper.
        p1 = self.point1.rmin(self.point2)
        p2 = self.point1.rmax(self.point2)
        object.__setattr__(self, 'point1', p1)
        object.__setattr__(self, 'point2', p2)

    def intersection(self, other):
        p1 = self.point1.rmax(other.point1)
        p2 = self.point2.rmin(other.point2)
        if p1.x > p2.x or p1.y > p2.y:
            # FIXME is this the right solution?
            p1 = p2 = self.point1
        return Rectangle(p1, p2)
