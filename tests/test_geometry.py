from attr import astuple
from ashlar.geometry import Vector, Rectangle


def rectangle(y1, x1, y2, x2):
    return Rectangle(Vector(y1, x1), Vector(y2, x2))

def test_rectangle_intersection():
    r1 = rectangle(0, 0, 10, 9)
    def intersect(*args):
        return astuple(r1.intersection(rectangle(*args)))
    # Rectangles that fully encompass r1.
    assert intersect(-5, -5, 15, 15) == ((0, 0), (10, 9))
    # Rectangles contained fully within r1.
    assert intersect(2, 3, 7, 9) == ((2, 3), (7, 9))
    # Rectangles that overlap partially.
    assert intersect(5, 5, 15, 15) == ((5, 5), (10, 9))
    assert intersect(5, -5, 15, 5) == ((5, 0), (10, 5))
    assert intersect(5, -5, 15, 15) == ((5, 0), (10, 9))
    # Rectangles that don't overlap at all.
    assert intersect(15, 15, 25, 25) == ((10, 9), (10, 9))
    assert intersect(15, -5, 25, -15) == ((10, 0), (10, 0))
    # Touching along one full edge.
    assert intersect(10, 0, 20, 9) == ((10, 0), (10, 9))
    assert intersect(0, -10, 10, 0) == ((0, 0), (10, 0))
    # Partial overlap along one axis only.
    assert intersect(15, 5, 25, 15) == ((10, 5), (10, 9))
    assert intersect(15, -5, 25, 5) == ((10, 0), (10, 5))
    assert intersect(5, -15, 15, -5) == ((5, 0), (10, 0))
    # Partial overlap along one axis only, and touching.
    assert intersect(10, 5, 20, 15) == ((10, 5), (10, 9))
    assert intersect(10, -5, 20, 5) == ((10, 0), (10, 5))
    assert intersect(5, 0, 15, -10) == ((5, 0), (10, 0))
    # Touching at one corner.
    assert intersect(10, 9, 20, 20) == ((10, 9), (10, 9))
    assert intersect(10, 0, 20, -10) == ((10, 0), (10, 0))
    # Rectangles with zero area.
    assert intersect(5, 5, 5, 5) == ((5, 5), (5, 5))
    assert intersect(10, 5, 10, 5) == ((10, 5), (10, 5))
    assert intersect(15, 5, 15, 5) == ((10, 5), (10, 5))
    assert intersect(15, -5, 15, -5) == ((10, 0), (10, 0))
