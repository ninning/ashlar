import os
import itertools
import concurrent.futures
import attr
import attr.validators as av
import numpy as np
from . import metadata, util, align
from .util import attrib, cached_property


@attr.s(frozen=True, kw_only=True)
class RegistrationProcess(object):
    """Container for parameters and algorithms for registration.

    The default `neighbor_overlap_cutoff` percentile of 50 was chosen to reject
    diagonally adjacent tiles in a regular grid. As there are slightly more
    up-down and left-right neighbors than diagonal neighbors in a grid, the 50th
    percentile will correspond to an up-down or left-right neighbor intersection
    area. Unusual tile position collections may require tuning of this
    parameter.

    The default `neighbor_overlap_bias` value of 0 will only consider strictly
    overlapping tiles. Increasing this parameter will also consider touching or
    even disjoint tiles. The typical use case for increasing `bias` is for data
    sets where neighboring stage positions just touch, but due to stage position
    error the imaged regions do have some actual overlap that can be
    registered. Specifying a small bias value will include these touching
    neighbors in the neighbors graph.

    For reproducible random number generation, `random_seed` may be set to a
    fixed value. For further control, a numpy RandomState instance may be passed
    via `random_state`. Use one or the other of these arguments, not both.

    """

    tileset = attrib(
        validator=av.instance_of(metadata.TileSet),
        doc="TileSet to register."
    )
    channel_number = attrib(
        converter=int,
        doc="Index of imaging channel to use for registration."
    )
    neighbor_overlap_cutoff = attrib(
        default=50, converter=float,
        validator=util.validate_range(0, 100),
        doc="Percentile cutoff for determining which tiles are neighbors."
    )
    neighbor_overlap_bias = attrib(
        default=0, converter=float,
        doc="Distance to expand/contract tile bounds before neighbor testing."
    )
    num_permutations = attrib(
        default=1000, converter=int,
        doc="Number of permutations used to sample the error distribution."
    )
    random_seed = attrib(
        default=None, validator=av.optional(av.instance_of(int)),
        doc="Seed for the pseudo-random number generator."
    )
    random_state = attrib(
        default=None,
        validator=av.optional(av.instance_of(np.random.RandomState)),
        doc="A numpy RandomState, constructed using `random_seed` by default."
    )

    def __attrs_post_init__(self):
        if self.random_seed is not None and self.random_state is not None:
            raise ValueError(
                "Can only specify random_seed or random_state, not both."
            )
        if self.random_state is None:
            rand = np.random.RandomState(self.random_seed)
            object.__setattr__(self, 'random_state', rand)

    @cached_property
    def graph(self):
        """Neighbors graph of the tileset."""
        return self.tileset.build_neighbors_graph(
            self.neighbor_overlap_cutoff, self.neighbor_overlap_bias
        )

    def random_tile_pair_index(self):
        return self.random_state.randint(len(self.tileset), size=2)

    def tile_random_neighbor_index(self, i):
        neighbors = list(self.graph.neighbors(i))
        return self.random_state.choice(neighbors)

    def neighbor_permutations(self):
        for i in range(self.num_permutations):
            while True:
                a, b = self.random_tile_pair_index()
                if a != b and (a, b) not in self.graph.edges:
                    break
            a_neighbor = self.tile_random_neighbor_index(a)
            new_b_bounds = self.tileset.rectangles[a_neighbor]
            tile_a = self.tileset.get_tile(a, self.channel_number)
            tile_b = self.tileset.get_tile(b, self.channel_number)
            tile_b = attr.evolve(tile_b, bounds=new_b_bounds)
            yield tile_a.intersection(tile_b), tile_b.intersection(tile_a)

    def neighbor_intersections(self):
        for a, b in self.graph.edges:
            tile_a = self.tileset.get_tile(a, self.channel_number)
            tile_b = self.tileset.get_tile(b, self.channel_number)
            yield a, b, tile_a.intersection(tile_b), tile_b.intersection(tile_a)

    def sample_neighbor_background(self):
        num_workers = len(os.sched_getaffinity(0))
        with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
            def task(args):
                return align.register_tiles(*args)
            results_iter = pool.map(task, self.neighbor_permutations())
            return [a.error for a in results_iter]

    def align_neighbors(self):
        num_workers = len(os.sched_getaffinity(0))
        with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
            def task(args):
                a, b, tile_a, tile_b = args
                alignment = align.register_tiles(tile_a, tile_b)
                return align.EdgeTileAlignment(alignment, a, b)
            results_iter = pool.map(task, self.neighbor_intersections())
            return list(results_iter)
