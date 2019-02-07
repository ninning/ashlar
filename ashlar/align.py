import itertools
import numbers
import attr
import attr.validators as av
import numpy as np
import scipy.ndimage as ndimage
import skimage.feature
import pyfftw

from . import geometry
from .util import cached_property

# Patch np.fft to use pyfftw so skimage utilities can benefit.
np.fft = pyfftw.interfaces.numpy_fft


@attr.s
class TileAlignment(object):
    shift = attr.ib(validator=av.instance_of(geometry.Vector))
    error = attr.ib()


@attr.s
class EdgeTileAlignment(object):
    alignment = attr.ib(validator=av.instance_of(TileAlignment))
    tile_index_1 = attr.ib()
    tile_index_2 = attr.ib()

    def __attrs_post_init__(self):
        # Normalize so that tile_index_1 < tile_index_2.
        if self.tile_index_1 > self.tile_index_2:
            t1 = self.tile_index_1
            t2 = self.tile_index_2
            new_shift = self.alignment.shift * -1
            new_alignment = attr.evolve(self.alignment, shift=new_shift)
            object.__setattr__(self, 'tile_index_1', t2)
            object.__setattr__(self, 'tile_index_2', t1)
            object.__setattr__(self, 'alignment', new_alignment)

    @cached_property
    def tile_indexes(self):
        return (self.tile_index_1, self.tile_index_2)


def register_tiles(tile1, tile2):
    if tile1.pixel_size != tile2.pixel_size:
        raise ValueError("tiles have different pixel sizes")
    if tile1.bounds.shape != tile2.bounds.shape:
        raise ValueError("tiles have different shapes")
    if tile1.bounds.area == 0:
        raise ValueError("tiles are empty")
    shift_pixels, error = register(tile1.image, tile2.image)
    shift = geometry.Vector.from_ndarray(shift_pixels) * tile1.pixel_size
    shift_adjusted = shift + (tile1.bounds.vector1 - tile2.bounds.vector1)
    return TileAlignment(shift_adjusted, error)


def register(img1, img2, upsample_factor=10):
    """Return translation shift from img2 to img2 and an error metric.

    This function wraps skimage registration to apply our conventions and
    enhancements. We pre-whiten the input images, use optimized FFTW FFT
    functions, always provide fourier-space input images, resolve the phase
    confusion problem, and report an improved (to us) error metric.

    """
    img1w = whiten(img1)
    img2w = whiten(img2)
    img1_f = fft2(img1w)
    img2_f = fft2(img2w)
    img1w = img1w.real
    img2w = img2w.real
    shift, _, _ = skimage.feature.register_translation(
        img1_f, img2_f, upsample_factor, 'fourier'
    )
    # At this point we may have a shift in the wrong quadrant since the FFT
    # assumes the signal is periodic. We test all four possibilities and return
    # the shift that gives the highest direct correlation (sum of products).
    shape = np.array(img1.shape)
    shift_pos = (shift + shape) % shape
    shift_neg = shift_pos - shape
    shifts = list(itertools.product(*zip(shift_pos, shift_neg)))
    correlations = [np.sum(img1w * ndimage.shift(img2w, s)) for s in shifts]
    idx = np.argmax(correlations)
    shift = np.array(shifts[idx])
    correlation = correlations[idx]
    total_amplitude = np.linalg.norm(img1w) * np.linalg.norm(img2w)
    if correlation > 0 and total_amplitude > 0:
        error = -np.log(correlation / total_amplitude)
    else:
        error = np.inf
    return shift, error


def fft2(img):
    return pyfftw.builders.fft2(img, planner_effort='FFTW_ESTIMATE',
                                avoid_copy=True, auto_align_input=True,
                                auto_contiguous=True)()


# Pre-calculate the Laplacian operator kernel. We'll always be using 2D images.
_laplace_kernel = skimage.restoration.uft.laplacian(2, (3, 3))[1]

def whiten(img):
    # Copied from skimage.filters.edges, with explicit aligned output from
    # convolve. Also the mask option was dropped.
    img = skimage.img_as_float(img)
    output = pyfftw.empty_aligned(img.shape, 'complex64')
    output.imag[:] = 0
    ndimage.convolve(img, _laplace_kernel, output.real)
    return output
