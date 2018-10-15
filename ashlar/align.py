import numbers
import attr
import numpy as np
import scipy.ndimage
import skimage.feature
import pyfftw

from . import geometry

# Patch np.fft to use pyfftw so skimage utilities can benefit.
np.fft = pyfftw.interfaces.numpy_fft


@attr.s
class TileAlignment(object):
    shift = attr.ib(validator=attr.validators.instance_of(geometry.Vector))
    error = attr.ib()


def register_tiles(tile1, tile2):
    if tile1.pixel_size != tile2.pixel_size:
        raise ValueError("tiles have different pixel sizes")
    shift_pixels, error = register(tile1.image, tile2.image)
    shift = geometry.Vector.from_ndarray(shift_pixels) * tile1.pixel_size
    # TODO Resolve phase confusion by either 1) Testing direct correlation at
    # the 4 possible shifts; or 2) Apply a windowing function before
    # registration.
    shift_adjusted = shift + (tile1.bounds.vector1 - tile2.bounds.vector1)
    return TileAlignment(shift_adjusted, error)


def register(img1, img2, upsample_factor=10):
    """Return translation shift from img2 to img2 and an error metric.

    This function wraps skimage registration to apply our conventions and
    enhancements. We pre-whiten the input images, use optimized FFTW FFT
    functions, always provide fourier-space input images, and mathematically
    transform the skimage-generated error metric in a way that makes it more
    useful for us.

    """
    img1_f = fft2(whiten(img1))
    img2_f = fft2(whiten(img2))
    shift, error, _ = skimage.feature.register_translation(
        img1_f, img2_f, upsample_factor, 'fourier'
    )
    # Recover the intensity-normalized correlation magnitude by inverting the
    # transformation applied by register_translation.
    correlation = np.sqrt(1 - error ** 2)
    # Log-transform and negate the correlation to produce a suitable distance
    # metric for the Dijkstra path calculation, as well as something that
    # produces sensible distributions for the threshold computation.
    error = -np.log(correlation)
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
    scipy.ndimage.convolve(img, _laplace_kernel, output.real)
    return output

    # Other possible whitening functions:
    #img = skimage.filters.roberts(img)
    #img = skimage.filters.scharr(img)
    #img = skimage.filters.sobel(img)
    #img = np.log(img)
    #img = img - scipy.ndimage.filters.gaussian_filter(img, 2) + 0.5
