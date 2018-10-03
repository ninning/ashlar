from __future__ import division, print_function
import warnings
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
import numpy as np
import attr
from .metadata import TileSetMetadata

import jnius_config
if not jnius_config.vm_running:
    pkg_root = pathlib.Path(__file__).parent.resolve()
    bf_jar_path = pkg_root / 'jars' / 'loci_tools.jar'
    if not bf_jar_path.exists():
        raise RuntimeError("loci_tools.jar missing from distribution"
                           " (expected it at %s)" % bf_jar_path)
    jnius_config.add_classpath(str(bf_jar_path))
import jnius


JString = jnius.autoclass('java.lang.String')
DebugTools = jnius.autoclass('loci.common.DebugTools')
IFormatReader = jnius.autoclass('loci.formats.IFormatReader')
MetadataStore = jnius.autoclass('loci.formats.meta.MetadataStore')
ServiceFactory = jnius.autoclass('loci.common.services.ServiceFactory')
OMEXMLService = jnius.autoclass('loci.formats.services.OMEXMLService')
ChannelSeparator = jnius.autoclass('loci.formats.ChannelSeparator')
UNITS = jnius.autoclass('ome.units.UNITS')

# Work around pyjnius #300. Passing a python string directly here corrupts the
# value under Python 3, but explicitly converting it into a Java string works.
DebugTools.enableLogging(JString("ERROR"))

pixel_dtypes = {
    'uint8': np.uint8,
    'uint16': np.uint16,
}
ome_dtypes = {v: k for k, v in pixel_dtypes.items()}


@attr.s(frozen=True)
class BioformatsReader(object):
    path = attr.ib()
    bf_reader = attr.ib()
    bf_metadata = attr.ib()

    @classmethod
    def from_path(cls, path):
        factory = ServiceFactory()
        service = jnius.cast(OMEXMLService, factory.getInstance(OMEXMLService))
        bf_metadata = service.createOMEXMLMetadata()
        bf_reader = ChannelSeparator()
        bf_reader.setMetadataStore(bf_metadata)
        # FIXME Workaround for pyjnius #300.
        bf_reader.setId(JString(path))
        return cls(path, bf_reader ,bf_metadata)

    @property
    def metadata(self):
        """Return a TileSetMetadata object representing this dataset."""
        return TileSetMetadata(
            self.pixel_dtype, self.pixel_size, self.tile_shape, self.positions
        )

    @property
    def pixel_dtype(self):
        # FIXME verify all images have the same dtype.
        ome_dtype = self.bf_metadata.getPixelsType(0).value
        dtype = pixel_dtypes.get(ome_dtype)
        if dtype is None:
            raise ValueError("can't handle pixel type: '{}'".format(ome_dtype))
        return dtype

    @property
    def pixel_size(self):
        # FIXME verify all images have the same pixel size.
        quantities = [
            self.bf_metadata.getPixelsPhysicalSizeY(0),
            self.bf_metadata.getPixelsPhysicalSizeX(0)
        ]
        values = [
            length_as_microns(q, "pixel size") for q in quantities
        ]
        if values[0] != values[1]:
            raise ValueError(
                "can't handle non-square pixels: ({}, {})".format(values)
            )
        return values[0]

    @property
    def tile_shape(self):
        # FIXME verify all images have the same shape.
        quantities = [
            self.bf_metadata.getPixelsSizeY(0),
            self.bf_metadata.getPixelsSizeX(0)
        ]
        shape = np.array([q.value for q in quantities], dtype=int)
        return shape

    @property
    def positions(self):
        positions = np.array([
            self.get_position(i) for i in range(self.num_tiles)
        ])
        return positions

    def get_position(self, idx):
        """Return stage position Y, X in microns of one image."""
        # FIXME verify all planes have the same X,Y position.
        quantities = [
            self.bf_metadata.getPlanePositionY(idx, 0),
            self.bf_metadata.getPlanePositionX(idx, 0)
        ]
        values = [
            length_as_microns(q, "stage coordinates") for q in quantities
        ]
        position = np.array(values, dtype=float)
        if not self.is_metamorph_stk:
            # Except for Metamorph STK, invert Y so that stage position
            # coordinates and image pixel coordinates are aligned.
            # FIXME Ask BioFormats team about handling this in the Reader API.
            position *= [-1, 1]
        return position

    @property
    def num_images(self):
        return self.bf_metadata.imageCount

    @property
    def num_tiles(self):
        num_tiles = self.num_images
        # Skip final overview slide in Metamorph Slide Scan data if present.
        if self.is_metamorph_stk and self.has_overview_image:
            num_tiles -= 1
        return num_tiles

    @property
    def format_name(self):
        return self.bf_reader.getFormat()

    @property
    def is_metamorph_stk(self):
        return self.format_name == 'Metamorph STK'

    @property
    def has_overview_image(self):
        last_image_name = self.bf_metadata.getImageName(self.num_images - 1)
        return 'overview' in last_image_name.lower()


def length_as_microns(quantity, name):
    """Return a length quantity's value in microns.

    The `name` of the quantity is used to format a warning message on conversion
    failure.

    """
    value = quantity.value(UNITS.MICROMETER)
    if value is None:
        # Conversion failed, which happens when the unit is "reference
        # frame". Take the bare value as microns, but emit a warning.
        # FIXME Figure out what "reference frame" means and handle this better.
        warnings.warn("No units for {}, assuming micrometers.".format(name))
        value = quantity.value()
    return value.doubleValue()
