import numpy as np
import matplotlib.pyplot as plt
import modest_image
from ashlar import reg

reader1 = reg.BioformatsReader('input/mmo_test/Original/Scan 20x obj 25.nd')
aligner1 = reg.EdgeAligner(reader1, channel=2, verbose=True)
aligner1.run()

reader2 = reg.BioformatsReader('input/mmo_test/Rescanned/25.nd')
reader2.metadata.positions
reader2.metadata._positions -= np.array([39670, 23238])
aligner2 = reg.LayerAligner(reader2, aligner1, verbose=True)
aligner2.run()

mosaic1 = reg.Mosaic(
    aligner1, aligner1.mosaic_shape, 'output/mmo_test/1_original_{channel}.tif',
    verbose=True, channels=[2]
)
(img1,) = mosaic1.run(mode='return')
#mosaic1.run()
mosaic2 = reg.Mosaic(
    aligner2, aligner1.mosaic_shape, 'output/mmo_test/2_rescanned_{channel}.tif',
    verbose=True, channels=[2]
)
(img2,) = mosaic2.run(mode='return')
#mosaic2.run()
