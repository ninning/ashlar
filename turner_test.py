from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import modest_image
from ashlar import reg

reader1 = reg.BioformatsReader(
    'input/mmo12_ashlar_test/Original_180328_Set_3/Scan 20x obj 4.nd',
)
aligner1 = reg.EdgeAligner(reader1, channel=0, max_shift=30, verbose=True)
aligner1.run()

reader2 = reg.BioformatsReader(
    'input/mmo12_ashlar_test/Rescanned_180328_Set_3/Scan 20x obj 4.nd',
)
reader2.metadata.positions
reader2.metadata._positions -= np.array([123.26375708, 80.16952213])
aligner2 = reg.LayerAligner(
    reader2, aligner1, angle_range=(-8, -6.5), verbose=True
)
aligner2.run()

mosaic1 = reg.Mosaic(
    aligner1, aligner1.mosaic_shape, 'output/mmo_180328/set_3/1_original_{channel}.tif',
    verbose=True, channels=[0]
)
imgs1 = mosaic1.run(mode='return')
#mosaic1.run()

mosaic2 = reg.Mosaic(
    aligner2, aligner1.mosaic_shape, 'output/mmo_180328/set_3/2_rescanned_{channel}.tif',
    verbose=True, channels=[0]
)
imgs2 = mosaic2.run(mode='return')
#mosaic2.run()

ic = np.empty(imgs1[0].shape+(3,), dtype=np.uint8)
ic[...,0] = imgs1[0] / 42
ic[...,1] = imgs2[0] / 14
