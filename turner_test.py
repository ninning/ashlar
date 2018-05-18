import numpy as np
import matplotlib.pyplot as plt
import modest_image
from ashlar import reg

reader1 = reg.BioformatsReader('/var/jmuhlich/mmo12_ashlar_test/Original_180328_Set_2/Scan 20x obj 3.nd')
aligner1 = reg.EdgeAligner(reader1, channel=0, max_shift=30, verbose=True)
aligner1.run()

reader2 = reg.BioformatsReader('/var/jmuhlich/mmo12_ashlar_test/Rescanned_180328_Set_2/Scan 20x obj 3.nd')
reader2.metadata.positions
reader2.metadata._positions -= np.array([76, -28])
aligner2 = reg.LayerAligner(
    reader2, aligner1, angle_range=(-7.5, -6.8), verbose=True
)
aligner2.run()

mosaic1 = reg.Mosaic(
    aligner1, aligner1.mosaic_shape, 'output/set_1/1_original_{channel}.tif',
    verbose=True, channels=[0]
)
imgs1 = mosaic1.run(mode='return')
#mosaic1.run()
mosaic2 = reg.Mosaic(
    aligner2, aligner1.mosaic_shape, 'output/set_1/2_rescanned_{channel}.tif',
    verbose=True, channels=[0]
)
imgs2 = mosaic2.run(mode='return')
#mosaic2.run()
