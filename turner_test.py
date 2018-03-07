import numpy as np
import matplotlib.pyplot as plt
import modest_image
from ashlar import reg

reader1 = reg.LooseFilesReader(
    path='input/mmo_test/Original/',
    pattern='Scan 20x obj 25_w{channel}_s{index}_t1.TIF',
    skip_images=(range(0,64,8) +  range(1,64,8) + [64])
)
reader2 = reg.LooseFilesReader(
    path='input/mmo_test/Rescanned/',
    pattern='25_w{channel}_s{index}_t1.TIF',
    skip_images=(range(6,56,7) + [56])
)

aligner1 = reg.EdgeAligner(reader1, channel=2, verbose=True)
aligner1.run()

shift = (reader2.metadata.positions - reader1.metadata.positions).mean(axis=0)
reader2.metadata._positions -= shift
aligner2 = reg.LayerAligner(reader2, aligner1, verbose=True)
aligner2.run()

mosaic1 = reg.Mosaic(
    aligner1, aligner1.mosaic_shape, 'output/mmo_test/1_original_{channel}.tif',
    verbose=True
)
mosaic1.run()
mosaic2 = reg.Mosaic(
    aligner2, aligner1.mosaic_shape, 'output/mmo_test/2_rescanned_{channel}.tif',
    verbose=True
)
mosaic2.run()
