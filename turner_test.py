import numpy as np
from ashlar import reg

reader1 = reg.LooseFilesReader(
    path='input/mmo_test/Original/',
    pattern='Scan 20x obj 25_w{channel}_s{index}_t1.TIF',
    skip_images=[64]
)
reader2 = reg.LooseFilesReader(
    path='input/mmo_test/Rescanned/',
    pattern='25_w{channel}_s{index}_t1.TIF',
    skip_images=[56]
)

aligner1 = reg.EdgeAligner(reader1, verbose=True)
aligner1.run()

mosaic1 = reg.Mosaic(aligner1, aligner1.mosaic_shape, '', channels=[0], verbose=True)
(img1,) = mosaic1.run(mode='return')

aligner2 = reg.EdgeAligner(reader2, verbose=True)
aligner2.run()

mosaic2 = reg.Mosaic(aligner2, aligner2.mosaic_shape, '', channels=[0], verbose=True)
(img2,) = mosaic2.run(mode='return')

