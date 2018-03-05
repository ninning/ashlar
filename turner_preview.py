from __future__ import print_function, division
import sys
import numpy as np
import matplotlib.pyplot as plt
import modest_image
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

for r in (reader1, reader2):

    metadata = r.metadata

    positions = metadata.positions - metadata.origin
    far_corners = metadata.positions + metadata.size - metadata.origin
    mshape = np.ceil(far_corners.max(axis=0)).astype(int)
    mosaic = np.zeros(mshape, dtype=np.uint16)

    total = r.metadata.num_images
    for i in range(total):
        sys.stdout.write("\rLoading %d/%d" % (i + 1, total))
        sys.stdout.flush()
        reg.paste(mosaic, r.read(c=0, series=i), positions[i])
    print()

    plt.figure()
    ax = plt.gca()

    modest_image.imshow(ax, mosaic)

    plt.show()
