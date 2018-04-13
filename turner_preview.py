from __future__ import print_function, division
import sys
import numpy as np
import matplotlib.pyplot as plt
import modest_image
from ashlar import reg

reader1 = reg.BioformatsReader('input/mmo_test/Original/Scan 20x obj 25.nd')
reader2 = reg.BioformatsReader('input/mmo_test/Rescanned/25.nd')

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
        reg.paste(mosaic, r.read(c=2, series=i), positions[i])
    print()

    plt.figure()
    ax = plt.gca()

    modest_image.imshow(ax, mosaic)

plt.show()
