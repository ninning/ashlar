from __future__ import print_function, division
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from ashlar import reg

def onclick(event):
    idx = axes.index(event.inaxes)
    print('%d: %f, %f' % (idx, event.xdata, event.ydata))
    coords[idx, :] = [event.ydata, event.xdata]

reader1 = reg.BioformatsReader(sys.argv[1])
reader2 = reg.BioformatsReader(sys.argv[2])

coords = np.zeros([2,2])
axes = []

for r in [reader1, reader2]:

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

    scale = np.max(mosaic.shape) / 2000
    mosaic = skimage.transform.rescale(mosaic, 1 / scale)
    x1 = metadata.origin
    x2 = metadata.positions.max(axis=0) + metadata.size
    extent = (x1[1], x2[1], x2[0], x1[0])

    fig, ax = plt.subplots()
    axes.append(ax)
    ax.imshow(mosaic, extent=extent)
    fig.canvas.mpl_connect('button_press_event', onclick)

plt.ioff()
plt.show()

print(coords[1] - coords[0])
