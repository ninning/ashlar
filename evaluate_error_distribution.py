from __future__ import print_function, division
import sys
import warnings
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from ashlar import reg

warnings.filterwarnings(
    'ignore', message=r'.*measurement unit is undefined', module=r'ashlar\.reg'
)

path = sys.argv[1]
name = sys.argv[2]

print("%s\n==========" % name)

reader = reg.BioformatsReader(path)

aligner = reg.EdgeAligner(reader, verbose=True, max_shift=30)
aligner.run()
with open('%s.pck' % name, 'wb') as f:
    pickle.dump(aligner, f)

mosaic = reg.Mosaic(
    aligner, aligner.mosaic_shape, '', channels=[0], verbose=True
)
img, = mosaic.run(mode='return')
fig1 = plt.gcf()
reg.plot_edge_quality(aligner, img, show_tree=False, pos='aligner')
plt.savefig('%s-graph.pdf' % name, dpi=600)
plt.close()

def finite(a):
    return a[np.isfinite(a)]

errors = finite(aligner.all_errors)
background = finite(aligner.errors_negative_sampled)
plt.hist(
    [errors, background], bins=40, range=(0, 10),
    histtype='stepfilled', ec='black', alpha=0.7
)
plt.vlines(aligner.max_error, 0, 10)
plt.savefig('%s-error.pdf' % name)
plt.close()

print()
