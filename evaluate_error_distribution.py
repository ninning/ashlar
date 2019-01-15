from __future__ import print_function, division
import sys
import warnings
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from ashlar import reg

warnings.filterwarnings(
    'ignore', message=r'.*measurement unit is undefined', module=r'ashlar\.reg'
)

path = sys.argv[1]
name = sys.argv[2]

max_shift = 15
if len(sys.argv) >= 4:
    max_shift = float(sys.argv[3])

channel = 0
if len(sys.argv) >= 5:
    channel = float(sys.argv[4])

print("%s\n==========" % name)

reader = reg.BioformatsReader(path)

aligner = reg.EdgeAligner(
    reader, verbose=True, max_shift=max_shift, channel=channel
)
aligner.run()
with open('%s.pck' % name, 'wb') as f:
    pickle.dump(aligner, f)

with PdfPages('%s.pdf' % name) as pdf:

    mosaic = reg.Mosaic(
        aligner, aligner.mosaic_shape, '', channels=[channel], verbose=True
    )
    img, = mosaic.run(mode='return')
    fig1 = plt.figure()
    reg.plot_edge_quality(
        aligner, img, show_tree=False, pos='aligner', use_mi=False,
        nx_kwargs=dict(node_size=6, font_size=2, width=1)
    )
    pdf.savefig(dpi=600)
    plt.close()

    def finite(a):
        return a[np.isfinite(a)]

    errors = finite(aligner.all_errors)
    background = finite(aligner.errors_negative_sampled)
    fig2 = plt.figure(figsize=(7,4))
    plt.hist(
        [errors, background], bins=40, range=(0, 10), density=True,
        histtype='stepfilled', ec='black', alpha=0.7
    )
    plt.axvline(aligner.max_error, c='black', linestyle=':')
    pdf.savefig()
    plt.close()

print()
