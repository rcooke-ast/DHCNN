import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

evalfile = 'mnum0.evaluation'
dat = Table.read(evalfile, format='ascii')

for ff in range(13):
    plt.subplot(4,4,ff+1)
    var = 'var_{0:d}'.format(ff+1)
    plt.plot(dat[var], dat['Y'], 'bx')
    plt.title(var)
plt.show()
