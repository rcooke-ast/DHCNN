import numpy as np
from astropy.table import Table

t = Table.read('DR1_quasars_master.csv', format='ascii.csv')
t_nondla = t[np.where(t['DLAzabs'].mask)]
t_nondla.write('DR1_quasars_master_NODLA.csv', format='ascii.csv', overwrite=True)
