import numpy as np
from astropy.table import Table

zmin, zmax = 2.5, 3.4
dispreq = 2.5  # Sampling size

# Load the catalogue
t = Table.read('DR1_quasars_master.csv', format='ascii.csv')
# Trim the table to only be quasars in a certain redshift range
t_trim = t[np.where((t['zem_Adopt'].data > zmin) & (t['zem_Adopt'].data < zmax) & (t['Dispersion'].data == dispreq))]
# Find those quasars with no DLAs in this redshift range
nodla = t_trim['DLAzabs'].mask
for dd in range(nodla.size):
    if not nodla[dd]:
        # This qso has a DLA... check the redshifts
        include = True
        dlaspl = t_trim['DLAzabs'].data[dd].split(",")
        for ll in range(len(dlaspl)):
            zabs = float(dlaspl[ll])
            if (zabs > zmin) and (zabs<zmax):
                include = False
        if include:
            nodla[dd] = True

t_final = t_trim[np.where(nodla)]
t_final.write('DR1_quasars_master_trimmed.csv', format='ascii.csv', overwrite=True)
