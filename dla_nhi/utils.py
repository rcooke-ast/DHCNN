import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

filename = '../data/DR1_quasars_master_NODLA.csv'
t = Table.read(filename, format='ascii.csv')
