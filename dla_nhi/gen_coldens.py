"""
Sample f(N) to generate a list of column densities that can be drawn from random.
"""

import numpy as np
from pyigm.fN.fnmodel import FNModel
from pyigm.fN.mockforest import monte_HIcomp
import utils as dhcnn_utils
import astropy.units as u
from astropy import cosmology
from IPython import embed

seed = 12345

# Get a random state so that the noise and components can be reproduced
rstate = np.random.RandomState(seed)
zmin, zmax = 2.6, 3.4
NHImin, NHImax = 19.2, 21.0
velstep = 2.5
numgen = 1000000  # Some big number!

# Get the CDDF
NHI = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
sply = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
params = dict(sply=sply)
fN_model = FNModel('Hspline', pivots=NHI, param=params, zmnx=(2., 5.))
cosmo = cosmology.core.FlatLambdaCDM(70., 0.3)

# Generate a fake spectrum
print("Note :: I made some changes in pyigm.fn.mockforest.py to avoid convolution and noise")
print("Note :: I made some changes in pyigm.fn.mockforest.py to perform my own subpixellation")
NHIvals = np.zeros(numgen)
cntr = 0
embed()
while True:
    try:
        HI_comps = monte_HIcomp((zmin, zmax), fN_model, NHI_mnx=(NHImin, NHImax), bfix=None, cosmo=cosmo, rstate=rstate)
    except ValueError:
        # No lines in that random generation
        continue
    nHI = HI_comps['lgNHI'].size
    if nHI == 0: continue
    NHIvals[cntr:cntr+nHI] = HI_comps['lgNHI'].data
    cntr += nHI
    if cntr > 30:
        break