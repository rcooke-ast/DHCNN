"""
Sample f(N) to generate a list of column densities that can be drawn from random.
"""

import numpy as np
from pyigm.fN.fnmodel import FNModel
from pyigm.fN.mockforest import mk_mock
import utils as dhcnn_utils
import astropy.units as u
from IPython import embed

seed = 12345

# Get a random state so that the noise and components can be reproduced
rstate = np.random.RandomState(seed)
zmax = 3.4
velstep = 2.5
numgen = 1000000  # Some big number!

# Get the CDDF
NHI = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
sply = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
params = dict(sply=sply)
fN_model = FNModel('Hspline', pivots=NHI, param=params, zmnx=(2., 5.))

wave, subwave = dhcnn_utils.generate_wave(wavemax=1240.0 * (1.0 + zmax), velstep=velstep, nsubpix=10)
wave *= u.AA
subwave *= u.AA

# Generate a fake spectrum
print("Note :: I made some changes in pyigm.fn.mockforest.py to avoid convolution and noise")
print("Note :: I made some changes in pyigm.fn.mockforest.py to perform my own subpixellation")
NHIvals = np.zeros(numgen)
cntr = 0
embed()
while True:
    _, HI_comps, mock_subspec = mk_mock(wave, zmax, fN_model, fwhm=0.0, s2n=0.0)

