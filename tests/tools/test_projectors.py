from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
from CDTools.tools import projectors
import numpy as np
import torch as t
from scipy.fftpack import fftshift, ifftshift

def test_modulus():
    # Create a complex array with modulus 12 and phase pi/4
    np_result = 6**.5*(np.ones((10,10))+1j*np.ones((10,10)))

    assert(np.allclose(cmath.torch_to_complex(projectors.modulus(t.ones((10,10,2)), 12*t.ones((10,10)))), np_result))


def test_support():
    # Test masking
    support = t.zeros((10,10))
    np_result = np.zeros((10,10))

    assert(np.allclose(cmath.torch_to_complex(projectors.support(t.ones((10,10,2)), support)), np_result))
