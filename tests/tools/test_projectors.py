from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
from CDTools.tools import projectors
import numpy as np
import torch as t
from scipy.fftpack import fftshift, ifftshift

def test_modulus():
    # Create a complex array with modulus 12 and phase pi/4
    np_result = np.sqrt(6) * (1 + 1j) * np.ones((10,10))
    # Test without masks
    assert np.allclose(cmath.torch_to_complex(projectors.modulus(t.ones((10,10,2)), 12*t.ones((10,10)))), np_result)
    
    # Test with mask
    mask = t.ones((10,10,2), dtype = t.uint8)
    mask[5]*=0
    np_result[5] = 1+1j
    print(mask)
    print(cmath.torch_to_complex(projectors.modulus(t.ones((10,10,2)), 12*t.ones((10,10)), mask = mask)))
    assert np.allclose(cmath.torch_to_complex(projectors.modulus(t.ones((10,10,2)), 12*t.ones((10,10)), mask = mask)), np_result)


def test_support():
    # Define a mask as uint8 to make sure it works when support is not
    # the same type as the wavefield
    support = t.zeros((10,10)).to(t.uint8)
    # some masked, some unmasked
    support[:3,:3] = 1
    
    np_result = np.zeros((10,10)).astype(np.complex128)
    np_result[:3,:3] = 1 + 1j
    
    assert(np.allclose(cmath.torch_to_complex(projectors.support(t.ones((10,10,2)), support)), np_result))
