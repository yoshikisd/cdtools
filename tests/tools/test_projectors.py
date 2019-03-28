from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
from CDTools.tools import projectors
import numpy as np
import torch as t


def test_modulus():
    # Create a complex array with random modulus and known phase
    np_result = np.sqrt(6) * (1 + 1j) * np.random.rand(10,10)
    projection_intensity =  t.from_numpy(np.abs(np_result)**2).to(t.float32)
    original_wavefront = cmath.complex_to_torch((1+1j) * np.random.rand(10,10)).to(t.float32)
    # Test without masks
    torch_result = projectors.modulus(original_wavefront,projection_intensity)
    assert np.allclose(cmath.torch_to_complex(torch_result),np_result)
    
    # Test with mask
    mask = t.ones((10,10,2), dtype = t.uint8)
    mask[5]*=0
    np_result[5] = cmath.torch_to_complex(original_wavefront[5])
    torch_result = projectors.modulus(original_wavefront,projection_intensity, mask=mask)
    print(np_result[5])
    print(cmath.torch_to_complex(torch_result)[5])
    assert np.allclose(cmath.torch_to_complex(torch_result),np_result)



def test_support():
    # Define a mask as uint8 to make sure it works when support is not
    # the same type as the wavefield
    support = t.zeros((10,10)).to(t.uint8)
    # some masked, some unmasked
    support[:3,:3] = 1
    
    np_result = np.zeros((10,10)).astype(np.complex128)
    np_result[:3,:3] = 1 + 1j
    
    assert(np.allclose(cmath.torch_to_complex(projectors.support(t.ones((10,10,2)), support)), np_result))
