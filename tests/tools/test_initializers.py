from __future__ import division, print_function, absolute_import

from CDTools.tools import initializers
from CDTools.tools import cmath
import numpy as np
import torch as t



def test_gaussian():
    # Generate gaussian as a numpy array (square array)
    shape = [10, 10]
    sigma = [2.5, 2.5]
    center = ((shape[0]-1)/2, (shape[1]-1)/2)
    y, x = np.mgrid[:shape[0], :shape[1]]
    np_result = 10*np.exp(-0.5*((x-center[1])/sigma[1])**2
                          -0.5*((y-center[0])/sigma[0])**2)
    init_result = cmath.torch_to_complex(initializers.gaussian([10, 10], 10, [2.5, 2.5]))
    assert np.allclose(init_result, np_result)

    # Generate gaussian as a numpy array (rectangular array)
    shape = [10, 5]
    sigma = [2.5, 2.5]
    center = ((shape[0]-1)/2, (shape[1]-1)/2)
    y, x = np.mgrid[:shape[0], :shape[1]]
    np_result = 10*np.exp(-0.5*((x-center[1])/sigma[1])**2
                          -0.5*((y-center[0])/sigma[0])**2)
    init_result = cmath.torch_to_complex(initializers.gaussian([10, 5], 10, [2.5, 2.5]))
    assert np.allclose(init_result, np_result)
