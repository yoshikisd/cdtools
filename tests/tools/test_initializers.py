from __future__ import division, print_function, absolute_import

from CDTools.tools import initializers
from CDTools.tools import cmath
import numpy as np
import torch as t

def test_exit_wave_geometry():
    # First test a simple case where nothing need change
    basis = t.Tensor([[0,-30e-6,0],
                      [-20e-6,0,0]]).transpose(0,1)
    shape = t.Size([73,56])
    wavelength = 1e-9
    distance = 1.
    rs_basis, full_shape, det_slice = \
        initializers.exit_wave_geometry(basis, shape, wavelength,
                                        distance, opt_for_fft=False)
    assert full_shape == shape
    assert t.ones(full_shape)[det_slice].shape ==  shape
    assert t.allclose(rs_basis[0,1],t.Tensor([-8.928571428571428e-07]))
    assert t.allclose(rs_basis[1,0],t.Tensor([-4.5662100456621004e-07]))
     
    # Then test it's expanding functionality for a non-optimal array
    rs_basis, full_shape, det_slice = \
        initializers.exit_wave_geometry(basis, shape, wavelength,
                                        distance, opt_for_fft=True)
    exp_shape = t.Size([75,60])
    assert full_shape == exp_shape
    assert t.ones(full_shape)[det_slice].shape ==  shape
    assert t.allclose(rs_basis[0,1],t.Tensor([-8.333333333333333e-07]))
    assert t.allclose(rs_basis[1,0],t.Tensor([-4.444444444444444e-07]))

    
    # Then test it's padding function
    rs_basis, full_shape, det_slice = \
        initializers.exit_wave_geometry(basis, shape, wavelength,
                                        distance, opt_for_fft=False,
                                        padding=2)
    exp_shape = t.Size([77,60])
    assert full_shape == exp_shape
    assert t.ones(full_shape)[det_slice].shape ==  shape

    
    # Finally test it off-center, without expanding
    center = t.Tensor([20,42])
    rs_basis, full_shape, det_slice = \
        initializers.exit_wave_geometry(basis, shape, wavelength,
                                        distance, center=center,
                                        opt_for_fft=False)
    exp_shape = t.Size([105,84])
    assert full_shape == exp_shape
    assert t.ones(full_shape)[det_slice].shape ==  shape



def test_calc_object_setup():
    # First just try a simple case
    probe_shape = t.Size([120,57])
    translations = t.rand((30,2)) * 300
    t_max = t.max(translations, dim=0)[0]
    t_min = t.min(translations, dim=0)[0]
    obj_shape, min_translation = initializers.calc_object_setup(probe_shape, translations)
    exp_shape = t.ceil(t_max - t_min).to(t.int32) + t.Tensor(list(probe_shape)).to(t.int32)
    assert t.allclose(min_translation, t_min)

    assert obj_shape == t.Size(exp_shape)
    
    # Then add some padding
    padding = 5
    obj_shape, min_translation = initializers.calc_object_setup(probe_shape, translations, padding=padding)
    assert t.allclose(min_translation, t_min - padding)
    assert obj_shape == t.Size(exp_shape + 2 * padding)
        

    
def test_gaussian():
    # Generate gaussian as a numpy array (square array)
    shape = [10, 10]
    sigma = [2.5, 2.5]
    center = ((shape[0]-1)/2, (shape[1]-1)/2)
    y, x = np.mgrid[:shape[0], :shape[1]]
    np_result = 10*np.exp(-0.5*((x-center[1])/sigma[1])**2
                          -0.5*((y-center[0])/sigma[0])**2)
    init_result = cmath.torch_to_complex(initializers.gaussian([10, 10], [2.5, 2.5], amplitude=10))
    assert np.allclose(init_result, np_result)

    # Generate gaussian as a numpy array (rectangular array)
    shape = [10, 5]
    sigma = [2.5, 3]
    center = ((shape[0]-1)/2, (shape[1]-1)/2)
    y, x = np.mgrid[:shape[0], :shape[1]]
    np_result = np.exp(-0.5*((x-center[1])/sigma[1])**2
                          -0.5*((y-center[0])/sigma[0])**2)
    init_result = cmath.torch_to_complex(initializers.gaussian(shape, sigma))
    assert np.allclose(init_result, np_result)
                                    
    # Generate gaussian with curvature
    shape = [20, 30]
    sigma = [2.5, 5]
    curvature = [1,0.6]
    center = ((shape[0]-1)/2 + 3, (shape[1]-1)/2 - 1.4)
    y, x = np.mgrid[:shape[0], :shape[1]]
    np_result = (10+0j)*np.exp(-0.5*((x-center[1])/sigma[1])**2
                               -0.5*((y-center[0])/sigma[0])**2)
    np_result *= np.exp(0.5j*curvature[1]*(x-center[1])**2
                        +0.5j*curvature[0]*(y-center[0])**2)
    init_result = cmath.torch_to_complex(initializers.gaussian(shape, sigma,
                        center=center, curvature=curvature, amplitude=10))
    assert np.allclose(init_result, np_result)
    
