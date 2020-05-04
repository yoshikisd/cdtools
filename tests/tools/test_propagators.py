from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
from CDTools.tools import initializers
from CDTools.tools import propagators

import numpy as np
import torch as t
import pytest
import scipy.misc
from scipy.fftpack import fftshift, ifftshift
from matplotlib import pyplot as plt


@pytest.fixture(scope='module')
def exit_waves_1():
    # Import scipy test image and add a random phase
    obj = scipy.misc.ascent()[0:64,0:64].astype(np.complex128)
    arr = np.random.random_sample((64,64))
    obj *= (arr+(1-arr**2)**.5*1j)
    obj = cmath.complex_to_torch(obj)

    # Construct wavefront from image
    probe = initializers.gaussian([64, 64], [5, 5], amplitude=1e3)
    return cmath.cmult(probe,obj)

    

def test_far_field(exit_waves_1):
    # Far field diffraction patterns calculated by numpy with zero frequency in center
    np_result = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cmath.torch_to_complex(exit_waves_1)),norm='ortho'))

    assert(np.allclose(np_result, cmath.torch_to_complex(propagators.far_field(exit_waves_1))))



def test_inverse_far_field(exit_waves_1):
    # We want the inverse far field to map back to the exit waves with no intensity corrections
    # Far field result for exit waves calculated with numpy
    far_field_np_result = cmath.complex_to_torch(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cmath.torch_to_complex(exit_waves_1)),norm='ortho')))

    assert(np.allclose(exit_waves_1, propagators.inverse_far_field(far_field_np_result)))


def test_generate_high_NA_k_intensity_map():

    # We need to generate a plausible scenario. I will start
    # by using the initializer to generate a reasonable exit wave geometry
    # and detector pair
    basis = t.Tensor([[0,-30e-6,0],
                      [-20e-6,0,0]]).transpose(0,1)
    shape = t.Size([478,573])
    #shape = t.Size([3,5])
    wavelength = 1e-9
    distance = 1#6e-3
    rs_basis, full_shape, det_slice = \
        initializers.exit_wave_geometry(basis, shape, wavelength,
                                        distance, opt_for_fft=False)

    k_map, intensity_map = propagators.generate_high_NA_k_intensity_map(
        rs_basis, basis, shape, distance, wavelength,
        dtype=t.float32)

    # generate a good test exit wave
    i = (np.arange(478) - 240)
    j = (np.arange(573) - 270)
    Is,Js = np.meshgrid(i,j,indexing='ij')
    wavefield = ((np.abs(Is) < 20) * (np.abs(Js) < 25)).astype(np.complex128)
    t_wavefield = cmath.complex_to_torch(wavefield).to(dtype=t.float32)

    high_NA_propagated = propagators.high_NA_far_field(
        t_wavefield, k_map, intensity_map=intensity_map)
    low_NA_propagated = propagators.far_field(t_wavefield)

    low_NA = cmath.torch_to_complex(low_NA_propagated)
    high_NA = cmath.torch_to_complex(high_NA_propagated)

    # Checking first that for a low-NA propagation they give the same result
    # 1e-4 tolerance seems to be reasonable in this comparison given my
    # exploration with the code
    #assert np.max(np.abs(high_NA-low_NA))/np.max(np.abs(low_NA)) < 1e-4

    # Now I will explore some results with a tilted sample
    #print(rs_basis)
    #print(rs_basis_tilted)
    distance = 0.01#6e-3
    rs_basis, full_shape, det_slice = \
        initializers.exit_wave_geometry(basis, shape, wavelength,
                                        distance, opt_for_fft=False)
    rs_basis_tilted = rs_basis.clone()
    rs_basis_tilted[2,1] = rs_basis_tilted[0,1]

    
    k_map, intensity_map = propagators.generate_high_NA_k_intensity_map(
        rs_basis_tilted, basis, shape, distance, wavelength,
        dtype=t.float32)
    
    high_NA_propagated = propagators.high_NA_far_field(
        t_wavefield, k_map, intensity_map=intensity_map)
    low_NA_propagated = propagators.far_field(t_wavefield)

    low_NA = cmath.torch_to_complex(low_NA_propagated)
    high_NA = cmath.torch_to_complex(high_NA_propagated)
    
    #plt.close('all')
    #plt.imshow(np.abs(low_NA)) 
    #plt.figure()
    #plt.imshow(np.abs(high_NA))
    #plt.colorbar()
    #plt.imshow(np.abs(wavefield))
    #plt.show()


    # Now I want to test that it doesn't crash for wavefields of various shapes
    propagators.high_NA_far_field(t_wavefield.unsqueeze(0),
                                  k_map, intensity_map=intensity_map)
    propagators.high_NA_far_field(t_wavefield.unsqueeze(0).unsqueeze(0),
                                  k_map, intensity_map=intensity_map)

    # I believe this works, but I still would like to get a second method for
    # simulating at least one diffraction pattern as an independent check
    
    assert 0
    
    

    
    
def test_near_field():

    # The strategy is to compare the propagation of a gaussian beam to
    # the propagation in the paraxial approximation.
    
    x = (np.arange(901) - 450) * 1.5e-9
    y = (np.arange(1200) - 600) * 1e-9
    Ys,Xs = np.meshgrid(y,x)
    Rs = np.sqrt(Xs**2+Ys**2)

    wavelength = 3e-9 #nm
    sigma = 20e-9 #nm
    z = 1000e-9 #nm

    k = 2 * np.pi / wavelength
    w0 = np.sqrt(2)*sigma
    zr = np.pi * w0**2 / wavelength
    wz = w0 * np.sqrt(1 + (z / zr)**2)
    Rz =  z * (1 + (zr / z)**2)
    
    E0 = np.exp(-Rs**2 / w0**2)

    # The analytical expression for propagation of a gaussian beam in the
    # paraxial approx
    Ez = w0 / wz * np.exp(-Rs**2 / wz**2) * np.exp(-1j * k * ( z + Rs**2 / (2 * Rz)) + 1j * np.arctan(z / zr))
    Ez_nozphase = Ez * np.exp(1j * k * z)
    

    # First we check it normally
    asp = propagators.generate_angular_spectrum_propagator(
        E0.shape,(1.5e-9,1e-9),wavelength,z,dtype=t.float64)

    
    Ez_t = propagators.near_field(cmath.complex_to_torch(E0),asp)
    Ez_t = cmath.torch_to_complex(Ez_t)
    
    # Check for at least 10^-3 relative accuracy in this scenario
    assert np.max(np.abs(Ez-Ez_t)) < 1e-3 * np.max(np.abs(Ez))


    Emz = np.conj(Ez)

    Emz_t = propagators.inverse_near_field(cmath.complex_to_torch(E0),asp)
    Emz_t = cmath.torch_to_complex(Emz_t)    

    # Again, 10^-3 is about all the accuracy we can expect
    assert np.max(np.abs(Emz-Emz_t)) < 1e-3 * np.max(np.abs(Emz))

    # Then, we check it with the phase correction
    asp = propagators.generate_angular_spectrum_propagator(
        E0.shape,(1.5e-9,1e-9),wavelength,z,remove_z_phase=True,
        dtype=t.float64)
    
    
    Ez_t = propagators.near_field(cmath.complex_to_torch(E0),asp)
    Ez_t = cmath.torch_to_complex(Ez_t)
    
    # Check for at least 10^-3 relative accuracy in this scenario
    assert np.max(np.abs(Ez_nozphase-Ez_t)) < 1e-3 * np.max(np.abs(Ez_nozphase))


    Emz = np.conj(Ez_nozphase)

    Emz_t = propagators.inverse_near_field(cmath.complex_to_torch(E0),asp)
    Emz_t = cmath.torch_to_complex(Emz_t)    

    # Again, 10^-3 is about all the accuracy we can expect
    assert np.max(np.abs(Emz-Emz_t)) < 1e-3 * np.max(np.abs(Emz))

    
def test_generalized_near_field():

    # The strategy is to compare the propagation of a gaussian beam to
    # the propagation in the paraxial approximation.

    # For this one, we want to test it on a rotated coordinate system
    # First, we should do a test with the phase ramp along the z direction
    # explicitly included

    basis= np.array([[0,-1.5e-9],[-1e-9,0],[0,0]])
    x = (np.arange(901) - 450) * 1.5e-9
    y = (np.arange(1200) - 600) * 1e-9
    Xs_0,Ys_0 = np.meshgrid(x,y)
    Zs_0 = np.zeros(Xs_0.shape)

    Positions = np.stack([Xs_0,Ys_0,Zs_0])
    
    # assert 0
    wavelength = 3e-9 #nm
    sigma = 20e-9 #nm
    z = 1000e-9 #nm

    k = 2 * np.pi / wavelength
    w0 = np.sqrt(2)*sigma
    zr = np.pi * w0**2 / wavelength
    

    # The analytical expression for propagation of a gaussian beam in the
    # paraxial approx
    def get_w(Zs):
        return w0 * np.sqrt(1 + (Zs / zr)**2)

    def get_inv_R(Zs):
        return Zs / (Zs**2 + zr**2)
    
    def get_E(Xs, Ys, Zs, correct=False):
        # if correct is True, remove the e^(-ikz) dependence
        Rs_sq = Xs**2 + Ys**2
        Wzs = get_w(Zs)
        E = w0 / Wzs * np.exp(-Rs_sq / Wzs**2) *\
            np.exp(-1j * k * ( Zs + Rs_sq * get_inv_R(Zs) / 2) + \
                   1j * np.arctan(Zs / zr))
        if correct:
            E = E * np.exp(1j * k * Zs)
        return E
    

    # We make some rotation matrices to test
    
    # This tests the straight ahead case
    I = np.eye(3)

    # This tests a rotation about the y axis
    th = np.deg2rad(5)
    Ry = np.array([[np.cos(th),0,np.sin(th)],
                   [0,1,0],
                   [-np.sin(th),0,np.cos(th)]])
    
    # This tests a rotation about two axes
    phi = np.deg2rad(2)
    Rx = np.array([[1,0,0],
                   [0,np.cos(phi),-np.sin(phi)],
                   [0,np.sin(phi),np.cos(phi)]])
    Rboth = np.matmul(Rx,Ry)

    # This tests a shearing
    shear = 0.23
    Rshear = np.array([[1,shear,0],
                       [0,1,0],
                       [0,0,1]])

    
    # This tests an inversion of the axes
    Rinv = np.array([[-1,0,0],
                     [0,-1,0],
                     [0,0,-1]])

    # This tests a reflection about the y-z plane
    Rrefl = np.array([[-1,0,0],
                      [0,1,0],
                      [0,0,-1]])
    
    # This tests a shearing and a rotation together
    Rall = np.matmul(Rrefl,np.matmul(Rboth,Rshear))


    # And we make some propagation vectors to test:

    # This is along the z direction
    z_dir = np.array([0,0,1])

    # This checks that it's not sensitive to the magnitude
    z_dir_large = np.array([0,0,10])
    
    # And finally some offset vectors

    # This checks straight ahead
    z_offset = np.array([0,0,z])

    
    # This checks with an offset in x and y
    shear_offset = np.array([0.1*z,-0.03*z,z])

    # This checks with an offset in x and y, with negative z
    shear_back_offset = np.array([0.1*z,-0.03*z,-z])

    
    rot_mats = [Rrefl,I,Rinv, Rboth, Rboth,Rboth, Rall, Rall, Rall, I, Rall]
    offset_vecs = [z_offset]*8 + [shear_offset] + [shear_back_offset]*2
    propagation_vecs = ['perp','offset',z_dir,
                        'perp','offset',z_dir_large,
                        'perp','offset',z_dir_large,
                        z_dir, z_dir_large]
    purposes = ['standard']*3 + ['both-rot']*3 + ['shear-rot']*3 + ['backward']*2
    
    for purpose,rot_mat,offset_vec, propagation_vec \
        in zip(purposes,rot_mats,offset_vecs,propagation_vecs):
        
        print('Testing', purpose)
        Xs,Ys,Zs_0 = np.tensordot(rot_mat,Positions,axes=1)
        new_basis = np.dot(rot_mat, basis)
        Xs_prop, Ys_prop, Zs_prop = np.stack([Xs,Ys,Zs_0]) \
            + offset_vec[:,None,None]
        
        print('Propagate Along',propagation_vec)
        
        if str(propagation_vec) == 'perp':
            E0 = get_E(Xs,Ys,Zs_0, correct=False)
            Ez = get_E(Xs_prop,Ys_prop,Zs_prop, correct=False)
            asp = propagators.generate_generalized_angular_spectrum_propagator(
                E0.shape,new_basis,wavelength,offset_vec,dtype=t.float64)
        elif str(propagation_vec) == 'offset':
            E0 = get_E(Xs,Ys,Zs_0, correct=True)
            Ez = get_E(Xs_prop,Ys_prop,Zs_prop, correct=True)
            asp = propagators.generate_generalized_angular_spectrum_propagator(
                E0.shape,new_basis,wavelength,offset_vec,
                dtype=t.float64, propagate_along_offset=True)
        else:
            E0 = get_E(Xs,Ys,Zs_0, correct=True)
            Ez = get_E(Xs_prop,Ys_prop,Zs_prop, correct=True)
            asp = propagators.generate_generalized_angular_spectrum_propagator(
                    E0.shape,new_basis,wavelength,offset_vec,
                    dtype=t.float64, propagation_vector=propagation_vec)

        Ez_t = propagators.near_field(cmath.complex_to_torch(E0),asp)
        Ez_t = cmath.torch_to_complex(Ez_t)
        # Check for at least 10^-3 relative accuracy in this scenario
        if not np.max(np.abs(Ez-Ez_t)) < 1e-3 * np.max(np.abs(Ez)):
            plt.close('all')
            plt.imshow(np.angle(Ez))
            plt.figure()
            plt.imshow(np.angle(Ez_t))
            plt.show()
            
        assert np.max(np.abs(Ez-Ez_t)) < 1e-3 * np.max(np.abs(Ez))
        
            
        Em0_t = propagators.inverse_near_field(cmath.complex_to_torch(Ez),asp)
        Em0_t = cmath.torch_to_complex(Em0_t)    
        
        # Again, 10^-3 is about all the accuracy we can expect
        assert np.max(np.abs(E0-Em0_t)) < 1e-3 * np.max(np.abs(E0))
            
        print('Test Successful')



def test_inverse_near_field():
    
    x = (np.arange(800) - 400) * 1.5e-9
    y = (np.arange(1200) - 600) * 1e-9
    Ys,Xs = np.meshgrid(y,x)
    Rs = np.sqrt(Xs**2+Ys**2)
    
    wavelength = 3e-9 #nm
    sigma = 20e-9 #nm
    z = 1000e-9 #nm

    w0 = np.sqrt(2)*sigma
    E0 = np.exp(-Rs**2 / w0**2)

    asp = propagators.generate_angular_spectrum_propagator(
        E0.shape,(1.5e-9,1e-9),wavelength,z,dtype=t.float64)

    E0 = cmath.complex_to_torch(E0)
    E_prop = propagators.near_field(E0,asp)
    
    E_backprop = propagators.inverse_near_field(E_prop, asp)

    # We just want to check that it actually is the inverse
    assert t.all(t.isclose(E0,E_backprop))
