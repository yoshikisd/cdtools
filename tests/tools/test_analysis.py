from __future__ import division, print_function, absolute_import

import pytest
import numpy as np
import torch as t
from itertools import combinations

from CDTools.tools import analysis, cmath, initializers


def test_orthogonalize_probes():

    # The test strategy should be to define a few non-orthogonal probes
    # and orthogonalize them. Then we can test two features of the results:

    # 1) Are they orthogonal?
    # 2) Is the total intensity at each point the same as it was originally?

    probe_xs = np.arange(128) - 64
    probe_ys = np.arange(150) - 75
    probe_Ys, probe_Xs = np.meshgrid(probe_ys, probe_xs)
    probe_Rs = np.sqrt(probe_Xs**2 + probe_Ys**2)

    probes = np.array([10*np.exp(-probe_Rs**2 / (2 * 10**2)),
                       3*np.exp(-probe_Rs**2 / (2 * 12**2)),
                       1*np.exp(-probe_Rs**2 / (2 * 15**2))]).astype(np.complex64)

    ortho_probes = cmath.torch_to_complex(analysis.orthogonalize_probes(probes))
    
    for p1,p2 in combinations(ortho_probes,2):
        assert np.sum(np.conj(p1)*p2) / np.sum(np.abs(p1)**2) < 1e-6

    for probe in ortho_probes:
        print(np.sum(np.abs(probe)**2))

    for probe in probes:
        print(np.sum(np.abs(probe)**2))
        
    probe_intensity = np.sum(np.abs(probes)**2,axis=0)
    ortho_probe_intensity = np.sum(np.abs(ortho_probes)**2,axis=0)

    assert np.allclose(probe_intensity,ortho_probe_intensity)



def test_standardize():

    # Start by making a probe and object that should meet the standardization
    # conditions
    probe = initializers.gaussian((230,240),(20,20),curvature=(0.01,0.01))
    probe = cmath.torch_to_complex(probe)
    probe = probe * np.sqrt(len(probe.ravel()) / np.sum(np.abs(probe)**2))
    probe = probe * np.exp(-1j * np.angle(np.sum(probe)))

    assert np.isclose(1, np.sum(np.abs(probe)**2)/ len(probe.ravel()))
    assert np.isclose(0,np.angle(np.sum(probe)))

    obj = 30 * np.random.rand(230,240) * np.exp(1j * (np.random.rand(230,240) - 0.5))
    obj_slice = np.s_[(obj.shape[0]//8)*3:(obj.shape[0]//8)*5,
                      (obj.shape[1]//8)*3:(obj.shape[1]//8)*5]

    obj = obj * np.exp(-1j * np.angle(np.sum(obj[obj_slice])))
    assert np.isclose(0,np.angle(np.sum(obj[obj_slice])))

    
    # Then make a nonstandard version of them and standardize it
    # First, don't add a phase ramp and test
    test_probe = probe * 37.6 * np.exp(1j*0.35)
    test_obj = obj / 37.6 * np.exp(1j*1.43)
    s_probe, s_obj = analysis.standardize(test_probe, test_obj)
    assert np.allclose(probe, s_probe)
    assert np.allclose(obj, s_obj)

    # Test that it works on torch tensors
    s_probe, s_obj = analysis.standardize(cmath.complex_to_torch(test_probe).to(t.float32), cmath.complex_to_torch(test_obj).to(t.float32))
    s_probe = cmath.torch_to_complex(s_probe)
    s_obj = cmath.torch_to_complex(s_obj)
    assert np.allclose(probe, s_probe)
    assert np.allclose(obj, s_obj)

    
    # Then do one with a phase ramp
    phase_ramp_dir = (np.random.rand(2) - 0.5)

    probe_Xs, probe_Ys = np.mgrid[:probe.shape[0],:probe.shape[1]]
    phase_ramp = np.exp(1j*probe_Ys * phase_ramp_dir[1]+
                        1j*probe_Xs * phase_ramp_dir[0])
    test_probe = test_probe * phase_ramp

    obj_Xs, obj_Ys = np.mgrid[:obj.shape[0],:obj.shape[1]]
    obj_phase_ramp = np.exp(-1j*obj_Ys * phase_ramp_dir[1]+
                        -1j*obj_Xs * phase_ramp_dir[0])
    test_obj = test_obj * obj_phase_ramp

    s_probe, s_obj = analysis.standardize(test_probe, test_obj, correct_ramp=True)

    assert np.max(s_probe - probe) / np.max(np.abs(probe)) < 1e-4
    assert np.max(s_obj - obj) / np.max(np.abs(obj)) < 1e-4

    # Finally a test with the phase ramp and multiple probes
    subdominant_probe = 0.1*np.random.rand(230,240) * np.exp(1j * (np.random.rand(230,240) - 0.5))
    subdominant_probe = subdominant_probe * np.exp(-1j * np.angle(np.sum(subdominant_probe)))
    test_subdominant_probe = subdominant_probe * 37.6
    test_subdominant_probe = test_subdominant_probe * phase_ramp

    incoh_probe = np.array([test_probe,test_subdominant_probe])

    s_probe, s_obj = analysis.standardize(incoh_probe, test_obj, correct_ramp=True)

    assert np.max(s_probe[0] - probe) / np.max(np.abs(probe)) < 1e-4
    assert np.max(s_obj - obj) / np.max(np.abs(obj)) < 1e-4
    assert np.max(s_probe[1] - subdominant_probe) / np.max(np.abs(subdominant_probe)) < 1e-4



from matplotlib import pyplot as plt
def test_synthesize_reconstructions():
    # Not really sure how to test this to be honest

    # Perhaps it's just best to test for a lack of failures?
    pass


def test_calc_consistency_prtf():

    # Create an object with a specific structure
    obj = 30 * np.random.rand(1030,1040) * np.exp(1j * (np.random.rand(1030,1040) - 0.5))

    #
    synth_obj = np.sqrt(0.7) * obj

    obj_stack = [obj]#,obj,obj,obj]

    basis = np.array([[0,2,0],
                      [3,0,0]])
    
    freqs, prtf = analysis.calc_consistency_prtf(synth_obj, obj_stack, basis)
    assert np.allclose(prtf, 0.7)
    
    freqs, prtf = analysis.calc_consistency_prtf(synth_obj, obj_stack, basis, nbins=30)
    assert np.allclose(prtf, 0.7)

    # Check that is uses the right number of bins
    assert len(prtf) == 30
    assert len(freqs) == 30
    
    # Check that the maximum frequency is correct for the basis
    assert np.isclose(freqs[-1]-freqs[-2] + freqs[-1], np.sqrt(1/4**2 + 1/6**2))
    
