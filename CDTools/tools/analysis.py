from __future__ import division, print_function

import torch as t
import numpy as np
from CDTools.tools import cmath
from CDTools.tools import image_processing as ip

__all__ = ['orthogonalize_probes','standardize']

from matplotlib import pyplot as plt
def orthogonalize_probes(probes):
    """Orthogonalizes a set of incoherently mixing probes
    
    The strategy is to define a reduced orthogonal basis that spans
    all of the retrieved probes, and then build the density matrix
    defined by the probes in that basis. After diagonalization, the
    eigenvectors can be recast into the original basis and returned
    
    Args:
        probes (t.Tensor) : n x (image) size tensor, a stack of probes
    
    Returns:
        (t.Tensor) : n x (image) size tensor, a stack of probes
    """

    try:
        probes = cmath.torch_to_complex(probes.detach().cpu())
    except:
        pass

    bases = []
    coefficients = np.zeros((probes.shape[0],probes.shape[0]), dtype=np.complex64)
    for i, probe in enumerate(probes):
        ortho_probe = np.copy(probe)
        for j, basis in enumerate(bases):
            coefficients[j,i] = np.sum(basis.conj()*ortho_probe)
            ortho_probe -= basis * coefficients[j,i]
             

        coefficients[i,i] = np.sqrt(np.sum(np.abs(ortho_probe)**2))
        bases.append(ortho_probe / coefficients[i,i])


    density_mat = coefficients.dot(np.conj(coefficients).transpose())
    eigvals, eigvecs = np.linalg.eigh(density_mat)

    ortho_probes = []
    for i in range(len(eigvals)):
        coefficients = np.sqrt(eigvals[i]) * eigvecs[:,i]
        print(coefficients)
        probe = np.zeros(bases[0].shape, dtype=np.complex64)
        for coefficient, basis in zip(coefficients, bases):
            probe += basis * coefficient
        ortho_probes.append(probe)

        
    return cmath.complex_to_torch(np.stack(ortho_probes[::-1]))
    
        


def standardize(probe, obj, obj_slice=None, correct_ramp=False):
    """Standardizes a probe and object to prepare them for comparison

    There are a number of ambiguities in the definition of a ptychographic
    reconstruction. This function makes an explicit choice for each ambiguity
    to allow comparisons between independent reconstructions without confusing
    these ambiguities for real differences between the reconstructions.

    The ambiguities and standardizations are:
    * Probe and object can be scaled inversely to one another
        * So we set the probe intensity to an average per-pixel value of 1
    * The probe and object can aquire equal and opposite phase ramps
        * So we set the centroid of the FFT of the probe to zero frequency
    * The probe and object can each acquire an arbitrary overall phase
        * So we set the phase of the sum of all values of both the probe and object to 0

    When dealing with the properties of the object, a slice is used by
    default as the edges of the object often are dominated by unphysical
    noise. The default slice is from 3/8 to 5/8 of the way across.

    Args:
        probe (t.tensor) : tensor or numpy array storing a retrieved probe
        obj (t.tensor) : tensor or numpy array storing a retrieved probe
        obj_slice (slice) : optional, a slice to take from the object for calculating normalizations
        correct_ramp (bool) : Default False, whether to correct for the relative phase ramps

    """
    # First, we normalize the probe intensity to a fixed value.
    probe_np = False
    if isinstance(probe, np.ndarray):
        probe = cmath.complex_to_torch(probe).to(t.float32)
        probe_np = True
    obj_np = False
    if isinstance(obj, np.ndarray):
        obj = cmath.complex_to_torch(obj).to(t.float32)
        obj_np = True
        
    normalization = t.sqrt(t.sum(cmath.cabssq(probe)) / (len(probe.view(-1))/2))
    probe = probe / normalization
    obj = obj * normalization

    # Default slice of the object to use for alignment, etc.
    if obj_slice is None:
        obj_slice = np.s_[(obj.shape[0]//8)*3:(obj.shape[0]//8)*5,
                          (obj.shape[1]//8)*3:(obj.shape[1]//8)*5]

    
    if correct_ramp:
        # Need to check if this is actually working and, if noy, why not
        center_freq = ip.centroid_sq(cmath.fftshift(t.fft(probe,2)),comp=True)
        center_freq -= (t.tensor(probe.shape[:-1]) // 2).to(t.float32)
        center_freq /= t.tensor(probe.shape[:-1]).to(t.float32)


    
        Is, Js = np.mgrid[:probe.shape[0],:probe.shape[1]]
        probe_phase_ramp = cmath.expi(2*np.pi *
                                      (center_freq[0] * t.tensor(Is).to(t.float32) +
                                       center_freq[1] * t.tensor(Js).to(t.float32)))
        probe = cmath.cmult(probe, cmath.cconj(probe_phase_ramp))
        Is, Js = np.mgrid[:obj.shape[0],:obj.shape[1]]
        obj_phase_ramp = cmath.expi(2*np.pi *
                                    (center_freq[0] * t.tensor(Is).to(t.float32) +
                                     center_freq[1] * t.tensor(Js).to(t.float32)))
        obj = cmath.cmult(obj, obj_phase_ramp)

    
    # Then, we set them to consistent absolute phases
    probe_angle = cmath.cphase(t.sum(probe,dim=(0,1)))
    obj_angle = cmath.cphase(t.sum(obj[obj_slice],dim=(0,1)))

    probe = cmath.cmult(probe, cmath.expi(-probe_angle))
    obj = cmath.cmult(obj, cmath.expi(-obj_angle))

    if probe_np:
        probe = cmath.torch_to_complex(probe.detach().cpu())
    if obj_np:
        obj = cmath.torch_to_complex(obj.detach().cpu())
    
    return probe, obj
    
