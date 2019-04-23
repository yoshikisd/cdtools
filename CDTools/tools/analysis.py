from __future__ import division, print_function

import torch as t
import numpy as np
from CDTools.tools import cmath



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
            ortho_probe -= basis * coefficients[i,j]
             

        coefficients[i,i] = np.sqrt(np.sum(np.abs(ortho_probe)**2))
        bases.append(ortho_probe / coefficients[i,i])

    density_mat = np.conj(coefficients).transpose().dot(coefficients)
    eigvals, eigvecs = np.linalg.eigh(density_mat)

    ortho_probes = []
    for i in range(len(eigvals)):
        coefficients = np.sqrt(eigvals[i]) * eigvecs[:,i]
        probe = np.zeros(bases[0].shape, dtype=np.complex64)
        for coefficient, basis in zip(coefficients, bases):
            probe += basis * coefficient
        ortho_probes.append(probe)

        
    return cmath.complex_to_torch(np.stack(ortho_probes))
    
        
            
