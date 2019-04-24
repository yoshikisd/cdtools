from __future__ import division, print_function, absolute_import

import pytest
import numpy as np
import torch as t
from itertools import combinations

from CDTools.tools import analysis, cmath


from matplotlib import pyplot as plt
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
