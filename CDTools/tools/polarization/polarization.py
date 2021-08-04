import numpy as numpy
import torch as t
import math
from math import sin
from math import cos

__all__ = ['apply_linear_polarizer',
           'apply_phase_retardance',
           'apply_half_wave_plate',
           'apply_quarter_wave_plate',
           'apply_circular_polarizer',
           'apply_jones_matrix']
print()
           
def apply_linear_polarizer(probe, polarizer, multiple_modes=True, transpose=True):
    """
    Applies a linear polarizer to the probe

    Parameters:
    ----------
    probe: t.Tensor
        A (N)(P)x2xMxL tensor representing the probe, MxL - the size of the probe
        The angle between the fast-axis of the linear polarizer and the horizontal axis
    polarizer: t.Tensor
        A 1D tensor (N) representing the polarizer angles for each of the patterns (or a single tensor of shape (1))

    Returns:
    --------
    linearly polarized probe: t.Tensor
        (N)(P)x2x1xMxL 
    """
    # if len(polarizer.shape) == 0:
    #     polarizer = t.tensor([polarizer])
    if len(polarizer,shape) == 0:
        polarizer = t.tensor([polarizer])
    pol_cos = lambda idx: cos(math.radians(polarizer[idx]))
    pol_sin = lambda idx: sin(math.radians(polarizer[idx]))
    jones_matrices = t.stack(([t.tensor([[(pol_cos(idx)) ** 2, pol_sin(idx) * pol_cos(idx)], [pol_sin(idx) * pol_cos(idx), (pol_sin(idx)) ** 2]]).to(dtype=t.cfloat) for idx in range(len(polarizer))]))

    return apply_jones_matrix(probe, jones_matrices, transpose=transpose, multiple_modes=multiple_modes)

def apply_jones_matrix(probe, jones_matrix, transpose=True, multiple_modes=True):
    """
    Applies a given Jones matrix to the probe

    Parameters:
    ----------
    probe: t.Tensor
        A (N)(P)x2xMxL tensor representing the probe
    jones_matrix: t.tensor
        (N)x2x2x(M)x(L) 

    Returns:
    --------
    a probe with the jones matrix applied: t.Tensor
        (N)(P)x2xMxL 

    Assume that if the probe has a dimension (N), so does the jones matrix
    """
    if multiple_modes:
        if transpose:
            if len(jones_matrix.shape) >= 4:
                jones_matrix = jones_matrix[..., None, :, :, :, :]  

            else:
                jones_matrix = jones_matrix[..., None, :, :, None, None]
            probe = probe[..., None, :, :]
            # if jones matrices do not differ from pattern to pattern
            if len(probe.shape) > len(jones_matrix.shape):
                jones_matrix = jones_matrix[None, ...]
            jones_matrix = jones_matrix.transpose(-1, -3).transpose(-2, -4) 
            # (N)1xMxLx2x2 or (N)1x1x1x2x2
            probe = probe.transpose(-1, -3).transpose(-2, -4)
            output = t.matmul(jones_matrix, probe).transpose(-2, -4).transpose(-1, -3).squeeze(-3)
            # (N)Px2xMxL

        else:
            if len(jones_matrix.shape) < 4:
                jones_matrix = jones_matrix[..., None, :, :, :, :]
            probe = t.stack((probe, probe), dim=-4)
            output = t.sum(jones_matrix * probe, dim=-3)
            #(N)x2xMxL

    else:
        if transpose:   
            if len(jones_matrix.shape) < 4:
                jones_matrix = jones_matrix[..., None, None]
            probe = probe[..., None, :, :]
            # if jones matrices do not differ from pattern to pattern
            if len(probe.shape) > len(jones_matrix.shape):
                jones_matrix = jones_matrix[None, ...]
            probe = probe.transpose(-1, -3).transpose(-2, -4)
            jones_matrix = jones_matrix.transpose(-1, -3).transpose(-2, -4)
            output = t.matmul(jones_matrix, probe).transpose(-2, -4).transpose(-1, -3).squeeze(-3)


        else:
            if len(jones_matrix.shape) < 4:
                jones_matrix = jones_matrix[..., None, None]
            probe = t.stack((probe, probe), dim=-4)
            output = t.sum(jones_matrix * probe, dim=-3)
    
    return output


def apply_phase_retardance(probe, phase_shift):
    """
    Shifts the y-component of the field wrt the x-component by a given phase shift 

    Parameters:
    ----------
    probe: t.Tensor
        A (...)x2x1xMxL tensor representing the probe
    phase_shift: float
        phase shift in degrees

    Returns:
    --------
    probe: t.Tensor
        (...)x2x1xMxL 
    """
    probe = probe.to(dtype=t.cfloat)
    jones_matrix = t.tensor([[1, 0], [0, phase_shift]])
    probe = probe.transpose(-1, -3).transpose(-2, -4)
    polarized_probe = t.matmul(jones_matrix.to(dtype=t.cfloat), probe)

    # Transpose it back
    return polarized_probe.transpose(-1, -3).transpose(-2, -4)

def apply_circular_polarizer(probe, left_polarized=True):
    """
    Applies a circular polarizer to the probe

    Parameters:
    ----------
    probe: t.Tensor
        A (...)x2x1xMxL tensor representing the probe
    left_polarizd: bool
        True for the left-polarization, False for the right
    
    Returns:
    --------
    circularly polarized probe: t.Tensor
        (...)x2x1xMxL 
    """
    probe = probe.to(dtype=t.cfloat)
    if left_polarized:
        jones_matrix = (1/2 * t.tensor([[1, -1j], [1j, 1]]))
    else:
        jones_matrix = 1/2 * t.tensor([[1, 1j], [-1j, 1]])
    probe = probe.transpose(-1, -3).transpose(-2, -4)
    polarized_probe = t.matmul(jones_matrix.to(dtype=t.cfloat), probe)

    # Transpose it back
    return polarized_probe.transpose(-1, -3).transpose(-2, -4)

def apply_quarter_wave_plate(probe, fast_axis_angle):
    """
    Parameters:
    ----------
    probe: t.Tensor
        A (...)x2x1xMxL tensor representing the probe, MxL - the size of the probe
    fast_axis_angle: float
        The angle between the fast-axis of the polarizer and the horizontal axis

    Returns:
    --------
    polarized probe: t.Tensor
        (...)x2x1xMxL 
    """
    probe = probe.to(dtype=t.cfloat)
    theta = math.radians(fast_axis_angle)
    exponent = t.exp(-1j * math.pi / 4 * t.ones(2, 2))
    jones_matrix = exponent* t.tensor([[(cos(theta))**2 + 1j * (sin(theta))**2, (1 - 1j) * sin(theta) * cos(theta)], [(1 - 1j) * sin(theta) * cos(theta), (sin(theta))**2 + 1j * (cos(theta))**2]])
    probe = probe.transpose(-1, -3).transpose(-2, -4)
    polarized_probe = t.matmul(jones_matrix.to(dtype=t.cfloat), probe)
    # Transpose it back
    return polarized_probe.transpose(-1, -3).transpose(-2, -4)


def apply_half_wave_plate(probe, fast_axis_angle):
    """
    Parameters:
    ----------
    probe: t.Tensor
        A (...)x2x1xMxL tensor representing the probe, MxL - the size of the probe
    fast_axis_angle: float
        The angle between the fast-axis of the polarizer and the horizontal axis

    Returns:
    --------
    polarized probe: t.Tensor
        (...)x2x1xMxL 
    """
    probe = probe.to(dtype=t.cfloat)
    theta = math.radians(fast_axis_angle)
    exponent = t.exp(-1j * math.pi / 2 * t.ones(2, 2))
    jones_matrix = exponent * t.tensor([[(cos(theta))**2 - (sin(theta))**2, 2 * sin(theta) * cos(theta)], [2 * sin(theta) * cos(theta), (sin(theta))**2 - (cos(theta))**2]])
    probe = probe.transpose(-1, -3).transpose(-2, -4)
    polarized_probe = t.matmul(jones_matrix.to(dtype=t.cfloat), probe)
    # Transpose it back
    return polarized_probe.transpose(-1, -3).transpose(-2, -4)

# probe = t.rand(17, 7, 2, 6, 4)
# polarizer = t.rand(7)
# out = apply_linear_polarizer(probe, polarizer)
# out2 = apply_linear_polarizer(probe, polarizer, transpose=False)

# print(out.shape)
# print(out2.shape)

# a = t.ones(17, 8, 2, 3, 4)
# b = t.ones(2, 1, 1)



# probe = t.ones(5, 2, 3, 3)

# polarizer = t.tensor([45])

# exitw = apply_linear_polarizer(probe, polarizer)
# print(exitw[:, 0, :, :])
# print('y', exitw[:, 1, :, :])

a = t.ones(2, 4)
print(t.sum(a, dim=1).shape)