import numpy as numpy
import torch as t
# Abe - again, we don't need math here. replace with torch-native functions
# in all the definitions here. We have to use torch here if we want to be
# able to calculate derivatives w.r.t. the polarizer angle, say.
import math
from math import sin
from math import cos

__all__ = ['apply_linear_polarizer',
           'apply_phase_retardance',
           'apply_half_wave_plate',
           'apply_quarter_wave_plate',
           'apply_circular_polarizer',
           'apply_jones_matrix',
           'generate_linear_polarizer',
           'generate_birefringent_obj']


# Abe - split these into two functions

# Note for the future: this function should
def generate_linear_polarizer(pol_angle):
    single_angle = False

    pol_angle = t.as_tensor(pol_angle).to(dtype=t.float32)
    if pol_angle.dim() == 0:
        pol_angle = t.unsqueeze(pol_angle,0)
        single_angle = True
        
    pol_angle_rad = t.deg2rad(pol_angle)
    a = t.cos(pol_angle_rad) ** 2
    b = t.sin(pol_angle_rad) * t.cos(pol_angle_rad)
    c = b
    d = t.sin(pol_angle_rad) ** 2
    ab = t.stack((a, b), dim=-1)
    cd = t.stack((c, d), dim=-1)
    jones_matrices = t.stack((ab, cd), dim=-2)
    if single_angle:
        return jones_matrices[0].to(dtype=t.cfloat)
    else:
        return jones_matrices.to(dtype=t.cfloat)


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
    jones_matrices = generate_linear_polarizer(polarizer)
    return apply_jones_matrix(probe, jones_matrices, transpose=transpose, multiple_modes=multiple_modes)


def apply_jones_matrix(probe, jones_matrix, transpose=True, multiple_modes=True):
    # print('probe', probe.shape, 'jones matrix', jones_matrix.shape)
    # if jones_matrix.shape == t.Size([5, 2, 2, 2, 2]):
    #     print('probe', probe.shape, 'jones', jones_matrix.shape)
    #     print('JONES', jones_matrix)
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

    """
    if transpose:
        if jones_matrix.dim() < 4:
            jones_matrix = jones_matrix[..., None, None]
        if multiple_modes:
            jones_matrix = jones_matrix.unsqueeze(-5)
        probe = probe[..., None, :, :]
        # if jones matrices do not differ from pattern to pattern
        if probe.dim() > jones_matrix.dim():
            jones_matrix = jones_matrix.unsqueeze(0)
        # vice versa
        elif jones_matrix.dim() > probe.dim():
            probe = probe.unsqueeze(0)
        # print('apply jonesmatrix: probe', probe.shape, 'matrix:', jones_matrix)
        jones_matrix = jones_matrix.transpose(-1, -3).transpose(-2, -4)
        probe = probe.transpose(-1, -3).transpose(-2, -4)
        output = t.matmul(jones_matrix, probe).transpose(-2, -4).transpose(-1, -3).squeeze(-3)

    else:
        raise NotImplementedError

    return output


def apply_phase_retardance(probe, phase_shift, multiple_modes=True):
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
    theta = t.as_tensor(phase_shift, dtype=t.float32)
    theta = t.deg2rad(theta)
    probe = probe.to(dtype=t.cfloat)
    jones_matrix = t.tensor([[1, 0], [0, t.exp(phase_shift)]]).to(dtype=t.cfloat)
    polarized = apply_jones_matrix(probe, jones_matrix, multiple_modes=multiple_modes)

    return polarized

def apply_circular_polarizer(probe, left_polarized=True, multiple_modes=True):
    """
    Applies a circular polarizer to the probe

    Parameters:
    ----------
    probe: t.Tensor
        A (...)x2xMxL tensor representing the probe
    left_polarizd: bool
        True for the left-polarization, False for the right

    Returns:
    --------
    circularly polarized probe: t.Tensor
        (...)x2xMxL
    """
    probe = probe.to(dtype=t.cfloat)
    if left_polarized:
        jones_matrix = (1/2 * t.tensor([[1, -1j], [1j, 1]])).to(dtype=t.cfloat)
    else:
        jones_matrix = 1/2 * t.tensor([[1, 1j], [-1j, 1]]).to(dtype=t.cfloat)
    polarized = apply_jones_matrix(probe, jones_matrix, multiple_modes=multiple_modes)
    return polarized

def apply_quarter_wave_plate(probe, fast_axis_angle, multiple_modes=True):
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
    jones_matrix = exponent* t.tensor([[(cos(theta))**2 + 1j * (sin(theta))**2, (1 - 1j) * sin(theta) * cos(theta)], [(1 - 1j) * sin(theta) * cos(theta), (sin(theta))**2 + 1j * (cos(theta))**2]]).to(dtype=t.cfloat)
    out = apply_jones_matrix(probe, jones_matrix, multiple_modes=multiple_modes)

    return out

def apply_half_wave_plate(probe, fast_axis_angle, multiple_modes=True):
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
    jones_matrix = exponent * t.tensor([[(cos(theta))**2 - (sin(theta))**2, 2 * sin(theta) * cos(theta)], [2 * sin(theta) * cos(theta), (sin(theta))**2 - (cos(theta))**2]]).to(dtype=t.cfloat)
    out = apply_jones_matrix(probe, jones_matrix, multiple_modes=multiple_modes)

    return out

def generate_birefringent_obj(fast_axis=90, phase_ret=10, atten_fast=1, atten_ret=1, global_phase=0):
    def to_rad(angle):
        angle = t.as_tensor(angle, dtype=t.float32)
        angle = t.deg2rad(angle)
        return angle

    fast_axis = to_rad(fast_axis)
    phase_ret = to_rad(phase_ret)
    global_phase = to_rad(global_phase)

    def coord_rot(angle):
        a = t.stack((t.cos(angle), t.sin(angle)), dim=-1)
        b = t.stack((-t.sin(angle), t.cos(angle)), dim=-1)
        return t.stack((a, b), dim=-2).to(dtype=t.cfloat)

    r1 = coord_rot(-fast_axis)
    r2 = coord_rot(fast_axis)
    p = t.exp(global_phase * 1j) * t.as_tensor([[atten_fast, 0], [0, atten_ret * t.exp(phase_ret*1j)]], dtype=t.cfloat)
    return t.matmul(r1, t.matmul(p, r2))
