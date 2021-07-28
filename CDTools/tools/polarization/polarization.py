import numpy as numpy
import torch as t
import math
from math import sin
from math import cos

def apply_linear_polarizer(probe, polar_angle):
	"""
	Applies a linear polarizer to the probe

	Parameters:
	----------
	probe: t.Tensor
		A (...)x2x1xMxL tensor representing the probe, MxL - the size of the probe
	polar_angle: float
		The angle between the fast-axis of the linear polarizer and the horizontal axis

	Returns:
	--------
	linearly polarized probe: t.Tensor
		(...)x2x1xMxL 
	"""
	probe = probe.to(dtype=t.cfloat)
	theta = math.radians(polar_angle)
	polarizer = t.tensor([[(cos(theta)) ** 2, sin(2 * theta) / 2], [sin(2 * theta) / 2, sin(theta) ** 2]]).to(dtype=t.cfloat)
	# I haven't figured out how to multiply tensors using tensordot yet, 
	# so we'll be temporarily using matmul on the previously tranposed vector 
	# (since it returns the matrix multiplication product over the last two dimensions

	#Swap the dimensions for the prober to be (...)xMxLx2x1 to perform matmul on it 
	probe = probe.transpose(-1, -3).transpose(-2, -4)
	polarized_probe = t.matmul(polarizer, probe)

	# Transpose it back
	return polarized_probe.transpose(-1, -3).transpose(-2, -4)

def apply_jones_matrix(probe, jones_matrix):
	"""
	Applies a given Jones matrix to the probe

	Parameters:
	----------
	probe: t.Tensor
		A (...)x2x1xMxL tensor representing the probe
	jones_matrix: t.tensor
		(...)x2x2

	Returns:
	--------
	linearly polarized probe: t.Tensor
		(...)x2x1xMxL 
	"""
	probe = probe.to(dtype=t.cfloat)
	probe = probe.transpose(-1, -3).transpose(-2, -4)
	polarized_probe = t.matmul(jones_matrix.to(dtype=t.cfloat), probe)

	# Transpose it back
	return polarized_probe.transpose(-1, -3).transpose(-2, -4)

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

# probe = t.rand(3, 2, 1, 5, 6)
# print(apply_linear_polarizer(probe, 30).shape)
# print(apply_circular_polarizer(probe).shape)
# print(apply_phase_retardance(probe, 29).shape)
# print(apply_half_wave_plate(probe, 29).shape)
# print(apply_quarter_wave_plate(probe, 29).shape)
