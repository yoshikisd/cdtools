import numpy as numpy
import torch as t

def polarize_probe_linear(probe, polar_angle):
	"""
	Applies a linear polarizer to the probe

	Parameters:
	----------
	probe: t.Tensor
		A (...)x2x1xMxL tensor representing the probe
	polar_angle: float
		The angle between the fast-axis of the linear polarizer and the x-axis
	"""

	polarizer = t.tensor([[(math.cos(theta)) ** 2, math.sin(2 * theta) / 2], [math.sin(2 * theta) / 2, (math.sin(theta)) ** 2]])
	return t.tensordot(polarizer, probe, dims=([-3], [-4]))

def apply_jones_matrix(probe, jones_matrix):
	"""
	Applies the given Jones matrix to the probe

	Parameters:
	----------
	probe: t.Tensor
		A (...)x2x1xMxL tensor representing the probe
	jones_matrix: t.tensor
		2x2
	"""