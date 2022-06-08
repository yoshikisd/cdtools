import numpy as np
import torch as t
from cdtools.tools.polarization import apply_linear_polarizer, generate_linear_polarizer
from cdtools.tools.polarization import apply_jones_matrix as jones


# Abe - I removed all the imports that didn't need to be here.

# Abe - A few issues. First, you could just write "from math import cos, sin"
# Second, we already have numpy imported, so better to use np.cos and np.sin
# from math import cos as cos, sin as sin
# numpy also has np.pi and np.deg2rad 

from numpy import cos, sin, deg2rad

# Further comments:
#
# 1) You should also introduce an assert statement for the output shape
# checking, e.g. assert out.shape == t.Size(<shape>). See example in the
# first test.
#
# 2) Instead of using "assert a and b and c and d...", use separate assert
# statements for each condition. This is both more readable, and it allows the
# testing system to pinpoint exactly which of the assert statements failed.
#
# 3) USE SPACES INSTEAD OF TABS!!! 4 spaces per indent is the convention for
# this project.
#
# 4) Docstrings should go after the function definition, rather than before
#
# 5) Still missing cases where Jones matrix has multiple entries (N) but probe
# doesn't. Low priority.


angle = 87
angle_2 = angle - 45

def polarizer(angle):
    theta = deg2rad(angle)
    polarizer = t.tensor([[(cos(theta)) ** 2, sin(2 * theta) / 2], [sin(2 * theta) / 2, sin(theta) ** 2]]).to(dtype=t.cfloat)
    return polarizer

exponent = t.exp(-1j * np.pi / 4 * t.ones(2, 2)).to(dtype=t.cfloat)
theta2 = deg2rad(angle_2)
quarter_plate = t.tensor([[(cos(theta2))**2 + 1j * (sin(theta2))**2, (1 - 1j) * sin(theta2) * cos(theta2)], 
                                   [(1 - 1j) * sin(theta2) * cos(theta2), (sin(theta2))**2 + 1j * (cos(theta2))**2]]).to(dtype=t.cfloat)

def build_from_quarters(jones1, jones2, jones3, jones4):
    x = t.cat((t.stack((jones1, jones1), dim=-1), t.stack((jones2, jones2), dim=-1)), dim=-1)
    y = t.cat((t.stack((jones3, jones3), dim=-1), t.stack((jones4, jones4), dim=-1)), dim=-1)
    x = t.stack((x, x), dim=-2)
    y = t.stack((y, y), dim=-2)
    return t.cat((x, y), dim=-2).to(dtype=t.cfloat)

jones_plate = t.matmul(quarter_plate, polarizer(angle))
jones0 = polarizer(0)
jones90 = polarizer(90)
jones45 = polarizer(45)

transpose = True

def test_apply_jones_matrix_no_modes_no_mult_patterns_one_jones_matr(): 
    ''' 
    after applying the polarizer and the quarter_plate, the probe should get circularly polarized
    probe: no multiple modes, 1 diffr pattern
        2xMxL
    jones_matrix: same jones matrix applied to all the pixels
        2x2
    '''
    probe = t.rand(2, 3, 4, dtype=t.cfloat)
    print(polarizer)
    out = jones(jones(probe, polarizer(angle), multiple_modes=False, transpose=transpose), 
        quarter_plate, multiple_modes=False, transpose=transpose)
    print('expected shape:(2, 3, 4)')
    print('actual:', out.shape)
    print('simulated:', out)

    assert np.allclose(np.real(out[0]), np.imag(out[1]))
    assert out.shape == t.Size((2, 3, 4))


def test_generate_linear_polarizer():
    pol_angles = [0, 45, 90]
    pol_angle1 = 45
    pol_angle2 = t.tensor(90)
    pol_angle3 = t.tensor([0])
    pols = generate_linear_polarizer(pol_angles)
    pol1 = generate_linear_polarizer(pol_angle1)
    pol2 = generate_linear_polarizer(pol_angle2)
    pol3 = generate_linear_polarizer(pol_angle3)
    print('polarizers 0, 45, 90 (1D tensor) shape:', pols.shape)
    print('shape of the polarizer generated from int:', pol1.shape)
    print('shape of the polarizer generated from 0D tensor:', pol2.shape)
    print('shape of the linear polarizer generated from t.Size(0) tensor:', pol3.shape)
    print('90', pol2)
    print(jones90)
    probe = t.ones(4, 4)
    jones_m = [jones0, jones45, jones90]
    jones_m = t.stack([matr for matr in jones_m])
    assert pols.shape == t.Size((3, 2, 2))
    assert pol1.shape == t.Size((2, 2))
    assert pol2.shape == t.Size((2, 2))
    assert pol3.shape == t.Size((1, 2, 2))
    assert t.allclose(pol1, jones45)


def test_apply_jones_matrix_no_modes_no_mult_patterns_diff_jones_matr():    
    '''probe: no multiple modes, 1 diffr pattern
        2xMxL = 2x4x4
    jones_matrix: jones matrices differ from pixel to pixel
        2x2xMxL = 2x2x4x4
    4 quarters: 
        1: [:, :, :-2, :-2] - circular_polarizer, 
        2: [:, :, :-2, -2:] - 0
        3: [:, :, -2:, :-2] - 90
        4: [:, :, -2:, -2:] - 45
    '''
    jones_matr = build_from_quarters(jones_plate, jones0, jones90, jones45)
    probe = t.ones(2, 4, 4).to(dtype=t.cfloat)
    print('jones:', jones_matr)
    # print('jones:', jones_matr)
    out = jones(probe, jones_matr, multiple_modes=False, transpose=transpose)

    # jones -> (2,2,x,y), probe, output probe
    # interaction -> 
    print('expected shape:(2, 4, 4)')
    print('simulated:', out.shape)
    print('simulated:', out)

    # Example of using multiple asserts
    assert np.allclose(np.real(out[0, :-2, :-2]), np.imag(out[1, :-2, :-2]))
    assert t.allclose(out[0, :-2, -2:], t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, -2:, :-2], t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, -2:, -2:], out[0, -2:, -2:])
    assert out.shape == t.Size((2, 4, 4))

def test_apply_jones_matrix_no_modes_mult_patterns_one_jones_matr():
    '''
    probe: no multiple modes, multiple diffr patterns
        Nx2xMxL = 3x2x4x4
    jones_matrix: same jones matrix applied to all the pixels
        Nx2x2 = 3x2x2
        3 different matrices for each probe:
            1: 0
            2: 45
            3: 90
    '''
    probe = t.ones(2, 4, 4, dtype=t.cfloat)
    probe = t.stack(([probe * (i + 1) for i in range(3)]), dim=0)
    jones_matr = t.stack(([polarizer(angle) for angle in [0, 45, 90]]), dim=0)
    print('probe shape:', probe.shape, 'jones shape', jones_matr.shape)
    out = jones(probe, jones_matr, multiple_modes=False, transpose=transpose)

    print('expected shape: (3, 2, 4, 4)')
    print('actual:', out.shape)
    print('simulated:', out)

    assert t.allclose(out[0, 0, :, :], t.ones(4, 4, dtype=t.cfloat))
    assert t.allclose(out[0, 1, :, :], t.zeros(4, 4, dtype=t.cfloat))
    assert t.allclose(out[1, 0, :, :], out[1, 1])
    assert t.allclose(out[2, 0, :, :], 3* t.zeros(4, 4, dtype=t.cfloat))
    assert t.allclose(out[2, 1, :, :], 3 * t.ones(4, 4, dtype=t.cfloat))
    assert out.shape == t.Size((3, 2, 4, 4))

def test_apply_jones_matrix_no_modes_mult_patterns_diff_jones_matr():
    '''
    probe: no multiple modes, multiple diffr pattern
        Nx2xMxL = 3x2x4x4 
    jones_matrix: jones matrices differ from pixel to pixel
        Nx2x2xMxL = 3x2x2x4x4
        jones matrix for the 1st pattern:
            1: quat plate, 2: 90, 3: 0, 4: 45
        jones matrix for the 2nd pattern:
            1: 0, 2: 45, 3: 90, 4: quat plate
        jones matrix for the 3rd pattern:
            1: 90, 2: 0, 3: 45, 4: quat plate
    '''
    jones_m = [build_from_quarters(jones_plate, jones90, jones0, jones45), 
                                 build_from_quarters(jones0, jones45, jones90, jones_plate),
                                 build_from_quarters(jones90, jones0, jones45, jones_plate)]
    jones_matr = t.stack(([i for i in jones_m]), dim=0)
    probe = t.ones(2, 4, 4, dtype=t.cfloat)
    probe = t.stack(([probe * (i + 1) for i in range(3)]), dim=0)
    out = jones(probe, jones_matr, multiple_modes=False, transpose=transpose)

    print('expected shape: (3, 2, 4, 4)')
    print('actual shape:', out.shape)
    print('simulated:', out)


    assert np.allclose(np.real(out[0, 0, :-2, :-2]), np.imag(out[0, 1, :-2, :-2]))
    assert t.allclose(out[0, 0, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 1, :-2, -2:], t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 0, -2:, :-2], t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 1, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 0, -2:, -2:], out[0, 1, -2:, -2:])

    assert t.allclose(out[1, 0, :-2, :-2], 2 * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 1, :-2, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 0, :-2, -2:], out[1, 1, :-2, -2:])
    assert t.allclose(out[1, 0, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 1, -2:, :-2], 2 * t.ones(2, 2, dtype=t.cfloat))
    assert np.allclose(np.real(out[1, 0, -2:, -2:]), np.real(out[1, 0, -2:, -2:]))

    assert t.allclose(out[2, 0, :-2, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 1, :-2, :-2], 3 * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 0, :-2, -2:], 3 * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 1, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 0, -2:, :-2], out[2, 1, -2:, :-2])
    assert np.allclose(np.real(out[2, 0, -2:, -2:]), np.real(out[2, 0, -2:, -2:]))
    assert out.shape == t.Size((3, 2, 4, 4))

def test_apply_jones_matrix_mult_modes_1_pattern_one_jones_matr():
    '''
    probe: multiple modes, 1 diffr pattern
        Px2xMxL = 2x2x3x4
    jones_matrix: same jones matrix applied to all the pixels
        2x2 - quarter plate
    '''
    probe = t.rand(2, 2, 3, 4, dtype=t.cfloat)
    out = jones(jones(probe, polarizer(angle), multiple_modes=True, transpose=transpose), quarter_plate, multiple_modes=True, transpose=transpose)

    print('expected shape: (2, 2, 3, 4)')
    print('actual:', out.shape)
    print('simulated:', out)

    assert np.allclose(np.real(out[0, 0, :, :]), np.imag(out[0, 1, :, :]))
    assert np.allclose(np.real(out[1, 0, :, :]), np.imag(out[1, 1, :, :]))
    assert out.shape == t.Size((2, 2, 3, 4))

def test_apply_jones_matrix_mult_modes_1_pattern_diff_jones_matr():
    '''
    probe: multiple modes, 1 diffr pattern
        Px2xMxL = 3x2x4x4 
    jones_matrix: jones matrices differ from pixel to pixel
        2xMxL = 2x4x4
    4 quarters: 
        1: [:, :, :-2, :-2] - circular_polarizer, 
        2: [:, :, :-2, -2:] - 0
        3: [:, :, -2:, :-2] - 90
        4: [:, :, -2:, -2:] - 45
    '''
    jones_matr = build_from_quarters(jones_plate, jones0, jones90, jones45)
    probe = t.ones(2, 4, 4, dtype=t.cfloat)
    probe = t.stack(([probe * (i + 1) for i in range(3)]), dim=0)
    out = jones(probe, jones_matr, multiple_modes=True, transpose=transpose)

    print('expected shape: (3, 2, 4, 4)')
    print('actual shape:', out.shape)
    print('simulated:', out)

    assert np.allclose(np.real(out[0, 0, :-2, :-2]), np.imag(out[0, 1, :-2, :-2]))
    assert t.allclose(out[0, 0, :-2, -2:], t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 1, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 0, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 1, -2:, :-2], t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 0, -2:, -2:], out[0, 1, -2:, -2:])

    assert np.allclose(np.real(out[1, 0, :-2, :-2]), np.imag(out[1, 1, :-2, :-2]))
    assert t.allclose(out[1, 0, :-2, -2:], 2 * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 1, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 0, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 1, -2:, :-2], 2 * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 0, -2:, -2:], out[1, 1, -2:, -2:])

    assert np.allclose(np.real(out[2, 0, :-2, :-2]), np.imag(out[2, 1, :-2, :-2]))
    assert t.allclose(out[2, 0, :-2, -2:], 3 * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 1, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 0, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 1, -2:, :-2], 3 * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 0, -2:, -2:], out[2, 1, -2:, -2:])
    assert out.shape == t.Size((3, 2, 4, 4))

def test_apply_jones_matrix_mult_modes_mult_pattern_one_jones_matr():
    '''
    probe: multiple modes, multiple diffr patterns
        NxPx2xMxL = 3x7x2x3x4
    jones_matrix: same jones matrix applied to all the pixels (although differs from pattern to pattern)
        Nx2x2 = 3x2x2
            3 different matrices for each probe in one mode:
            1: 0
            2: 45
            3: 90
    '''
    probe = t.ones(2, 3, 4, dtype=t.cfloat)
    # 1st mode
    probe_mode1 = t.stack(([probe * (i + 1) for i in range(3)]), dim=0)
    # 2nd mode
    probe_mode2 = 10 * t.stack(([probe * (i + 1) for i in range(3)]), dim=0)    
    probe = t.stack(([probe_mode1 * 10 ** i for i in range(7)]), dim=1)
    jones_matr = t.stack(([polarizer(angle) for angle in [0, 45, 90]]), dim=0)
    print('probe shape:', probe.shape, 'jones shape', jones_matr.shape)
    out = jones(probe, jones_matr, multiple_modes=True, transpose=transpose)

    print('expected shape: (3, 7, 2, 3, 4)')
    print('actual:', out.shape)
    print('simulated (patterns in one mode):', out[:, 6, :, :, :])
    
    # we'll be checking only one mode
    assert t.allclose(out[0, 6, 0, :, :], (10**6) * t.ones(3, 4, dtype=t.cfloat))
    assert t.allclose(out[0, 6,  1, :, :], t.zeros(3, 4, dtype=t.cfloat))
    assert t.allclose(out[1, 6, 0, :, :], out[1, 6, 1, :, :])
    assert t.allclose(out[2, 6, 0, :, :], 3 * (10**6) *  t.zeros(3, 4, dtype=t.cfloat))
    assert t.allclose(out[2, 6, 1, :, :], 3 * (10**6) * t.ones(3, 4, dtype=t.cfloat))
    assert out.shape == t.Size((3, 7, 2, 3, 4))

def test_apply_jones_matrix_mult_modes_mult_patterns_diff_jones_matr():
    '''
    probe: multiple modes, multiple diffr pattern
        NxPx2xMxL = 3x4x2x4x4 
    jones_matrix: jones matrices differ from pixel to pixel
        Nx2x2xMxL = 3x2x2x4x4
        (differs across the patterns in each mode)
        jones matrix for the 1st pattern:
            1: quat plate, 2: 90, 3: 0, 4: 45
        jones matrix for the 2nd pattern:
            1: 0, 2: 45, 3: 90, 4: quat plate
        jones matrix for the 3rd pattern:
            1: 90, 2: 0, 3: 45, 4: quat plate
    '''
    jones_m = [build_from_quarters(jones_plate, jones90, jones0, jones45), 
                         build_from_quarters(jones0, jones45, jones90, jones_plate),
                         build_from_quarters(jones90, jones0, jones45, jones_plate)]
    jones_matr = t.stack(([i for i in jones_m]), dim=0)
    probe = t.ones(2, 4, 4, dtype=t.cfloat)
    # one mode
    probe_mode = t.stack(([probe * (i + 1) for i in range(3)]), dim=0)      
    probe = t.stack(([probe_mode * (10 ** i) for i in range(4)]), dim=1)
    print('probe:', probe.shape)
    print('jones:', jones_matr.shape)
    out = jones(probe, jones_matr, multiple_modes=True, transpose=transpose)

    print('expected shape: (3, 4, 2, 4, 4)')
    print('actual shape:', out.shape)
    print('simulated patterns in one mode:', out[:, 3, :, :, :])

    o = 10 ** 3
    # we'll be checking only one mode (4th)
    assert np.allclose(np.real(out[0, 3, 0, :-2, :-2]), np.imag(out[0, 3, 1, :-2, :-2]))
    assert t.allclose(out[0, 3, 0, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 3, 1, :-2, -2:], o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 3, 0, -2:, :-2], o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 3, 1, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 3, 0, -2:, -2:], out[0, 3, 1, -2:, -2:])

    assert t.allclose(out[1, 3, 0, :-2, :-2], 2 * o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 3, 1, :-2, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 3, 0, :-2, -2:], out[1, 3, 1, :-2, -2:])
    assert t.allclose(out[1, 3, 0, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 3, 1, -2:, :-2], 2 * o * t.ones(2, 2, dtype=t.cfloat))
    assert np.allclose(np.real(out[1, 3, 0, -2:, -2:]), np.real(out[1, 3, 0, -2:, -2:]))

    assert t.allclose(out[2, 3, 0, :-2, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 3, 1, :-2, :-2], 3 * o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 3, 0, :-2, -2:], 3 * o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 3, 1, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 3, 0, -2:, :-2], out[2, 3, 1, -2:, :-2])
    assert np.allclose(np.real(out[2, 3, 0, -2:, -2:]), np.real(out[2, 3, 0, -2:, -2:]))
    assert out.shape == t.Size((3, 4, 2, 4, 4))

def test_apply_jones_matrix_no_modes_mult_patterns_one_jones_matr_1():
    '''
    probe: no multiple modes, multiple diffr patterns
        Nx2xMxL = 3x2x4x4
    jones_matrix: same jones matrix applied to all the pixels
        2x2 = 2x2 - a quarter waveplate 
    '''
    probe = t.ones(2, 4, 4, dtype=t.cfloat)
    probe = t.stack(([probe * (i + 1) for i in range(3)]), dim=0)
    jones_matr = jones_plate
    print('probe shape:', probe.shape, 'jones shape', jones_matr.shape)
    out = jones(probe, jones_matr, multiple_modes=False, transpose=transpose)

    print('expected shape: (3, 2, 4, 4)')
    print('actual:', out.shape)
    print('simulated:', out)

    assert np.allclose(np.real(out[:, 0, :, :]), np.imag(out[:, 1, :, :]))
    assert out.shape == t.Size((3, 2, 4, 4))

def test_apply_jones_matrix_no_modes_mult_patterns_diff_jones_matr_1():
    '''
    probe: no multiple modes, multiple diffr pattern
        Nx2xMxL = 3x2x4x4 
    jones_matrix: jones matrices differ from pixel to pixel
        2x2xMxL = 2x2x4x4
        1: quat plate, 2: 90, 3: 0, 4: 45

    '''
    jones_matr = build_from_quarters(jones_plate, jones90, jones0, jones45)
    probe = t.ones(2, 4, 4, dtype=t.cfloat)
    probe = t.stack(([probe * (i + 1) for i in range(3)]), dim=0)
    out = jones(probe, jones_matr, multiple_modes=False, transpose=transpose)

    print('expected shape: (3, 2, 4, 4)')
    print('actual shape:', out.shape)
    print('simulated:', out)

# check the 3rd pattern
    assert np.allclose(np.real(out[2, 0, :-2, :-2]), np.imag(out[2, 1, :-2, :-2]))
    assert t.allclose(out[2, 0, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 1, :-2, -2:], 3 * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 0, -2:, :-2], 3 * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 1, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 0, -2:, -2:], out[2, 1, -2:, -2:])
    assert out.shape == t.Size((3, 2, 4, 4))

def test_apply_jones_matrix_mult_modes_mult_pattern_one_jones_matr_1():
    '''
    probe: multiple modes, multiple diffr patterns
        NxPx2xMxL = 3x7x2x3x4
    jones_matrix: same jones matrix applied to all the pixels (although differs from pattern to pattern)
        2x2 = 2x2
        quarter wave plate
    '''
    probe = t.ones(2, 3, 4, dtype=t.cfloat)
    # 1st mode
    probe_mode1 = t.stack(([probe * (i + 1) for i in range(3)]), dim=0)
    probe = t.stack(([probe_mode1 * 10 ** i for i in range(7)]), dim=1)
    jones_matr = jones_plate
    print('probe shape:', probe.shape, 'jones shape', jones_matr.shape)
    out = jones(probe, jones_matr, multiple_modes=True, transpose=transpose)

    print('expected shape: (3, 7, 2, 3, 4)')
    print('actual:', out.shape)
    print('simulated (patterns in one mode):', out[:, 6, :, :, :])
    
    # we'll be checking only one mode
    assert np.allclose(np.real(out[:, :, 0, :, :]), np.imag(out[:, :, 1, :, :]))
    assert out.shape == t.Size((3, 7, 2, 3, 4))

def test_apply_jones_matrix_mult_modes_mult_patterns_diff_jones_matr_1():
    '''
    probe: multiple modes, multiple diffr pattern
        NxPx2xMxL = 3x4x2x4x4 
    jones_matrix: jones matrices differ from pixel to pixel
        2x2xMxL = 2x2x4x4
        jones matrix:
        1: quat plate, 2: 90, 3: 0, 4: 45
    '''
    jones_matr = build_from_quarters(jones_plate, jones90, jones0, jones45)
    probe = t.ones(2, 4, 4, dtype=t.cfloat)
    # one mode
    probe_mode = t.stack(([probe * (i + 1) for i in range(3)]), dim=0)      
    probe = t.stack(([probe_mode * (10 ** i) for i in range(4)]), dim=1)
    print('probe:', probe.shape)
    print('jones:', jones_matr.shape)
    out = jones(probe, jones_matr, multiple_modes=True, transpose=transpose)

    print('expected shape: (3, 4, 2, 4, 4)')
    print('actual shape:', out.shape)
    print('simulated patterns in one mode:', out[:, 3, :, :, :])

    o = 10 ** 2
    # we'll be checking only one mode (3th)
    assert np.allclose(np.real(out[0, 2, 0, :-2, :-2]), np.imag(out[0, 2, 1, :-2, :-2]))
    assert t.allclose(out[0, 2, 0, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 2, 1, :-2, -2:], o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 2, 0, -2:, :-2], o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 2, 1, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 2, 0, -2:, -2:], out[0, 2, 1, -2:, -2:])
    assert out.shape == t.Size((3, 4, 2, 4, 4))

    return None


def test_apply_jones_matrix_no_mult_modes_one_pattern_probe_mult_patterns_jones_1():
    '''
    probe: 
        2xMxL = 2x3x4 
    jones_matrix: 
        Nx2x2 = 3x2x2
        jones matrices:
        1: quat plate, 2: 90, 3: 0
    '''
    jones_matr = t.stack((jones_plate, jones90, jones0))
    probe = t.ones(2, 3, 4, dtype=t.cfloat)
    out = jones(probe, jones_matr, multiple_modes=False, transpose=transpose)

    print('expected shape: (3, 2, 3, 4)')
    print('actual shape:', out.shape)
    print('simulated:', out)

    # we'll be checking only one mode (3th)
    assert np.allclose(np.real(out[0, 0, :, :]), np.imag(out[0, 1, :, :]))
    assert t.allclose(out[1, 0, :, :], t.zeros(3, 4, dtype=t.cfloat))
    assert t.allclose(out[1, 1, :, :], t.ones(3, 4, dtype=t.cfloat))
    assert t.allclose(out[2, 0, :, :], t.ones(3, 4, dtype=t.cfloat))
    assert t.allclose(out[2, 1, :, :], t.zeros(3, 4, dtype=t.cfloat))
    assert out.shape == t.Size((3, 2, 3, 4))

    return None


def test_apply_jones_matrix_no_mult_modes_one_pattern_probe_mult_patterns_jones_2():
    '''
    probe: 
        2xMxL = 2x4x4
    jones_matrix: jones matrices differ from pixel to pixel
        Nx2x2xMxL = 3x2x2x4x4
        jones matrix for the 1st pattern:
            1: quat plate, 2: 90, 3: 0, 4: 45
        jones matrix for the 2nd pattern:
            1: 0, 2: 45, 3: 90, 4: quat plate
        jones matrix for the 3rd pattern:
            1: 90, 2: 0, 3: 45, 4: quat plate
    '''
    jones_m = [build_from_quarters(jones_plate, jones90, jones0, jones45), 
                         build_from_quarters(jones0, jones45, jones90, jones_plate),
                         build_from_quarters(jones90, jones0, jones45, jones_plate)]
    jones_matr = t.stack(([i for i in jones_m]), dim=0)
    probe = t.ones(2, 4, 4, dtype=t.cfloat)
    out = jones(probe, jones_matr, multiple_modes=False, transpose=transpose)

    print('expected shape: (3, 2, 4, 4)')
    print('actual shape:', out.shape)
    print('simulated:', out)

    assert np.allclose(np.real(out[0, 0, :-2, :-2]), np.imag(out[0, 1, :-2, :-2]))
    assert t.allclose(out[0, 0, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 1, :-2, -2:], t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 0, -2:, :-2], t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 1, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 0, -2:, -2:], out[0, 1, -2:, -2:])

    assert t.allclose(out[1, 0, :-2, :-2], t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 1, :-2, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 0, :-2, -2:], out[1, 1, :-2, -2:])
    assert t.allclose(out[1, 0, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 1, -2:, :-2], t.ones(2, 2, dtype=t.cfloat))
    assert np.allclose(np.real(out[1, 0, -2:, -2:]), np.real(out[1, 0, -2:, -2:]))

    assert t.allclose(out[2, 0, :-2, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 1, :-2, :-2], t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 0, :-2, -2:], t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 1, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 0, -2:, :-2], out[2, 1, -2:, :-2])
    assert np.allclose(np.real(out[2, 0, -2:, -2:]), np.real(out[2, 0, -2:, -2:]))
    assert out.shape == t.Size((3, 2, 4, 4))


def test_apply_jones_matrix_mult_modes_one_pattern_probe_mult_patterns_jones_1():
    '''
    probe: multiple modes, multiple diffr patterns
        Px2xMxL = 7x2x3x4
    jones_matrix: same jones matrix applied to all the pixels (although differs from pattern to pattern)
        Nx2x2 = 3x2x2
            3 different matrices for each pattern:
            1: 0
            2: 45
            3: 90
    '''
    probe = t.ones(2, 3, 4, dtype=t.cfloat)  
    probe = t.stack(([probe * 10 ** i for i in range(7)]), dim=0)
    jones_matr = t.stack(([polarizer(angle) for angle in [0, 45, 90]]), dim=0)
    print('probe shape:', probe.shape, 'jones shape', jones_matr.shape)
    out = jones(probe, jones_matr, multiple_modes=True, transpose=transpose)

    print('expected shape: (3, 7, 2, 3, 4)')
    print('actual:', out.shape)
    print('simulated (patterns in one mode):', out[:, 6, :, :, :])
    
    # we'll be checking only one mode
    assert t.allclose(out[0, 6, 0, :, :], (10**6) * t.ones(3, 4, dtype=t.cfloat))
    assert t.allclose(out[0, 6,  1, :, :], t.zeros(3, 4, dtype=t.cfloat))
    assert t.allclose(out[1, 6, 0, :, :], out[1, 6, 1, :, :])
    assert t.allclose(out[2, 6, 0, :, :], (10**6) *  t.zeros(3, 4, dtype=t.cfloat))
    assert t.allclose(out[2, 6, 1, :, :], (10**6) * t.ones(3, 4, dtype=t.cfloat))
    assert out.shape == t.Size((3, 7, 2, 3, 4))


def test_apply_jones_matrix_mult_modes_one_pattern_probe_mult_patterns_jones_2():
    '''
    probe: multiple modes, multiple diffr pattern
        Px2xMxL = 4x2x4x4 
    jones_matrix: jones matrices differ from pixel to pixel
        Nx2x2xMxL = 3x2x2x4x4
        (differs across the patterns in each mode)
        jones matrix for the 1st pattern:
            1: quat plate, 2: 90, 3: 0, 4: 45
        jones matrix for the 2nd pattern:
            1: 0, 2: 45, 3: 90, 4: quat plate
        jones matrix for the 3rd pattern:
            1: 90, 2: 0, 3: 45, 4: quat plate
    '''
    jones_m = [build_from_quarters(jones_plate, jones90, jones0, jones45), 
                         build_from_quarters(jones0, jones45, jones90, jones_plate),
                         build_from_quarters(jones90, jones0, jones45, jones_plate)]
    jones_matr = t.stack(([i for i in jones_m]), dim=0)
    probe = t.ones(2, 4, 4, dtype=t.cfloat)    
    probe = t.stack(([probe * (10 ** i) for i in range(4)]), dim=0)
    print('probe:', probe.shape)
    print('jones:', jones_matr.shape)
    out = jones(probe, jones_matr, multiple_modes=True, transpose=transpose)

    print('expected shape: (3, 4, 2, 4, 4)')
    print('actual shape:', out.shape)
    print('simulated patterns in one mode:', out[:, 3, :, :, :])

    o = 10 ** 3
    # we'll be checking only one mode (4th)
    assert np.allclose(np.real(out[0, 3, 0, :-2, :-2]), np.imag(out[0, 3, 1, :-2, :-2]))
    assert t.allclose(out[0, 3, 0, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 3, 1, :-2, -2:], o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 3, 0, -2:, :-2], o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 3, 1, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[0, 3, 0, -2:, -2:], out[0, 3, 1, -2:, -2:])

    assert t.allclose(out[1, 3, 0, :-2, :-2], o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 3, 1, :-2, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 3, 0, :-2, -2:], out[1, 3, 1, :-2, -2:])
    assert t.allclose(out[1, 3, 0, -2:, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[1, 3, 1, -2:, :-2], o * t.ones(2, 2, dtype=t.cfloat))
    assert np.allclose(np.real(out[1, 3, 0, -2:, -2:]), np.real(out[1, 3, 0, -2:, -2:]))

    assert t.allclose(out[2, 3, 0, :-2, :-2], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 3, 1, :-2, :-2], o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 3, 0, :-2, -2:], o * t.ones(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 3, 1, :-2, -2:], t.zeros(2, 2, dtype=t.cfloat))
    assert t.allclose(out[2, 3, 0, -2:, :-2], out[2, 3, 1, -2:, :-2])
    assert np.allclose(np.real(out[2, 3, 0, -2:, -2:]), np.real(out[2, 3, 0, -2:, -2:]))
    assert out.shape == t.Size((3, 4, 2, 4, 4))
