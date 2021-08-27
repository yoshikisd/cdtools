import torch as t
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import Slider
from matplotlib import ticker, patheffects
from CDTools.tools.polarization import polarization

all = ['visualize_components_amplitudes',
        'visualize_attenuations',
        'visualize_fast_axes',
        'visualize_global_axes',
        'visualize_phase_ret',
        'visualize_probe']

def iterator(a, b):
    '''
    A helper function we can use to facilitate to process of iterating over 2D arrays
    '''
    x, y = t.arange(a), t.arange(b)
    x, y = t.meshgrid(x, y)
    x, y = t.ravel(x), t.ravel(y)
    return (x, y)

def plot_probe_ellipse(a=1, b=1, phase_ret=0, scale=1, x0=4, y0=5):
    '''
    Given a probe vector at a point (x0, y0), visualizes ellipticity
    of its polarization

    Parameters:
    -----------
    a : 1D np.array
        An amplitude of the horizontal component of the probe vector
    b : 1D np.array
        An amplitude of the vertical component of the probe vector
    phase_ret : 1D np.array
        A phase difference in radians bettween the phases of the y and x components
    scale : int
        A scaling factor
    x0, y0: 1D np.array or float
        Defines the location of the vector to be plotted

    Returns:
    --------
    x, y set of points to plot a single ellipse
    '''
    theta = np.linspace(0, 2*np.pi, 20)
    x = x0 + scale * a * np.real(np.exp(1j * theta))
    y = y0 + scale * b * np.real(np.exp(1j * (theta + phase_ret)))
    return x, y

def plot_attenuations(atten_slow=1, atten_fast=1, fast_ax_angle=0, scale=1, x0=4, y0=4):
    '''
    Plots attenuations along the fast and slow axes

    Parameters:
    -----------
    atten_slow: 1D np.array
        Attenuation along the slow axis
    atten_fast: 1D np.array
        Attenuation along the fast axis
    fast_ax_angle: 1D np.array
        An angle between the fast horizontal and fast axes
    phase_ret: 1D np.array
        A difference in phases gained by the slow and the fast components
        The most clockwise axis is always considered to be the fast one
    scale: 1D np.array
        A scaling factor
    x0, y0: 1D np.array or float
        Defines the location to be plotted at

    Returns:
    --------
    x, y set of points to plot a single Jones matrix of the object
    '''
    angle = fast_ax_angle
    theta = np.linspace(0, 2*np.pi, 20)
    # collection of points to plot a fast axis
    print('angle', np.rad2deg(angle))
    x = x0 + scale * atten_fast * np.real(np.exp(1j * theta))
    x_f = x0 + (x - x0) * np.cos(angle)
    y_f = y0 + (x - x0) * np.sin(angle)
    # collection of point to plot a slow axis
    y = y0 + scale * atten_slow * np.real(np.exp(1j * theta))
    x_s = x0 - (y - y0) * np.sin(angle)
    y_s = y0 + (y - y0) * np.cos(angle)
    return x_f, y_f, x_s, y_s

def plot_fast_axis(fast_ax_angle=0, scale=1, x0=4, y0=4):
    '''
    Plots directions of the fast axes only
    '''
    xf, yf, xs, ys = plot_attenuations(fast_ax_angle=fast_ax_angle, atten_fast=1, atten_slow=0, scale=scale, x0=x0, y0=y0)
    return xf, yf

def plot_figures(shape, num_of_el_along_x=20, num_of_el_along_y=20,
                 phases=None, fast_ax_angles=None,
                 atten_fast=None, atten_slow=None, scale=5,
                 probe=False, attenuations=False, fast_axes=False):
    """
    All the parameters - np.arrays of shape (shape)
    """
    x_centers = np.linspace(1, shape[0] - 1, num_of_el_along_x)
    y_centers = np.linspace(1, shape[1] - 1, num_of_el_along_y)
    X, Y = np.meshgrid(x_centers, y_centers)
    X, Y = np.ravel(X), np.ravel(Y)
    xs, ys = np.array([]), np.array([])
    for x, y in zip(X, Y):
        k, m = int(x), int(y)
        if probe:
            xx, yy = plot_probe_ellipse(a=atten_fast[k, m], b=atten_slow[k, m],
                                        phase_ret=phases[k, m], scale=scale, x0=x, y0=y)
            title = 'Polarized Probe'
        elif attenuations:
            xf, yf, xs, ys = plot_attenuations(atten_slow=atten_slow[k, m], atten_fast=atten_fast[k, m],
                                       fast_ax_angle=fast_ax_angles[k, m], scale=scale, x0=x, y0=y)
        elif fast_axes:
            xx, yy = plot_fast_axis(fast_ax_angle=fast_ax_angles[k, m], scale=scale, x0=x, y0=y)
            title = 'Fast Axes'

        if attenuations:
            plt.plot(xf, yf, c='b')
            plt.plot(xs, ys, c='b')
            plt.axis('equal')
            plt.title('Attenuations')
        else:
            plt.plot(xx, yy, c='b')
            plt.axis('equal')
            plt.title(title)

"""
VISUALIZATION FUNCTIONS (and helper functions)
"""
def determine_fast_axis(angle0, angle1):
    # in radians
    w0, w1 = t.as_tensor(angle0, dtype=t.float32), t.as_tensor(angle1, dtype=t.float32)
    diff = (w0 - w1) % (2 * np.pi)
    if t.tensor(0, dtype=t.float32) <= diff and diff <= t.tensor(np.pi, dtype=t.float32):
        fast = 1
        phase_ret = (w0 - w1) % (2 * np.pi)
        gl_phase = w1
    else:
        fast = 0
        phase_ret = (w1 - w0) % (2 * np.pi)
        gl_phase = w0

    if t.allclose((w0 - w1) % (2 * np.pi), phase_ret):
        gl_phase = w1
        fast = 1

    return fast, phase_ret, gl_phase

def retrieve_obj_info(obj):
    obj_t = obj.transpose(-1, -3).transpose(-2, -4)
    w, v = t.linalg.eig(obj_t)
    b = obj.shape[-1]
    a = obj.shape[-2]
    eigenvectors = t.empty(a, b, 2, dtype=t.cfloat)
    ret_phases = t.empty(a, b, dtype=t.float32)
    global_phases = t.empty(a, b, dtype=t.float32)
    atten_fast = t.empty(a, b, dtype=t.float32)
    atten_slow = t.empty(a, b, dtype=t.float32)
    fast_ax_angles = t.empty(a, b, dtype=t.float32)

    for k, m in zip(*iterator(a, b)):
        angle0, angle1 = t.angle(w[k, m, 0]).to(dtype=t.float32), t.angle(w[k, m, 1]).to(dtype=t.float32)
        fst, phase_ret, gl_phase = determine_fast_axis(angle0, angle1)
        atten0, atten1 = t.abs(w[k, m, 0]), t.abs(w[k, m, 1])
        ret_phases[k, m] = phase_ret
        def fast(b):
            if b == 0:
                eigenvectors[k, m, :] = v[k, m, 0, :]
                atten_fast[k, m] = atten0
                atten_slow[k, m] = atten1
                global_phases[k, m] = angle0
            elif b == 1:
                eigenvectors[k, m, :] = v[k, m, 1, :]
                atten_fast[k, m] = atten1
                atten_slow[k, m] = atten0
                global_phases[k, m] = angle1

        if t.allclose(angle0, angle1):
        # then it's a linear polarizer, fast ax - the one for which attenuation is bigger
            if atten0 > atten1:
                fst = 0
            else:
                fst = 1
        fast(fst)

        cos = np.abs(eigenvectors[k, m, 0])
        sin = np.abs(eigenvectors[k, m, 1])
        if t.allclose(cos, t.zeros(1, dtype=t.float32)):
            fast_ax_angles[k, m] = 90
        else:
            fast_ax_angles[k, m] = t.atan(sin/cos)

    fast_ax_angles = np.asarray(fast_ax_angles, dtype=np.float32)
    ret_phases = np.asarray(ret_phases, dtype=np.float32)
    global_phases = np.asarray(global_phases, dtype=np.float32)
    atten_fast = np.asarray(atten_fast, dtype=np.float32)
    atten_slow = np.asarray(atten_slow, dtype=np.float32)

    return fast_ax_angles, ret_phases, global_phases, atten_fast, atten_slow


def visualize_components_amplitudes(obj, rot_angle=0, logarithmic=False):
    # coord_rot angle is the only angle in degrees here
    def coord_rot(angle):
            angle = t.as_tensor(angle, dtype=t.float32)
            angle = t.deg2rad(angle)
            a = t.stack((t.cos(angle), t.sin(angle)), dim=-1)
            b = t.stack((-t.sin(angle), t.cos(angle)), dim=-1)
            return t.stack((a, b), dim=-2).to(dtype=t.cfloat)
    for k, m in zip(*iterator(obj.shape[-2], obj.shape[-1])):
        obj[:, :, k, m] = t.matmul(coord_rot(rot_angle), obj[:, :, k, m])
    components = [obj[i, j, :, :] for i, j in zip(*iterator(2, 2))]
    if logarithmic:
        components = [np.log(comp)/np.log(10) for comp in components]
    titles = ['Amptlitudes of the a components', 'Amplitudes of the b components',
              'Amplitudes of the c components', 'Amplitudes of the d components']

    for i in range(4):
        amplitude = np.abs(components[i])
        plt.imshow(module)
        plt.colorbar()
        plt.title(titles[i])

def visualize_phase_ret(obj, logarithmic=False):
    print('DHSBCUYLIWGBCGLWIYV')
    fast_ax_angles, ret_phases, global_phases, atten_fast, atten_slow = retrieve_obj_info(obj)
    if logarithmic:
        ret_phases = np.log(ret_phases)/np.log(10)
    plt.imshow(ret_phases)
    plt.colorbar()
    plt.show()

def visuallize_global_phases(obj, logarithmic=False):
    fast_ax_angles, ret_phases, global_phases, atten_fast, atten_slow = retrieve_obj_info(obj)
    if logarithmic:
        global_phases = np.log(global_phases)/np.log(10)
    plt.imshow(global_phases)
    plt.colorbar()
    plt.show()

def visualize_fast_axes(obj, num_of_el_along_x=20, num_of_el_along_y=20, scale=1):
    fast_ax_angles, ret_phases, global_phases, atten_fast, atten_slow = retrieve_obj_info(obj)
    A, B = obj.shape[-2], obj.shape[-1]
    plot_figures((A, B), num_of_el_along_x=num_of_el_along_x, num_of_el_along_y=num_of_el_along_y,
                 fast_ax_angles=fast_ax_angles, scale=scale, fast_axes=True)

def visualize_attenuations(obj, num_of_el_along_x=20, num_of_el_along_y=20, scale=1):
    fast_ax_angles, ret_phases, global_phases, atten_fast, atten_slow = retrieve_obj_info(obj)
    A, B = obj.shape[-2], obj.shape[-1]
    plot_figures((A, B), num_of_el_along_x=num_of_el_along_x, num_of_el_along_y=num_of_el_along_y,
                 phases=ret_phases, atten_slow=atten_slow, atten_fast=atten_fast,
                 fast_ax_angles=fast_ax_angles, scale=scale, attenuations=True)

def visualize_probe(probe, scale=1, num_of_el_along_x=20, num_of_el_along_y=20):
    a = np.abs(probe[..., 0, :, :])
    b = np.abs(probe[..., 1, :, :])
    phases = np.angle(probe[..., 1, :, :]) - np.angle(probe[..., 0, :, :])
    A, B = np.asarray(probe.shape[-2]), np.asarray(probe.shape[-1])
    plot_figures((A, B), num_of_el_along_x=num_of_el_along_x, num_of_el_along_y=num_of_el_along_y,
                     phases=phases, atten_fast=a, atten_slow=b, scale=scale,
                     probe=True)


#
# def object_from_components(a, b, c, d):
#     ab = t.stack((a, b), dim=-3)
#     cd = t.stack((c, d), dim=-3)
#     return t.stack((ab, cd), dim=-4)
#
# def object_from_quarters(a, b, c, d):
#     ab = t.cat((a, b), dim=-1)
#     cd = t.cat((c, d), dim=-1)
#     return t.cat((ab, cd), dim=-2)
#
# def generate_birefringent_obj(shape, func_axes=None, func_ret_phases=None, func_global_phases=None, fast_axes=[0, 0, 90, 90], phases=[0, 18, 40, 18]):
#     ret = [polarization.generate_birefringent_obj(fast_axis=i, phase_ret=j) for i, j in zip(fast_axes, phases)]
#     A = shape[0]
#     B = shape[1]
#     def to_rad(angle):
#         angle = t.as_tensor(angle)
#         return t.deg2rad(angle)
#
#     if func_axes is None:
#         # 4 sets of components for each quarter
#         components = [[ret[i][j][k].repeat(A//2, B//2) for j, k in zip(*iterator(2, 2))] for i in range(4)]
#         # build the quarters from the components
#         quarters = [object_from_components(*comp) for comp in components]
#         obj = object_from_quarters(*quarters)
#     else:
#         X, Y = t.arange(A), t.arange(B)
#         X, Y = t.meshgrid(X, Y)
#         obj = t.empty(2, 2, A, B, dtype=t.cfloat)
#         for i, j in zip(*iterator(A, B)):
#             x, y = X[i, j], Y[i, j]
#             axis = func_axes(x, y)
#             ret = func_ret_phases(x, y)
#             gl = func_global_phases(x, y)
#             obj[:, :, i, j] = polarization.generate_birefringent_obj(fast_axis=axis, phase_ret=ret, global_phase=gl)
#             w, v = t.linalg.eig(obj[:, :, i, j])
#
#     # t.tensor of shape (2, 2, A, B)
#     return obj
#
# def axes(x, y):
#     # return (x-25)**2 + (y-25)**2
#     return  x * 20
#
# def phases(x, y):
#     # return ((x - 5) ** 2 + (y - 5) ** 2) * 20
#     return  (x) * 10
#
# def glob(x, y):
#     return x * 10
#
# def amp(x, y):
#     return 1 + 0.05 * (x + y)
#
# def build_probe(shape, phase_ret_func=None, amps_func=None):
#     probe = t.empty(2, shape[0], shape[1], dtype=t.cfloat)
#     for x, y in zip(*iterator(probe.shape[-2], probe.shape[-1])):
#         phase = phase_ret_func(x, y)
#         phase = t.deg2rad(phase).to(dtype=t.cfloat)
#         abs = amps_func(x, y).to(dtype=t.cfloat)
#         probe[0, x, y] = abs
#         probe[1, x, y] = abs * t.exp(phase * 1j)
#
#     return probe
#
#
# obj = generate_birefringent_obj((10, 10), func_axes=axes, func_ret_phases=phases, func_global_phases=glob)
# probe = build_probe((10, 10), phase_ret_func=phases, amps_func=amp)
# # visualize_probe(probe, num_of_el_along_x=10, num_of_el_along_y=10, scale=0.1)
# visualize_fast_axes(obj, num_of_el_along_x=10, num_of_el_along_y=10, scale=0.1)
# # visualize_phase_ret(obj)
# # visuallize_global_phases(obj)
# # visualize_attenuations(obj, num_of_el_along_x=10, num_of_el_along_y=10, scale=0.3)
# plt.show()
