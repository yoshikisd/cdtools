"""Contains functions to sensibly initialize reconstructions

The functions in this module both do the geometric calculations needed to
initialize the reconstrucions, and the heuristic calculations for
geierating sensible initializations for the probe guess.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t

__all__ = ['exit_wave_geometry', 'calc_object_setup', 'gaussian',
           'gaussian_probe', 'SHARP_style_probe', 'RPI_spectral_init'] 

from CDTools.tools import cmath
from CDTools.tools.propagators import *
from CDTools.tools.analysis import orthogonalize_probes
from scipy.fftpack import next_fast_len
from scipy.sparse import linalg as spla
from torch.nn.functional import pad
import numpy as np


def exit_wave_geometry(det_basis, det_shape, wavelength, distance, center=None, opt_for_fft=True, padding=0, oversampling=1):
    """Returns an exit wave basis and shape, as well as a detector slice for the given detector geometry
    
    It takes in the parameters for a given detector - the basis defining
    the pixel pitch and the shape, as well as the wavelength and propagation
    distance. Optionally, it accepts a defined "center", the zero-frequency
    pixel's location. It then will automatically define a larger detector
    if necessary, define the exit wave basis associated with a far-field
    diffraction experiment, and return that basis, shape, and detector slice
    
    Parameters
    ----------
    det_basis : torch.Tensor
        The detector basis, as defined elsewhere
    det_shape : torch.Size
        The (i,j) shape of the detector
    wavelength : float)
        The wavelength of light for the experiment, in m
    distance : float
        The sample-detector distance, in m
    center : torch.Tensor
        If defined, the location of the zero frequency pixel
    opt_for_fft : bool
        Default is true, whether to increase detector size to improve fft performance
    padding : int
        Default is 0, the size of an extra border of nonphysical pixels around the detector
    oversampling : int
        Default is 1, the amount to multiply the exit wave shape by.
    
    Returns
    -------
    basis : torch.Tensor
        The exit wave basis
    shape : torch.Tensor
        The exit wave's shape
    slice : slice
        The slice corresponding to the physical detector
    """
    
    det_shape = t.Tensor(tuple(det_shape)).to(t.int32)
    det_basis = t.Tensor(det_basis)
    # First, set the center if it's not already specified
    # This definition matches the center pixel of an fftshifted array
    if center is None:
        center = det_shape // 2
    else:
        center = t.Tensor(center).to(t.int32)
        
    # Then, calculate the required detector size from the centering
    # This is a bit opaque but was worth doing accurately
    min_left = center * 2
    min_right = (det_shape - center) * 2 - 1
    full_shape = t.max(min_left,min_right).to(t.int32) + 2 * padding

    # In some edge cases this shape can be smaller than the detector shape
    full_shape = t.max(full_shape, det_shape)

    
    if opt_for_fft:
        full_shape = t.Tensor([next_fast_len(dim) for dim in full_shape]).to(t.int32)
    
    # Then, generate a slice that pops the actual detector from the full
    # detector shape
    full_center = full_shape // 2
    det_slice = np.s_[int(full_center[0]-center[0]):
                      int(full_center[0]-center[0]+det_shape[0]),
                      int(full_center[1]-center[1]):
                      int(full_center[1]-center[1]+det_shape[1])]

    
    # Finally, generate the basis for the exit wave in real space

    # This method should work for a general parallelogram
    # shaped detector
    det_shape = det_basis * full_shape.to(t.float32)
    pinv_basis = t.Tensor(np.linalg.pinv(det_shape).transpose()).to(t.float32)
    real_space_basis = pinv_basis * wavelength * distance

    # This is definitely correct, but less simple. Included here
    # So future me can check that both versions are consistent.
    #oop_dir = np.cross(det_basis[:,0],det_basis[:,1])
    #oop_dir /= np.linalg.norm(oop_dir)
    #full_basis = np.array([np.array(det_basis[:,0]),np.array(det_basis[:,1]),oop_dir]).transpose()
    #inv_basis = t.Tensor(np.linalg.inv(full_basis)[:2,:].transpose()).to(t.float32)
    #real_space_basis = inv_basis*wavelength * distance / \
    #    full_shape.to(t.float32)

    # Finally, convert the shape back to a torch.Size
    full_shape = t.Size([dim * oversampling for dim in full_shape]) 


    return real_space_basis, full_shape, det_slice


def calc_object_setup(probe_shape, translations, padding=0):
    """Returns an object shape and minimum pixel translation

    Based on the given pixel-space translations, it will calculate the
    required size for an object array and calculate the pixel translation
    that corresponds to a shift by (0,0) of the probe. 
    
    Optionally a small extra border can be defined via the padding
    attribute. If this is done, the calculated pixel translation will
    correspond to (padding,padding)
    
    Parameters
    ----------
    probe_shape : torch.Size
        The size of the probe array
    translations : torch.Tensor
        A Jx2 stack of pixel-valued (i,j) translations
    padding : int
        Optional, the size of an extra border to include
    
    Returns
    -------
    obj_shape : torch.Size
        The minimum required size of the object array
    min_translation : torch.Tensor
        The minimum pixel-valued translation
    """

    # First we look at the translations to find the minimum translation
    # and the range of translations
    min_translation = t.min(translations, dim=0)[0]
    translation_range = t.max(translations, dim=0)[0] - min_translation
    
    # Calculate the required shape
    translation_range = t.ceil(translation_range).numpy().astype(np.int32)
    shape = translation_range + np.array(probe_shape) + 2 * padding
    shape = t.Size(shape)
    
    # And the minimum translation
    min_translation = min_translation - padding
    
    return shape, min_translation
    
    

def gaussian(shape, sigma, amplitude=1, center = None, curvature=[0,0]):
    """Returns an array with a centered Gaussian

    Takes in the shape and standard deviation of a gaussian
    and returns a complex torch tensor (trailing dimension is 2) with
    values corresponding to a two-dimensional gaussian function

    Note that [0, 0] is taken to be at the upper left corner of the array.
    Default is centered at ((shape[0]-1)/2, (shape[1]-1)/2)) because x and y are zero-indexed.
    By default, the phase is uniformly 0, however a curvature can be
    specified to simulate a probe that has been propagated a known distance
    from it's focal point. The curvature is implemented by adding a quadratic
    phase phi = exp(i*curvature/2 r^2) to the Gaussian

    Parameters
    ----------
    shape : array
        A 1x2 array specifying the dimensions of the output array in the form (i shape, j shape)
    sigma : array
        A 1x2 array specifying the i- and j- standard deviation of the gaussian in the form (i stdev, j stdev)
    amplitude : float
        Default 1, the amplitude the gaussian to simulate
    center : array
        Optional, a 1x2 array specifying the location of the center of the gaussian (i center, j center)
    curvature : array
        Optional, a complex part to add to the gaussian coefficient

    Returns
    -------
    torch.Tensor 
        The complex-style tensor storing the Gaussian
    """
    if center is None:
        center = ((shape[0]-1)/2, (shape[1]-1)/2)
        
    i, j = np.mgrid[:shape[0], :shape[1]]
    isq = (i - center[0])**2
    jsq = (j - center[1])**2
    result = np.exp((1j*curvature[0] / 2 - 1 / (2 * sigma[0]**2)) * isq + \
        (1j*curvature[1] / 2 - 1 / (2 * sigma[1]**2)) * jsq)
    return cmath.complex_to_torch(amplitude*result)



def gaussian_probe(dataset, basis, shape, sigma, propagation_distance=0):
    """Initializes a gaussian probe based on experimental parameters

    This function generates a gaussian probe initialization which has a
    total fluence matching the order of magnitude of the intensity in
    the observed dataset, provided the object function is of order 1.
    
    The initialization is done using parameters defined in physical units,
    such as sigma (in meters) and the propagation distance (in meters).
    The internal conversion to pixel space is done with a provided probe
    basis and probe shape.
    
    TODO: Should be updated to accept a mask
    
    Sigma can be provided either as a scalar for a uniform beam, or as 
    an iterable of length 2 with [sigma_i, sigma_j] being the components
    of sigma in the directions parallel to the i and j basis vectors of
    the probe basis
    
    Parameters
    ----------
    dataset : Ptycho_2D_Dataset
        The dataset whose intensity we want to match
    basis : array
        The real space basis for exit waves in our experiment
    shape : array
        The shape of the simulated real space arrays
    sigma : array
        The standard deviation of the probe at it's focus
    propagation_distance : float
        Default 0, a distance to propagate the gaussian from it's focus
    
    Returns
    -------
    torch.Tensor
        The complex-style tensor storing the Gaussian
    """
    # First, we want to generate the parameters (sigma and curvature) for the
    # propagated gaussian. Ignore the purely z-dependent phases
    wavelength = dataset.wavelength
    if propagation_distance is  None:
        propagation_distance = 0
    
    z = propagation_distance # for shorthand
    
    sigma = np.array(sigma)
    k = 2 * np.pi / wavelength
    zr = k * sigma**2
    sigmaz = sigma * np.sqrt(1 + (z / zr)**2)
    curvature = -k * z / (z**2 + zr**2)

    # The conversion must then be done to pixel space
    sigma_pix = sigmaz / np.array([np.linalg.norm(basis[:,0]),
                                   np.linalg.norm(basis[:,1])])
    curvature_pix = curvature * np.array([np.linalg.norm(basis[:,0]),
                                          np.linalg.norm(basis[:,1])])**2

    # Then we can generate the gaussian array
    probe = gaussian(shape, sigma=sigma_pix, curvature=curvature_pix)
        
    # Finally, we should calculate the average pattern intensity from the
    # dataset and normalize the gaussian probe. This should be done by
    avg_intensities = [t.sum(dataset[idx][1]) for idx in range(len(dataset))]
    avg_intensity = t.mean(t.Tensor(avg_intensities))
    probe_intensity = t.sum(cmath.cabssq(probe))
    return avg_intensity / probe_intensity * probe
    

def SHARP_style_probe(dataset, shape, det_slice, propagation_distance=None, oversampling=1):
    """Generates a SHARP style probe guess from a dataset

    What we call the "SHARP" style probe guess is to take a mean of all
    the diffraction patterns and use that as an initial guess of the
    Fourier space distribution of the probe. We set all the phases to
    zero, which would for many simple beams (like a zone plate) generate
    a first guess of the probe that is very close to the focal spot of
    the probe beam.

    If the probe is simulated in higher resolution than the detector,
    a common occurence, these undefined pixels are set to zero for the
    purposes of defining the guess

    We make a small tweak to this procedure to lower the central pixel of
    the probe generated this way, which can often overwhelm the rest of the
    probe if there is significant noise on the detector
        
    Parameters
    ----------
    dataset : Ptycho_2D_Dataset
        The dataset to work from
    shape : torch.Size
        The size of the probe array to simulate
    det_slice : slice
        A slice or tuple of slices corresponding to the detector region in Fourier space
    propagation_distance : float
        Default is no propagation, an amount to propagate the guessed probe from it's focal point
    oversampling : int 
        Default 1, the width of the region of pixels in the wavefield to bin into a single detector pixel
 
    Returns
    -------
    torch.Tensor
        The complex-style tensor storing the probe guess
    """


    # to use the mask or not?
    intensities = np.zeros([dim // oversampling for dim in shape])
    for params, im in dataset:
        if hasattr(dataset,'mask') and dataset.mask is not None:
            intensities[det_slice] += dataset.mask.cpu().numpy() * im.cpu().numpy()
        else:
            intensities[det_slice] += im.cpu().numpy()         
    intensities /= len(dataset)

    # Subtract off a known background if it's stored
    if hasattr(dataset, 'background') and dataset.background is not None:
        intensities[det_slice] = np.clip(intensities[det_slice] - dataset.background.cpu().numpy(), a_min=0,a_max=None)
    
    probe_fft = cmath.complex_to_torch(np.sqrt(intensities))
    
    probe_guess = cmath.torch_to_complex(inverse_far_field(probe_fft))
    
    # Now we remove the central pixel
    center = np.array(probe_guess.shape) // 2
    
    # I'm always divided on whether to use this modification:
    
    probe_guess[center[0], center[1]]=np.mean([
        probe_guess[center[0]-1, center[1]],
        probe_guess[center[0]+1, center[1]],
        probe_guess[center[0], center[1]-1],
        probe_guess[center[0], center[1]+1]])

    probe_guess = cmath.complex_to_torch(probe_guess)
    
    if propagation_distance is not None:
        # First generate the propagation array

        probe_shape = t.Tensor(tuple(probe_guess.shape))[:-1]

        # Start by recalculating the probe basis from the given information
        det_basis = t.Tensor(dataset.detector_geometry['basis'])
        basis_dirs = det_basis / t.norm(det_basis, dim=0)
        distance = dataset.detector_geometry['distance']
        probe_basis = basis_dirs * dataset.wavelength * distance / \
            (probe_shape * t.norm(det_basis,dim=0))

        # Then package everything as it's needed
        probe_spacing = t.norm(probe_basis,dim=0).numpy()
        probe_shape = probe_shape.numpy().astype(np.int32)

        #assert 0
        # And generate the propagator
        AS_prop = generate_angular_spectrum_propagator(probe_shape, probe_spacing, dataset.wavelength, propagation_distance)

        probe_guess = near_field(probe_guess,AS_prop)

    
    # Finally, place this probe in a full-sized array if there is oversampling
    final_probe = t.zeros([dim for dim in shape] + [2])
    left = shape[0]//2 - probe_guess.shape[0] // 2
    top = shape[1]//2 - probe_guess.shape[1] // 2 
    final_probe[left:left+probe_guess.shape[0],
                top:top+probe_guess.shape[1],:] = probe_guess
    
    return final_probe


def RPI_spectral_init(pattern, probe, obj_shape, n_modes=1, mask=None, background=None):

    # First, check if the probe is a single mode or many modes.
    # If the probe is many modes, orthogonalize it and use the top mode
    # for initialization
    if probe.dim() == 4:
        probe = orthogonalize_probes(probe)[0]
    
    pad0l = (probe.shape[-3] - obj_shape[0])//2
    pad0r = probe.shape[-3] - obj_shape[0] - pad0l
    pad1l = (probe.shape[-2] - obj_shape[1])//2
    pad1r = probe.shape[-2] - obj_shape[1] - pad1l
    
    def a_dagger(im):
        im = cmath.complex_to_torch(im.reshape(obj_shape)).to(dtype=t.float32)
        im = inverse_far_field(pad(far_field(im), (0,0,pad1l,pad1r,pad0l,pad0r)))
        exit_wave = cmath.cmult(probe,im)
        farfield = cmath.torch_to_complex(far_field(exit_wave))
        return farfield.ravel()

    def a(measured):
        measured = cmath.complex_to_torch(measured.reshape(pattern.shape[0],pattern.shape[1])).to(dtype=t.float32)
        im = inverse_far_field(measured)
        multiplied = cmath.cmult(cmath.cconj(probe), im)
        backplane = far_field(multiplied)
        clipped = backplane[pad0l:pad0l+obj_shape[0],
                            pad1l:pad1l+obj_shape[1],:]
        return cmath.torch_to_complex(inverse_far_field(clipped)).ravel()

    patsize = pattern.shape[0]*pattern.shape[1]
    imsize = obj_shape[0]*obj_shape[1]
    probesize = probe.shape[0]*probe.shape[1]
    A_dagger = spla.LinearOperator((patsize, imsize),matvec=a_dagger)
    A = spla.LinearOperator((imsize,patsize),matvec=a)

    # Correct the pattern for the background and mask
    np_pattern = pattern.numpy()
    if background is not None:
        np_pattern = np_pattern - background.numpy()
    if mask is not None:
        np_pattern = np_pattern * mask.numpy()
    np_pattern = np.abs(np_pattern.ravel())
        
    realspace_intensities = A * A_dagger * np.ones(imsize)
    vec = A_dagger * realspace_intensities
    def y(measured):
        #return measured * np_pattern
        # This normalizes the intensities to account for the fact
        # that some pixels draw from way more spots on the detector than
        # others
        return measured * np_pattern / np.abs(vec)

    Y = spla.LinearOperator((patsize, patsize),matvec=y)
    eigval, z0 = spla.eigs(A * Y * A_dagger, k=n_modes, which='LM')
    z0 = z0.transpose().reshape(n_modes,obj_shape[0], obj_shape[1])

    # Now we set the overall scale and relative weights of the guess
    scale_factor = np.sqrt(np.sum(np_pattern) /
                           t.sum(cmath.cabssq(probe)).numpy())
    relative_weights = eigval / np.sum(eigval**2)

    z0 = z0 * (scale_factor * relative_weights[:,None,None])
    # Now we have to normalize the modes by their eigenvalues

    return cmath.complex_to_torch(z0).to(dtype=t.float32)
