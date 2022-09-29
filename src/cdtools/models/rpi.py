import torch as t
from cdtools.models import CDIModel
from cdtools import tools
from cdtools.tools import plotting as p
from cdtools.tools.interactions import RPI_interaction
from cdtools.tools import initializers
from scipy.ndimage.morphology import binary_dilation
import numpy as np
from copy import copy
import time

__all__ = ['RPI']

#
# This model has to work a bit differently from a ptychography model
# because a typical RPI dataset will have lots of images, each of which
# can be reconstructed on it's own. I can see a few ways to approach this:
#
# 1) Enfore 1 image per dataset as a restriction for internal consistency
# of the "model" idea
#
# 2) Override some of the base functions of the CDIModel to accept optional
# parameters that make it work on larger datasets
#
# I think #1 is basically untenable, because it would require creating
# a new dataset and model for each reconstruction - which means instantiating
# tons of stuff and storing all sorts of excess information for each frame,
# when that could easily be reused for other frames. As a result, I think I
# will try the following pattern:
#
# 1) A model will contain a single object "guess" at all times
# 2) Constructing a model from a data will automatically instantiate the object guess from the first diffraction pattern in the dataset
# 3) All the optimization functions will get an additional argument for the index of the diffraction pattern to reconstruct. This could either be handled by editing the CDIModel base class to pass through some kwargs, or by overriding all the optimization functions explicitly.
# 4) A few convenience functions can be written to re-initialize the object array from any image / index in the dataset
# 5) A new function can be written to reconstruct the entire dataset by running through each pattern one at a time.
#
# The advantage of this approach is that I can start by writing a class that
# will only reconstruct the first diffraction pattern from a dataset and then
# extend it.
#
# Final note: It is worth seeing whether it is possible to include the
# probe propagation explicitly as a parameter which can be reconstructed
# via gradient descent.
#
#

__all__ = ['RPI']

class RPI(CDIModel):

    @property
    def obj(self):
        return t.complex(self.obj_real, self.obj_imag)

    @property
    def weights(self):
        ws = t.complex(self.weights_real, self.weights_imag) 
        return ws / 10# / self.obj_real.size().numel()


    
    def __init__(self, wavelength, detector_geometry, probe_basis,
                 probe, obj_guess, detector_slice=None,
                 background=None, mask=None, saturation=None,
                 obj_support=None, oversampling=1, weight_matrix=False,
                 simulate_finite_pixels=False):

        super(RPI, self).__init__()

        self.wavelength = t.tensor(wavelength)
        self.detector_geometry = copy(detector_geometry)
        
        det_geo = self.detector_geometry
        if hasattr(det_geo, 'distance'):
            det_geo['distance'] = t.tensor(det_geo['distance'])
        if hasattr(det_geo, 'basis'):
            det_geo['basis'] = t.tensor(det_geo['basis'])
        if hasattr(det_geo, 'corner'):
            det_geo['corner'] = t.tensor(det_geo['corner'])

        self.probe_basis = t.tensor(probe_basis)

        scale_factor = t.tensor([probe.shape[-1]/obj_guess.shape[-1],
                                 probe.shape[-2]/obj_guess.shape[-2]])
        self.obj_basis = self.probe_basis * scale_factor
        self.detector_slice = detector_slice

        # Maybe something to include in a bit
        # self.surface_normal = t.tensor(surface_normal)
        
        self.saturation = saturation
        
        if mask is None:
            self.mask = mask
        else:
            self.mask = t.tensor(mask, dtype=t.bool)

            
        self.probe = t.tensor(probe, dtype=t.complex64)
        
        if obj_guess.dim() == 2:
            obj_guess = obj_guess[None, :, :]

        obj_guess = t.tensor(obj_guess, dtype=t.complex64)
        
        self.obj_real = t.nn.Parameter(obj_guess.real)
        self.obj_imag = t.nn.Parameter(obj_guess.imag)
        
        self.weights_real = t.nn.Parameter(t.eye(probe.shape[0])* 10)# * self.obj_real.size().numel())
        self.weights_imag = t.nn.Parameter(t.zeros(probe.shape[0]))

        if not weight_matrix:
            self.weights_real.requires_grad=False
            self.weights_imag.requires_grad=False
        
        # Wait for LBFGS to be updated for complex-valued parameters
        # self.obj = t.nn.Parameter(obj_guess.to(t.float32))

        if background is None:
            if detector_slice is not None:
                background = 1e-6 * t.ones(
                    self.probe[0][self.detector_slice].shape,
                    dtype=t.float32)
            else:
                background = 1e-6 * t.ones(self.probe[0].shape,
                                           dtype=t.float32)
                
        self.background = t.tensor(background, dtype=t.float32)

        if obj_support is not None:
            self.obj_support = obj_support
            self.obj.data = self.obj * obj_support[None, ...]
        else:
            self.obj_support = t.ones_like(self.obj[0, ...])

        self.oversampling = oversampling

        self.simulate_finite_pixels=simulate_finite_pixels


    @classmethod
    def from_dataset(cls, dataset, probe, obj_size=None, background=None, mask=None, padding=0, n_modes=1, saturation=None, scattering_mode=None, oversampling=1, auto_center=False, initialization='random', opt_for_fft=False, weight_matrix=False, probe_threshold=0, simulate_finite_pixels=False):
        
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # always do this on the cpu
        get_as_args = dataset.get_as_args
        dataset.get_as(device='cpu')
        # We only need the patterns here, not the inputs associated with them.
        _, patterns = dataset[:]
        dataset.get_as(*get_as_args[0],**get_as_args[1])

        # Set to none to avoid issues with things outside the detector
        if auto_center:
            center = tools.image_processing.centroid(t.sum(patterns,dim=0))
        else:
            center = None
            
        # Then, generate the probe geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        probe_basis, probe_shape, det_slice =  ewg(det_basis,
                                                   det_shape,
                                                   wavelength,
                                                   distance,
                                                   center=center,
                                                   padding=padding,
                                                   opt_for_fft=opt_for_fft,
                                                   oversampling=oversampling)

        if not isinstance(probe,t.Tensor):
            probe = t.as_tensor(probe)
        
        # Potentially need all of this orientation stuff later
        
        #if hasattr(dataset, 'sample_info') and \
        #   dataset.sample_info is not None and \
        #   'orientation' in dataset.sample_info:
        #    surface_normal = dataset.sample_info['orientation'][2]
        #else:
        #    surface_normal = np.array([0.,0.,1.])

        # If this information is supplied when the function is called,
        # then we override the information in the .cxi file
        #if scattering_mode in {'t', 'transmission'}:
        #    surface_normal = np.array([0.,0.,1.])
        #elif scattering_mode in {'r', 'reflection'}:
        #    outgoing_dir = np.cross(det_basis[:,0], det_basis[:,1])
        #    outgoing_dir /= np.linalg.norm(outgoing_dir)
        #    surface_normal = outgoing_dir + np.array([0.,0.,1.])
        #    surface_normal /= np.linalg.norm(surface_normal)


        if background is None and hasattr(dataset, 'background') \
           and dataset.background is not None:
            background = t.sqrt(dataset.background)
        elif background is not None:
            background = t.sqrt(t.Tensor(background).to(dtype=t.float32))

        det_geo = dataset.detector_geometry

        # If no mask is given, but one exists in the dataset, load it.
        if mask is None and hasattr(dataset, 'mask') \
           and dataset.mask is not None:
            mask = dataset.mask.to(t.bool)

        # Now we initialize the object
        if obj_size is None:
            # This is a standard size for a well-matched probe and detector
            obj_size = (np.array(probe_shape) // 2).astype(int)

        if initialization.lower().strip() == 'random':
            # I think something to do with the fact that the object is defined
            # on a coarser grid needs to be accounted for here that is not
            # accounted for yet
            scale = t.sum(patterns[0]) / t.sum(t.abs(probe)**2)
            obj_guess = scale * t.exp(2j * np.pi * t.rand([n_modes,]+list(obj_size)))
        elif initialization.lower().strip() == 'uniform':
            # I think something to do with the fact that the object is defined
            # on a coarser grid needs to be accounted for here that is not
            # accounted for yet
            scale = t.sum(patterns[0]) / t.sum(t.abs(probe)**2)
            obj_guess = scale * t.ones([n_modes,]+list(obj_size), dtype=probe.dtype)
        elif initialization.lower().strip() == 'spectral':
            if background is not None:
                obj_guess = initializers.RPI_spectral_init(
                    patterns[0], probe, obj_size, mask=mask,
                    background=background**2, n_modes=n_modes)
            else:
                obj_guess = initializers.RPI_spectral_init(
                    patterns[0], probe, obj_size, mask=mask,
                    n_modes=n_modes)
                
        else:
            raise KeyError('Initialization "' + str(initialization) + \
                           '" invalid - use "spectral", "uniform", or "random"')

        probe_intensity = t.sqrt(t.sum(t.abs(probe)**2,axis=0))
        probe_fft = tools.propagators.far_field(probe_intensity)
        pad0l = (probe.shape[-2] - obj_size[-2])//2
        pad0r = probe.shape[-2] - obj_size[-2] - pad0l
        pad1l = (probe.shape[-1] - obj_size[-1])//2
        pad1r = probe.shape[-1] - obj_size[-1] - pad1l
        probe_lr_fft = probe_fft[pad0l:-pad0r,pad1l:-pad1r]
        probe_lr = t.abs(tools.propagators.inverse_far_field(probe_lr_fft))

        obj_support = probe_lr > t.max(probe_lr) * probe_threshold
        obj_support = t.as_tensor(binary_dilation(obj_support))

        return cls(wavelength, det_geo, probe_basis,
                   probe, obj_guess, detector_slice=det_slice,
                   background=background, mask=mask, saturation=saturation,
                   obj_support=obj_support, oversampling=oversampling,
                   weight_matrix=weight_matrix,
                   simulate_finite_pixels=simulate_finite_pixels)

    @classmethod
    def from_calibration(cls, calibration, obj_size=None, n_modes=1, saturation=None):
        
        wavelength = calibration['wavelength']
        probe_basis = t.as_tensor(calibration['basis'])
        probe = t.as_tensor(calibration['probe'])
        if 'background' in calibration:
            background = t.as_tensor(calibration['background'])
        else:
            background = t.zeros_like(probe.real)
        if 'mask' in calibration:
            mask = t.as_tensor(calibration['mask'])
        else:
            mask = t.ones_like(probe.real, dtype=t.bool)

        # Now we initialize the object
        if obj_size is None:
            # This is a standard size for a well-matched probe and detector
            obj_size = (np.array(probe.shape[-2:]) // 2).astype(int)

        obj_guess = t.exp(2j * np.pi * t.rand([n_modes,]+obj_size))

        det_geo = {'distance': 1,
                   'basis': wavelength / probe_basis} # Is this even right?
                
        return cls(wavelength, det_geo, probe_basis,
                   probe, obj_guess,
                   background=background, mask=mask)

    
    def random_init(self, pattern):
        scale = t.sum(pattern) / t.sum(t.abs(self.probe)**2)
        self.obj.data = scale * t.exp(
            2j * np.pi * t.rand(self.obj.shape)).to(
                dtype=self.obj.dtype, device=self.obj.device)
        
    def spectral_init(self, pattern):
        if self.background is not None:
            self.obj.data = initializers.RPI_spectral_init(
                pattern, self.probe, self.obj.shape[-3:-1], mask=self.mask,
                background=self.background**2, n_modes=self.obj.shape[0]).to(
                    dtype=self.obj.dtype, device=self.obj.device)
        else:
            self.obj.data = initializers.RPI_spectral_init(
                pattern, self.probe, self.obj.shape[-3:-1], mask=self.mask,
                n_modes=self.obj.shape[0]).to(
                    dtype=self.obj.dtype, device=self.obj.device)
    
    # Needs work
    def interaction(self, index, *args):
        # including *args allows this to work with all sorts of datasets
        # that might include other information in with the index in their
        # "input" parameters (such as translations for a ptychography dataset).
        # This makes it seamless to use such a dataset even though those
        # extra arguments will not be used.

        
        all_exit_waves = []

        # Mix the probes with the weight matrix
        prs = t.sum(self.weights[..., None, None] * self.probe, axis=-3)
        
        for i in range(self.probe.shape[0]):
            pr = prs[i]
            # Here we have a 3D probe (one single mode)
            # and a 4D object (multiple modes mixing incoherently)
            exit_waves = RPI_interaction(pr,
                                         self.obj_support * self.obj)
                
            all_exit_waves.append(exit_waves)

        # This creates a bunch of modes generated from all possible combos
        # of the probe and object modes all strung out along the first index
        output = t.cat(all_exit_waves)
        
        # If we have multiple indexes input, we unsqueeze and repeat the stack
        # of wavefields enough times to simulate each requested index. This
        # seems silly, but it enables (for example) one to do a reconstruction
        # from a set of diffraction patterns that are all known to be from the
        # same object.
        try:
            # will fail if index has no length, for example when index
            # is just an int. In this case, we just do nothing instead
            output = output.unsqueeze(0).repeat(1,len(index),1,1,1)
        except TypeError:
            pass
        return output


    def forward_propagator(self, wavefields):
        p = tools.propagators.far_field(wavefields)
        return p


    def backward_propagator(self, wavefields):
        return tools.propagators.inverse_far_field(wavefields)

    
    def measurement(self, wavefields):
        # Here I'm taking advantage of an undocumented feature in the
        # incoherent_sum measurement function where it will work with
        # a 4D wavefield array as well as a 5D array.
        m = tools.measurements.quadratic_background(wavefields,
                            self.background,
                            detector_slice=self.detector_slice,
                            measurement=tools.measurements.incoherent_sum,
                            saturation=self.saturation,
                            oversampling=self.oversampling,
                            simulate_finite_pixels=self.simulate_finite_pixels)
        return m
    
    def loss(self, sim_data, real_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data, mask=mask)
        #return tools.losses.poisson_nll(real_data, sim_data, mask=mask)

    def regularizer(self, factors):
        if self.obj.shape[0] == 1:
            return factors[0] * t.sum(t.abs(self.obj[0,:,:])**2)
        else:
            return factors[0] * t.sum(t.abs(self.obj[0,:,:])**2) \
                + factors[1] * t.sum(t.abs(self.obj[1:,:,:])**2)
        
    def to(self, *args, **kwargs):
        super(RPI, self).to(*args, **kwargs)
        self.wavelength = self.wavelength.to(*args,**kwargs)
        # move the detector geometry too
        det_geo = self.detector_geometry
        if hasattr(det_geo, 'distance'):
            det_geo['distance'] = det_geo['distance'].to(*args,**kwargs)
        if hasattr(det_geo, 'basis'):
            det_geo['basis'] = det_geo['basis'].to(*args,**kwargs)
        if hasattr(det_geo, 'corner'):
            det_geo['corner'] = det_geo['corner'].to(*args,**kwargs)

        if self.mask is not None:
            self.mask = self.mask.to(*args, **kwargs)

        self.probe = self.probe.to(*args,**kwargs)
        self.probe_basis = self.probe_basis.to(*args,**kwargs)
        self.obj_basis = self.obj_basis.to(*args,**kwargs)
        self.obj_support = self.obj_support.to(*args,**kwargs)
        self.background = self.background.to(*args, **kwargs)
        
        # Maybe include in a bit
        #self.surface_normal = self.surface_normal.to(*args, **kwargs)

    def sim_to_dataset(self, args_list):
        raise NotImplementedError('No sim to dataset yet, sorry!')

    plot_list = [
        ('Root Sum Squared Amplitude of all Probes',
         lambda self, fig: p.plot_amplitude(
             np.sqrt(np.sum((t.abs(t.sum(self.weights[..., None, None].detach() * self.probe, axis=-3))**2).cpu().numpy(),axis=0)),
             fig=fig, basis=self.probe_basis)),
        ('Dominant Object Amplitude', 
         lambda self, fig: p.plot_amplitude(self.obj[0], fig=fig,
                                            basis=self.obj_basis)),
        ('Dominant Object Phase',
         lambda self, fig: p.plot_phase(self.obj[0], fig=fig,
                                        basis=self.obj_basis)),
        ('Subdominant Object Amplitude',
         lambda self, fig: p.plot_amplitude(self.obj[1], fig=fig,
                                            basis=self.obj_basis),
         lambda self: len(self.obj) >=2),
        ('Subdominant Object Phase',
         lambda self, fig: p.plot_phase(self.obj[1], fig=fig,
                                        basis=self.obj_basis),
         lambda self: len(self.obj) >=2)
    ]


    def save_results(self, dataset=None, full_obj=False):
        # dataset is set as a kwarg here because it isn't needed, but the
        # common pattern is to pass a dataset. This makes it okay if one
        # continues to use that standard pattern
        probe_basis = self.probe_basis.detach().cpu().numpy()
        obj_basis = self.obj_basis.detach().cpu().numpy()
        probe = self.probe.detach().cpu().numpy()
        # Provide the option to save out the subdominant objects or
        # just the dominant one
        if full_obj:
            obj = self.obj.detach().cpu().numpy()
        else:
            obj = self.obj[0].detach().cpu().numpy()
        background = self.background.detach().cpu().numpy()**2
        
        return {'probe_basis': probe_basis, 'obj_basis': obj_basis,
                'probe': probe,'obj': obj,
                'background': background}

