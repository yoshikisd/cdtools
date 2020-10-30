from __future__ import division, print_function, absolute_import

import torch as t
from CDTools.models import CDIModel
from CDTools.datasets import Ptycho2DDataset
from CDTools import tools
from CDTools.tools import cmath
from CDTools.tools import plotting as p
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from copy import copy


class SMatrixPtycho(CDIModel):

    def __init__(self, wavelength, detector_geometry,
                 probe_basis, probe_guess, probe_fourier_support,
                 s_matrix_guess,
                 detector_slice=None,
                 surface_normal=np.array([0.,0.,1.]),
                 min_translation = t.Tensor([0,0]),
                 background = None, translation_offsets=None, mask=None,
                 weights = None, translation_scale = 1, saturation=None,
                 oversampling=1):
        
        super(SMatrixPtycho,self).__init__()
        self.wavelength = t.Tensor([wavelength])
        self.detector_geometry = copy(detector_geometry)
        det_geo = self.detector_geometry
        if hasattr(det_geo, 'distance'):
            det_geo['distance'] = t.Tensor(det_geo['distance'])
        if hasattr(det_geo, 'basis'):
            det_geo['basis'] = t.Tensor(det_geo['basis'])
        if hasattr(det_geo, 'corner'):
            det_geo['corner'] = t.Tensor(det_geo['corner'])

        self.min_translation = t.Tensor(min_translation)

        self.probe_basis = t.Tensor(probe_basis)
        self.detector_slice = detector_slice
        self.surface_normal = t.Tensor(surface_normal)
        
        self.saturation = saturation
        
        if mask is None:
            self.mask = mask
        else:
            self.mask = t.BoolTensor(mask)
        
        # We rescale the probe here so it learns at the same rate as the
        # object
        if probe_guess.dim() > 3:
            self.probe_norm = 1 * t.max(tools.cmath.cabs(probe_guess[0].to(t.float32)))
        else:
            self.probe_norm = 1 * t.max(tools.cmath.cabs(probe_guess.to(t.float32)))         
            
        self.probe = t.nn.Parameter(probe_guess.to(t.float32)
                                    / self.probe_norm)
        
        self.s_matrix = t.nn.Parameter(s_matrix_guess.to(t.float32))
        
        if background is None:
            ew_shape = [s_matrix_guess.shape[0] - 1 + probe_guess.shape[1],
                        s_matrix_guess.shape[1] - 1 + probe_guess.shape[2]]
            if detector_slice is not None:
                background = 1e-6 * t.ones(t.ones(ew_shape)[self.detector_slice].shape).to(t.float32)
            else:
                background = 1e-6 * t.ones(ew_shape).to(t.float32)
                
        self.background = t.nn.Parameter(t.Tensor(background).to(t.float32))

        if weights is None:
            self.weights = None
        else:
            self.weights = t.nn.Parameter(t.Tensor(weights).to(t.float32))
        
        if translation_offsets is None:
            self.translation_offsets = None
        else:
            self.translation_offsets = t.nn.Parameter(t.Tensor(translation_offsets).to(t.float32)/ translation_scale) 

        self.translation_scale = translation_scale

        self.probe_fourier_support = t.Tensor(probe_fourier_support).to(t.float32)
        
        self.oversampling = oversampling
        
        
    @classmethod
    def from_dataset(cls, dataset, probe_convergence_radius, locality_radius=1, probe_size=None, randomize_ang=0, padding=0, n_modes=1, translation_scale = 1, saturation=None, propagation_distance=None, scattering_mode=None, oversampling=1, auto_center=True):
        
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # always do this on the cpu
        get_as_args = dataset.get_as_args
        dataset.get_as(device='cpu')
        (indices, translations), patterns = dataset[:]
        dataset.get_as(*get_as_args[0],**get_as_args[1])

        # Set to none to avoid issues with things outside the detector
        if auto_center:
            center = tools.image_processing.centroid(t.sum(patterns,dim=0))
        else:
            center = None
            
        # Then, generate the probe geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        probe_basis, ew_shape, det_slice =  ewg(det_basis,
                                                   det_shape,
                                                   wavelength,
                                                   distance,
                                                   center=center,
                                                   padding=padding,
                                                   opt_for_fft=False,
                                                   oversampling=oversampling)

        # This shrinks the probe to ensure that the output wavefield
        # is the correct shape
        probe_shape = t.Size(np.array(ew_shape) - (2*locality_radius))
        
        if hasattr(dataset, 'sample_info') and \
           dataset.sample_info is not None and \
           'orientation' in dataset.sample_info:
            surface_normal = dataset.sample_info['orientation'][2]
        else:
            surface_normal = np.array([0.,0.,1.])


        # If this information is supplied when the function is called,
        # then we override the information in the .cxi file
        if scattering_mode in {'t', 'transmission'}:
            surface_normal = np.array([0.,0.,1.])
        elif scattering_mode in {'r', 'reflection'}:
            outgoing_dir = np.cross(det_basis[:,0], det_basis[:,1])
            outgoing_dir /= np.linalg.norm(outgoing_dir)
            surface_normal = outgoing_dir + np.array([0.,0.,1.])
            surface_normal /= np.linalg.norm(outgoing_dir)


        # Next generate the object geometry from the probe geometry and
        # the translations
        pix_translations = tools.interactions.translations_to_pixel(probe_basis, translations, surface_normal=surface_normal)

        # The locality radius correction is probably not needed because
        # it will always be way less than 200, but it ensures that there
        # is no wrapping in the s-matrix
        obj_size, min_translation = tools.initializers.calc_object_setup(probe_shape, pix_translations, padding=200+2*locality_radius)

        if hasattr(dataset, 'background') and dataset.background is not None:
            background = t.sqrt(dataset.background)
        else:
            background = None

        # Finally, initialize the probe and  object using this information
        if probe_size is None:
            if locality_radius != 0:
                probe = tools.initializers.SHARP_style_probe(dataset, ew_shape, det_slice, propagation_distance=propagation_distance, oversampling=oversampling)[locality_radius:-locality_radius,locality_radius:-locality_radius]
            else:
                probe = tools.initializers.SHARP_style_probe(dataset, ew_shape, det_slice, propagation_distance=propagation_distance, oversampling=oversampling)
        else:
            probe = tools.initializers.gaussian_probe(dataset, probe_basis, probe_shape, probe_size, propagation_distance=propagation_distance)


        # Now we initialize all the subdominant probe modes
        probe_max = t.max(cmath.cabs(probe))
        probe_stack = [0.01 * probe_max * t.rand(probe.shape,dtype=probe.dtype) for i in range(n_modes - 1)]
        probe = t.stack([tools.propagators.inverse_far_field(probe),] + probe_stack)

        s_matrix = t.zeros([2*locality_radius+1,2*locality_radius+1,obj_size[0],
                            obj_size[1],2])
        s_matrix[locality_radius,locality_radius,:,:,:] = \
            tools.cmath.expi(randomize_ang * (t.rand(obj_size)-0.5))


        
        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset),2)) - 0.5)

        weights = t.ones(len(dataset))
        
        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.bool)
        else:
            mask = None

        
        probe_support = t.zeros_like(probe[0].to(dtype=t.float32))
        xs, ys = np.mgrid[:probe.shape[1],:probe.shape[2]]
        xs = xs - np.mean(xs)
        ys = ys - np.mean(ys)
        Rs = np.sqrt(xs**2 + ys**2)
        probe_support[Rs<probe_convergence_radius] = 1
        probe = probe * probe_support[None,:,:]
        

        return cls(wavelength, det_geo, probe_basis, probe, probe_support,
                   s_matrix,
                   detector_slice=det_slice,
                   surface_normal=surface_normal,
                   min_translation=min_translation,
                   translation_offsets = translation_offsets,
                   weights=weights, mask=mask, background=background,
                   translation_scale=translation_scale,
                   saturation=saturation,
                   oversampling=oversampling)
                   
    
    def interaction(self, index, translations):
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations,
                                                             surface_normal=self.surface_normal)
        pix_trans -= self.min_translation

        if self.translation_offsets is not None:
            pix_trans += self.translation_scale * self.translation_offsets[index]
            
        all_exit_waves = []
        for i in range(self.probe.shape[0]):
            pr = tools.propagators.inverse_far_field(self.probe[i] * self.probe_fourier_support)
            exit_waves = self.probe_norm * tools.interactions.ptycho_2D_sinc_s_matrix(
                pr, self.s_matrix, pix_trans, shift_probe=True)
            
            exit_waves = exit_waves 


            if exit_waves.dim() == 4:
                exit_waves =  self.weights[index][:,None,None,None] * exit_waves
            else:
                exit_waves =  self.weights[index] * exit_waves
                
            all_exit_waves.append(exit_waves)

        
        return t.stack(all_exit_waves)
    
        
    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)


    def backward_propagator(self, wavefields):
        return tools.propagators.inverse_far_field(wavefields)

    
    def measurement(self, wavefields):
        return tools.measurements.quadratic_background(wavefields,
                            self.background,
                            detector_slice=self.detector_slice,
                            measurement=tools.measurements.incoherent_sum,
                            saturation=self.saturation,
                            oversampling=self.oversampling)

    
    def loss(self, sim_data, real_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data, mask=mask)

    
    def to(self, *args, **kwargs):
        super(SMatrixPtycho, self).to(*args, **kwargs)
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

            
        self.min_translation = self.min_translation.to(*args,**kwargs)
        self.probe_basis = self.probe_basis.to(*args,**kwargs)
        self.probe_norm = self.probe_norm.to(*args,**kwargs)
        self.probe_fourier_support = self.probe_fourier_support.to(*args,**kwargs)
        self.surface_normal = self.surface_normal.to(*args, **kwargs)

        
    def sim_to_dataset(self, args_list):
        # In the future, potentially add more control
        # over what metadata is saved (names, etc.)
        
        # First, I need to gather all the relevant data
        # that needs to be added to the dataset
        entry_info = {'program_name': 'CDTools',
                      'instrument_n': 'Simulated Data',
                      'start_time': datetime.now()}

        surface_normal = self.surface_normal.detach().cpu().numpy()
        xsurfacevec = np.cross(np.array([0.,1.,0.]), surface_normal)
        xsurfacevec /= np.linalg.norm(xsurfacevec)
        ysurfacevec = np.cross(surface_normal, xsurfacevec)
        ysurfacevec /= np.linalg.norm(ysurfacevec)
        orientation = np.array([xsurfacevec, ysurfacevec, surface_normal])
        
        sample_info = {'description': 'A simulated sample',
                       'orientation': orientation}

        
        detector_geometry = self.detector_geometry
        mask = self.mask
        wavelength = self.wavelength
        indices, translations = args_list
        
        # Then we simulate the results
        data = self.forward(indices, translations)

        # And finally, we make the dataset
        return Ptycho2DDataset(translations, data,
                                 entry_info = entry_info,
                                 sample_info = sample_info,
                                 wavelength=wavelength,
                                 detector_geometry=detector_geometry,
                                 mask=mask)

    
    def corrected_translations(self,dataset):
        translations = dataset.translations.to(dtype=self.probe.dtype,device=self.probe.device)
        t_offset = tools.interactions.pixel_to_translations(self.probe_basis,self.translation_offsets*self.translation_scale,surface_normal=self.surface_normal)
        return translations + t_offset


    # Needs to be updated to allow for plotting to an existing figure
    plot_list = [
        ('Dominant Probe Amplitude',
         lambda self, fig: p.plot_amplitude(self.probe[0], fig=fig, basis=self.probe_basis)),
        ('Dominant Probe Phase',
         lambda self, fig: p.plot_phase(self.probe[0], fig=fig, basis=self.probe_basis)),
        ('Subdominant Probe Amplitude',
         lambda self, fig: p.plot_amplitude(self.probe[1], fig=fig, basis=self.probe_basis),
         lambda self: len(self.probe) >=2),
        ('Subdominant Probe Phase',
         lambda self, fig: p.plot_phase(self.probe[1], fig=fig, basis=self.probe_basis),
         lambda self: len(self.probe) >=2),
        ('Exit Wave Amplitude under Uniform Illumination', 
         lambda self, fig: p.plot_amplitude(t.sum(self.s_matrix.data,dim=(0,1)), fig=fig, basis=self.probe_basis)),
        ('Exit Wave Phase under Uniform Illumination',
         lambda self, fig: p.plot_phase(t.sum(self.s_matrix.data,dim=(0,1)), fig=fig, basis=self.probe_basis)),
        ('Corrected Translations',
         lambda self, fig, dataset: p.plot_translations(self.corrected_translations(dataset), fig=fig)),
        ('Background',
         lambda self, fig: plt.figure(fig.number) and plt.imshow(self.background.detach().cpu().numpy()**2))
    ]

    
    def save_results(self, dataset):
        basis = self.probe_basis.detach().cpu().numpy()
        translations = self.corrected_translations(dataset).detach().cpu().numpy()
        probe = cmath.torch_to_complex(self.probe.detach().cpu())
        probe = probe * self.probe_norm.detach().cpu().numpy()
        s_matrix = cmath.torch_to_complex(self.s_matrix.detach().cpu())
        background = self.background.detach().cpu().numpy()**2
        weights = self.weights.detach().cpu().numpy()
        wavelength = self.wavelength.cpu().numpy()
        
        return {'basis':basis, 'translation':translations,
                'probe':probe,'s_matrix':s_matrix,
                'background':background,
                'weights':weights, 'wavelength':wavelength}
