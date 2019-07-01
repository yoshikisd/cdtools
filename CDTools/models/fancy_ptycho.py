from __future__ import division, print_function, absolute_import

import torch as t
from CDTools.models import CDIModel
from CDTools.datasets import Ptycho_2D_Dataset
from CDTools import tools
from CDTools.tools import cmath
from CDTools.tools import plotting as p
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from copy import copy


class FancyPtycho(CDIModel):

    def __init__(self, wavelength, detector_geometry,
                 probe_basis, detector_slice,
                 probe_guess, obj_guess,
                 surface_normal=np.array([0.,0.,1.]),
                 min_translation = t.Tensor([0,0]),
                 background = None, translation_offsets=None, mask=None,
                 weights = None, translation_scale = 1, saturation=None,
                 probe_support = None, obj_support=None):
        
        super(FancyPtycho,self).__init__()
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
            self.mask = t.ByteTensor(mask)
        
        # We rescale the probe here so it learns at the same rate as the
        # object
        if probe_guess.dim() > 3:
            self.probe_norm = 1 * t.max(tools.cmath.cabs(probe_guess[0].to(t.float32)))
        else:
            self.probe_norm = 1 * t.max(tools.cmath.cabs(probe_guess.to(t.float32)))         
            
        self.probe = t.nn.Parameter(probe_guess.to(t.float32)
                                    / self.probe_norm)
        
        self.obj = t.nn.Parameter(obj_guess.to(t.float32))
        
        if background is None:
            background = 1e-6 * t.ones(self.probe[0][self.detector_slice].shape[:-1])

                
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

        if probe_support is not None:
            self.probe_support = probe_support
        else:
            self.probe_support = t.ones_like(self.probe[0])

        if obj_support is not None:
            self.obj_support = obj_support
            self.obj.data = self.obj * obj_support
        else:
            self.obj_support = t.ones_like(self.obj)
        
        
    @classmethod
    def from_dataset(cls, dataset, probe_size=None, randomize_ang=0, padding=0, n_modes=1, translation_scale = 1, saturation=None, probe_support_radius=None, propagation_distance=None, restrict_obj=-1, scattering_mode=None):
        
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
        center = tools.image_processing.centroid(t.sum(patterns,dim=0))
        
        # Then, generate the probe geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        probe_basis, probe_shape, det_slice =  ewg(det_basis,
                                                   det_shape,
                                                   wavelength,
                                                   distance,
                                                   center=center,
                                                   padding=padding,
                                                   opt_for_fft=False)


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
        
        obj_size, min_translation = tools.initializers.calc_object_setup(probe_shape, pix_translations, padding=50)

        if hasattr(dataset, 'background') and dataset.background is not None:
            background = t.sqrt(dataset.background)
        else:
            background = None

        # Finally, initialize the probe and  object using this information
        if probe_size is None:
            probe = tools.initializers.SHARP_style_probe(dataset, probe_shape, det_slice, propagation_distance=propagation_distance)
        else:
            probe = tools.initializers.gaussian_probe(dataset, probe_basis, probe_shape, probe_size, propagation_distance=propagation_distance)


        # Now we initialize all the subdominant probe modes
        probe_max = t.max(cmath.cabs(probe))
        probe_stack = [0.01 * probe_max * t.rand(probe.shape,dtype=probe.dtype) for i in range(n_modes - 1)]
        probe = t.stack([probe,] + probe_stack)

        obj = tools.cmath.expi(randomize_ang * (t.rand(obj_size)-0.5))
                              
        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset),2)) - 0.5)

        weights = t.ones(len(dataset))
        
        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.uint8)
        else:
            mask = None

        if probe_support_radius is not None:
            probe_support = t.zeros_like(probe[0].to(dtype=t.float32))
            p_cent = np.array(probe.shape[1:3]).astype(int) // 2
            psr = int(probe_support_radius)
            probe_support[p_cent[0]-psr:p_cent[0]+psr,
                          p_cent[1]-psr:p_cent[1]+psr] = 1
        else:
            probe_support = None;

        if restrict_obj != -1:
            ro = restrict_obj
            os = np.array(obj_size)
            ps = np.array(probe_shape)
            obj_support = t.zeros_like(obj.to(dtype=t.float32))
            obj_support[ps[0]//2-ro:os[0]+ro-ps[0]//2,
                        ps[1]//2-ro:os[1]+ro-ps[1]//2] = 1
        else:
            obj_support = None

        return cls(wavelength, det_geo, probe_basis, det_slice, probe, obj,
                   surface_normal=surface_normal,
                   min_translation=min_translation,
                   translation_offsets = translation_offsets,
                   weights=weights, mask=mask, background=background,
                   translation_scale=translation_scale,
                   saturation=saturation,
                   probe_support=probe_support,
                   obj_support=obj_support)
                   
    
    def interaction(self, index, translations):
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations,
                                                             surface_normal=self.surface_normal)
        pix_trans -= self.min_translation

        if self.translation_offsets is not None:
            pix_trans += self.translation_scale * self.translation_offsets[index]
            
        all_exit_waves = []
        for i in range(self.probe.shape[0]):
            pr = self.probe[i] * self.probe_support
            exit_waves = self.probe_norm * tools.interactions.ptycho_2D_sinc(pr,
                                                           self.obj_support * self.obj,
                                                           pix_trans,
                                                           shift_probe=True)
            exit_waves = exit_waves * self.probe_support[...,:,:]


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
                            saturation=self.saturation )

    
    def loss(self, sim_data, real_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data, mask=mask)

    
    def to(self, *args, **kwargs):
        super(FancyPtycho, self).to(*args, **kwargs)
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
        self.probe_support = self.probe_support.to(*args,**kwargs)
        self.obj_support = self.obj_support.to(*args,**kwargs)
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
        return Ptycho_2D_Dataset(translations, data,
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
        ('Object Amplitude', 
         lambda self, fig: p.plot_amplitude(self.obj, fig=fig, basis=self.probe_basis)),
        ('Object Phase',
         lambda self, fig: p.plot_phase(self.obj, fig=fig, basis=self.probe_basis)),
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
        obj = cmath.torch_to_complex(self.obj.detach().cpu())
        background = self.background.detach().cpu().numpy()**2
        weights = self.weights.detach().cpu().numpy()
        
        return {'basis':basis, 'translation':translations,
                'probe':probe,'obj':obj,
                'background':background,
                'weights':weights}
