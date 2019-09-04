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


class PinholePlanePtycho(CDIModel):

    def __init__(self, wavelength, detector_geometry,
                 probe_basis,
                 pinhole_guess, obj_guess, illumination_guess,
                 pinhole_propagation,
                 detector_slice=None,
                 surface_normal=np.array([0.,0.,1.]),
                 min_translation = t.Tensor([0,0]),
                 background = None, translation_offsets=None, mask=None,
                 weights = None, translation_scale = 1, saturation=None,
                 pinhole_support = None, obj_support=None, oversampling=1):
        
        super(PinholePlanePtycho,self).__init__()
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
        
        # We rescale the pinhole here so it learns at the same rate as the
        # object
        self.pinhole_norm = 1 * t.max(tools.cmath.cabs(pinhole_guess.to(t.float32)))         
            
        self.pinhole = t.nn.Parameter(pinhole_guess.to(t.float32)
                                    / self.pinhole_norm)
        
        self.obj = t.nn.Parameter(obj_guess.to(t.float32))
        self.illumination = t.nn.Parameter(illumination_guess.to(t.float32))

        spacing = np.abs([self.probe_basis.numpy()[0,1], self.probe_basis.numpy()[1,0]])

        
        self.pinhole_propagator = tools.propagators.generate_angular_spectrum_propagator(self.pinhole.shape[:-1], spacing, self.wavelength, pinhole_propagation).to(dtype=self.pinhole.dtype)
        
        
        if background is None:
            if detector_slice is not None:
                background = 1e-6 * t.ones(self.pinhole[self.detector_slice].shape[:-1])
            else:
                background = 1e-6 * t.ones(self.pinhole.shape[:-1])

                
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

        if pinhole_support is not None:
            self.pinhole_support = pinhole_support
        else:
            self.pinhole_support = t.ones_like(self.pinhole)

        if obj_support is not None:
            self.obj_support = obj_support
            self.obj.data = self.obj * obj_support
        else:
            self.obj_support = t.ones_like(self.obj)

        self.oversampling = oversampling
        
        
    @classmethod
    def from_dataset(cls, dataset, pinhole_propagation, pinhole_size=None, randomize_ang=0, padding=0, translation_scale = 1, saturation=None, pinhole_support_radius=None, propagation_distance=None, restrict_obj=-1, scattering_mode=None, oversampling=1):
        
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
        
        # Then, generate the pinhole geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        probe_basis, pinhole_shape, det_slice =  ewg(det_basis,
                                                     det_shape,
                                                     wavelength,
                                                     distance,
                                                     center=center,
                                                     padding=padding,
                                                     opt_for_fft=False,
                                                     oversampling=oversampling)


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

        obj_size, min_translation = tools.initializers.calc_object_setup(pinhole_shape, pix_translations, padding=200)

        if hasattr(dataset, 'background') and dataset.background is not None:
            background = t.sqrt(dataset.background)
        else:
            background = None

        # Finally, initialize the pinhole and  object using this information
        if pinhole_size is None:
            pinhole = tools.initializers.SHARP_style_probe(dataset, pinhole_shape, det_slice, propagation_distance=propagation_distance, oversampling=oversampling)
        else:
            pinhole = tools.initializers.gaussian_probe(dataset, probe_basis, pinhole_shape, pinhole_size, propagation_distance=propagation_distance)


        illumination = tools.cmath.expi(t.zeros(obj_size))
        obj = tools.cmath.expi(randomize_ang * (t.rand(obj_size)-0.5))
                              
        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset),2)) - 0.5)

        weights = t.ones(len(dataset))
        
        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.uint8)
        else:
            mask = None

        if pinhole_support_radius is not None:
            pinhole_support = t.zeros_like(pinhole.to(dtype=t.float32))
            p_cent = np.array(pinhole.shape[:2]).astype(int) // 2
            psr = int(pinhole_support_radius)
            pinhole_support[p_cent[0]-psr:p_cent[0]+psr,
                            p_cent[1]-psr:p_cent[1]+psr] = 1
            pinhole = pinhole_support * pinhole
        else:
            pinhole_support = None;

        if restrict_obj != -1:
            ro = restrict_obj
            os = np.array(obj_size)
            ps = np.array(pinhole_shape)
            obj_support = t.zeros_like(obj.to(dtype=t.float32))
            obj_support[ps[0]//2-ro:os[0]+ro-ps[0]//2,
                        ps[1]//2-ro:os[1]+ro-ps[1]//2] = 1
        else:
            obj_support = None

        return cls(wavelength, det_geo, probe_basis, pinhole, obj,
                   illumination, pinhole_propagation,
                   detector_slice=det_slice,
                   surface_normal=surface_normal,
                   min_translation=min_translation,
                   translation_offsets = translation_offsets,
                   weights=weights, mask=mask, background=background,
                   translation_scale=translation_scale,
                   saturation=saturation,
                   pinhole_support=pinhole_support,
                   obj_support=obj_support,
                   oversampling=oversampling)
                   
    
    def interaction(self, index, translations):
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations,
                                                             surface_normal=self.surface_normal)
        pix_trans -= self.min_translation

        if self.translation_offsets is not None:
            pix_trans += self.translation_scale * self.translation_offsets[index]

        pr = self.pinhole * self.pinhole_support
        pinhole_exit_waves = self.pinhole_norm * \
            tools.interactions.ptycho_2D_sinc(pr,
                                              self.illumination,
                                              pix_trans,
                                              shift_probe=True)
        sample_illumination = tools.propagators.near_field(pinhole_exit_waves,
                                                self.pinhole_propagator)
        exit_waves = []
        if pinhole_exit_waves.dim() == 4:
            for pix_tran, illum in zip(pix_trans, sample_illumination):
                exit_waves.append(tools.interactions.ptycho_2D_sinc(illum,
                                        self.obj_support * self.obj,
                                        pix_tran,
                                        shift_probe=True))
            exit_waves = t.stack(exit_waves)
        else:
            exit_waves = tools.interactions.ptycho_2D_sinc(sample_illumination,
                                        self.obj_support * self.obj,
                                        pix_trans,
                                        shift_probe=True)
                
        
        
        if exit_waves.dim() == 4:
            exit_waves =  self.weights[index][:,None,None,None] * exit_waves
        else:
            exit_waves =  self.weights[index] * exit_waves
            
            
        return exit_waves
    
        
    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)


    def backward_propagator(self, wavefields):
        return tools.propagators.inverse_far_field(wavefields)

    
    def measurement(self, wavefields):
        return tools.measurements.quadratic_background(wavefields,
                            self.background,
                            detector_slice=self.detector_slice,
                            saturation=self.saturation,
                            oversampling=self.oversampling)

    
    def loss(self, sim_data, real_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data, mask=mask)

    
    def to(self, *args, **kwargs):
        super(PinholePlanePtycho, self).to(*args, **kwargs)
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
        self.pinhole_norm = self.pinhole_norm.to(*args,**kwargs)
        self.pinhole_support = self.pinhole_support.to(*args,**kwargs)
        self.obj_support = self.obj_support.to(*args,**kwargs)
        self.surface_normal = self.surface_normal.to(*args, **kwargs)
        self.pinhole_propagator = self.pinhole_propagator.to(*args, **kwargs)

        
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
        translations = dataset.translations.to(dtype=self.pinhole.dtype,device=self.pinhole.device)
        t_offset = tools.interactions.pixel_to_translations(self.probe_basis,self.translation_offsets*self.translation_scale,surface_normal=self.surface_normal)
        return translations + t_offset


    # Needs to be updated to allow for plotting to an existing figure
    plot_list = [
        ('Pinhole Amplitude',
         lambda self, fig: p.plot_amplitude(self.pinhole, fig=fig, basis=self.probe_basis)),
        ('Pinhole Phase',
         lambda self, fig: p.plot_phase(self.pinhole, fig=fig, basis=self.probe_basis)),
        ('Illumination Amplitude',
         lambda self, fig: p.plot_amplitude(self.illumination, fig=fig, basis=self.probe_basis),
         lambda self: len(self.pinhole) >=2),
        ('Illumination Phase',
         lambda self, fig: p.plot_phase(self.illumination, fig=fig, basis=self.probe_basis),
         lambda self: len(self.pinhole) >=2),
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
        probe = cmath.torch_to_complex(self.pinhole.detach().cpu())
        probe = probe * self.pinhole_norm.detach().cpu().numpy()
        obj = cmath.torch_to_complex(self.obj.detach().cpu())
        background = self.background.detach().cpu().numpy()**2
        weights = self.weights.detach().cpu().numpy()
        
        return {'basis':basis, 'translation':translations,
                'probe':probe,'obj':obj,
                'background':background,
                'weights':weights}
