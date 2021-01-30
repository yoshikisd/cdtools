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
from functools import reduce

__all__ = ['Multislice2DPtycho']

class Multislice2DPtycho(CDIModel):

    def __init__(self, wavelength, detector_geometry,
                 probe_basis,
                 probe_guess, obj_guess, dz, nz,
                 detector_slice=None,
                 surface_normal=np.array([0.,0.,1.]),
                 min_translation = t.Tensor([0,0]),
                 background = None, translation_offsets=None, mask=None,
                 weights = None, translation_scale = 1, saturation=None,
                 #probe_support = None,
                 probe_fourier_support=None,
                 oversampling=1,
                 bandlimit=None,
                 subpixel=True,
                 exponentiate_obj=True,
                 fourier_probe=False, units='um'):
        
        super(Multislice2DPtycho,self).__init__()
        self.wavelength = t.Tensor([wavelength])
        self.detector_geometry = copy(detector_geometry)
        self.dz = -dz
        self.nz = nz
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
        self.subpixel = subpixel
        self.exponentiate_obj = exponentiate_obj
        self.fourier_probe = fourier_probe
        self.units = units
        
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
        
        self.obj = t.nn.Parameter(obj_guess.to(t.float32))
        
        if background is None:
            if detector_slice is not None:
                background = 1e-6 * t.ones(self.probe[0][self.detector_slice].shape[:-1])
            else:
                background = 1e-6 * t.ones(self.probe[0].shape[:-1])

                
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

        spacing = np.linalg.norm(self.probe_basis,axis=0)
        shape = np.array(self.probe.shape[1:-1])
        
        self.bandlimit = bandlimit
        
        self.as_prop = tools.propagators.generate_angular_spectrum_propagator(shape, spacing, self.wavelength, self.dz, bandlimit=self.bandlimit)

        
    @classmethod
    def from_dataset(cls, dataset, dz, nz, probe_convergence_radius, probe_size=None, padding=0, n_modes=1, translation_scale = 1, saturation=None, propagation_distance=None, scattering_mode=None, oversampling=1, auto_center=True, bandlimit=None, replicate_slice=False, subpixel=True, exponentiate_obj=True, units='um', fourier_probe=False):
        
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
        probe_basis, probe_shape, det_slice =  ewg(det_basis,
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

        obj_size, min_translation = tools.initializers.calc_object_setup(probe_shape, pix_translations, padding=200)

        if hasattr(dataset, 'background') and dataset.background is not None:
            background = t.sqrt(dataset.background)
        else:
            background = None

        # Finally, initialize the probe and  object using this information
        if probe_size is None:
            probe = tools.initializers.SHARP_style_probe(dataset, probe_shape, det_slice, propagation_distance=propagation_distance, oversampling=oversampling)
        else:
            probe = tools.initializers.gaussian_probe(dataset, probe_basis, probe_shape, probe_size, propagation_distance=propagation_distance)

        # For a Fourier space probe
        if fourier_probe:
            probe = tools.propagators.far_field(probe)
            
        # Now we initialize all the subdominant probe modes
        probe_max = t.max(cmath.cabs(probe))
        probe_stack = [0.01 * probe_max * t.rand(probe.shape,dtype=probe.dtype) for i in range(n_modes - 1)]
        probe = t.stack([probe,] + probe_stack)

        # Consider a different start
        if exponentiate_obj:
            obj = t.zeros(obj_size+(2,))
        else:
            obj = tools.cmath.expi(t.zeros(obj_size))
        # If we will use a separate object per slice
        if not replicate_slice:
            obj = t.stack([obj]*nz)
            
        
        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset),2)) - 0.5)

        weights = t.ones(len(dataset))
        
        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.bool)
        else:
            mask = None

        probe_support = t.zeros_like(probe[0])
        xs, ys = np.mgrid[:probe.shape[-3],:probe.shape[-2]]
        xs = xs - np.mean(xs)
        ys = ys - np.mean(ys)
        Rs = np.sqrt(xs**2 + ys**2)
        
        probe_support[Rs<probe_convergence_radius] = 1
        probe = probe * probe_support[None,:,:]
        
        return cls(wavelength, det_geo, probe_basis, probe, obj, dz, nz,
                   detector_slice=det_slice,
                   surface_normal=surface_normal,
                   min_translation=min_translation,
                   translation_offsets = translation_offsets,
                   weights=weights, mask=mask, background=background,
                   translation_scale=translation_scale,
                   saturation=saturation,
                   #probe_support=probe_support,
                   probe_fourier_support=probe_support,
                   oversampling=oversampling,
                   bandlimit=bandlimit,
                   subpixel=subpixel,
                   exponentiate_obj=exponentiate_obj,
                   units=units, fourier_probe=fourier_probe)
                   
    
    def interaction(self, index, translations):
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations,
                                                             surface_normal=self.surface_normal)
        pix_trans -= self.min_translation

        if self.translation_offsets is not None:
            pix_trans += self.translation_scale * self.translation_offsets[index]

        # For a Fourier-space probe
        if self.fourier_probe:
            prs  =tools.propagators.inverse_far_field(self.probe*self.probe_fourier_support[None,:,:])
        else:
            prs = self.probe*self.probe_fourier_support[None,:,:]
        # Here is where the mixing would happen, if it happened

        if self.exponentiate_obj:
            obj = cmath.cexpi(self.obj/self.nz)
        else:
            obj = self.obj
        
        exit_waves = self.probe_norm * prs
        for i in range(self.nz):
            # If only one object slice
            if self.obj.dim() == 3:
                if i == 0 and self.subpixel:
                    # We only need to apply the subpixel shift to the first
                    # slice, because it shifts the probe
                    exit_waves = tools.interactions.ptycho_2D_sinc(
                        exit_waves, obj, pix_trans,
                        shift_probe=True, multiple_modes=True)
                else:
                    exit_waves = tools.interactions.ptycho_2D_round(
                        exit_waves, obj, pix_trans,
                        multiple_modes=True)
                    
            elif self.obj.dim() == 4:
                # If separate slices
                if i == 0 and self.subpixel:
                    exit_waves = tools.interactions.ptycho_2D_sinc(
                        exit_waves, obj[i], pix_trans,
                        shift_probe=True, multiple_modes=True)
                else:
                    exit_waves = tools.interactions.ptycho_2D_round(
                        exit_waves, obj[i], pix_trans,
                        multiple_modes=True)

            if i < self.nz-1: #on all but the last iteration
                exit_waves = tools.propagators.near_field(
                    exit_waves,self.as_prop)
                    
        
        if exit_waves.dim() == 5:
            # If the index is a list and not a single index
            exit_waves =  self.weights[index][...,None,None,None,None] * exit_waves
        else:
            # If the index a single index
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
                            measurement=tools.measurements.incoherent_sum,
                            saturation=self.saturation,
                            oversampling=self.oversampling)

    
    def loss(self, sim_data, real_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data, mask=mask)
        #return tools.losses.poisson_nll(real_data, sim_data, mask=mask,eps=0.5)

    
    def to(self, *args, **kwargs):
        super(Multislice2DPtycho, self).to(*args, **kwargs)
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
        #self.probe_support = self.probe_support.to(*args,**kwargs)
        self.probe_fourier_support = self.probe_fourier_support.to(*args,**kwargs)
        
        self.surface_normal = self.surface_normal.to(*args, **kwargs)
        self.as_prop = self.as_prop.to(*args, **kwargs)
        
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
        ('Dominant Probe Fourier Space Amplitude',
         lambda self, fig: p.plot_amplitude(self.probe[0] if self.fourier_probe else tools.propagators.inverse_far_field(self.probe[0]), fig=fig)),
        ('Dominant Probe Fourier Space Phase',
         lambda self, fig: p.plot_phase(self.probe[0] if self.fourier_probe else tools.propagators.inverse_far_field(self.probe[0]), fig=fig)),
        ('Dominant Probe Real Space Amplitude',
         lambda self, fig: p.plot_amplitude(self.probe[0] if not self.fourier_probe else tools.propagators.inverse_far_field(self.probe[0]), fig=fig, basis=self.probe_basis, units=self.units)),
        ('Dominant Probe Real Space Phase',
         lambda self, fig: p.plot_phase(self.probe[0] if not self.fourier_probe else tools.propagators.inverse_far_field(self.probe[0]), fig=fig, basis=self.probe_basis, units=self.units)),
        ('Subdominant Probe Real Space Amplitude',
         lambda self, fig: p.plot_amplitude(self.probe[1] if not self.fourier_probe else tools.propagators.inverse_far_field(self.probe[1]), fig=fig, basis=self.probe_basis, units=self.units),
         lambda self: len(self.probe) >=2),
        ('Subdominant Probe Real Space Phase',
         lambda self, fig: p.plot_phase(self.probe[1] if not self.fourier_probe else tools.propagators.inverse_far_field(self.probe[1]), fig=fig, basis=self.probe_basis, units=self.units),
         lambda self: len(self.probe) >=2),
        ('Integrated Real Part of T', 
         lambda self, fig: p.plot_real(t.sum(self.obj.detach().cpu(),dim=0), fig=fig, basis=self.probe_basis, units=self.units),
         lambda self: self.exponentiate_obj),
        ('Integrated Imaginary Part of T',
         lambda self, fig: p.plot_imag(t.sum(self.obj.detach().cpu(),dim=0), fig=fig, basis=self.probe_basis, units=self.units),
         lambda self: self.exponentiate_obj),
        ('Amplitude of Stacked Object Function', 
         lambda self, fig: p.plot_amplitude(reduce(cmath.cmult, self.obj.detach().cpu()), fig=fig, basis=self.probe_basis, units=self.units),
         lambda self: not self.exponentiate_obj),
        ('Phase of Stacked Object Function',
         lambda self, fig: p.plot_phase(reduce(cmath.cmult, self.obj.detach().cpu()), fig=fig, basis=self.probe_basis, units=self.units),
         lambda self: not self.exponentiate_obj),
        ('Corrected Translations',
         lambda self, fig, dataset: p.plot_translations(self.corrected_translations(dataset), fig=fig, units=self.units)),
        ('Background',
         lambda self, fig: plt.figure(fig.number) and plt.imshow(self.background.detach().cpu().numpy()**2))
    ]

    
    def save_results(self, dataset):
        basis = self.probe_basis.detach().cpu().numpy()
        translations = self.corrected_translations(dataset).detach().cpu().numpy()
        if self.fourier_probe:
            probe = tools.propagators.inverse_far_field(self.probe)
        else:
            probe = self.probe
            
        probe = cmath.torch_to_complex(probe.detach().cpu())
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = cmath.torch_to_complex(self.obj.detach().cpu())
        background = self.background.detach().cpu().numpy()**2
        weights = self.weights.detach().cpu().numpy()
        dz = self.dz
        nz = self.nz
        prop = self.as_prop
        
        return {'basis':basis, 'translation':translations,
                'probe':probe,'obj':obj,
                'background':background,
                'weights':weights, 'dz':dz, 'nz':nz,
                'interlayer propagator': prop}
