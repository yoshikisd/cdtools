import torch as t
from cdtools.models import CDIModel
from cdtools.datasets import Ptycho2DDataset
from cdtools import tools
from cdtools.tools import plotting as p
from cdtools.tools import analysis
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from scipy import linalg as sla
from copy import copy

__all__ = ['FancyPtycho']

class PolarizationSweptPtycho(CDIModel):

    def __init__(self, wavelength, detector_geometry,
                 probe_basis,
                 probe_guess,
                 obj_guess,
                 polarization_states,
                 detector_slice=None,
                 surface_normal=t.tensor([0., 0., 1.], dtype=t.float32),
                 min_translation=t.tensor([0, 0], dtype=t.float32),
                 background=None,
                 translation_offsets=None,
                 mask=None,
                 weights=None,
                 translation_scale=1,
                 saturation=None,
                 probe_support=None,
                 oversampling=1,
                 fourier_probe=False,
                 loss='amplitude mse',
                 units='um',
                 simulate_probe_translation=False,
                 simulate_finite_pixels=False,
                 dtype=t.float32,
                 obj_view_crop=0
                 ):

        super(PolarizationSweptPtycho, self).__init__()
        self.register_buffer('wavelength',
                             t.tensor(wavelength, dtype=dtype))
        self.store_detector_geometry(detector_geometry,
                                     dtype=dtype)

        self.register_buffer('min_translation',
                             t.tensor(min_translation, dtype=dtype))

        self.register_buffer('probe_basis',
                             t.tensor(probe_basis, dtype=dtype))
        
        self.detector_slice = copy(detector_slice)
        self.register_buffer('surface_normal',
                             t.tensor(surface_normal, dtype=dtype))
        if saturation is None:
            self.saturation = None
        else:
            self.register_buffer('saturation',
                                 t.tensor(saturation, dtype=dtype))
        # Not sure how to make this a buffer...
        self.units = units

        self.fourier_probe = fourier_probe

        if mask is None:
            self.mask = None
        else:
            self.register_buffer('mask',
                                 t.tensor(mask, dtype=t.bool))
        
        probe_guess = t.tensor(probe_guess, dtype=t.complex64)
        obj_guess = t.tensor(obj_guess, dtype=t.complex64)

        # We rescale the probe here so it learns at the same rate as the
        # object 
        if probe_guess.dim() > 2:
            probe_norm = 1 * t.max(t.abs(probe_guess[0]))
        else:
            probe_norm = 1 * t.max(t.abs(probe_guess))
        self.register_buffer('probe_norm', probe_norm.to(dtype))
                
        self.probe = t.nn.Parameter(probe_guess / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess)


        self.obj_view_slice = np.s_[obj_view_crop:-obj_view_crop,
                                    obj_view_crop:-obj_view_crop]
        
        if background is None:
            if detector_slice is not None:
                dummy_det = t.empty([s//oversampling
                                     for s in self.probe.shape[-2:]])
                shape = dummy_det[self.detector_slice].shape
                #shape = self.probe[0][self.detector_slice].shape
            else:
                shape = [s//oversampling for s in self.probe.shape[-2:]]
            background = 1e-6 * t.ones(shape, dtype=t.float32)
            
        self.background = t.nn.Parameter(background)

        if weights is None:
            self.weights = None
        else:
            self.weights = t.nn.Parameter(t.tensor(weights,
                                                   dtype=t.float32))

        if translation_offsets is None:
            self.translation_offsets = None
        else:
            t_o = t.tensor(translation_offsets, dtype=t.float32)
            t_o = t_o / translation_scale
            self.translation_offsets = t.nn.Parameter(t_o)

        self.register_buffer('translation_scale',
                             t.tensor(translation_scale, dtype=dtype))

        if probe_support is None:
            probe_support = t.ones_like(self.probe[0], dtype=t.bool)
        self.register_buffer('probe_support',
                             t.tensor(probe_support, dtype=t.bool))
            
        self.oversampling = oversampling

        self.simulate_probe_translation = simulate_probe_translation

        if simulate_probe_translation:
            Is = t.arange(self.probe.shape[-2], dtype=dtype)
            Js = t.arange(self.probe.shape[-1], dtype=dtype)
            Is, Js = t.meshgrid(Is/t.max(Is), Js/t.max(Js))
            
            I_phase = 2 * np.pi* Is * self.oversampling
            J_phase = 2 * np.pi* Js * self.oversampling
            self.register_buffer('I_phase', I_phase)
            self.register_buffer('J_phase', J_phase)
            

        self.simulate_finite_pixels = simulate_finite_pixels

        self.polarization_states = t.nn.Parameter(
            t.tensor(polarization_states, dtype=dtype))
        # by default, don't optimize this, but I think it might be
        # interesting to try it because the polarization states
        # are not very pure
        self.polarization_states.requires_grad = False
            
            
        # Here we set the appropriate loss function
        if (loss.lower().strip() == 'amplitude mse'
                or loss.lower().strip() == 'amplitude_mse'):
            self.loss = tools.losses.amplitude_mse
        elif (loss.lower().strip() == 'poisson nll'
                or loss.lower().strip() == 'poisson_nll'):
            self.loss = tools.losses.poisson_nll
        else:
            raise KeyError('Specified loss function not supported')


    @classmethod
    def from_dataset(cls,
                     dataset,
                     probe_size=None,
                     randomize_ang=0,
                     padding=0,
                     n_modes=1,
                     translation_scale=1,
                     saturation=None,
                     probe_support_radius=None,
                     probe_fourier_crop=None,
                     propagation_distance=None,
                     scattering_mode=None,
                     oversampling=1,
                     auto_center=False,
                     opt_for_fft=False,
                     fourier_probe=False,
                     loss='amplitude mse',
                     units='um',
                     simulate_probe_translation=False,
                     simulate_finite_pixels=False,
                     obj_view_crop=None,
                     obj_padding=200
                     ):

        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # always do this on the cpu
        get_as_args = dataset.get_as_args
        dataset.get_as(device='cpu')

        # We include the *extras to make this work even with datasets, like
        # polarization dependent datasets, that might toss out extra inputs
        ((indices, translations, *extras), patterns) = dataset[:]
        polarization_states = dataset.polarization_states
        
        dataset.get_as(*get_as_args[0], **get_as_args[1])

        # Set to none to avoid issues with things outside the detector
        if auto_center:
            center = tools.image_processing.centroid(t.sum(patterns, dim=0))
        else:
            center = None

        # Then, generate the probe geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        probe_basis, probe_shape, det_slice = ewg(det_basis,
                                                  det_shape,
                                                  wavelength,
                                                  distance,
                                                  center=center,
                                                  padding=padding,
                                                  opt_for_fft=opt_for_fft,
                                                  oversampling=oversampling)

        if hasattr(dataset, 'sample_info') and \
           dataset.sample_info is not None and \
           'orientation' in dataset.sample_info:
            surface_normal = dataset.sample_info['orientation'][2]
        else:
            surface_normal = np.array([0., 0., 1.])

        # If this information is supplied when the function is called,
        # then we override the information in the .cxi file
        if scattering_mode in {'t', 'transmission'}:
            surface_normal = np.array([0., 0., 1.])
        elif scattering_mode in {'r', 'reflection'}:
            outgoing_dir = np.cross(det_basis[:, 0], det_basis[:, 1])
            outgoing_dir /= np.linalg.norm(outgoing_dir)
            surface_normal = outgoing_dir + np.array([0., 0., 1.])
            surface_normal /= -np.linalg.norm(surface_normal)

        # Next generate the object geometry from the probe geometry and
        # the translations

        pix_translations = tools.interactions.translations_to_pixel(probe_basis, translations, surface_normal=surface_normal)

        obj_size, min_translation = tools.initializers.calc_object_setup(probe_shape, pix_translations, padding=obj_padding)

        if hasattr(dataset, 'background') and dataset.background is not None:
            background = t.sqrt(dataset.background)
        else:
            background = None

        # Finally, initialize the probe and  object using this information
        if probe_size is None:
            probe = tools.initializers.SHARP_style_probe(dataset, probe_shape, det_slice, propagation_distance=propagation_distance, oversampling=oversampling)
        else:
            probe = tools.initializers.gaussian_probe(dataset, probe_basis, probe_shape, probe_size, propagation_distance=propagation_distance)

        if probe_fourier_crop is not None:
            probe = tools.propagators.far_field(probe)
            probe = probe[probe_fourier_crop:-probe_fourier_crop,
                          probe_fourier_crop:-probe_fourier_crop]
            probe = tools.propagators.inverse_far_field(probe)
            
        # Now we initialize all the subdominant probe modes
        probe_max = t.max(t.abs(probe))
        probe_stack = [0.01 * probe_max * t.rand(probe.shape, dtype=probe.dtype) for i in range(n_modes - 1)]

        # For a Fourier space probe
        if fourier_probe:
            probe = tools.propagators.far_field(probe)

        probe = t.stack([probe, ] + probe_stack)
        # The probe gets one extra dimension, so we can simulate one probe
        # per polarization state
        n_states = polarization_states.shape[0]
        probe = t.stack([probe] * n_states, dim=0)

        # Looks like an identity matrix
        obj_base = t.exp(1j * randomize_ang * (t.rand(obj_size)-0.5))
        obj_top = t.stack([obj_base, obj_base*0], dim=0)
        obj_bottom = t.stack([obj_base*0, obj_base], dim=0)
        obj = t.stack([obj_top, obj_bottom], dim=0)

        if obj_view_crop is None:
            obj_view_crop = min(probe.shape[-2], probe.shape[-1]) // 2
        if obj_view_crop < 0:
            obj_view_crop += min(probe.shape[-2], probe.shape[-1]) // 2
        obj_view_crop += obj_padding
            
        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset), 2)) - 0.5)

        Ws = t.ones(len(dataset))

        if hasattr(dataset, 'intensities') and dataset.intensities is not None:
            Ws *= (dataset.intensities.to(dtype=Ws.dtype)[:,...]
                   / t.mean(dataset.intensities))

        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.bool)
        else:
            mask = None

        if probe_support_radius is not None:
            probe_support = t.zeros(probe[0].shape, dtype=t.bool)
            xs, ys = np.mgrid[:probe.shape[-2], :probe.shape[-1]]
            xs = xs - np.mean(xs)
            ys = ys - np.mean(ys)
            Rs = np.sqrt(xs**2 + ys**2)

            probe_support[Rs < probe_support_radius] = 1
            probe = probe * probe_support[None, :, :]

        else:
            probe_support = None

        return cls(wavelength, det_geo, probe_basis, probe, obj,
                   polarization_states=polarization_states,
                   detector_slice=det_slice,
                   surface_normal=surface_normal,
                   min_translation=min_translation,
                   translation_offsets=translation_offsets,
                   weights=Ws, mask=mask, background=background,
                   translation_scale=translation_scale,
                   saturation=saturation,
                   probe_support=probe_support,
                   fourier_probe=fourier_probe,
                   oversampling=oversampling,
                   loss=loss, units=units,
                   simulate_probe_translation=simulate_probe_translation,
                   simulate_finite_pixels=simulate_finite_pixels,
                   obj_view_crop=obj_view_crop)


    def interaction(self, index, translations, polarization_indices, *args):

        # The *args is included so that this can work even when given, say,
        # a polarized ptycho dataset that might spit out more inputs.

        # Step 1 is to convert the translations for each position into a
        # value in pixels
        pix_trans = tools.interactions.translations_to_pixel(
            self.probe_basis,
            translations,
            surface_normal=self.surface_normal)
        pix_trans -= self.min_translation
        # We then add on any recovered translation offset, if they exist
        if self.translation_offsets is not None:
            pix_trans += (self.translation_scale *
                          self.translation_offsets[index])
        # This restricts the basis probes to stay within the probe support
        basis_prs = self.probe * self.probe_support[..., :, :]
        print(basis_prs.shape)
        # Now we expand the polarization states explicitly.
        # pol_basis_prs has dimensions n_states x 2 x n_modes x m x l
        pol_basis_prs = (
            self.polarization_states[...,None,None,None]
            * basis_prs[...,None,:,:,:])
        print(pol_basis_prs.shape)
        # For a Fourier-space probe, we take an IFT
        if self.fourier_probe:
            basis_prs = tools.propagators.inverse_far_field(basis_prs)
            
        # So, first we get the appropriate probe for each shot.
        # prs is now n_frames x 2 x n_modes x m x l
        prs = pol_basis_prs[polarization_indices]
        print(prs.shape)
        # Now we construct the probes for each shot from the basis probes
        if self.weights is not None:
            Ws = self.weights[index]
            # And then we multiply by the weights, along the 0th dimension
            prs = Ws[..., None, None, None, None] * basis_prs
                
        if self.simulate_probe_translation:
            det_pix_trans = tools.interactions.translations_to_pixel(
                    self.det_basis,
                    translations,
                    surface_normal=self.surface_normal)
            
            probe_masks = t.exp(1j* (det_pix_trans[:,0,None,None] *
                                     self.I_phase[None,...] +
                                     det_pix_trans[:,1,None,None] *
                                     self.J_phase[None,...]))
            prs = prs * probe_masks[...,None,None,:,:]

        # We automatically rescale the probe to match the background size,
        # which allows us to do stuff like let the object be super-resolution,
        # while restricting the probe to the detector resolution but still
        # doing an explicit real-space limitation of the probe
        padding = [self.oversampling * self.background.shape[-2] - prs.shape[-2],
                   self.oversampling * self.background.shape[-1] - prs.shape[-1]]

        if any([p != 0 for p in padding]): # For probe_fourier_crop != 0.
            padding = [padding[-1]//2, padding[-1]-padding[-1]//2,
                       padding[-2]//2, padding[-2]-padding[-2]//2]
            prs = tools.propagators.far_field(prs)
            prs = t.nn.functional.pad(prs, padding)
            prs = tools.propagators.inverse_far_field(prs)
        # Now we actually do the interaction, using the sinc subpixel
        # translation model as per usual
        print('hi')
        print(prs[...,0,:,:,:].shape)
        print(self.obj[:,0].shape)
        exit_waves = self.probe_norm * tools.interactions.ptycho_2D_sinc(
            prs[...,0,:,:,:], self.obj[:,0], pix_trans,
            shift_probe=True, multiple_modes=True)
        exit_waves = exit_waves + (
            self.probe_norm * tools.interactions.ptycho_2D_sinc(
            prs[...,1,:,:,:], self.obj[:,1], pix_trans,
            shift_probe=True, multiple_modes=True)
        )
        print(exit_waves.shape)
        # After the object, no point in treating the polarization modes
        # any differently from the object/probe modes
        print(exit_waves.flatten(start_dim=1,end_dim=-3).shape)
        print('sup')
        return exit_waves.flatten(start_dim=1,end_dim=-3)


    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)


    def backward_propagator(self, wavefields):
        return tools.propagators.inverse_far_field(wavefields)


    def measurement(self, wavefields):
        return tools.measurements.quadratic_background(
            wavefields,
            self.background,
            detector_slice=self.detector_slice,
            measurement=tools.measurements.incoherent_sum,
            saturation=self.saturation,
            oversampling=self.oversampling,
            simulate_finite_pixels=self.simulate_finite_pixels)


    # Note: No "loss" function is defined here, because it is added
    # dynamically during object creation in __init__

    def sim_to_dataset(self, args_list, calculation_width=None):
        # In the future, potentially add more control
        # over what metadata is saved (names, etc.)

        # First, I need to gather all the relevant data
        # that needs to be added to the dataset
        entry_info = {'program_name': 'cdtools',
                      'instrument_n': 'Simulated Data',
                      'start_time': datetime.now()}

        surface_normal = self.surface_normal.detach().cpu().numpy()
        xsurfacevec = np.cross(np.array([0., 1., 0.]), surface_normal)
        xsurfacevec /= np.linalg.norm(xsurfacevec)
        ysurfacevec = np.cross(surface_normal, xsurfacevec)
        ysurfacevec /= np.linalg.norm(ysurfacevec)
        orientation = np.array([xsurfacevec, ysurfacevec, surface_normal])

        sample_info = {'description': 'A simulated sample',
                       'orientation': orientation}


        mask = self.mask
        wavelength = self.wavelength
        indices, translations = args_list

        data = []
        len(indices)
        if calculation_width is None:
            calculation_width = len(indices)
        index_chunks = [indices[i:i + calculation_width]
                        for i in range(0, len(indices),
                                       calculation_width)]
        translation_chunks = [translations[i:i + calculation_width]
                              for i in range(0, len(indices),
                                             calculation_width)]
        
            
        # Then we simulate the results
        data = [self.forward(idx, trans).detach()
                for idx, trans in zip(index_chunks, translation_chunks)]

        data = t.cat(data, dim=0)
        # And finally, we make the dataset
        return Ptycho2DDataset(
            translations, data,
            entry_info=entry_info,
            sample_info=sample_info,
            wavelength=wavelength,
            detector_geometry=self.get_detector_geometry(),
            mask=mask)


    def corrected_translations(self, dataset):
        translations = dataset.translations.to(
            dtype=t.float32, device=self.probe.device)
        if (hasattr(self, 'translation_offsets') and
            self.translation_offsets is not None):
            t_offset = tools.interactions.pixel_to_translations(
                self.probe_basis,
                self.translation_offsets * self.translation_scale,
                surface_normal=self.surface_normal)
            return translations + t_offset
        else:
            return translations


    def get_rhos(self):
        # If this is the general unified mode model
        if self.weights.dim() >= 2:
            Ws = self.weights.detach().cpu().numpy()
            rhos_out = np.matmul(np.swapaxes(Ws, 1, 2), Ws.conj())
            return rhos_out
        # This is the purely incoherent case
        else:
            return np.array([np.eye(self.probe.shape[0])]*self.weights.shape[0],
                            dtype=np.complex64)

    def tidy_probes(self, normalization=1, normalize=False):
        """Tidies up the probes
        
        What we want to do here is use all the information on all the probes
        to calculate a natural basis for the experiment, and update all the
        density matrices to operate in that updated basis
        """

        # First we treat the purely incoherent case

        # I don't love this pattern of using an if statement with a return
        # to catch this case, but because it's so much simpler than the
        # unified mode case I think it's appropriate
        probe = self.probe.detach().cpu().numpy()
        ortho_probes = analysis.orthogonalize_probes(probe)
        self.probe.data = t.as_tensor(
            ortho_probes,
            device=self.probe.device,
            dtype=self.probe.dtype)

        
    plot_list = [
        ('Basis Probe Fourier Space Amplitudes',
         lambda self, fig: p.plot_amplitude(
             (self.probe if self.fourier_probe
              else tools.propagators.inverse_far_field(self.probe)),
              fig=fig)),
        ('Basis Probe Fourier Space Phases',
         lambda self, fig: p.plot_phase(
             (self.probe if self.fourier_probe
              else tools.propagators.inverse_far_field(self.probe))
             , fig=fig)),
        ('Basis Probe Real Space Amplitudes',
         lambda self, fig: p.plot_amplitude(
             (self.probe if not self.fourier_probe
              else tools.propagators.inverse_far_field(self.probe)),
             fig=fig,
             basis=self.probe_basis,
             units=self.units)),
        ('Basis Probe Real Space Phases',
         lambda self, fig: p.plot_phase(
             (self.probe if not self.fourier_probe
              else tools.propagators.inverse_far_field(self.probe)),
             fig=fig,
             basis=self.probe_basis,
             units=self.units)),
        ('Average Density Matrix Amplitudes',
         lambda self, fig: p.plot_amplitude(
             np.nanmean(np.abs(self.get_rhos()), axis=0),
             fig=fig),
         lambda self: len(self.weights.shape) >= 2),
        ('% Power in Top Mode (only accurate after tidy_probes)',
         lambda self, fig, dataset: p.plot_nanomap(
             self.corrected_translations(dataset),
             analysis.calc_top_mode_fraction(self.get_rhos()),
             fig=fig,
             units=self.units),
         lambda self: len(self.weights.shape) >= 2),
        ('Object Amplitude',
         lambda self, fig: p.plot_amplitude(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.probe_basis,
             units=self.units)),
        ('Object Phase',
         lambda self, fig: p.plot_phase(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.probe_basis,
             units=self.units)),
        ('Corrected Translations',
         lambda self, fig, dataset: p.plot_translations(self.corrected_translations(dataset), fig=fig, units=self.units)),
        ('Background',
         lambda self, fig: plt.figure(fig.number) and plt.imshow(self.background.detach().cpu().numpy()**2))
    ]

#    def plot_errors(self, dataset):
        
    
    
    def save_results(self, dataset):
        basis = self.probe_basis.detach().cpu().numpy()
        translations = self.corrected_translations(dataset).detach().cpu().numpy()
        probe = self.probe.detach().cpu().numpy()
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = self.obj.detach().cpu().numpy()
        background = self.background.detach().cpu().numpy()**2
        weights = self.weights.detach().cpu().numpy()
        oversampling = self.oversampling
        wavelength = self.wavelength.cpu().numpy()

        return {'basis': basis, 'translation': translations,
                'probe': probe, 'obj': obj,
                'background': background,
                'oversampling': oversampling,
                'weights': weights, 'wavelength': wavelength}
