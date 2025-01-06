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

class FancyPtycho(CDIModel):

    def __init__(self,
                 wavelength,
                 detector_geometry,
                 obj_basis,
                 probe_guess,
                 obj_guess,
                 surface_normal=t.tensor([0., 0., 1.], dtype=t.float32),
                 min_translation=t.tensor([0, 0], dtype=t.float32),
                 background=None,
                 probe_basis=None,
                 translation_offsets=None,
                 probe_fourier_shifts=None,
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
                 exponentiate_obj=False,
                 phase_only=False,
                 dtype=t.float32,
                 obj_view_crop=0
                 ):

        super(FancyPtycho, self).__init__()
        self.register_buffer('wavelength',
                             t.as_tensor(wavelength, dtype=dtype))
        self.store_detector_geometry(detector_geometry,
                                     dtype=dtype)

        self.register_buffer('min_translation',
                             t.as_tensor(min_translation, dtype=dtype))

        self.register_buffer('obj_basis',
                             t.as_tensor(obj_basis, dtype=dtype))
        if probe_basis is None:
            self.register_buffer('probe_basis',
                                 t.as_tensor(obj_basis, dtype=dtype))
        else:
            self.register_buffer('probe_basis',
                                 t.as_tensor(probe_basis, dtype=dtype))
            
        self.register_buffer('surface_normal',
                             t.as_tensor(surface_normal, dtype=dtype))

        if saturation is None:
            self.saturation = None
        else:
            self.register_buffer('saturation',
                                 t.as_tensor(saturation, dtype=dtype))

        self.register_buffer('fourier_probe',
                             t.as_tensor(fourier_probe, dtype=bool))

        self.register_buffer('exponentiate_obj',
                             t.as_tensor(exponentiate_obj, dtype=bool))

        self.register_buffer('phase_only',
                             t.as_tensor(phase_only, dtype=bool))

        # Not sure how to make this a buffer...
        self.units = units

        if mask is None:
            self.mask = None
        else:
            self.register_buffer('mask',
                                 t.as_tensor(mask, dtype=t.bool))
        
        probe_guess = t.as_tensor(probe_guess, dtype=t.complex64)
        obj_guess = t.as_tensor(obj_guess, dtype=t.complex64)

            
        # We rescale the probe here so it learns at the same rate as the
        # object
        if probe_guess.dim() > 2:
            probe_norm = 1 * t.max(t.abs(probe_guess[0]))
        else:
            probe_norm = 1 * t.max(t.abs(probe_guess))
        self.register_buffer('probe_norm', probe_norm.to(dtype))
        
        self.probe = t.nn.Parameter(probe_guess / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess)

        # NOTE: I think it makes sense to protect against obj_view_crop
        # being zero or below, because there is nothing else to show outside
        # the object array. No reason to throw an error if, e.g., the user
        # asks for a big padding which goes outside of the actual object array.
        # Just show the full array.
        if obj_view_crop > 0:
            self.obj_view_slice = np.s_[obj_view_crop:-obj_view_crop,
                                        obj_view_crop:-obj_view_crop]
        else:
            self.obj_view_slice = np.s_[:,:]
        
        # TODO: perhaps not working anymore for fourier cropped probes
        if background is None:
            raise NotImplementedError('Issues with this due to probe fourier padding')
            shape = [s//oversampling for s in self.probe[0]]
            background = 1e-6 * t.ones(shape, dtype=t.float32)
            
        self.background = t.nn.Parameter(background)

        if weights is None:
            self.weights = None
        else:
            # We now need to distinguish between real-valued per-image
            # weights and complex-valued per-mode weight matrices
            if len(weights.shape) == 1:
                # This is if it's just a list of numbers
                self.weights = t.nn.Parameter(t.as_tensor(weights,
                                                       dtype=t.float32))
            else:
                # Now this is a matrix of weights, so it needs to be complex
                self.weights = t.nn.Parameter(t.as_tensor(weights,
                                                       dtype=t.complex64))

        if translation_offsets is None:
            self.translation_offsets = None
        else:
            t_o = t.as_tensor(translation_offsets, dtype=t.float32)
            t_o = t_o / translation_scale
            self.translation_offsets = t.nn.Parameter(t_o)
            
        if probe_fourier_shifts is None:
            self.probe_fourier_shifts = None
        else:
            self.probe_fourier_shifts = t.nn.Parameter(
                t.as_tensor(translation_offsets, dtype=t.float32)
            )

        self.register_buffer('translation_scale',
                             t.as_tensor(translation_scale, dtype=dtype))

        if probe_support is None:
            probe_support = t.ones_like(self.probe[0], dtype=t.bool)
        self.register_buffer('probe_support',
                             t.as_tensor(probe_support, dtype=t.bool))
        self.probe.data *= self.probe_support
            
        self.register_buffer('oversampling',
                             t.as_tensor(oversampling, dtype=int))

        self.register_buffer(
            'simulate_probe_translation',
            t.as_tensor(simulate_probe_translation, dtype=bool)
        )

        if simulate_probe_translation or (self.probe_fourier_shifts is not None):
            Is = t.arange(self.probe.shape[-2], dtype=dtype)
            Js = t.arange(self.probe.shape[-1], dtype=dtype)
            Is, Js = t.meshgrid(Is/t.max(Is), Js/t.max(Js))
            
            I_phase = 2 * np.pi* Is * self.oversampling
            J_phase = 2 * np.pi* Js * self.oversampling
            self.register_buffer('I_phase', I_phase)
            self.register_buffer('J_phase', J_phase)
            

        self.register_buffer('simulate_finite_pixels',
                             t.as_tensor(simulate_finite_pixels, dtype=bool))
            
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
                     n_modes=1,
                     n_obj_modes=1,
                     dm_rank=None,
                     translation_scale=1,
                     saturation=None,
                     probe_support_radius=None,
                     probe_fourier_crop=None,
                     propagation_distance=None,
                     scattering_mode=None,
                     oversampling=1,
                     fourier_probe=False,
                     loss='amplitude mse',
                     units='um',
                     allow_probe_fourier_shifts=False,
                     simulate_probe_translation=False,
                     simulate_finite_pixels=False,
                     exponentiate_obj=False,
                     phase_only=False,
                     obj_view_crop=None,
                     obj_padding=200,
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
        (indices, translations, *extras), patterns = dataset[:]

        dataset.get_as(*get_as_args[0], **get_as_args[1])

        # Then, generate the probe geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        obj_basis = ewg(
            det_basis,
            det_shape,
            wavelength,
            distance,
            oversampling=oversampling,
        )

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

        pix_translations = tools.interactions.translations_to_pixel(
            obj_basis,
            translations,
            surface_normal=surface_normal,
        )

        obj_size, min_translation = tools.initializers.calc_object_setup(
            [s * oversampling for s in det_shape],
            pix_translations,
            padding=obj_padding,
        )

        # Finally, initialize the probe and  object using this information
        if probe_size is None:
            probe = tools.initializers.SHARP_style_probe(
                dataset,
                propagation_distance=propagation_distance,
                oversampling=oversampling,
            )
        else:
            probe = tools.initializers.gaussian_probe(
                dataset,
                obj_basis,
                probe_shape,
                probe_size,
                propagation_distance=propagation_distance,
            )

        if hasattr(dataset, 'background') and dataset.background is not None:
            background = t.sqrt(dataset.background)
        else:
            background = 1e-6 * t.ones(
                dataset.patterns.shape[-2:], dtype=t.float32)
            
        if probe_fourier_crop is not None:
            probe = tools.propagators.far_field(probe)
            probe = probe[probe_fourier_crop : probe.shape[-2]
                          - probe_fourier_crop,
                          probe_fourier_crop : probe.shape[-1]
                          - probe_fourier_crop]
            probe = tools.propagators.inverse_far_field(probe)

            scale_factor = np.array(det_shape) / np.array(probe.shape)
            probe_basis = obj_basis * scale_factor[None,:]
        else:
            probe_basis = obj_basis.clone()
            
        # Now we initialize all the subdominant probe modes
        probe_max = t.max(t.abs(probe))
        probe_stack = [0.01 * probe_max * t.rand(probe.shape, dtype=probe.dtype) for i in range(n_modes - 1)]

        # For a Fourier space probe
        if fourier_probe:
            probe = tools.propagators.far_field(probe)

        probe = t.stack([probe, ] + probe_stack)

        obj = (randomize_ang * (t.rand(obj_size)-0.5)).to(dtype=t.complex64)
        if not exponentiate_obj:
            obj = t.exp(1j * obj)

        if n_obj_modes != 1:
            obj = t.stack([obj,] + [0.05*t.ones_like(obj),]*(n_obj_modes-1))

        if phase_only:
            obj.imag[:] = 0

        pfc = (probe_fourier_crop if probe_fourier_crop else 0)
        if obj_view_crop is None:
            obj_view_crop = min(probe.shape[-2], probe.shape[-1]) // 2 + pfc
        if obj_view_crop < 0:
            obj_view_crop += min(probe.shape[-2], probe.shape[-1]) // 2 + pfc

        obj_view_crop += obj_padding
            
        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset), 2)) - 0.5)

        if allow_probe_fourier_shifts:
            probe_fourier_shifts = t.zeros((len(dataset), 2), dtype=t.float32)
        else:
            probe_fourier_shifts = None

        if dm_rank is not None and dm_rank != 0:
            if dm_rank > n_modes:
                raise KeyError('Density matrix rank cannot be greater than the number of modes. Use dm_rank = -1 to use a full rank matrix.')
            elif dm_rank == -1:
                # dm_rank == -1 is defined to mean full-rank
                dm_rank = n_modes

            Ws = t.zeros(len(dataset), dm_rank, n_modes, dtype=t.complex64)
            # Start with as close to the identity matrix as possible,
            # cutting of when we hit the specified maximum rank
            for i in range(0, dm_rank):
                Ws[:, i, i] = 1
        else:
            # dm_rank == None or dm_rank = 0 triggers a special case where
            # a standard incoherent multi-mode model is used. This is the
            # default, because it is so common.
            # In this case, we define a set of weights which only has one index
            Ws = t.ones(len(dataset))

        if hasattr(dataset, 'intensities') and dataset.intensities is not None:
            intensities = dataset.intensities.to(dtype=Ws.dtype)[:,...]
            weights = t.sqrt(intensities)
            Ws *= (weights / t.mean(weights))

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

        return cls(wavelength, det_geo, obj_basis, probe, obj,
                   surface_normal=surface_normal,
                   min_translation=min_translation,
                   translation_offsets=translation_offsets,
                   weights=Ws, mask=mask, background=background,
                   translation_scale=translation_scale,
                   saturation=saturation,
                   probe_basis=probe_basis,
                   probe_support=probe_support,
                   fourier_probe=fourier_probe,
                   oversampling=oversampling,
                   loss=loss, units=units,
                   probe_fourier_shifts=probe_fourier_shifts,
                   simulate_probe_translation=simulate_probe_translation,
                   simulate_finite_pixels=simulate_finite_pixels,
                   phase_only=phase_only,
                   exponentiate_obj=exponentiate_obj,
                   obj_view_crop=obj_view_crop)


    def interaction(self, index, translations, *args):

        # The *args is included so that this can work even when given, say,
        # a polarized ptycho dataset that might spit out more inputs.

        # Step 1 is to convert the translations for each position into a
        # value in pixels
        pix_trans = tools.interactions.translations_to_pixel(
            self.obj_basis,
            translations,
            surface_normal=self.surface_normal)
        pix_trans -= self.min_translation
        # We then add on any recovered translation offset, if they exist
        if self.translation_offsets is not None:
            pix_trans += (self.translation_scale *
                          self.translation_offsets[index])

        # This restricts the basis probes to stay within the probe support
        basis_prs = self.probe * self.probe_support[..., :, :]

        # For a Fourier-space probe, we take an IFT
        if self.fourier_probe:
            basis_prs = tools.propagators.inverse_far_field(basis_prs)
            
        # Now we construct the probes for each shot from the basis probes
        if self.weights is not None:
            Ws = self.weights[index]
        else:
            try:
                Ws = t.ones(len(index)) # I'm positive this introduced a bug
            except:
                Ws = 1

        if self.weights is None or len(self.weights[0].shape) == 0:
            # If a purely stable coherent illumination is defined
            prs = Ws[..., None, None, None] * basis_prs
        else:
            # If a frame-by-frame weight matrix is defined
            # This takes the dot product of all the weight matrices with
            # the probes. The output has dimensions of translation, then
            # coherent mode index, then x,y, and then complex index
            # Maybe this can be done with a matmul now?
            prs = t.sum(Ws[..., None, None] * basis_prs, axis=-3)
        
        if self.simulate_probe_translation or (self.probe_fourier_shifts is not None):
            if self.probe_fourier_shifts is not None:
                det_pix_trans = self.probe_fourier_shifts[index]
            else:
                det_pix_trans = t.zeros_like(translations)

            if self.simulate_probe_translation:
                det_pix_trans = det_pix_trans +  tools.interactions.translations_to_pixel(
                    self.det_basis,
                    translations,
                    surface_normal=self.surface_normal)

                
            probe_masks = t.exp(1j* (det_pix_trans[:,0,None,None] *
                                     self.I_phase[None,...] +
                                     det_pix_trans[:,1,None,None] *
                                     self.J_phase[None,...]))
            prs = prs * probe_masks[...,None,:,:]


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

            
        if self.exponentiate_obj:
            if self.phase_only:
                obj = t.exp(1j*self.obj.real)
            else:
                obj = t.exp(1j*self.obj)
        else:
            obj = self.obj


        # Now we actually do the interaction, using the sinc subpixel
        # translation model as per usual
        exit_waves = self.probe_norm * tools.interactions.ptycho_2D_sinc(
            prs, obj, pix_trans,
            shift_probe=True,
            multiple_modes=True,
            probe_support=self.probe_support)
        
        return exit_waves


    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)


    def backward_propagator(self, wavefields):
        return tools.propagators.inverse_far_field(wavefields)


    def measurement(self, wavefields):
        return tools.measurements.quadratic_background(
            wavefields,
            self.background,
            measurement=tools.measurements.incoherent_sum,
            saturation=self.saturation,
            oversampling=self.oversampling,
            simulate_finite_pixels=self.simulate_finite_pixels,
        )


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
                self.obj_basis,
                self.translation_offsets * self.translation_scale,
                surface_normal=self.surface_normal)
            return translations + t_offset
        else:
            return translations

        
    def center_probes(self, iterations=4):
        """Centers the probes
        
        Note that this does not compensate for the centering by adjusting
        the object, so it's a good idea to reset the object after centering
        the probes
        """
        centered_probe = tools.image_processing.center(
            self.probe.data.cpu(), iterations=iterations)
        self.probe.data = centered_probe.to(device=self.probe.data.device)


    def tidy_probes(self):
        """Tidies up the probes
        
        What we want to do here is use all the information on all the probes
        to calculate a natural basis for the experiment, and update all the
        density matrices to operate in that updated basis

        As a first step, we calculate the state of the light field across the
        full experiment, using the weight matrices and basis probes. Then, we
        use an SVD to update the basis probes so they form an eigenbasis of
        the implied density matrix for the full experiment.

        Next, the weight matrices for each shot are recalculated so that the
        probes generated by weights * basis_probes for each shot are themselves
        an eigenbasis for that individual shot's density matrix.
        """
        
        # First we treat the incoherent but stable  case, where the weights are
        # just one per-shot overall weight
        if self.weights.dim() == 1:
            probe = self.probe.detach().cpu().numpy()
            ortho_probes = analysis.orthogonalize_probes(self.probe.detach())
            self.probe.data = ortho_probes
            return

        # What follows is for the unified OPRP and incoherent multi-mode model,
        # where each shot has it's own matrix of weights such that the probe
        # state for each shot is self.weights @ self.probe
        
        # We concatenate all the weight matrices, to come up with a state
        # corresponding to the summed light field across all the exposures.
        # This state will have a large number of modes, but all built from
        # the same small number of basis modes
        all_weights = t.cat(t.unbind(self.weights.detach(), dim=0), dim=0)

        # We generate the orthogonal probes based on this full-experiment
        # representation of the light field.
        ortho_probes, reexpressed_weights = \
            analysis.orthogonalize_probes(
                self.probe.detach(),
                weight_matrix=all_weights,
                return_reexpressed_weights=True
            )

        # We just orthogonalized the incoherent sum of all the exposures
        # across the full experiment, so the output probes are normalized so
        # that their intensity matches the summed intensity across the full
        # experiment. We divide their amplitudes by the square root of the
        # number of shots so that we now have a set of probes corresponding
        # to the mean shot
        ortho_probes /= np.sqrt(self.weights.shape[0])
        reexpressed_weights *= np.sqrt(self.weights.shape[0])
        
        # We now replace the shot-to-shot weights with the versions that have
        # been re-expressed in the new basis. 
        new_weights = t.stack(t.split(reexpressed_weights,
                                      self.weights.shape[1]), dim=0)

        # And we save it back to the model
        self.probe.data = ortho_probes.to(
            device=self.probe.device, dtype=self.probe.dtype)        
        self.weights.data = new_weights.to(
            device=self.weights.device, dtype=self.weights.dtype)

        # NOTE: I used to have this part as an option, with "tidy_each_frame",
        # because it took such a long time. Now that I've rewritten it properly,
        # it's quite fast and so I removed the kwarg because there's really
        # no situation where you woudn't want to do this.

        # Now, we seek to edit the shot-to-shot weight matrices such that
        # self.weights[i] @ self.probes will be properly orthogonalized for
        # all i.

        # All we need to know about the probes is that they are orthogonalized
        # and the intensity within each probe mode
        probe_sqrt_intensities = t.linalg.norm(self.probe.data, dim=(-2,-1))

        # This does a super fast batched computation
        U, S, Vh = t.linalg.svd(self.weights.data * probe_sqrt_intensities,
                                full_matrices=False)

        # We discard the U matrix and re-multiply S & Vh
        self.weights.data = S[:,:,None] * (Vh / probe_sqrt_intensities)

    
    def plot_wavefront_variation(self, dataset, fig=None, mode='amplitude', **kwargs):
        def get_probes(idx):
            basis_prs = self.probe * self.probe_support[..., :, :]
            prs = t.sum(self.weights[idx, :, :, None, None] * basis_prs,
                        axis=-3)
            ortho_probes = analysis.orthogonalize_probes(prs)

            if mode.lower() == 'amplitude':
                return np.abs(ortho_probes.detach().cpu().numpy())
            if mode.lower() == 'root_sum_intensity':
                return np.sum(np.abs(ortho_probes.detach().cpu().numpy())**2,
                              axis=0)
            if mode.lower() == 'phase':
                return np.angle(ortho_probes.detach().cpu().numpy())

        probe_matrix = np.zeros([self.probe.shape[0]]*2,
                                dtype=np.complex64)
        np_probes = self.probe.detach().cpu().numpy()
        for i in range(probe_matrix.shape[0]):
            for j in range(probe_matrix.shape[0]):
                probe_matrix[i,j] = np.sum(np_probes[i]*np_probes[j].conj())

        weights = self.weights.detach().cpu().numpy()

        probe_intensities = np.sum(np.tensordot(weights, probe_matrix, axes=1)
                                   * weights.conj(), axis=2)

        # Imaginary part is already essentially zero up to rounding error
        probe_intensities = np.real(probe_intensities)

        values = np.sum(probe_intensities, axis=1)
        if mode.lower() == 'amplitude' or mode.lower() == 'root_sum_intensity':
            cmap = 'viridis'
        else:
            cmap = 'twilight'

        p.plot_nanomap_with_images(
            self.corrected_translations(dataset),
            get_probes,
            values=values,
            fig=fig,
            units=self.units,
            basis=self.obj_basis,
            nanomap_colorbar_title='Total Probe Intensity',
            cmap=cmap,
            **kwargs),

        
    plot_list = [
        ('',
         lambda self, fig, dataset: self.plot_wavefront_variation(
             dataset,
             fig=fig,
             mode='root_sum_intensity',
             image_title='Root Summed Probe Intensities',
             image_colorbar_title='Square Root of Intensity'),
         lambda self: len(self.weights.shape) >= 2),
        ('',
         lambda self, fig, dataset: self.plot_wavefront_variation(
             dataset,
             fig=fig,
             mode='amplitude',
             image_title='Probe Amplitudes (scroll to view modes)',
             image_colorbar_title='Probe Amplitude'),
         lambda self: len(self.weights.shape) >= 2),
        ('',
         lambda self, fig, dataset: self.plot_wavefront_variation(
             dataset,
             fig=fig,
             mode='phase',
             image_title='Probe Phases (scroll to view modes)',
             image_colorbar_title='Probe Phase'),
         lambda self: len(self.weights.shape) >= 2),
        ('Basis Probe Fourier Space Amplitudes',
         lambda self, fig: p.plot_amplitude(
             (self.probe if self.fourier_probe
              else tools.propagators.far_field(self.probe)),
              fig=fig)),
        ('Basis Probe Fourier Space Colorized',
         lambda self, fig: p.plot_colorized(
             (self.probe if self.fourier_probe
              else tools.propagators.far_field(self.probe))
             , fig=fig)),
        ('Basis Probe Real Space Amplitudes',
         lambda self, fig: p.plot_amplitude(
             (self.probe if not self.fourier_probe
              else tools.propagators.inverse_far_field(self.probe)),
             fig=fig,
             basis=self.probe_basis,
             units=self.units)),
        ('Basis Probe Real Space Colorized',
         lambda self, fig: p.plot_colorized(
             (self.probe if not self.fourier_probe
              else tools.propagators.inverse_far_field(self.probe)),
             fig=fig,
             basis=self.probe_basis,
             units=self.units)),
        ('Average Weight Matrix Amplitudes',
         lambda self, fig: p.plot_amplitude(
             np.nanmean(np.abs(self.weights.data.cpu().numpy()), axis=0),
             fig=fig),
         lambda self: len(self.weights.shape) >= 2),
        ('% of Power in Top Mode',
         lambda self, fig, dataset: p.plot_nanomap(
             self.corrected_translations(dataset),
             100 * t.stack([
                 analysis.calc_mode_power_fractions(
                     self.probe.data,
                     weight_matrix=self.weights.data[i])[0]
                 for i in range(self.weights.shape[0])
             ], dim=0),
             fig=fig,
             units=self.units),
         lambda self: len(self.weights.shape) >= 2),
        ('Object Amplitude',
         lambda self, fig: p.plot_amplitude(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.obj_basis,
             units=self.units),
         lambda self: not self.exponentiate_obj),
        ('Object Phase',
         lambda self, fig: p.plot_phase(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.obj_basis,
             units=self.units),
         lambda self: not self.exponentiate_obj),
        ('Real Part of T', 
         lambda self, fig: p.plot_real(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.obj_basis,
             units=self.units,
             cmap='cividis'),
         lambda self: self.exponentiate_obj),
        ('Imaginary Part of T',
         lambda self, fig: p.plot_imag(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.obj_basis,
             units=self.units),
         lambda self: self.exponentiate_obj),

        ('Corrected Translations',
         lambda self, fig, dataset: p.plot_translations(self.corrected_translations(dataset), fig=fig, units=self.units)),
        ('Background',
         lambda self, fig: p.plot_amplitude(self.background**2, fig=fig))
    ]
    
    
    def save_results(self, dataset):
        # This will save out everything needed to recreate the object
        # in the same state, but it's not the best formatted. For example,
        # "background" stores the square root of the background, etc.
        base_results = super().save_results()

        # We also save out the main results in a more readable format
        obj_basis = self.obj_basis.detach().cpu().numpy()
        probe_basis = self.probe_basis.detach().cpu().numpy()
        translations=self.corrected_translations(dataset).detach().cpu().numpy()
        original_translations = dataset.translations.detach().cpu().numpy()
        probe = self.probe.detach().cpu().numpy()
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = self.obj.detach().cpu().numpy()
        background = self.background.detach().cpu().numpy()**2
        weights = self.weights.detach().cpu().numpy()
        oversampling = self.oversampling.cpu().numpy()
        wavelength = self.wavelength.cpu().numpy()

        results = {
            'obj_basis': obj_basis,
            'probe_basis': probe_basis,
            'translations': translations,
            'original_translations': original_translations,
            'probe': probe,
            'obj': obj,
            'background': background,
            'oversampling': oversampling,
            'weights': weights,
            'wavelength': wavelength,
        }

        return {**base_results, **results}

class FancyPtychoMultilayer(FancyPtycho):
    """ A subclass of FancyPtycho modified to handle the appearance of a
    rocking-curve produced by a multilayer Bragg mirror.

    This model should largely be identical to FancyPtycho but utilizes 3
    parameters: the object (obj_guess), probe (probe_guess), and a signal
    representing the contribution from the multilayer (ml_guess)

    The parameter ml_guess has been added in the  __init__ and is, by default,
    defined as None. If the ml_guess is still None, FancyPtychoMultilayer
    should behave as if it was just FancyPtycho. If, however, ml_guess is
    defined as a tensor, then the tensor will be added as a t.nn.Parameter.
    """

    def __init__(self,
                 wavelength,
                 detector_geometry,
                 obj_basis,
                 probe_guess,
                 obj_guess,
                 ml_guess = None,
                 surface_normal=t.tensor([0., 0., 1.], dtype=t.float32),
                 min_translation=t.tensor([0, 0], dtype=t.float32),
                 background=None,
                 probe_basis=None,
                 translation_offsets=None,
                 probe_fourier_shifts=None,
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
                 exponentiate_obj=False,
                 phase_only=False,
                 dtype=t.float32,
                 obj_view_crop=0
                 ):
        """ Initializes the class FancyPtychoMultilayer.

        The initialization done here is nearly identical to the one performed
        in FancyPtycho. The main exception being...
        - ...the addition of ml_guess as a parameter, representing the
            signal contribution
        - ...the last few lines of code which adds ml_guess as a
            t.nn.Parameter if it is not None

        """
        # Call FancyPtycho's __init__ method
        super(FancyPtychoMultilayer, self).__init__(wavelength=wavelength,
                                                    detector_geometry=detector_geometry,
                                                    obj_basis=obj_basis,
                                                    probe_guess=probe_guess,
                                                    obj_guess=obj_guess,
                                                    surface_normal=surface_normal,
                                                    min_translation=min_translation,
                                                    background=background,
                                                    probe_basis=probe_basis,
                                                    translation_offsets=translation_offsets,
                                                    probe_fourier_shifts=probe_fourier_shifts,
                                                    mask=mask,
                                                    weights=weights,
                                                    translation_scale=translation_scale,
                                                    saturation=saturation,
                                                    probe_support=probe_support,
                                                    oversampling=oversampling,
                                                    fourier_probe=fourier_probe,
                                                    loss=loss,
                                                    units=units,
                                                    simulate_probe_translation=simulate_probe_translation,
                                                    simulate_finite_pixels=simulate_finite_pixels,
                                                    exponentiate_obj=exponentiate_obj,
                                                    phase_only=phase_only,
                                                    dtype=dtype,
                                                    obj_view_crop=obj_view_crop)

        # Store the representation of the multilayer signal
        # TODO: Double check to make sure that the None case works
        if ml_guess is not None:
            ml_guess = t.as_tensor(ml_guess, dtype=t.complex64)
        self.ml = ml_guess

    @classmethod
    def from_dataset(cls,
                     dataset,
                     probe_size=None,
                     randomize_ang=0,
                     n_modes=1,
                     n_obj_modes=1,
                     dm_rank=None,
                     translation_scale=1,
                     saturation=None,
                     probe_support_radius=None,
                     probe_fourier_crop=None,
                     propagation_distance=None,
                     scattering_mode=None,
                     oversampling=1,
                     fourier_probe=False,
                     loss='amplitude mse',
                     units='um',
                     allow_probe_fourier_shifts=False,
                     simulate_probe_translation=False,
                     simulate_finite_pixels=False,
                     exponentiate_obj=False,
                     phase_only=False,
                     multilayer=False,
                     obj_view_crop=None,
                     obj_padding=200,
                     ):
        """Creates a FancyPtychoMultilayer class from a dataset.

        This function is similar to its counterpart in FancyPtycho with
        the exception of its utilization of an additonal t.nn.Parameter
        to represent the signal contribtution of the multilayer.

        """
        # Call the OG from_dataset method from FancyPtycho
        super(FancyPtychoMultilayer,  cls).from_dataset(dataset=dataset,
                                                        probe_size=probe_size,
                                                        randomize_ang=randomize_ang,
                                                        n_modes=n_modes,
                                                        n_obj_modes=n_obj_modes,
                                                        dm_rank=dm_rank,
                                                        translation_scale=translation_scale,
                                                        saturation=saturation,
                                                        probe_support_radius=probe_support_radius,
                                                        probe_fourier_crop=probe_fourier_crop,
                                                        propagation_distance=propagation_distance,
                                                        scattering_mode=scattering_mode,
                                                        oversampling=oversampling,
                                                        fourier_probe=fourier_probe,
                                                        loss=loss,
                                                        units=units,
                                                        allow_probe_fourier_shifts=allow_probe_fourier_shifts,
                                                        simulate_probe_translation=simulate_probe_translation,
                                                        simulate_finite_pixels=simulate_finite_pixels,
                                                        exponentiate_obj=exponentiate_obj,
                                                        phase_only=phase_only,
                                                        multilayer=multilayer,
                                                        obj_view_crop=obj_view_crop,
                                                        obj_padding=obj_padding)

        

        # Additional consideration for a multilayer:
        # Assuming that the probe has been initialized as a SHARP style probe,
        # we would like to remove the loud sinc signal.
        if not multilayer:
            # TODO: Check if we can do a regular ptycho reconstruction by setting ml_guess
            #       to None while incorporating a multilayer into the interaction model
            ml_guess = None
        else:
            if probe_size is not None:
                # Throw an error if the beam is initialized as a gaussian; we're not
                # considering this situation.
                raise NotImplementedError(
                    "Multilayer consideration is not supported for probes initialized as gaussians.")

            # First, get the main probe mode in the far-field
            # TODO: Generalize this to accept multiple probes
            #probe_fft = tools.propagators.far_field(probe)[0].numpy()

            # Grab a vertical line scan showing the sinc-pulse
            # TODO: Make the determination of the vertical
            #       line scan position automated. It's currently manual
            #line_scan = np.mean(probe_fft[:, 150:160], axis=1)
            #intensity_scan = np.abs(line_scan)

            # Fit a sinc function
            # TODO: Before this step, use shrinkwrap (or somethin) to create a mask
            #       out of the annulus.
            # Question: What would be the optimal way to handle the phase here?
            """ Note to self: sinc does not nicely describe the multilayer
            def sinc(x, a, b, c, d):
                return a * np.sinc(b * (x - c)) + d
            """

            # But, where a sinc fails, maybe an interference function
            # (or a bit from it) may work
            #def custom_fn(x, a, b):
            #    return (np.sin(a * (x - b)) ** 2) / (np.sin(x - b) ** 2)

            #x_data = np.linspace(0, intensity_scan.shape[0],
            #                     intensity_scan.shape[0])
            #k = intensity_scan.argmax()
            #p0 = [intensity_scan[k], 20, x_data[k], intensity_scan.mean()]

            #params, _ = curve_fit(sinc, x_data, intensity_scan, p0=p0)

            #line_scan = 1 + 1  # Here to trigger a break-point

        return cls