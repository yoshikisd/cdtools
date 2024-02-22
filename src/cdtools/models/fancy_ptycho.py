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

# TODO The information on bases is all wrong when the probe_fourier_crop is
# used, this affects display but also saveout of data and maybe even stuff
# like the probe translation simulation

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

        super(FancyPtycho, self).__init__()
        self.register_buffer('wavelength',
                             t.tensor(wavelength, dtype=dtype))
        self.store_detector_geometry(detector_geometry,
                                     dtype=dtype)

        self.register_buffer('min_translation',
                             t.tensor(min_translation, dtype=dtype))

        self.register_buffer('obj_basis',
                             t.tensor(obj_basis, dtype=dtype))
        if probe_basis is None:
            self.register_buffer('probe_basis',
                                 t.tensor(obj_basis, dtype=dtype))
        else:
            self.register_buffer('probe_basis',
                                 t.tensor(probe_basis, dtype=dtype))
            
        self.register_buffer('surface_normal',
                             t.tensor(surface_normal, dtype=dtype))

        if saturation is None:
            self.saturation = None
        else:
            self.register_buffer('saturation',
                                 t.tensor(saturation, dtype=dtype))

        self.register_buffer('fourier_probe',
                             t.tensor(fourier_probe, dtype=bool))

        # Not sure how to make this a buffer...
        self.units = units

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
                self.weights = t.nn.Parameter(t.tensor(weights,
                                                       dtype=t.float32))
            else:
                # Now this is a matrix of weights, so it needs to be complex
                self.weights = t.nn.Parameter(t.tensor(weights,
                                                       dtype=t.complex64))

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
        self.probe.data *= self.probe_support
            
        self.register_buffer('oversampling',
                             t.tensor(oversampling, dtype=int))

        self.register_buffer('simulate_probe_translation',
                             t.tensor(simulate_probe_translation, dtype=bool))

        if simulate_probe_translation:
            Is = t.arange(self.probe.shape[-2], dtype=dtype)
            Js = t.arange(self.probe.shape[-1], dtype=dtype)
            Is, Js = t.meshgrid(Is/t.max(Is), Js/t.max(Js))
            
            I_phase = 2 * np.pi* Is * self.oversampling
            J_phase = 2 * np.pi* Js * self.oversampling
            self.register_buffer('I_phase', I_phase)
            self.register_buffer('J_phase', J_phase)
            

        self.register_buffer('simulate_finite_pixels',
                             t.tensor(simulate_finite_pixels, dtype=bool))
            
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
                     simulate_probe_translation=False,
                     simulate_finite_pixels=False,
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
            det_shape,
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
            shape = [s//oversampling 
                     for s in probe.shape[-2:]]
            background = 1e-6 * t.ones(shape, dtype=t.float32)
            
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

        obj = t.exp(1j * randomize_ang * (t.rand(obj_size)-0.5))
        if n_obj_modes != 1:
            obj = t.stack([obj,] + [0.05*t.ones_like(obj),]*(n_obj_modes-1))

        if obj_view_crop is None:
            obj_view_crop = min(probe.shape[-2], probe.shape[-1]) // 2
        if obj_view_crop < 0:
            obj_view_crop += min(probe.shape[-2], probe.shape[-1]) // 2
        obj_view_crop += obj_padding
            
        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset), 2)) - 0.5)

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
                   simulate_probe_translation=simulate_probe_translation,
                   simulate_finite_pixels=simulate_finite_pixels,
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
        
        if self.simulate_probe_translation:
            det_pix_trans = tools.interactions.translations_to_pixel(
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
        
        # Now we actually do the interaction, using the sinc subpixel
        # translation model as per usual
        exit_waves = self.probe_norm * tools.interactions.ptycho_2D_sinc(
            prs, self.obj, pix_trans,
            shift_probe=True, multiple_modes=True)
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
                self.obj_basis,
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

    def center_probes(self, iterations=4):
        """Centers the probes
        
        Note that this does not compensate for the centering by adjusting
        the object, so it's a good idea to reset the object after centering
        the probes
        """
        centered_probe = tools.image_processing.center(
            self.probe.data.cpu(), iterations=iterations)
        self.probe.data = centered_probe.to(device=self.probe.data.device)


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
        if self.weights.dim() == 1:
            probe = self.probe.detach().cpu().numpy()
            ortho_probes = analysis.orthogonalize_probes(probe)
            self.probe.data = t.as_tensor(
                ortho_probes,
                device=self.probe.device,
                dtype=self.probe.dtype)
            return

        # This is for the unified mode case

        # Note to future: We could probably do this more cleanly with an
        # SVD directly on the Ws matrix, instead of an eigendecomposition
        # of the rho matrix.

        rhos = self.get_rhos()
        overall_rho = np.mean(rhos, axis=0)
        probe = self.probe.detach().cpu().numpy()
        ortho_probes, A = analysis.orthogonalize_probes(
            probe, density_matrix=overall_rho,
            keep_transform=True, normalize=normalize)
        Aconj = A.conj()
        Atrans = np.transpose(A)
        new_rhos = np.matmul(Atrans, np.matmul(rhos, Aconj))

        new_rhos /= normalization
        ortho_probes *= np.sqrt(normalization)

        dm_rank = self.weights.shape[1]

        new_Ws = []
        for rho in new_rhos:
            # These are returned from smallest to largest - we want to keep
            # the largest ones
            w, v = sla.eigh(rho)
            w = w[::-1][:dm_rank]
            v = v[:, ::-1][:, :dm_rank]
            # For situations where the rank of the density matrix is not
            # full in reality, but we keep more modes around than needed,
            # some ws can go negative due to numerical error! This is
            # extremely rare, but comon enough to cause crashes occasionally
            # when there are thousands of individual matrices to transform
            # every time this is called.
            w = np.maximum(w, 0)

            new_Ws.append(np.dot(np.diag(np.sqrt(w)), v.transpose()))

        new_Ws = np.array(new_Ws)

        self.weights.data = t.as_tensor(
            new_Ws, dtype=self.weights.dtype, device=self.weights.device)

        self.probe.data = t.as_tensor(
            ortho_probes, device=self.probe.device, dtype=self.probe.dtype)


    def plot_wavefront_variation(self, dataset, fig=None, mode='amplitude', **kwargs):
        def get_probes(idx):
            basis_prs = self.probe * self.probe_support[..., :, :]
            prs = t.sum(self.weights[idx, :, :, None, None] * basis_prs,
                        axis=-4)
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
             basis=self.obj_basis,
             units=self.units)),
        ('Object Phase',
         lambda self, fig: p.plot_phase(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.obj_basis,
             units=self.units)),
        ('Corrected Translations',
         lambda self, fig, dataset: p.plot_translations(self.corrected_translations(dataset), fig=fig, units=self.units)),
        ('Background',
         lambda self, fig: plt.figure(fig.number) and plt.imshow(self.background.detach().cpu().numpy()**2))
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
