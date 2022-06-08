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

#
# Basic points:
# Just one probe mode, no need to overcomplicate things.
# Weights is only a list of numbers, no matrices or anything like that
# Mandatory probe support in Fourier space
#
# When loading from a dataset, we need information on the zone plate geometry
#

__all__ = ['TimeResolvedPtychoCalibration']


class TimeResolvedPtychoCalibration(CDIModel):

    def __init__(self, wavelength, detector_geometry,
                 probe_basis,
                 probe_guess, obj_guess,
                 fourier_times, probe_fourier_support,
                 times, time_dependence, frame_delays,
                 detector_slice=None,
                 surface_normal=t.tensor([0., 0., 1.], dtype=t.float32),
                 min_translation=t.tensor([0, 0], dtype=t.float32),
                 background=None, translation_offsets=None, mask=None,
                 weights=None, translation_scale=1, saturation=None,
                 oversampling=1,
                 loss='amplitude mse', units='um',
                 simulate_probe_translation=False):

        super(TimeResolvedPtychoCalibration, self).__init__()
        self.wavelength = t.tensor(wavelength)
        self.detector_geometry = copy(detector_geometry)
        det_geo = self.detector_geometry
        if 'distance' in det_geo:
            det_geo['distance'] = t.tensor(det_geo['distance'], dtype=t.float32)
        if 'basis' in det_geo:
            det_geo['basis'] = t.tensor(det_geo['basis'], dtype=t.float32)
        if 'corner' in det_geo:
            det_geo['corner'] = t.tensor(det_geo['corner'], dtype=t.float32)

        self.min_translation = t.tensor(min_translation)

        self.probe_basis = t.tensor(probe_basis)
        self.detector_slice = copy(detector_slice)
        self.surface_normal = t.tensor(surface_normal)

        self.saturation = saturation
        self.units = units

        if mask is None:
            self.mask = mask
        else:
            self.mask = t.tensor(mask, dtype=t.bool)

        probe_guess = t.tensor(probe_guess, dtype=t.complex64)
        obj_guess = t.tensor(obj_guess, dtype=t.complex64)

        self.probe_norm = 1 * t.max(t.abs(probe_guess))        

        self.probe = t.nn.Parameter(probe_guess / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess)

        if background is None:
            if detector_slice is not None:
                background = 1e-6 * t.ones(
                    self.probe[self.detector_slice].shape,
                    dtype=t.float32)
            else:
                background = 1e-6 * t.ones(self.probe[0].shape,
                                           dtype=t.float32)

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

        self.translation_scale = translation_scale

        self.probe_fourier_support = probe_fourier_support
        self.fourier_times = fourier_times
        self.times = times
        self.time_dependence = t.nn.Parameter(time_dependence)
        self.frame_delays = frame_delays

        self.oversampling = oversampling

        self.simulate_probe_translation = simulate_probe_translation
        if simulate_probe_translation:
            Is = t.arange(self.probe.shape[-2], dtype=t.float32)
            Js = t.arange(self.probe.shape[-1], dtype=t.float32)
            Is, Js = t.meshgrid(Is/t.max(Is), Js/t.max(Js))
            self.I_phase = 2 * np.pi* Is
            self.J_phase = 2 * np.pi* Js
            
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
    def from_dataset(cls, dataset, zp_geometry, time_window, n_times, n_frames, randomize_ang=0, padding=0, translation_scale=1, saturation=None, propagation_distance=None, scattering_mode=None, oversampling=1, auto_center=False, opt_for_fft=False, loss='amplitude mse', units='um', simulate_probe_translation=False):

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

        obj_size, min_translation = tools.initializers.calc_object_setup(probe_shape, pix_translations, padding=200)

        if hasattr(dataset, 'background') and dataset.background is not None:
            background = t.sqrt(dataset.background)
        else:
            background = None

        # Finally, initialize the probe and  object using this information
        probe = tools.initializers.SHARP_style_probe(dataset, probe_shape, det_slice, propagation_distance=propagation_distance, oversampling=oversampling)

        probe = tools.propagators.far_field(probe)
        
        obj = t.exp(1j * randomize_ang * (t.rand(obj_size)-0.5))

        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset), 2)) - 0.5)
        # we define a set of weights which only has one index
        Ws = t.ones(len(dataset))

        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.bool)
        else:
            mask = None

        # What do we know about the zp?
        # delta_r, N, and beamstop ratio. It's probably best, though,
        # to just read diameter, beamstop_diameter and focal_length directly
        # because that is the most general even if the optic isn't truly
        # a zone plate.
        zp_distance = zp_geometry['focal_length']
        probe_size = probe_basis * t.as_tensor(probe_shape, dtype=t.float32)
        pinv_basis = t.tensor(np.linalg.pinv(probe_size).transpose()).to(t.float32)
        zone_plate_basis = pinv_basis * wavelength * zp_distance
        zone_plate_steps = t.sum(zone_plate_basis,axis=1)
        # This may very well mix up x & y and fail on non-square detectors
        probe_fourier_support = t.zeros(probe.shape, dtype=t.bool)
        xs, ys = np.mgrid[:probe.shape[-2], :probe.shape[-1]]
        xs = zone_plate_steps[1] * (xs - np.mean(xs))
        ys = zone_plate_steps[0] * (ys - np.mean(ys))
        Rs = np.sqrt(xs**2 + ys**2)

        distances = np.sqrt(zp_distance**2 + xs**2 + ys**2)
        times = (distances - t.min(distances)) / 2.99792e8

        # This sets the support of the probe and also restricts the
        # timing matrix so it only considers times that are actually
        # in the window defined by the probe fourier support
        probe_fourier_support[Rs < zp_geometry['diameter']/2] = 1
        times[Rs > zp_geometry['diameter']/2] = 0
        times[Rs > zp_geometry['diameter']/2] = t.max(times)        
        
        probe_fourier_support[Rs < zp_geometry['beamstop_diameter']/2] = 0
        times[Rs < zp_geometry['beamstop_diameter']/2] = t.max(times)
        times[Rs < zp_geometry['beamstop_diameter']/2] = t.min(times)

        fourier_times = times - t.min(times)
        probe = probe * probe_fourier_support

        # This is now the time axis for the probe's envelope
        times = t.linspace(0, time_window, n_times+1)
        time_dependence = t.ones(n_times, dtype=t.complex64)
        frame_delays = t.linspace(0, t.max(fourier_times) + time_window, n_frames+2)
        frame_delays -= time_window
        frame_delays = frame_delays[1:-1]
        
        return cls(wavelength, det_geo, probe_basis, probe, obj,
                   fourier_times, probe_fourier_support,
                   times, time_dependence, frame_delays,
                   detector_slice=det_slice,
                   surface_normal=surface_normal,
                   min_translation=min_translation,
                   translation_offsets=translation_offsets,
                   weights=Ws, mask=mask, background=background,
                   translation_scale=translation_scale,
                   saturation=saturation,
                   oversampling=oversampling,
                   loss=loss, units=units,
                   simulate_probe_translation=simulate_probe_translation)


    def interaction(self, index, translations, *args):
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

        probes = self.get_probes(space='real')        
        Ws = self.weights[index]

        # This might not work well
        prs = Ws[...,None,None,None] * probes
        #prs = t.sum(Ws[..., None, None, None] * probes, axis=-3)
        #print(prs.shape)

        if self.simulate_probe_translation:
            det_pix_trans = tools.interactions.translations_to_pixel(
                    self.detector_geometry['basis'],
                    translations,
                    surface_normal=self.surface_normal)
            
            probe_masks = t.exp(1j* (det_pix_trans[:,0,None,None] *
                                     self.I_phase[None,...] +
                                     det_pix_trans[:,1,None,None] *
                                     self.J_phase[None,...]))
            prs = prs * probe_masks[...,None,:,:]


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
            detector_slice=self.detector_slice,
            measurement=tools.measurements.incoherent_sum,
            saturation=self.saturation,
            oversampling=self.oversampling)


    # Note: No "loss" function is defined here, because it is added
    # dynamically during object creation in __init__

    def to(self, *args, **kwargs):
        super(TimeResolvedPtychoCalibration, self).to(*args, **kwargs)
        self.wavelength = self.wavelength.to(*args, **kwargs)
        # move the detector geometry too
        det_geo = self.detector_geometry
        if 'distance' in det_geo:
            det_geo['distance'] = det_geo['distance'].to(*args, **kwargs)
        if 'basis' in det_geo:
            det_geo['basis'] = det_geo['basis'].to(*args, **kwargs)
        if 'corner' in det_geo:
            det_geo['corner'] = det_geo['corner'].to(*args, **kwargs)

        if self.mask is not None:
            self.mask = self.mask.to(*args, **kwargs)

        if self.simulate_probe_translation:
            self.I_phase = self.I_phase.to(*args, **kwargs)
            self.J_phase = self.J_phase.to(*args, **kwargs)

        self.min_translation = self.min_translation.to(*args, **kwargs)
        self.probe_basis = self.probe_basis.to(*args, **kwargs)
        self.probe_norm = self.probe_norm.to(*args, **kwargs)
        self.probe_fourier_support = self.probe_fourier_support.to(*args,
                                                                   **kwargs)
        self.fourier_times = self.fourier_times.to(*args, **kwargs)
        self.times = self.times.to(*args, **kwargs)
        self.surface_normal = self.surface_normal.to(*args, **kwargs)


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


        detector_geometry = self.detector_geometry
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
            detector_geometry=detector_geometry,
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


    def get_probes(self, space='real'):
        # This is the part where I need to create the probe modes using the
        # time-dependent stuff

        # This restricts the basis probes to stay within the probe support
        optic_mask = self.probe * self.probe_fourier_support

        probes = t.zeros([len(self.frame_delays)] + list(self.probe.shape),
                         dtype=self.probe.dtype, device=self.probe.device)
        for i, delay in enumerate(self.frame_delays):
            indices = t.bucketize(self.fourier_times, self.times + delay)
            clamped_indices = t.clamp(indices-1, max=len(self.time_dependence)-1)
            illumination = t.take(self.time_dependence, clamped_indices)
            illumination[indices==0] = 0
            illumination[indices==len(self.time_dependence)+1] = 0
            probes[i] = illumination * optic_mask

        if space.lower()=='real':
            return tools.propagators.inverse_far_field(probes)
        elif space.lower()=='fourier' or space.lower()=='reciprocal':
            return probes

        
    plot_list = [
        ('Probe Amplitudes (scroll to view modes)',
         lambda self, fig: p.plot_amplitude(self.get_probes(space='real'), fig=fig, basis=self.probe_basis, units=self.units)),
        ('Probe Phases (scroll to view modes)',
         lambda self, fig: p.plot_phase(self.get_probes(space='real'), fig=fig, basis=self.probe_basis, units=self.units)),
        ('Fourier Probe Amplitudes (scroll to view modes)',
         lambda self, fig: p.plot_amplitude(self.get_probes(space='fourier'), fig=fig, basis=self.probe_basis, units=self.units)),
        ('Fourier Probe Phases (scroll to view modes)',
         lambda self, fig: p.plot_phase(self.get_probes(space='fourier'), fig=fig, basis=self.probe_basis, units=self.units)),
        ('Optic Amplitude',
         lambda self, fig: p.plot_amplitude(self.probe, fig=fig, basis=self.probe_basis, units=self.units)),
        ('Optic Phase',
         lambda self, fig: p.plot_phase(self.probe, fig=fig, basis=self.probe_basis, units=self.units)),
        ('Average Density Matrix Amplitudes',
         lambda self, fig: p.plot_amplitude(np.nanmean(np.abs(self.get_rhos()), axis=0), fig=fig),
         lambda self: len(self.weights.shape) >= 2),
        ('% Power in Top Mode (only accurate after tidy_probes)',
         lambda self, fig, dataset: p.plot_nanomap(self.corrected_translations(dataset), analysis.calc_top_mode_fraction(self.get_rhos()), fig=fig, units=self.units),
         lambda self: len(self.weights.shape) >= 2),
        ('Object Amplitude',
         lambda self, fig: p.plot_amplitude(self.obj, fig=fig, basis=self.probe_basis, units=self.units)),
        ('Object Phase',
         lambda self, fig: p.plot_phase(self.obj, fig=fig, basis=self.probe_basis, units=self.units)),
        ('Corrected Translations',
         lambda self, fig, dataset: p.plot_translations(self.corrected_translations(dataset), fig=fig, units=self.units)),
        ('Background',
         lambda self, fig: plt.figure(fig.number) and plt.imshow(self.background.detach().cpu().numpy()**2)),
        ('Time Structure',
         lambda self, fig: (plt.figure(fig.number) and (plt.clf() or True) and plt.plot(self.time_dependence.real.detach().cpu().numpy()) and plt.plot(self.time_dependence.imag.detach().cpu().numpy())))
    ]

#    def plot_errors(self, dataset):
        
    
    
    def save_results(self, dataset):
        basis = self.probe_basis.detach().cpu().numpy()
        translations = self.corrected_translations(dataset).detach().cpu().numpy()
        optic = self.probe.detach().cpu().numpy()
        optic = optic * self.probe_norm.detach().cpu().numpy()
        time_dependence = self.time_dependence.detach().cpu().numpy()
        times = self.times.detach().cpu().numpy()
        fourier_times = self.fourier_times.detach().cpu().numpy()
        probes = self.get_probes().detach().cpu().numpy()
        obj = self.obj.detach().cpu().numpy()
        background = self.background.detach().cpu().numpy()**2
        weights = self.weights.detach().cpu().numpy()

        return {'basis': basis, 'translation': translations,
                'probes': probes, 'optic': optic,
                'times': times, 'time_dependence': time_dependence,
                'fourier_times': fourier_times,
                'obj': obj,
                'background': background,
                'weights': weights,
                }
