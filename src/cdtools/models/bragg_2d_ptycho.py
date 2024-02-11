import torch as t
from cdtools.models import CDIModel
from cdtools.datasets import Ptycho2DDataset
from cdtools import tools
from cdtools.tools import plotting as p
from cdtools.tools.propagators import generate_generalized_angular_spectrum_propagator as ggasp
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from copy import copy

__all__ = ['Bragg2DPtycho']

#
# Key ideas:
# 1) To a first approximation, do the reconstruction on a parallelogram
#    shaped grid on the sample which is conjugate to the detector coordinates
# 2) Simulate a probe in those same coordinates, but propagate it back and
#    forth (along with the translations), using the angular spectrum method
# 3) Apply a correction to the simulated data to account for the tilt of the
#    sample with respect to the detector
# 4) Include a correction for the thickness of the sample
#

#
# How to do this properly?
# First thing to note is that the two corrections (probe propagation before
# interaction and high-NA correction for the final diffraction measurement)
# should be able to be turned on separately, since they show up in different
# situations. In fact, I would like to focus on the first aspect initially
# since I think that's the dominant issue we will contend with at CSX.
#

#
# It also should be possible to choose an "auto" setting for the two
# corrections, since the geometry information given should be enough to
# decide if the correction is needed.
# Probably the automatic check will have to be very conservative for the
# probe propagation side since the model has no information about the
# expected numerical aperture of the probe.
#

#
# I'm worried that the propagation doesn't happen along the correct
# direction, if a phase ramp is expected to be baked in to the
# retrieved focal spot. Unclear if this is the case though.
#
# Retrieved focal spot should have the implicit phase ramp subtracted
# (that is, it should be the focal spot along the sample plane, but with
# the e^ikz dependence removed). Therefore, the original e^ikz dependence
# should be easy to re-add jusy by using the propagate_along feature
# in ggasp. So I believe this should not be a problem
#

class Bragg2DPtycho(CDIModel):

    # Needed to do the real/complex split
    #@property
    #def obj(self):
    #    return t.complex(self.obj_real, self.obj_imag)

    #@property
    #def probe(self):
    #    return t.complex(self.probe_real, self.probe_imag)


    def __init__(self, wavelength, detector_geometry,
                 probe_basis, probe_guess, obj_guess,
                 detector_slice=None,
                 min_translation=t.tensor([0, 0], dtype=t.float32),
                 median_propagation=t.tensor(0, dtype=t.float32),
                 background=None, translation_offsets=None, mask=None,
                 weights=None, translation_scale=1, saturation=None,
                 probe_support=None, oversampling=1,
                 propagate_probe=True, correct_tilt=True, lens=False, units='um'):

        # We need the detector geometry
        # We need the probe basis (but in this case, we don't need the surface
        # normal because it comes implied by the probe basis
        # we do need the detector slice I suppose
        # The min translation is also needed
        # The median propagation should be needed as well
        # translation_offsets can stay 2D for now
        # propagate_probe and correct_tilt are important!

        super(Bragg2DPtycho, self).__init__()
        self.units = units
        self.wavelength = t.tensor(wavelength)
        self.detector_geometry = copy(detector_geometry)
        det_geo = self.detector_geometry
        if hasattr(det_geo, 'distance'):
            det_geo['distance'] = t.tensor(det_geo['distance'])
        if hasattr(det_geo, 'basis'):
            det_geo['basis'] = t.tensor(det_geo['basis'])
        if hasattr(det_geo, 'corner'):
            det_geo['corner'] = t.tensor(det_geo['corner'])

        self.min_translation = t.tensor(min_translation)
        self.median_propagation = t.tensor(median_propagation)
        
        self.probe_basis = t.tensor(probe_basis)
        self.detector_slice = copy(detector_slice)

        # calculate the surface normal from the probe basis
        surface_normal =  np.cross(np.array(probe_basis)[:,1],
                           np.array(probe_basis)[:,0])
        surface_normal /= np.linalg.norm(surface_normal)
        self.surface_normal = t.tensor(surface_normal)
        
        self.saturation = saturation
        
        if mask is None:
            self.mask = mask
        else:
            self.mask = t.tensor(mask, dtype=t.bool)

        probe_guess = t.tensor(probe_guess, dtype=t.complex64)
        obj_guess = t.tensor(obj_guess, dtype=t.complex64)

        # We rescale the probe here so it learns at the same rate as the
        # object
        if probe_guess.dim() > 2:
            self.probe_norm = 1 * t.max(t.abs(probe_guess[0]))
        else:
            self.probe_norm = 1 * t.max(t.abs(probe_guess))        

        # Not strictly necessary but otherwise it will return
        # a probe with the stuff outside of the support unchanged after
        # optimization
        if probe_support is not None:
            self.probe_support = t.tensor(probe_support, dtype=t.bool)
            probe_guess = probe_guess * probe_support
            # This seems dumb, but otherwise it winds up with a mixture
            # of negative and positive zeros and it's super annoying when
            # you look at the phase map
            probe_guess[probe_guess == 0] = 0
        else:
            self.probe_support = t.ones(probe_guess[0].shape, dtype=t.bool)

        #self.probe_real = t.nn.Parameter(probe_guess.real / self.probe_norm)
        #self.probe_imag = t.nn.Parameter(probe_guess.imag / self.probe_norm)

        #self.obj_real = t.nn.Parameter(obj_guess.real )
        #self.obj_imag = t.nn.Parameter(obj_guess.imag)

        self.probe = t.nn.Parameter(probe_guess / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess)

        if background is None:
            if detector_slice is not None:
                background = 1e-6 * t.ones(
                    self.probe[0][self.detector_slice].shape,
                    dtype=t.float32)
            else:
                background = 1e-6 * t.ones(self.probe[0].shape,
                                           dtype=t.float32)

        self.background = t.nn.Parameter(background)

        if weights is None:
            self.weights = None
        else:
            # No incoherent + unstable here yet
            self.weights = t.nn.Parameter(t.tensor(weights,
                                                   dtype=t.float32))

        if translation_offsets is None:
            self.translation_offsets = None
        else:
            t_o = t.tensor(translation_offsets, dtype=t.float32)
            t_o = t_o / translation_scale
            self.translation_offsets = t.nn.Parameter(t_o)

        self.translation_scale = translation_scale

        self.oversampling = oversampling

        self.propagate_probe = propagate_probe
        self.correct_tilt = correct_tilt
        if correct_tilt:
            # recall that here we always want the shape of the detector
            # before it's cut down by the detector slice to match the
            # physical detector region
            probe_shape = self.probe[0].shape
            
            self.k_map, self.intensity_map = \
                tools.propagators.generate_high_NA_k_intensity_map(
                    self.probe_basis,
                    self.detector_geometry['basis'] / oversampling,
                    probe_shape,
                    self.detector_geometry['distance'],
                    self.wavelength,dtype=t.float32,
                    lens=lens)
        else:
            self.k_map = None
            self.intensity_map = None

        self.prop_dir = t.tensor([0, 0, 1], dtype=t.float32)

        # This propagator should be able to be multiplied by the propagation
        # distance each time to get a propagator
        self.universal_propagator = t.angle(ggasp(
            self.background.shape,#background in case of a fourier cropped probe
            self.probe_basis, self.wavelength,
            t.tensor([0, 0, self.wavelength/(2*np.pi)], dtype=t.float32),
            propagation_vector=self.prop_dir,
            dtype=t.complex64,
            propagate_along_offset=True))


    @classmethod
    def from_dataset(
            cls,
            dataset,
            probe_size=None,
            randomize_ang=0,
            padding=0,
            n_modes=1,
            translation_scale = 1,
            saturation=None,
            probe_support_radius=None,
            propagation_distance=None,
            scattering_mode=None,
            oversampling=1,
            auto_center=False,
            probe_fourier_crop=None,
            propagate_probe=True,
            correct_tilt=True,
            lens=False):
        
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
            
        # Then, generate the exit wave geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        ew_basis, ew_shape, det_slice =  ewg(det_basis,
                                                   det_shape,
                                                   wavelength,
                                                   distance,
                                                   center=center,
                                                   padding=padding,
                                                   oversampling=oversampling)
        # now we grab the sample surface normal
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

        # and we use that to generate the probe basis

        ew_normal = np.cross(np.array(ew_basis)[:,1],
                           np.array(ew_basis)[:,0])
        ew_normal /= np.linalg.norm(ew_normal)

        # This is a bit of an odd way to do a projection but I think it's
        # the most compact and reliable. We set up two matrix-vector equations
        # to enforce the two conditions (on the plane normal to the surface
        # normal and in the ew_normal direction from the original point

        mat = np.vstack([np.eye(3) - np.outer(ew_normal,ew_normal),
                         surface_normal])

        # We know that this matrix multiplied by the final result must
        # equal the input vector with a trailing 0, so we can do the
        # projection with a pseudoinverse and removing the last column

        projector = np.linalg.pinv(mat)[:, :3]
        probe_basis = t.Tensor(np.dot(projector, ew_basis))

        # Now we need a much better way to handle the translations here
        # than translations_to_pixel
        
        # Next generate the object geometry from the probe geometry and
        # the translations
        p2s = tools.interactions.project_translations_to_sample
        pix_translations, propagations = p2s(probe_basis, translations)


        obj_size, min_translation = tools.initializers.calc_object_setup(ew_shape, pix_translations, padding=200)

        median_propagation = t.median(propagations)

        # Finally, initialize the probe and  object using this information
        # Because the grid we defined on the sample is projected from the
        # detector conjugate space, we can pretend that the grid is just in
        # that space and use the standard initializations anyway
        
        if probe_size is None:
            probe = tools.initializers.SHARP_style_probe(dataset, ew_shape, det_slice, propagation_distance=propagation_distance, oversampling=oversampling)
        else:
            probe = tools.initializers.gaussian_probe(dataset, ew_basis, ew_shape, probe_size, propagation_distance=propagation_distance)

            
        if hasattr(dataset, 'background') and dataset.background is not None:
            background = t.sqrt(dataset.background)
        elif det_slice is not None:
            background = 1e-6 * t.ones(
                probe[det_slice].shape,
                dtype=t.float32)
        else:
            background = 1e-6 * t.ones(probe.shape[-2:],
                                       dtype=t.float32)

        if probe_fourier_crop is not None:
            probe = tools.propagators.far_field(probe)
            probe = probe[...,
                          probe_fourier_crop[0]:-probe_fourier_crop[0],
                          probe_fourier_crop[1]:-probe_fourier_crop[1]]
            probe = tools.propagators.inverse_far_field(probe)

        # Now we initialize all the subdominant probe modes
        probe_max = t.max(t.abs(probe))
        probe_stack = [0.01 * probe_max * t.rand(probe.shape,dtype=probe.dtype) for i in range(n_modes - 1)]
        probe = t.stack([probe,] + probe_stack)


        obj = t.exp(1j*(randomize_ang * (t.rand(obj_size)-0.5)))
                              
        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset),2)) - 0.5)

        weights = t.ones(len(dataset))
        
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

        # Here we need to implement a simple condition to choose whether
        # to propagate the probe or not
        if not( propagate_probe is True or propagate_probe is False):
            raise NotImplementedError('No auto option implemented yet')

        if not(correct_tilt is True or correct_tilt is False):
            raise NotImplementedError('No auto option implemented yet')

            
        return cls(wavelength, det_geo, probe_basis, probe, obj,
                   detector_slice=det_slice,
                   min_translation=min_translation,
                   median_propagation =median_propagation,
                   translation_offsets = translation_offsets,
                   weights=weights, mask=mask, background=background,
                   translation_scale=translation_scale,
                   saturation=saturation,
                   probe_support=probe_support,
                   oversampling=oversampling,
                   propagate_probe=propagate_probe,
                   correct_tilt=correct_tilt,
                   lens=lens)
                   
    
    def interaction(self, index, translations):
        pix_trans, props = tools.interactions.project_translations_to_sample(
            self.probe_basis, translations)

        
        pix_trans -= self.min_translation
        props -= self.median_propagation
        
        if self.translation_offsets is not None:
            pix_trans += self.translation_scale * self.translation_offsets[index]

        Ws = self.weights[index]
        prs = Ws[...,None,None,None] * self.probe * self.probe_support[...,:,:]
        # Now we need to propagate each of the probes


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

        # we do the upscaling before the propagation because the propagator
        # is currently calculated based on the object pixel pitch, not the
        # probe one

        if self.propagate_probe:
            for j in range(prs.shape[0]):
                # I believe this -1 sign is in error, but I need a dataset with
                # well understood geometry to figure it out
                propagator = t.exp(
                    1j*(props[j]*(2*np.pi)/self.wavelength)
                    * self.universal_propagator)
                prs[j] = tools.propagators.near_field(prs[j], propagator)

        exit_waves = self.probe_norm * tools.interactions.ptycho_2D_sinc(
            prs, self.obj,pix_trans,
            shift_probe=True, multiple_modes=True)

        return exit_waves

        
    def forward_propagator(self, wavefields):
        if self.correct_tilt:
            return tools.propagators.high_NA_far_field(
                wavefields,self.k_map,intensity_map=self.intensity_map)
        else:
            return tools.propagators.far_field(wavefields)


    def backward_propagator(self, wavefields):
        if self.correct_tilt:
            assert NotImplementedError('Backward propagator not defined with tilt correction')
        else:
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
        super(Bragg2DPtycho, self).to(*args, **kwargs)
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

        if self.k_map is not None:
            self.k_map = self.k_map.to(*args,**kwargs)
        if self.intensity_map is not None:
            self.intensity_map = self.intensity_map.to(*args,**kwargs)
            
        self.min_translation = self.min_translation.to(*args,**kwargs)
        self.probe_basis = self.probe_basis.to(*args,**kwargs)
        self.probe_norm = self.probe_norm.to(*args,**kwargs)
        self.probe_support = self.probe_support.to(*args,**kwargs)
        self.surface_normal = self.surface_normal.to(*args, **kwargs)
        self.prop_dir = self.prop_dir.to(*args, **kwargs)
        self.universal_propagator = self.universal_propagator.to(*args,**kwargs)


        
    def sim_to_dataset(self, args_list):
        # In the future, potentially add more control
        # over what metadata is saved (names, etc.)
        
        # First, I need to gather all the relevant data
        # that needs to be added to the dataset
        entry_info = {'program_name': 'cdtools',
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
        translations = dataset.translations.to(dtype=self.probe.real.dtype,
                                               device=self.probe.device)
        t_offset = tools.interactions.pixel_to_translations(self.probe_basis,self.translation_offsets*self.translation_scale,surface_normal=self.surface_normal)
        return translations + t_offset


    plot_list = [
        ('Basis Probe Fourier Space Amplitudes',
         lambda self, fig: p.plot_amplitude(tools.propagators.inverse_far_field(self.probe), fig=fig)),
        ('Basis Probe Fourier Space Phases',
         lambda self, fig: p.plot_phase(tools.propagators.inverse_far_field(self.probe), fig=fig)),
        ('Basis Probe Real Space Amplitudes',
         lambda self, fig: p.plot_amplitude(self.probe, fig=fig, basis=self.probe_basis, units=self.units)),
        ('Basis Probe Real Space Phases',
         lambda self, fig: p.plot_phase(self.probe, fig=fig, basis=self.probe_basis, units=self.units)),
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
        # This will save out everything needed to recreate the object
        # in the same state, but it's not the best formatted. For example,
        # "background" stores the square root of the background, etc.
        base_results = super().save_results()

        # We also save out the main results in a more readable format
        basis = self.probe_basis.detach().cpu().numpy()
        translations=self.corrected_translations(dataset).detach().cpu().numpy()
        original_translations = dataset.translations.detach().cpu().numpy()
        probe = self.probe.detach().cpu().numpy()
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = self.obj.detach().cpu().numpy()
        background = self.background.detach().cpu().numpy()**2
        weights = self.weights.detach().cpu().numpy()
        oversampling = self.oversampling
        wavelength = self.wavelength.cpu().numpy()

        results = {
            'basis': basis,
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
