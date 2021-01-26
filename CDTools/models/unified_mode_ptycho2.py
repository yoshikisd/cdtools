from __future__ import division, print_function, absolute_import

import torch as t
from CDTools.models import CDIModel
from CDTools.datasets import Ptycho2DDataset
from CDTools import tools
from CDTools.tools import cmath
from CDTools.tools import analysis
from CDTools.tools import plotting as p
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from copy import copy

__all__ = ['UnifiedModePtycho2']

class UnifiedModePtycho2(CDIModel):

    def __init__(self, wavelength, detector_geometry,
                 probe_basis,
                 probe_guess, obj_guess, Ws_guess,
                 detector_slice=None,
                 surface_normal=np.array([0.,0.,1.]),
                 min_translation = t.Tensor([0,0]),
                 background = None, translation_offsets=None, mask=None,
                 translation_scale = 1, saturation=None,
                 probe_support = None, obj_support=None, oversampling=1):
        
        super(UnifiedModePtycho2,self).__init__()
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
        
        self.obj = t.nn.Parameter(obj_guess.to(t.float32))
        
        if background is None:
            if detector_slice is not None:
                background = 1e-6 * t.ones(self.probe[0][self.detector_slice].shape[:-1])
            else:
                background = 1e-6 * t.ones(self.probe[0].shape[:-1])

                
        self.background = t.nn.Parameter(t.Tensor(background).to(t.float32))

        if type(Ws_guess) == type(t.zeros(1)):
            self.Ws = t.nn.Parameter(Ws_guess.to(t.float32))        
        else:
            self.Ws = t.nn.Parameter(cmath.complex_to_torch(Ws_guess).to(t.float32))
        
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

        self.oversampling = oversampling

        
    @classmethod
    def from_dataset(cls, dataset, probe_size=None, randomize_ang=0, padding=0, n_modes=1, translation_scale = 1, saturation=None, probe_support_radius=None, propagation_distance=None, restrict_obj=-1, scattering_mode=None, oversampling=1, auto_center=True, opt_for_fft=False, dm_rank=0):
        
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
                                                   opt_for_fft=opt_for_fft,
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
        if probe_size is None:
            probe = tools.initializers.SHARP_style_probe(dataset, probe_shape, det_slice, propagation_distance=propagation_distance, oversampling=oversampling)
        else:
            probe = tools.initializers.gaussian_probe(dataset, probe_basis, probe_shape, probe_size, propagation_distance=propagation_distance)


        # Now we initialize all the subdominant probe modes
        probe_max = t.max(cmath.cabs(probe))
        probe_stack = [0.01 * probe_max * t.rand(probe.shape,dtype=probe.dtype) for i in range(n_modes - 1)]
        probe = t.stack([probe,] + probe_stack)
        #probe = t.stack([tools.propagators.far_field(probe),] + probe_stack)

        obj = tools.cmath.expi(randomize_ang * (t.rand(obj_size)-0.5))
                              
        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset),2)) - 0.5)

        # dm_rank defines the rank of the shot-by-shot density matrices
        if dm_rank > n_modes:
            raise KeyError('Density matrix rank cannot be greater than the number of modes')
        elif dm_rank != 0:
            if dm_rank == -1:
                dm_rank = n_modes
            Ws = t.zeros(len(dataset),dm_rank,n_modes,2)
            Ws[:,0,0,0] = 1
            for i in range(1,dm_rank):
                Ws[:,i,i,0] = 1/np.sqrt(n_modes)
        else:
            # dm_rank=0 is a special case defining a purely stable, incoherent
            # mode mixing model. This is passed on by defining a set of weights
            # which only has one index
            Ws = t.ones(len(dataset))
                
        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.bool)
        else:
            mask = None

        if probe_support_radius is not None:
            probe_support = t.zeros_like(probe[0].to(dtype=t.float32))
            p_cent = np.array(probe.shape[1:3]).astype(int) // 2
            psr = int(probe_support_radius)
            probe_support[p_cent[0]-psr:p_cent[0]+psr,
                          p_cent[1]-psr:p_cent[1]+psr] = 1
            probe = probe * probe_support[None,:,:]
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

        return cls(wavelength, det_geo, probe_basis, probe, obj, Ws,
                   detector_slice=det_slice,
                   surface_normal=surface_normal,
                   min_translation=min_translation,
                   translation_offsets = translation_offsets,
                   mask=mask, background=background,
                   translation_scale=translation_scale,
                   saturation=saturation,
                   probe_support=probe_support,
                   obj_support=obj_support,
                   oversampling=oversampling)
                   
    
    def interaction(self, index, translations):
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations,
                                                             surface_normal=self.surface_normal)
        pix_trans -= self.min_translation

        if self.translation_offsets is not None:
            pix_trans += self.translation_scale * self.translation_offsets[index]

        Ws = self.Ws[index]
        
        if type(index) == type(0):
            index = [index]
            Ws = [Ws]
            pix_trans = [pix_trans]
            single_frame = True
        else:
            single_frame = False

        probes = []
        all_exit_waves = []

        # This is the case if a purely stable, incoherent model is defined.
        if len(Ws[0].shape) == 0:
            # What we do here is generate an identity matrix, and multiply
            # that identity matrix by the per-frame weight.
            Ws = [W * t.stack([t.eye(self.probe.shape[0]),
                               t.zeros([self.probe.shape[0]]*2)],dim=-1).to(
                                   dtype=W.dtype, device=W.device)
                  for W in Ws]
            
        
        # Outer iteration is the mode index iteration
        for i in range(Ws[0].shape[0]):
            # Now we need to separately treat each mode
            exit_waves = []
            for W, pix_tran in zip(Ws, pix_trans):
                # from storing the probe in Fourier space
                pr = [cmath.cmult(W[i,j,:], self.probe[j] * self.probe_support)
                      for j in range(self.probe.shape[0])]
                
                pr = t.sum(t.stack(pr), axis=0)
                
                exit_waves.append(self.probe_norm *
                                  tools.interactions.ptycho_2D_sinc(
                                      pr, self.obj_support * self.obj,
                                      pix_tran, shift_probe=True))
                
            exit_waves = t.stack(exit_waves)

            if single_frame:
                exit_waves = exit_waves[0]
            # Multiply again by probe support to suppress the fringes from the
            # sinc-interpolated shift
            exit_waves = exit_waves * self.probe_support[...,:,:]

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
        #return tools.losses.poisson_nll(real_data, sim_data, mask=mask)

    
    def to(self, *args, **kwargs):
        super(UnifiedModePtycho2, self).to(*args, **kwargs)
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
        return Ptycho2DDataset(translations, data,
                                 entry_info = entry_info,
                                 sample_info = sample_info,
                                 wavelength=wavelength,
                                 detector_geometry=detector_geometry,
                                 mask=mask)

    def get_rhos(self):
        # If this is not a purely stable model
        if len(self.Ws.shape) >= 2:
            Ws = cmath.torch_to_complex(self.Ws.detach().cpu())
            rhos_out = np.matmul(np.swapaxes(Ws,1,2), Ws.conj())
            return rhos_out
        else:
            return np.array([np.eye(self.probe.shape[0])]*self.Ws.shape[0],
                            dtype=np.complex64)
        
    def tidy_probes(self, normalization=1):
        """Tidies up the probes
        
        What we want to do here is use all the information on all the probes
        to calculate a natural basis for the experiment, and update all the
        density matrices to operate in that updated basis
        
        """

        # Must also implement a version that works appropriately with
        # a purely incoherent model
        
        #
        # Note to future: We could probably do this more cleanly with an
        # SVD directly on the Ws matrix, instead of an eigendecomposition
        # of the rho matrix. This could avoid potential stability issues
        # due to the existence of zero eigenvalues in the full rho matrix
        # when dm_rank < n_modes
        # 
        
        rhos = self.get_rhos()
        overall_rho = np.mean(rhos,axis=0)
        probe = cmath.torch_to_complex(self.probe.detach().cpu())
        ortho_probes, A = analysis.orthogonalize_probes(probe,
                                        density_matrix=overall_rho,
                                        keep_transform=True,
                                        normalize=True)
        Aconj = A.conj()
        Atrans = np.transpose(A)
        new_rhos = np.matmul(Atrans,np.matmul(rhos,Aconj))

        new_rhos /= normalization
        ortho_probes *= np.sqrt(normalization)

        dm_rank = self.Ws.shape[1]
        
        new_Ws = []
        for rho in new_rhos:
            # These are returned from smalles to largest - we want to keep
            # the largest ones
            w,v = np.linalg.eigh(rho)
            w = w[::-1][:dm_rank]
            v = v[:,::-1][:,:dm_rank]
            new_Ws.append(np.dot(np.diag(np.sqrt(w)),v.transpose()))

        new_Ws = np.array(new_Ws)

        self.Ws.data = cmath.complex_to_torch(new_Ws).to(
            dtype=self.Ws.dtype,device=self.Ws.device)
        
        self.probe.data = cmath.complex_to_torch(ortho_probes).to(
            device=self.probe.device,dtype=self.probe.dtype)
        
    
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
        ('Average Density Matrix Amplitudes',
         lambda self, fig: p.plot_amplitude(np.mean(np.abs(self.get_rhos()),axis=0), fig=fig),
         lambda self: len(self.Ws.shape) >=2),
        ('% Power in Top Mode (only accurate after tidy_probes)',
         lambda self, fig, dataset: p.plot_nanomap(self.corrected_translations(dataset), analysis.calc_top_mode_fraction(self.get_rhos()), fig=fig),
         lambda self: len(self.Ws.shape) >=2),
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
        Ws = cmath.torch_to_complex(self.Ws.detach().cpu())
        
        return {'basis':basis, 'translation':translations,
                'probe':probe,'obj':obj,
                'background':background,
                'Ws':Ws}