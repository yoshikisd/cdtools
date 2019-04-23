from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from matplotlib import pyplot as plt
import pickle
import argparse
from scipy import fftpack

from CDTools.tools import cmath, plotting
from CDTools.tools import image_processing as ip


def standardize(probe, obj, obj_slice=None):
    # First, we normalize the probe intensity to a fixed value.
    # Should this be the maximum or the integrated intensity? I think
    # probably the integrated intensity. We set the average per-[ixel
    # intensity in the probe to be one
    normalization = np.sqrt(np.sum(np.abs(probe)**2) / len(probe.ravel()))
    probe = cmath.complex_to_torch(probe / normalization)
    obj = cmath.complex_to_torch(obj * normalization)


    # Default slice of the object to use for alignment, etc.
    if obj_slice is None:
        obj_slice = np.s_[(obj.shape[0]//8)*3:(obj.shape[0]//8)*5,
                          (obj.shape[1]//8)*3:(obj.shape[1]//8)*5]


    
    # Now we get rid of the probe's phase ramp
    # Currently disabled
    #center_freq = ip.centroid_sq(cmath.fftshift(t.fft(probe,2)),comp=True)
    #center_freq -= (t.tensor(probe.shape[:-1]) // 2).to(t.float32)
    #center_freq /= t.tensor(probe.shape[:-1]).to(t.float32)


    
    #Is, Js = np.mgrid[:probe.shape[0],:probe.shape[1]]
    #probe_phase_ramp = cmath.expi(2*np.pi *
    #                              (center_freq[0] * t.tensor(Is).to(t.float32) +
    #                               center_freq[1] * t.tensor(Js).to(t.float32)))
    #probe = cmath.cmult(probe, cmath.cconj(probe_phase_ramp))
    #Is, Js = np.mgrid[:obj.shape[0],:obj.shape[1]]
    #obj_phase_ramp = cmath.expi(2*np.pi *
    #                            (center_freq[0] * t.tensor(Is).to(t.float32) +
    #                             center_freq[1] * t.tensor(Js).to(t.float32)))
    #obj = cmath.cmult(obj, obj_phase_ramp)

    
    # Then, we set them to consistent absolute phases
    probe_angle = cmath.cphase(t.sum(probe,dim=(0,1)))
    obj_angle = cmath.cphase(t.sum(obj[obj_slice],dim=(0,1)))

    probe = cmath.cmult(probe, cmath.expi(-probe_angle))
    obj = cmath.cmult(obj, cmath.expi(-obj_angle))

    return probe, obj
    

def synthesize_reconstructions(probes, objects, use_probe=False, obj_slice=None):

    if obj_slice is None:
        obj_slice = np.s_[(objects[0].shape[0]//8)*3:(objects[0].shape[0]//8)*5,
                          (objects[0].shape[1]//8)*3:(objects[0].shape[1]//8)*5]
    
    
    synth_probe, synth_obj = standardize(probes[0], objects[0])
    obj_stack = [cmath.torch_to_complex(synth_obj)]
    for i, (probe, obj) in enumerate(zip(probes[1:],objects[1:])):
        probe, obj = standardize(probe, obj)
        probe = probe[0]
        print(i)
        #plt.imshow(np.angle(cmath.torch_to_complex(obj[obj_slice])))
        #plt.show()

        if use_probe:
            shift = ip.find_shift(synth_probe,probe, resolution=50)
        else:
            shift = ip.find_shift(synth_obj[obj_slice],obj[obj_slice], resolution=50)
        

        obj = ip.sinc_subpixel_shift(obj,np.array(shift))
        probe = ip.sinc_subpixel_shift(probe,tuple(shift))
        #obj = t.roll(obj,tuple(int(s) for s in shift),dims=(0,1)) 
        #probe = t.roll(probe,tuple(int(s) for s in shift),dims=(0,1))

        synth_probe += probe
        synth_obj += obj
        obj_stack.append(cmath.torch_to_complex(obj))



    # If there only was one image
    try:
        i
    except:
        i = -1

    synth_probe = cmath.torch_to_complex(synth_probe)
    synth_obj = cmath.torch_to_complex(synth_obj)
    return synth_probe/(i+2), synth_obj/(i+2), obj_stack



def calc_prtf(synth_obj, objects, basis, obj_slice=None):
    if obj_slice is None:
        obj_slice = np.s_[(objects[0].shape[0]//8)*3:(objects[0].shape[0]//8)*5,
                          (objects[0].shape[1]//8)*3:(objects[0].shape[1]//8)*5]

    synth_obj = cmath.complex_to_torch(synth_obj[obj_slice])
    
    synth_fft = cmath.cabssq(cmath.fftshift(t.fft(synth_obj,2))).numpy()

    prtfs = []
    for obj in objects:
        obj = cmath.complex_to_torch(obj[obj_slice])
        single_fft = cmath.cabssq(cmath.fftshift(t.fft(obj,2))).numpy()

        
        di = np.linalg.norm(basis[:,0]) 
        dj = np.linalg.norm(basis[:,1])

        i_freqs = fftpack.fftshift(fftpack.fftfreq(single_fft.shape[0],d=di))
        j_freqs = fftpack.fftshift(fftpack.fftfreq(single_fft.shape[1],d=dj))

        Js,Is = np.meshgrid(j_freqs,i_freqs)
        Is = Is - np.mean(Is)
        Js = Js - np.mean(Js)
        Rs = np.sqrt(Is**2+Js**2)

        single_ints, bins = np.histogram(Rs,bins=100,weights=single_fft)
        synth_ints, bins = np.histogram(Rs,bins=100,weights=synth_fft)

        prtfs.append(synth_ints/single_ints)

    return bins[:-1], np.mean(prtfs,axis=0)
        
        

def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('file', help='The reconstruction file to calculate metrics for')
    parser.add_argument('--use-probe', '-up', action='store_true', help='Use the probe instead of the object to align the reconstructions')
    return parser


if __name__ == '__main__':

    args = make_argparser().parse_args()

    with open(args.file, 'rb') as f:
        dataset = pickle.load(f)

    synth_probe, synth_obj, aligned_objs = synthesize_reconstructions(
        dataset['probe'], dataset['obj'], args.use_probe)

    freqs, prtf = calc_prtf(synth_obj, aligned_objs, dataset['basis'])

    print(np.linalg.norm(dataset['basis'],axis=0))
    plotting.plot_phase(dataset['probe'][0][0],basis=1e6*dataset['basis'])
    plotting.plot_amplitude(dataset['probe'][0][0],basis=1e6*dataset['basis'])
    plotting.plot_colorized(dataset['probe'][0][0],basis=1e6*dataset['basis'])
    #plotting.plot_amplitude(synth_obj[400:750,450:850],basis=1e6*dataset['basis'])
    #plotting.plot_colorized(synth_obj[400:750,450:850],basis=1e6*dataset['basis'])
    #plotting.plot_phase(synth_obj[400:750,450:850],basis=1e6*dataset['basis'])
    plotting.plot_amplitude(synth_obj[::-1,::-1][450:900,325:775],basis=1e6*dataset['basis'])
    plotting.plot_phase(synth_obj[::-1,::-1][450:900,325:775],basis=1e6*dataset['basis'])
    plotting.plot_colorized(synth_obj[::-1,::-1][450:900,325:775],basis=1e6*dataset['basis'])

    plt.figure()
    real_translations = dataset['basis'].dot(dataset['translation'][0].transpose())
    real_translations -= np.min(real_translations,axis=1)[:,None]
    plt.plot(real_translations[0]*1e6,real_translations[1]*1e6,'k.')
    plt.plot(real_translations[0]*1e6,real_translations[1]*1e6,'b-',linewidth=0.5)
    plt.figure()
    plt.plot(freqs*1e-6, prtf)
    plt.show()
