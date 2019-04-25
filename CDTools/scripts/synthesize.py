from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from matplotlib import pyplot as plt
import pickle
import argparse
from scipy import fftpack

from CDTools.tools import cmath, plotting
from CDTools.tools import image_processing as ip
from CDTools.tools.analysis import *




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

    
    plotting.plot_phase(dataset['probe'][0][0],basis=1e6*dataset['basis'])
    plotting.plot_amplitude(dataset['probe'][0][0],basis=1e6*dataset['basis'])
    plotting.plot_colorized(dataset['probe'][0][0],basis=1e6*dataset['basis'])

    
    plotting.plot_phase(synth_probe[1],basis=1e6*dataset['basis'])
    plotting.plot_amplitude(synth_probe[1],basis=1e6*dataset['basis'])
    plotting.plot_colorized(synth_probe[1],basis=1e6*dataset['basis'])

    
    
    plotting.plot_amplitude(synth_obj,basis=1e6*dataset['basis'])
    plotting.plot_colorized(synth_obj,basis=1e6*dataset['basis'])
    plotting.plot_phase(synth_obj,basis=1e6*dataset['basis'])
    

    plt.figure()
    plt.show()
    exit()
    real_translations = dataset['basis'].dot(dataset['translation'][0].transpose())
    real_translations -= np.min(real_translations,axis=1)[:,None]
    plt.plot(real_translations[0]*1e6,real_translations[1]*1e6,'k.')
    plt.plot(real_translations[0]*1e6,real_translations[1]*1e6,'b-',linewidth=0.5)
    plt.figure()
    plt.plot(freqs*1e-6, prtf)
    plt.show()
