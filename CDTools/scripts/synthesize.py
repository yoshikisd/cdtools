from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from matplotlib import pyplot as plt
import pickle
import argparse

from CDTools.tools import cmath, plotting
from CDTools.tools.analysis import *


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
    print(dataset['basis'])

    print(synth_probe.shape)
    freqs, prtf = calc_consistency_prtf(synth_obj, aligned_objs, dataset['basis'])
    
    plotting.plot_phase(synth_probe,basis=dataset['basis'])
    plotting.plot_amplitude(synth_probe,basis=dataset['basis'])
    plotting.plot_colorized(synth_probe,basis=dataset['basis'])
    
    try:
        plotting.plot_phase(synth_probe[1],basis=dataset['basis'])
        plotting.plot_amplitude(synth_probe[1],basis=dataset['basis'])
        plotting.plot_colorized(synth_probe[1],basis=dataset['basis'])
    except:
        pass
      
    plotting.plot_amplitude(synth_obj,basis=dataset['basis'])
    plotting.plot_colorized(synth_obj,basis=dataset['basis'])
    plotting.plot_phase(synth_obj,basis=dataset['basis'])
    

    plt.figure()
    try:
        real_translations = dataset['basis'].dot(dataset['translation'][0].transpose())
        real_translations -= np.min(real_translations,axis=1)[:,None]
        real_translations = real_translations.transpose()
        plotting.plot_translations(real_translations)
        plt.figure()
    except:
        pass
        
    plt.plot(freqs*1e-6, prtf)
    plt.show()
