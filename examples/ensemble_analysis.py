from __future__ import division, print_function, absolute_import

from matplotlib import pyplot as plt
import pickle

from CDTools.tools import plotting
from CDTools.tools import analysis



with open('example_reconstructions/gold_balls_ensemble.pickle', 'rb') as f:
    dataset = pickle.load(f)

# This converts from a list of dictionaries to a dictionary of lists
# It's safe to assume that all elements have the same set of keys
if type(dataset) == type([]):
    dataset = {key: [element[key] for element in dataset]
               for key in dataset[0]}


# Now we synthesize an average reconstruction
synth_probe, synth_obj, aligned_objs = analysis.synthesize_reconstructions(
    dataset['probe'], dataset['obj'])

# And then we calculate the consistency PRTF from this
freqs, prtf = analysis.calc_consistency_prtf(synth_obj, aligned_objs, dataset['basis'][0])


# Plot the first mode in detail
plotting.plot_phase(synth_probe[0],basis=dataset['basis'][0])
plotting.plot_amplitude(synth_probe[0],basis=dataset['basis'][0])
plotting.plot_colorized(synth_probe[0],basis=dataset['basis'][0])

# Just plot the colorized version of the subdominant modes
plotting.plot_colorized(synth_probe[1],basis=dataset['basis'][0])
plotting.plot_colorized(synth_probe[2],basis=dataset['basis'][0])

# And now we plot the object
plotting.plot_amplitude(synth_obj,basis=dataset['basis'][0])
plotting.plot_colorized(synth_obj,basis=dataset['basis'][0])
plotting.plot_phase(synth_obj,basis=dataset['basis'][0])
    
# Finally, plot the consistency PRTF
plt.figure()
plt.plot(freqs*1e-6, prtf)
plt.xlabel('Spatial Frequency (cycles/um)')
plt.ylabel('Consistency Based PRTF')

plt.show()
