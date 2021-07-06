import CDTools
import numpy as np
from scipy import io

# Load the data
filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)

results = []
n = 25

for idx in range(n):
    print('Starting Reconstruction', idx+1, 'of',n)
    
    # Create a new model each time
    model = CDTools.models.FancyPtycho.from_dataset(dataset,n_modes=3,
                                                    randomize_ang=0.1*np.pi)

    # Work on the GPU
    model.to(device='cuda')
    dataset.get_as(device='cuda')

    # Run the reconstruction
    for loss in model.Adam_optimize(30, dataset, batch_size=100):
        print(model.report(), end='\r')

    # Print a summary that won't be overwritten by the next line
    print('Finished:',model.report())
    
    # And add the results to the ensemble dictionary
    results.append(model.save_results(dataset))

# This converts from a list of dictionaries to a dictionary of lists
# It's safe to assume that all elements have the same set of keys
# After this, e.g. dataset['probe'] will return a list of all probes.
results = {key: np.array([element[key] for element in results])
           for key in results[0].keys()}

print(results['probe'].shape)
# Save out the ensemble
io.savemat('example_reconstructions/gold_balls_ensemble.mat', results)
