import cdtools
from matplotlib import pyplot as plt
import torch as t
from cdtools.models.complex_adam import MyAdam
from torch.utils import data as torchdata
import time

# First, we load an example dataset from a .cxi file
filename = 'example_data/lab_ptycho_data.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)
dataset.translations = t.cat([dataset.translations]*5)
dataset.patterns = t.cat([dataset.patterns]*5)

# Next, we create a ptychography model from the dataset
model = cdtools.models.SimplePtycho.from_dataset(dataset)

#class MyDataParallel(t.nn.DataParallel):
#    def __getattr__(self, name):
#        return getattr(self.module, name)

model = t.nn.DataParallel(model, device_ids=[0,1,2,3])


# Make a dataloader
data_loader = torchdata.DataLoader(dataset, batch_size=20,
                                   shuffle=True)

device = 'cuda'
model.to(device=device)
#dataset.get_as(device='cuda')

# Define the optimizer
optimizer = MyAdam(model.parameters(), lr=0.01)

normalization=0
for inputs, patterns in data_loader:
    normalization += t.sum(patterns).cpu().numpy()

def run_iteration(stop_event=None):
    loss = 0
    N = 0
    t0 = time.time()
    for inputs, patterns in data_loader:
        inp = (x.to(device) for x in inputs)#inputs.to(device)
        pats = patterns.to(device)
        N += 1
        def closure():
            optimizer.zero_grad()

            sim_patterns = model.forward(*inp)
            
            if hasattr(model, 'mask'):
                loss = model.module.loss(pats,sim_patterns, mask=model.module.mask)
            else:
                loss = model.module.loss(pats,sim_patterns)
                
            loss.backward()
            return loss.detach()

        loss += optimizer.step(closure).detach().cpu().numpy()
    print('time', time.time()-t0)
    return loss / normalization


for it in range(20):
    print(run_iteration())

print(model.module.save_results().keys())
exit()
# Finally, we plot the results
model.module.inspect(dataset)
model.module.compare(dataset)
plt.show()
