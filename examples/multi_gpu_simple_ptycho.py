import cdtools
import torch as t
from cdtools.models.complex_adam import MyAdam
from torch.utils import data as torchdata
import time
import  torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def demo():
    # First, we load an example dataset from a .cxi file
    filename = 'example_data/lab_ptycho_data.cxi'
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)
    
    # Next, we create a ptychography model from the dataset
    model = cdtools.models.SimplePtycho.from_dataset(dataset)
    
    
    print(dist.is_torchelastic_launched())
    dist.init_process_group(backend="nccl", init_method='env://')
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % t.cuda.device_count()
    print('device id', device_id)
    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    print(model.loss_train)
    print(model.Adam_optimize)
    print(ddp_model.Adam_optimize)
    
    sampler = torchdata.distributed.DistributedSampler(dataset)
    # Make a dataloader
    #data_loader = torchdata.DataLoader(dataset, batch_size=20,
    #                                   shuffle=True, num_workers=0)
    data_loader = torchdata.DataLoader(dataset, batch_size=20,
                                       shuffle=False, sampler=sampler)
    
    device = 'cuda:'+str(device_id)
    dataset.get_as(device=device)
    
    # Define the optimizer
    #optimizer = MyAdam(model.parameters(), lr=0.01)
    optimizer = t.optim.Adam(model.parameters(), lr=0.01)
    
    normalization=0
    for inputs, patterns in dataset:#data_loader:
        normalization += t.sum(patterns)
        
    dist.all_reduce(normalization)

    for it in range(10):
        print(it)
        sampler.set_epoch(it)
        loss = 0
        N = 0
        t0 = time.time()
        for inputs, patterns in data_loader:
            #print(inputs)
            N += 1
            def closure():
                optimizer.zero_grad()
                
                sim_patterns = model.forward(*inputs)
                
                if hasattr(model, 'mask'):
                    loss = model.loss(patterns,sim_patterns, mask=model.mask)
                else:
                    loss = model.loss(patternss,sim_patterns)
                    
                loss.backward()
                return loss.detach()

            loss += optimizer.step(closure)#.cpu().numpy()
        #dist.all_reduce(loss)
        #loss = dist.all_reduce(loss)
        #print(it, 'time', time.time()-t0)
        #print(loss / normalization)
        if rank==0:
            model.inspect()
    return loss#.cpu().numpy()

if __name__=='__main__':
    demo()
