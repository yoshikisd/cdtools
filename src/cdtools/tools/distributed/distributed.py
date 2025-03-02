"""Contains wrapper functions to make reconstruction scripts compatible
with multi-GPU distributive approaches in PyTorch.


"""

import numpy as np
import torch as t
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import datetime
import os
import functools

__all__ = ['spawn']



def reconstructor_wrapper(rank, 
                          reconstructor, 
                          model, 
                          dataset, 
                          world_size, 
                          backend, 
                          timeout):
    """Wraps functions containing reconstruction loops (i.e., for loss in model.Adam_optimize)
    to enable multi-GPU operations to be set up. The wrapped function needs to passed to
    torch.multiprocessing.spawn or cdtools.tools.distributed.distributed.spawn

    Parameters:
        rank: int
            Rank of the GPU, with value ranging from [0, world_size-1]
        reconstructor:
            The reconstruction loop function
        model: t.nn.Module
            The CDIModel
        dataset: t.utils.data.Dataset
            The CDataset
        world_size: int
            Number of GPUs to use
        master_addr: str
            IP address of the machine that will host the process with rank 0
        master_port: str
            A free port on the machine that will host the process with rank 0
        nccl_p2p_disable: bool
            Disable NCCL peer-2-peer communication
    """
    # Initialize the process group
    init_process_group(backend=backend,
                        rank=rank,
                        world_size=world_size,
                        timeout=timeout)
    
    # Load the model to the appropriate GPU rank the process is using
    device = f'cuda:{rank}'
    model.to(device=device)
    dataset.get_as(device=device) 

    # Wrap the model with DistributedDataParallel
    model = DDP(model,
                device_ids=[rank],  # Tells DDP which GPU the model lives in
                output_device=rank, # Tells DDP which GPU to output to
                find_unused_parameters=True) # TODO: Understand what this is really doing...
    
    # Dayne's special sanity check: Don't start reconstructing until all GPUs have synced.
    barrier()
    
    reconstructor(model, dataset, rank, world_size)         # Start the reconstruction loop
    barrier()                               # Wait for all GPUs to finish reconstructing
    destroy_process_group()                 # Destroy the process group


def spawn(reconstructor,
          model,
          dataset,
          world_size: int,
          backend: str = 'nccl',
          timeout: int = 60,
          master_addr: str = 'localhost',
          master_port: str = '8888',
          nccl_p2p_disable: bool = True):
    """Spawns world_size processes that runs a reconstructor loop function.
    A wrapper around torch.multiprocessing.spawn. 
    
    It includes the setup of OS environmental variables needed for
    initializing the distributed backend

    Parameters:
        reconstructor:
            The wrapped reconstruction loop
        world_size: int
            Number of GPUs to use
        master_addr: str
            IP address of the machine that will host the process with rank 0
        master_port: str
            A free port on the machine that will host the process with rank 0
        nccl_p2p_disable: bool
            Disable NCCL peer-2-peer communication
    """
    # Convert timeout from int to datetime
    timeout = datetime.timedelta(seconds=timeout)

    # Set up environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['NCCL_P2P_DISABLE'] = str(int(nccl_p2p_disable))

    # Ensure a "graceful" termination of subprocesses if something goes wrong.
    try:
        print('\nStarting up multi-GPU reconstructions...')
        mp.spawn(reconstructor_wrapper,
                 args=(reconstructor, 
                       model,
                       dataset,
                       world_size,
                       backend, 
                       timeout),
                 nprocs=world_size,
                 join=True)
        print('Reconstructions complete. Stopping processes...')

    except Exception as e:
        # If something breaks, we try to make sure that the
        # process group is destroyed before the program fully
        # terminates
        print(e)
        destroy_process_group()
