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

__all__ = ['spawn', 'multi_gpu']


# Set the default timeout to 60s
WAIT_TIME = datetime.timedelta(seconds=60)


def process_manager(rank, reconstructor, world_size, backend, timeout):
    """A wrapper around the reconstruction script defined in a high-level
    user interface. 
    
    This wraps around the reconstructor function, enabling multi-GPU operations
    to be set up afterwards through torch.multiprocessing.spawn or
    cdtools.tools.distributed.distributed.spawn

    Parameters:
        rank: int
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
    init_process_group(backend=backend,
                        rank=rank,
                        world_size=world_size,
                        timeout=timeout)
    
    reconstructor(rank, world_size)
    barrier()
    destroy_process_group()


def spawn(reconstructor,
          world_size: int,
          backend: str = 'nccl',
          timeout: datetime = WAIT_TIME,
          master_addr: str = 'localhost',
          master_port: str = '8888',
          nccl_p2p_disable: bool = True):
    """A wrapper around torch.multiprocessing.spawn. 
    
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
    # Set up environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['NCCL_P2P_DISABLE'] = str(int(nccl_p2p_disable))

    # Ensure a "graceful" termination of subprocesses if something goes wrong.
    try:
        print('\nStarting up multi-GPU reconstructions...')
        mp.spawn(process_manager,
                 args=(reconstructor, world_size, backend, timeout),
                 nprocs=world_size,
                 join=True)
        print('Reconstructions complete. Stopping processes...')

    except Exception as e:
        # If something breaks, we try to make sure that the
        # process group is destroyed before the program fully
        # terminates
        print(e)
        destroy_process_group()
