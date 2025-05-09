"""Adam optimizer class

"""
import torch as t
from torch.utils import data as torchdata
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import ticker
import numpy as np
import threading
import queue
import time
from scipy import io
from contextlib import contextmanager
from cdtools.tools.data import nested_dict_to_h5, h5_to_nested_dict, nested_dict_to_numpy, nested_dict_to_torch
from cdtools.datasets.ptycho_2d_dataset import Ptycho2DDataset
from cdtools.models import CDIModel
from typing import Tuple, List
from cdtools.optimizer import Reconstructor

__all__ = ['Adam']

class Adam(Reconstructor):
    def __init__(self,
                 model: CDIModel,
                 dataset: Ptycho2DDataset,
                 schedule: bool = False,
                 subset: List[int] = None,
                 thread: bool = True):
        
        super().__init__(model, 
                         dataset, 
                         subset, 
                         thread)
        
        # Define the optimizer
        self.optimizer = t.optim.Adam(self.model.parameters())
        

    def setup_dataloader(self,
                         batch_size):
        # Make a dataloader suited for either single-GPU use or cases
        # where a process group (i.e., multiple GPUs) has been initialized
        if self.multi_gpu_used:
            # First, create a sampler to load subsets of dataset to the GPUs
            self.sampler = DistributedSampler(self.dataset,
                                            num_replicas=self.world_size,
                                            rank=self.rank,
                                            shuffle=True,
                                            drop_last=False)
            # Now create the dataloader
            self.data_loader = torchdata.DataLoader(self.dataset,
                                                    batch_size=batch_size//self.world_size,
                                                    num_workers=0, # Creating extra threads in children processes may cause problems. Leave this at 0.
                                                    drop_last=False,
                                                    pin_memory=False,# I'm not 100% sure what this does, but apparently making this True can cause bugs
                                                    sampler=self.sampler)
        else:
            self.data_loader = torchdata.DataLoader(self.dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
        # Store the optimizer parameters
        #self.hyperparameters['lr'] = lr
        #self.hyperparameters['betas'] = betas
        #self.hyperparameters['amsgrad'] = amsgrad
    
    def adjust_optimizer(self,
                         lr=0.005,
                         betas=(0.9, 0.999),
                         amsgrad=False):

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['betas'] = betas
            param_group['amsgrad'] = amsgrad


    def optimize(self,
                 iterations,
                 batch_size=15,
                 lr=0.005,
                 betas=(0.9, 0.999),
                 schedule=False,
                 amsgrad=False,
                 subset=None,
                 regularization_factor = None,
                 thread=True,
                 calculation_width=10):
        """Runs a round of reconstruction using the Adam optimizer

        Formerly CDIModel.Adam_optimize
        
        This calls the base Reconstructor.optimize method
        (formerly CDIModel.AD_optimize) to run a round of reconstruction

        NOTE: A decision should be made regarding whether self.optimize
              should only be allowed to adjust reconstruction
              hyperparameters rather than initialize them
        """
        self.model.training_history += (
            f'Planning {iterations} epochs of Adam, with a learning rate = '
            f'{lr}, batch size = {batch_size}, regularization_factor = '
            f'{regularization_factor}, and schedule = {schedule}.\n'
        )

        #############################################################
        # The subset statement is contained in Reconstructor.__init__
        #############################################################

        #############################################################
        # The dataloader step is handled by self.dataloader
        # TODO: Figure out a way to adjust the batch_size without
        #       creating a brand-spanking-new one each time
        #############################################################
        self.setup_dataloader(batch_size)

        #############################################################
        # The optimizer is created in self.__init__, but the 
        # hyperparameters need to be set up with self.adjust_optimizer
        #############################################################
        self.adjust_optimizer(lr,
                              betas,
                              amsgrad)

        #############################################################
        # Define the scheduler
        # NOTE: We may want to define this in __init__ and simply
        #
        if schedule:
            self.scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                    factor=0.2,
                                                                    threshold=1e-9)
        else:
            self.scheduler = None

        # This is analagous to making a call to CDIModel.AD_optimize
        return super(Adam, self).optimize(iterations,
                                            regularization_factor,
                                            thread,
                                            calculation_width)