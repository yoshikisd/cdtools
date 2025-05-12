"""This module contains the Adam Reconstructor subclass for performing
optimization ('reconstructions') on ptychographic/CDI models using
the Adam optimizer.

The Reconstructor class is designed to resemble so-called
'Trainer' classes that (in the language of the AI/ML folks) handles
the 'training' of a model given some dataset and optimizer.
"""
import torch as t
from torch.utils import data as torchdata
from torch.utils.data.distributed import DistributedSampler
from scipy import io
from contextlib import contextmanager
from cdtools.tools.data import nested_dict_to_h5, h5_to_nested_dict, nested_dict_to_numpy, nested_dict_to_torch
from cdtools.datasets.ptycho_2d_dataset import Ptycho2DDataset
from cdtools.models import CDIModel
from typing import Tuple, List
from cdtools.optimizer import Reconstructor

__all__ = ['Adam']

class Adam(Reconstructor):
    """
    The Adam Reconstructor subclass handles the optimization ('reconstruction') of 
    ptychographic models and datasets using the Adam optimizer.

    Parameters
    ----------
    model: CDIModel
        Model for CDI/ptychography reconstruction
    dataset: Ptycho2DDataset
        The dataset to reconstruct against
    subset : list(int) or int
        Optional, a pattern index or list of pattern indices to use

    Important attributes:
    - **model** -- Always points to the core model used.
    - **multi_gpu_used** -- Whether or not multi-GPU computation will be performed
      using a distributed data approach. This attribute will be pulled from the
      CDIModel (this flag is automatically set when using cdtools.tools.distributed.spawn).
    - **optimizer** -- This class by default uses `torch.optim.Adam` to perform
      optimizations.
    - **scheduler** -- A `torch.optim.lr_scheduler` that must be defined when creating the
      `Reconstructor` subclass through the `setup_scheduler` method.
    - **data_loader** -- A torch.utils.data.DataLoader that must be defined when creating
      the Reconstructor subclass through the `setup_dataloader` method.
    """
    def __init__(self,
                 model: CDIModel,
                 dataset: Ptycho2DDataset,
                 subset: List[int] = None):

        super().__init__(model, dataset, subset)
        
        # Define the optimizer for use in this subclass
        self.optimizer = t.optim.Adam(self.model.parameters())
        

    def setup_dataloader(self,
                         batch_size: int = 15,
                         shuffle: bool = True):
        """
        Sets up the dataloader.

        Parameters
        ----------
        batch_size : int
            Optional, the size of the minibatches to use
        shuffle : bool
            Optional, enable/disable shuffling of the dataset. This option
            is intended for diagnostic purposes and should be left as True.
        """
        # Make a dataloader suited for either single-GPU use or cases
        # where a process group (i.e., multiple GPUs) has been initialized
        if self.multi_gpu_used:
            # First, create a sampler to load subsets of dataset to the GPUs
            self.sampler = DistributedSampler(self.dataset,
                                              num_replicas=self.world_size,
                                              rank=self.rank,
                                              shuffle=shuffle,
                                              drop_last=False)
            # Now create the dataloader
            self.data_loader = torchdata.DataLoader(self.dataset,
                                                    batch_size=batch_size//self.world_size,
                                                    num_workers=0, # Creating extra threads in children processes may cause problems. Leave this at 0.
                                                    drop_last=False,
                                                    pin_memory=False,
                                                    sampler=self.sampler)
        else:
            self.data_loader = torchdata.DataLoader(self.dataset,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle)
        # Store the optimizer parameters
        #self.hyperparameters['lr'] = lr
        #self.hyperparameters['betas'] = betas
        #self.hyperparameters['amsgrad'] = amsgrad
    
    def adjust_optimizer(self,
                         lr=0.005,
                         betas=(0.9, 0.999),
                         amsgrad=False):
        """
        Change hyperparameters for the utilized optimizer.

        Parameters
        ----------
        lr : float
            Optional, The learning rate (alpha) to use. Default is 0.005. 0.05 is 
            typically the highest possible value with any chance of being stable
        betas : tuple
            Optional, the beta_1 and beta_2 to use. Default is (0.9, 0.999).
        amsgrad: bool
            Optional, whether to use the AMSGrad variant of this algorithm
        """
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
        """
        Runs a round of reconstruction using the Adam optimizer

        Formerly `CDIModel.Adam_optimize`
        
        This calls the Reconstructor.optimize superclass method
        (formerly `CDIModel.AD_optimize`) to run a round of reconstruction
        once the dataloader and optimizer hyperparameters have been
        set up.
        """
        # Update the training history
        self.model.training_history += (
            f'Planning {iterations} epochs of Adam, with a learning rate = '
            f'{lr}, batch size = {batch_size}, regularization_factor = '
            f'{regularization_factor}, and schedule = {schedule}.\n'
        )

        # The subset statement is contained in Reconstructor.__init__

        # The dataloader step is handled by self.dataloader
        # TODO: Figure out a way to adjust the batch_size without
        #       creating a brand-spanking-new one each time
        self.setup_dataloader(batch_size)

        # The optimizer is created in self.__init__, but the 
        # hyperparameters need to be set up with self.adjust_optimizer
        self.adjust_optimizer(lr,
                              betas,
                              amsgrad)

        # Define the scheduler
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