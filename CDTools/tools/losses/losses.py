"""Contains various loss functions to be used for optimization

It exposes three losses, one returning the mean squared amplitude error, one
that returns the mean squared intensity error, and one that returns the
maximum likelihood metric for a system with Poisson statistics.

"""
from __future__ import division, print_function, absolute_import
import torch as t


__all__ = ['amplitude_mse', 'intensity_mse', 'poisson_nll']


def amplitude_mse(intensities, sim_intensities, mask=None):
    """ Returns the mean squared error of a simulated dataset's amplitudes

    Calculates the mean squared error between a given set of 
    measured diffraction intensities and a simulated set.

    This function calculates the mean squared error between their
    associated amplitudes. Because this is not well defined for negative
    numbers, make sure that all the intensities are >0 before using this
    loss. Note that this is actually a sum-squared error, because this
    formulation makes it vastly simpler to compare error calculations
    between reconstructions with different minibatch size. I hope to
    find a better way to do this that is more honest with this
    cost function, though.

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    This is empirically the most useful loss function for most cases

    Parameters
    ----------
    intensities : torch.Tensor
        A tensor with measured detector values
    sim_intensities : torch.Tensor
        A tensor of simulated detector intensities
    mask : torch.Tensor
        A mask with ones for pixels to include and zeros for pixels to exclude

    Returns
    -------
    loss : torch.Tensor
        A single value for the mean amplitude mse
    """

    # I know it would be more efficient if this function took in the
    # amplitudes instead of the intensities, but I want to be consistent
    # with all the errors working off of the same inputs

    if mask is None:
        return t.sum((t.sqrt(sim_intensities) -
                      t.sqrt(intensities))**2)
    else:
        masked_intensities = intensities.masked_select(mask)
        return t.sum((t.sqrt(sim_intensities.masked_select(mask)) -
                      t.sqrt(masked_intensities))**2)


    
def intensity_mse(intensities, sim_intensities, mask=None):
    """ Returns the mean squared error of a simulated dataset's intensities

    Calculates the summed mean squared error between a given set of 
    diffraction intensities - the measured set of detector intensities -
    and a simulated set of diffraction intensities. This function
    calculates the mean squared error between the intensities.

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    Parameters
    ----------
    intensities : torch.Tensor
        A tensor with measured detector intensities.
    sim_intensities : torch.Tensor
        A tensor of simulated detector intensities
    mask : torch.Tensor
        A mask with ones for pixels to include and zeros for pixels to exclude

    Returns
    -------
    loss : torch.Tensor
        A single value for the mean intensity mse

    """
    if mask is None:
        return t.sum((sim_intensities - intensities)**2) \
            / intensities.view(-1).shape[0]
    else:
        masked_intensities = intensities.masked_select(mask)
        return t.sum((sim_intensities.masked_select(mask) -
                      masked_intensities)**2) \
                      / masked_intensities.shape[0]


    
def poisson_nll(intensities, sim_intensities, mask=None, eps=1e-6):
    """ Returns the Poisson negative log likelihood for a simulated dataset's intensities

    Calculates the overall Poisson maximum likelihood metric using
    diffraction intensities - the measured set of detector intensities -
    and a simulated set of intensities. This loss would be appropriate
    for detectors in a single-photon counting mode, with their output
    scaled to number of photons

    Note that this calculation ignores the log(intensities!) term in the
    full expression for Poisson negative log likelihood. This term doesn't
    change the calculated gradients so isn't worth taking the time to compute

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    The default value of eps is 1e-6 - a nonzero value here helps avoid
    divergence of the log function near zero.

    Parameters
    ----------
    intensities : torch.Tensor
        A tensor with measured detector intensities.
    sim_intensities : torch.Tensor
        A tensor of simulated detector intensities
    mask : torch.Tensor
        A mask with ones for pixels to include and zeros for pixels to exclude
    eps : float
        Optional, a small number to add to the simulated intensities
    
    Returns
    -------
    loss : torch.Tensor
        A single value for the poisson negative log likelihood

    """
    if mask is None:
        return t.sum(sim_intensities+eps -
                     intensities * t.log(sim_intensities+eps)) \
                     / intensities.view(-1).shape[0]
    
    else:
        masked_intensities = intensities.masked_select(mask)
        masked_sims = sim_intensities.masked_select(mask)
        return t.sum(masked_sims - masked_intensities *
                     t.log(masked_sims+eps)) / masked_intensities.shape[0]
