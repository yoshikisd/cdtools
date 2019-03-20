"""Contains various loss functions to be used for optimization

It exposes three losses, one returning the mean squared amplitude error, one
that returns the mean squared intensity error, and one that returns the
maximum likelihood metric for a system with Poisson statistics.

"""
from __future__ import division, print_function, absolute_import
import torch as t


__all__ = ['amplitude_mse', 'intensity_mse', 'poisson_ml']


def amplitude_mse(intensities, sim_intensities, mask=None):
    """ Returns the mean squared error of a simulated dataset's amplitudes

    Calculates the summed mean squared error between a given set of 
    measured diffraction intensities and a simulated set.

    This function calculates the mean squared error between their
    associated amplitudes. Because this is not well defined for negative
    numbers, make sure that all the intensities are >0 before using this
    loss.

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    This is empirically the most useful loss function

    Args:
        intensities (torch.Tensor) : A tensor with measured detector values
        sim_intensities (torch.Tensor) : A tensor of simulated detector intensities
        mask (torch.Tensor) : A mask with ones for pixels to include and zeros for pixels to exclude

    Returns:
        loss (torch.Tensor) : A single value for the summed mse

    """

    # I know it would be more efficient if this function took in the
    # amplitudes instead of the intensities, but I want to be consistent
    # with all the errors working off of the same inputs

    if mask is None:
        return t.sum((t.sqrt(sim_intensities) -
                      t.sqrt(intensities))**2)
    else:
        return t.sum((t.sqrt(sim_intensities.masked_select(mask)) -
                      t.sqrt(intensities.masked_select(mask)))**2)


    
def intensity_mse(intensities, sim_intensities, mask=None):
    """ Returns the mean squared error of a simulated dataset's intensities

    Calculates the summed mean squared error between a given set of 
    diffraction intensities - the measured set of detector intensities -
    and a simulated set of diffraction intensities. This function
    calculates the mean squared error between the intensities.

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    Args:
        intensities (torch.Tensor) : A tensor with measured detector intensities.
        sim_intensities (torch.Tensor) : A tensor of simulated detector intensities
        mask (torch.Tensor) : A mask with ones for pixels to include and zeros for pixels to exclude

    Returns:
        loss (torch.Tensor) : A single value for the summed mse

    """
    if mask is None:
        return t.sum((sim_intensities - intensities)**2)
    else:
        return t.sum((sim_intensities.masked_select(mask) -
                      intensities.masked_select(mask))**2)


    
def poisson_ml(intensities, sim_intensities, mask=None):
    """ Returns the Poisson maximum likelihood metric for a simulated dataset's intensities

    Calculates the overall Poisson maximum likelihood metric using
    diffraction intensities - the measured set of detector intensities -
    and a simulated set of intensities. This loss would be appropriate
    for detectors in a single-photon counting mode, with their output
    scaled to number of photons

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    Args:
        intensities (torch.Tensor) : A tensor with measured detector intensities.
        sim_intensities (torch.Tensor) : A tensor of simulated detector intensities
        mask (torch.Tensor) : A mask with ones for pixels to include and zeros for pixels to exclude

    Returns:
        loss (torch.Tensor) : A single value for the poisson ML metric

    """
    if mask is None:
        t.sum(simulated_intensities -
              intensities * t.log(simulated_intensities))
    else:
        return t.sum(simulated_intensities.masked_select(mask) -
                     intensities.masked_select(mask) *
                     t.log(simulated_intensities.masked_select(mask)))
