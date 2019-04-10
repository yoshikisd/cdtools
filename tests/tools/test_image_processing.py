from __future__ import division, print_function, absolute_import


import pytest
import numpy as np
import torch as t

from CDTools.tools import image_processing, cmath, initializers, interactions
from scipy import ndimage


def test_centroid():
    # Test single im
    im = t.rand((30,40))
    sp_centroid = ndimage.measurements.center_of_mass(im.numpy())
    centroid = image_processing.centroid(im)
    assert t.allclose(centroid, t.Tensor(sp_centroid))
    
    # Test stack o' ims
    ims = t.rand((5,30,40))
    sp_centroids = [ndimage.measurements.center_of_mass(im.numpy())
                    for im in ims]
    centroids = image_processing.centroid(ims)
    assert t.allclose(centroids, t.Tensor(sp_centroids))


def test_centroid_sq():
    # Test single im
    im = t.rand((30,40))
    sp_centroid = ndimage.measurements.center_of_mass(im.numpy()**2)
    centroid = image_processing.centroid_sq(im)
    assert t.allclose(centroid, t.Tensor(sp_centroid))

    # Test complex with multiple ims
    ims = t.rand((5,30,40,2))
    np_ims = cmath.torch_to_complex(ims)
    sp_centroids = [ndimage.measurements.center_of_mass(np.abs(im)**2)
                    for im in np_ims]
    centroids = image_processing.centroid_sq(ims, comp=True)
    assert t.allclose(centroids, t.Tensor(np.array(sp_centroids)))


def test_find_pixel_shift():

    # Test two real ims
    big_im = t.rand((30,70))
    im1 = big_im[3:,:-20]
    im2 = big_im[:-3,20:]
    assert t.all(image_processing.find_pixel_shift(im1,im2) == t.LongTensor([-3,20]))

    # Test a real and complex im
    big_im = t.rand((30,70))
    im1 = t.stack((big_im[:-5,10:],t.zeros_like(big_im[:-5,10:])),dim=-1)
    im2 = big_im[5:,:-10]
    assert t.all(image_processing.find_pixel_shift(im1,im2) == t.LongTensor([5,-10]))
    assert t.all(image_processing.find_pixel_shift(im2,im1) == t.LongTensor([-5,10]))
    
    # Test two complex ims
    big_im = t.rand((45,45,2))
    im1 = big_im[:-5,:-4]
    im2 = big_im[5:,4:]
    assert t.all(image_processing.find_pixel_shift(im1,im2) == t.LongTensor([5,4]))


def test_find_subpixel_shift():
    # We can do this by creating a test probe and a test object
    test_probe = t.rand((70,70,2))
    test_obj = t.ones((300,300,2))

    shift = t.tensor((0.8,0.75))
    
    im = interactions.ptycho_2D_sinc(test_probe, test_obj, shift)

    retrieved_shift = image_processing.find_subpixel_shift(im, test_probe, search_around=(0,0), resolution=50)
    # tolerance of 0.03 on this measurement
    assert t.all(t.abs(shift - retrieved_shift) < 0.03)

    
def test_find_shift():

    # We can do this by creating a test probe and a test object
    test_probe = t.rand((200,200,2))
    test_obj = t.ones((300,300,2))

    shift = t.tensor((0.8,0.75))
    
    im = interactions.ptycho_2D_sinc(test_probe, test_obj, shift)[:-40,:-6]

    retrieved_shift = image_processing.find_shift(im, test_probe[40:,6:], resolution=50)
    # tolerance of 0.03 on this measurement
    assert t.all(t.abs(shift + t.Tensor((40,6)) - retrieved_shift) < 0.03)
    
