"""
Purpose: Convert file collection from Jim Lebeau's TITAN microscope to .CXI
Author:  Abe Levitan
Date:    January 2019
"""

import numpy as np
import pickle
import h5py
import os
import CDTools
from CDTools.tools import data as cdtdata
from CDTools.datasets import Ptycho2DDataset
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from datetime import datetime
import xml.etree.ElementTree as ET

def load_raw_image_stack(filename):
    # The resulting data is an array of (exposure, image-i, image-j),
    # with image0i corresponding to y and image-j corresponding to x
    # Note that the real-space scanning is done from the bottom right
    # corner, first heading left (in x) then scanning up.
    rawdata = np.fromfile(filename,dtype='<f4')
    if len(rawdata) % (128*130) != 0:
        raise IndexError('The raw data file doesn\'t seem to have the right number of values stored in it')
    numshots = len(rawdata) // (128*130)
    # One of the directions seems to be flipped
    return rawdata.reshape(numshots,130,128)[:,:128,:][:,:,::-1].copy()

def load_metadata(filename):
    return ET.parse(filename).getroot()

def get_scan_shape(metadata):
    sp = metadata.find("scan_parameters[@mode='acquire']")
#    print(sp)
#    exit()
    shape_x = int(sp.find('scan_resolution_x').text)
    shape_y = int(sp.find('scan_resolution_y').text)
    return [shape_x,shape_y]

def get_camera_length(metadata):
    iomm = metadata.find('iom_measurements')
    ncl = iomm.find('nominal_camera_length')
    return float(ncl.text)

def get_scan_steps(metadata):
    shape = get_scan_shape(metadata)
    iomm = metadata.find('iom_measurements')
    fov = iomm.find('full_scan_field_of_view')
    xfov = float(fov.find('x').text)
    yfov = float(fov.find('y').text)
    return np.array([xfov,yfov]) / np.array(shape)

    

def gen_scan_grid(shape, step):
    ys, xs = np.mgrid[:shape[0],:shape[1]]
    xs = xs * step[0]
    ys = ys * step[1]
    return np.stack((xs.ravel(),ys.ravel(),np.zeros(ys.ravel().shape))).transpose()

h = 6.626e-34
c = 2.998e8
me =  9.109e-31
def calculate_wavelength(electron_energy):
    return h*c /  np.sqrt(electron_energy**2 + 2*c**2 * me * electron_energy)


def generate_detector_geometry(distance, pitches):
    basis = np.array([[0,-pitches[1]],[-pitches[0],0],[0,0]])
    return {'basis':basis, 'distance':distance}

def generate_dataset(translations, patterns, detector_geometry, electron_energy):
    wavelength = calculate_wavelength(electron_energy)
    print(wavelength)
    return Ptycho2DDataset(translations, patterns, wavelength=wavelength, detector_geometry=det_geo)



data_folder = '/media/Data Bank/ptychography_firsttry/out_of_focus_58Mx_1ms_reso80x80_ss1'
image_filename = 'scan_x80_y80.raw'
save_filename = 'test_defocus_newcalibration.cxi'
metadata_filename = 'out_of_focus_58Mx_1ms_reso80x80_ss1.xml'


#data_folder = '/media/Data Bank/ptychography_firsttry/acquisition_3'
#image_filename = 'scan_x80_y80.raw'
#save_filename = 'test_acq3_newcalibration.cxi'
#metadata_filename = 'acquisition_3.xml'


metadata = load_metadata(data_folder + '/' + metadata_filename)

scan_shape = get_scan_shape(metadata)
#scan_shape = 80

# These are reasonable initial guesses, until we get calibration data
#scan_step = 0.2e-10 #Angstrom, old value from manual measurement
scan_steps = get_scan_steps(metadata)

# This is something I can calculate from the detector length
# A good calibration is to assume that the pixel size is 0.2276 mm and
# the detector distance is equal to the nomninal camera length
camera_length = get_camera_length(metadata)
detector_distance = camera_length
pixel_pitches = [0.231e-3,0.231e-3] # best guess near length=0.230
# pixel_pitches = [0.2276e-3,0.2276e-3] # best overall average

# old manual calibration
#pixel_pitches = [150e-6,150e-6]
#detector_distance = 100e-3 # mm
#print([pp / detector_distance for pp in pixel_pitches])
#exit()

electron_energy = 200 * 1.602e-16 # Joules
# Important question: Check which side the images fill in from

data = load_raw_image_stack(data_folder + '/' + image_filename)
#data[:,30:-30,30:-30] = 0 # For HAADF
scan_points = gen_scan_grid(scan_shape,scan_steps)

det_geo = generate_detector_geometry(detector_distance, pixel_pitches)
dataset = generate_dataset(scan_points[1:], data[1:], det_geo, electron_energy)
dataset.inspect(units='nm')
plt.show()

dataset.to_cxi(data_folder + '/' + save_filename)



