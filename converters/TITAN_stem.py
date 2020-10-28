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
    #print(sp)
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
    scale = float(fov.find('scale_factor').text)
    # Not sure why I need to divide by this scale factor or why it exists
    # but this seems to produce the correct numbers
    xfov = float(fov.find('x').text) / scale
    yfov = float(fov.find('y').text) / scale
    return np.array([xfov,yfov]) / np.array(shape)
    

def gen_scan_grid(shape, step):
    ys, xs = np.mgrid[:shape[0],:shape[1]]
    xs = xs * step[0]
    ys = ys * step[1]
    return np.stack((xs.ravel(),ys.ravel(),np.zeros(ys.ravel().shape))).transpose()

def get_electron_energy(metadata):
    iomm = metadata.find('iom_measurements')
    energy = float(iomm.find('high_voltage').text) / 1000 # to keV    
    return energy * 1.602e-16 # to Joules

    
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
    print('Wavelength:',wavelength)
    print('Pixel NA:',(-detector_geometry['basis'][0,1]/detector_geometry['distance']))
    exit()
    return Ptycho2DDataset(translations, patterns, wavelength=wavelength, detector_geometry=det_geo)


# Change this to allow for command-line introduction of the data folder

# I think that the data folder should be the first command-line arg, and
# be defaulted to the current folder

# Then, the image filename should by default be the only .raw file in the
# folder if there is exactly one. If none or more than one, throw an error

# Then, the metadata filename should be the only .xml file in the folder
# if there is exactly one, otherwise it should throw an error.

# Next, there should be some .csv file or similar containing the calibration
# of the scan size and pixel size. The program should give a report of how
# well the calibrated and naive values match. If no calibration is given,
# should indicate that it is using the naive values.

# Finally, the output filename should by default be the xml filename,
# and can be overriden by a clarg


#data_folder = '/media/Data Bank/Electron Ptycho MoS2/Session2/MoS2_Pty_session2/acquisition_1_50_nm_positive'
#image_filename = 'scan_x128_y128.raw'
#metadata_filename = 'acquisition_1_50_nm_positive.xml'

data_folder = '/media/Data Bank/Electron Ptycho MoS2/Session2/MoS2_Pty_session2/acquisition_1_20_nm_positive'
image_filename = 'scan_x128_y128.raw'
metadata_filename = 'acquisition_1_20_nm_positive.xml'

#data_folder = '/media/Data Bank/Electron Ptycho MoS2/Session2/MoS2_Pty_session2/acquisition_1_100nm_positive'
#image_filename = 'scan_x128_y128.raw'
#metadata_filename = 'acquisition_1_100nm_positive.xml'


#data_folder = '/media/Data Bank/Electron Ptycho MoS2/Lower_Convergence_angle/Smaller_sampling/acquisition_1_convergence_28mrad_14oMx_285mm'
#image_filename = 'scan_x128_y128.raw'
#metadata_filename = 'acquisition_1_convergence_28mrad_14oMx_285mm.xml'

#data_folder = '/media/Data Bank/Electron Ptycho MoS2/Lower_Convergence_angle/Larger_sampling/acquisition_1_14o5Mx_285mm_28mrad'
#image_filename = 'scan_x256_y256.raw'
#metadata_filename = 'acquisition_1_14o5Mx_285mm_28mrad.xml'


#data_folder = '/media/Data Bank/Electron Ptycho MoS2/Lower_Convergence_angle/Defocus_series/Defocus_positive/acquisition_2_60nm_positive_defocus_140Mx_28mrad_285mm'
#image_filename = 'scan_x128_y128.raw'
#metadata_filename = 'acquisition_2_60nm_positive_defocus_140Mx_28mrad_285mm.xml'


#data_folder = '/media/Data Bank/Electron Ptycho MoS2/Lower_Convergence_angle/Defocus_series/Defocus_negative/acquisition_2_14oMx_defocus_negative_50nm_28mrad_285mm'
#image_filename = 'scan_x128_y128.raw'
#metadata_filename = 'acquisition_2_14oMx_defocus_negative_50nm_28mrad_285mm.xml'


save_filename = 'Initial_CXI_Generation.cxi'


#data_folder = '/media/Data Bank/ptychography_firsttry/out_of_focus_58Mx_1ms_reso80x80_ss1'
#image_filename = 'scan_x80_y80.raw'
#save_filename = 'test_defocus_newcalibration.cxi'
#metadata_filename = 'out_of_focus_58Mx_1ms_reso80x80_ss1.xml'

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

# This gets the electron energy
electron_energy = get_electron_energy(metadata)


pixel_pitches = [500e-6,500e-6] # best match to Abinash's calibration
# Also feels right, even though the docs I find for the EMPAD shows 150um pixels

# These came from a calibration done by Xi
#pixel_pitches = [0.231e-3,0.231e-3] # best guess near length=0.230
# pixel_pitches = [0.2276e-3,0.2276e-3] # best overall average

# old manual calibration
#pixel_pitches = [150e-6,150e-6]
#detector_distance = 100e-3 # mm
#print([pp / detector_distance for pp in pixel_pitches])
#exit()


# Important question: Check which side the images fill in from

data = load_raw_image_stack(data_folder + '/' + image_filename)
#data[:,30:-30,30:-30] = 0 # For HAADF
scan_points = gen_scan_grid(scan_shape,scan_steps)

det_geo = generate_detector_geometry(detector_distance, pixel_pitches)

# The first image will be something like 20% larger than the rest...
#dataset = generate_dataset(scan_points, data, det_geo, electron_energy)
dataset = generate_dataset(scan_points[1:], data[1:], det_geo, electron_energy)
dataset.inspect(units='nm')
plt.show()

dataset.to_cxi(data_folder + '/' + save_filename)



