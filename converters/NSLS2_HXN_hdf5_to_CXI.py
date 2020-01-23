#!/home/david/.conda/envs/CDToolsEnv/bin/python
"""
Purpose: Convert NSLSII HXN hdf5 files to CXI files for analysis with CDTools.
Author:  David Rower
Date:    December 2019
"""

import numpy as np
import pickle
import h5py
import os
import CDTools
from CDTools.tools import data as cdtdata
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

def create_cxi_from_NSLS2_HXN_2DFly(data_dir, save_str, scan_number,
                                wavelength, theta, ROI_corner_xy):
    """Converts NSLS2 HXN 2D Fly scan data (from pickle and hdf5) to CXI format

    Assumes scan files will live in data_dir with naming convention
    pickle: <data_dir>/scan_<scan_number>.pickle,
    hdf5: <data_dir>/scan_<scan_number>.hdf5,
    and will create the file <data_dir>/scan_<scan_number>.cxi.

    Parameters
    ----------
    data_dir : str
        Input data directory
    save_str : str
        Output data name
    scan_number : int
        A scan index number
    theta : float
        Rotation angle of sample in HXN convention, in degrees
    ROI_corner_xy : np.array
        1x2 array containing x, y corner of detector ROI
    """

    ## Load in pickle and hdf5 files
    scan_str = "scan_" + scan_number
    print('Scan #:', scan_number)

    # Load pickle (includes useful data about scan not in .hdf5 file)
    with open(os.path.join(data_dir, scan_str+".pickle"), 'rb') as f:
        scan_pickle = pickle.load(f)

    assert scan_pickle['plan_type'] == "FlyPlan2D", "Code only for FlyPlan2D."

    # Load hdf5 file
    scan_hdf5 = h5py.File(os.path.join(data_dir, scan_str+".h5"), 'r')


    ## Let's attempt to convert this bad boy
    print(80*'-'+'\nCreating cxi file.')


    # We save as .h5 in order to inspect with panalopy GUI utility
    scan_cxi = cdtdata.create_cxi(save_str)


    ## Add source
    cdtdata.add_source(scan_cxi, wavelength=wavelength)
    scan_cxi['entry_1/instrument_1/source_1']['name'] = scan_pickle['beamline_id']


    ## Add sample
    theta = np.radians(theta)
    sample_unit_vecs = Rotation.from_rotvec(theta * np.array([0,1,0])).as_dcm()
    orientation = np.hstack((sample_unit_vecs[:,0], sample_unit_vecs[:,1]))
    translation = np.zeros(3)
    sample_info_dict = {
        "name"        : "TaTe4",
        "orientation" : orientation,
        "translation" : translation
    }
    cdtdata.add_sample_info(scan_cxi, sample_info_dict)


    ## Add detector

    # Constant detector parameters
    detector_pixel_size = 55e-6 # meters
    detector_height_px = 515 # px ### WARNING: NEED TO CHECK THIS
    detector_width_px = 515 # px

    # Geometry parameters from scan files
    distance = scan_pickle['dist_detector'] * 1e-3 # assuming mm, almost sure
    gamma = np.radians(scan_pickle['gamma_detector'])
    delta = np.radians(scan_pickle['delta_detector'])
    Rg = Rotation.from_rotvec(-gamma * np.array([0,1,0])).as_dcm() # cw about y
    Rd = Rotation.from_rotvec(-delta * Rg[:,0]).as_dcm() # cw about rotated x
    RdRg = np.matmul(Rd, Rg)

    # Define detector basis: row vectors for y and x detector axes
    basis = detector_pixel_size * np.array([[0.,-1.,0.],[-1.,0.,0.]])
    basis = np.matmul(RdRg,basis.T).T

    # Define corner posiiton: first find center, then offset it
    corner_pos = np.dot(RdRg, distance * np.array([0.,0.,1.]))
    if ROI_corner_xy[0] is None:
        ROI_corner_xy[0] = 0.
    if ROI_corner_xy[1] is None:
        ROI_corner_xy[1] = 0.
    corner_pos -= basis[0,:] * (detector_width_px/2. - ROI_corner_xy[0])
    corner_pos -= basis[1,:] * (detector_height_px/2. - ROI_corner_xy[1])

    # Add detector data finally
    cdtdata.add_detector(scan_cxi, distance, basis.T, corner=corner_pos)


    ## Add data
    axes = ['translation'] + scan_pickle['axes'] # THIS IS ONLY FOR FLY2D
    data = np.copy(scan_hdf5['entry']['instrument']['detector']['data'])
    data[data == 0] = 1 # to prevent divide by zero in log error
    cdtdata.add_data(scan_cxi, data, axes)


    ## Add translations
    x_bounds = scan_pickle['scan_range'][0]
    y_bounds = scan_pickle['scan_range'][1]
    xx, yy = np.meshgrid(np.linspace(*x_bounds, scan_pickle['num1']),
                               np.linspace(*y_bounds, scan_pickle['num2']))
    translations = (1e-6 *
        np.stack((xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())), axis=1))
    cdtdata.add_ptycho_translations(scan_cxi, translations)


    ## Close hdf5 file
    scan_hdf5.close()
