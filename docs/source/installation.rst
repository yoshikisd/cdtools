Installation
============

CDTools can be downloaded from it's `MIT github page`_, and the relevant prerequisites can be downloaded via pip, conda, or most python package managers.

.. _`MIT github page`: https://github.mit.edu/Scattering/ADCD


Access to the MIT github page can be granted to any member of the MIT community - if you are not a member, access can be arranged through a guest account on the github enterprise server

Prerequisites
-------------

CDTools has the following prerequisites:

   * `numpy <http://www.numpy.org>`_
   * `scipy <http://www.scipy.org>`_
   * `matplotlib <https://matplotlib.org>`_
   * `pytorch <https://pytorch.org>`_
   * `python-dateutil <https://github.com/dateutil/dateutil/>`_
   * `h5py <https://www.h5py.org/>`_

All of these can be installed via pip or conda. It is required that pytorch is ilt with MKL, as that enables FFTs. Additionally, installing pytorch with CUDA support is recommended for running any serious reconstructions with the package. The code is written to be python 2.7+ compatible, although it is only tested in python 3.

Finally, to run the tests, pytest is required, and to build the docs, sphinx and sphinx-argparse are required.


Installation
------------

CDTools can be installed via pip, although it is recommended to install it in development mode as changes to the code are pushed to the MIT github quite frequently.

To install in developer mode, run the following command from the top level directory (the directory including the setup.py file)

.. code:: bash
	  
   $ pip install -e .


Run The Tests
-------------

To ensure that the installation has worked correctly, it is recommended to run the unit tests. After ensuring that `pytest <https://docs.pytest.org/en/latest/>`_ is installed, run the following command from the top level directory:

.. code:: bash

   $ pytest


