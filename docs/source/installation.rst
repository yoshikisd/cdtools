Installation
============

Step 1: Download
----------------

The source code for CDTools is hosted on it's `MIT github page`_. Access to the MIT github page can be granted to any member of the MIT community - if you are not a member, access can be arranged through a guest account on MIT's GitHub enterprise server.

.. _`MIT github page`: https://github.mit.edu/Scattering/CDTools

It is recommended that you clone the repository, rather than just downloading the contents, as it remains under heavy development. Cloning the repository will allow you to get access to new updates.

Step 2: Install Dependencies
----------------------------

CDTools depends on the following packages:

   * `numpy <http://www.numpy.org>`_
   * `scipy <http://www.scipy.org>`_
   * `matplotlib <https://matplotlib.org>`_
   * `pytorch <https://pytorch.org>`_
   * `python-dateutil <https://github.com/dateutil/dateutil/>`_
   * `h5py <https://www.h5py.org/>`_

And has optional dependencies on

   * `pytest <https://docs.pytest.org/>`_
   * `sphinx <https://www.sphinx-doc.org/>`_
   * `sphinx-argparse <https://sphinx-argparse.readthedocs.io>`_
     
All of these can be installed via pip or conda. Finally, CDTools is written to be python 2.7+ compatible, but is only actively tested on python 3.

**It is required that pytorch is built with MKL**, as that enables FFTs. Additionally, installing pytorch **with CUDA support** is recommended, if you intend to run any serious reconstructions with the package.

Finally, the optional depencency on pytest enables the tests to be run to confirm a successful installation. Sphinx and sphinx-argparse are only required if you plan on building the documentation.


Step 3: Install
---------------

To install in CDTools in developer mode (recommended, to allow any updates to be pushed immediately through your system), run the following command from the top level directory (the directory including the setup.py file)

.. code:: bash
	  
   $ pip install -e .

If you prefer to use a tool other than pip, CDTools can be installed via any other package management tool that works with a setup.py file.

  
Step 4: Run The Tests
---------------------

To ensure that the installation has worked correctly, it is recommended, although not required, that you run the unit tests. After ensuring that pytest is installed, run the following command from the top level directory:

.. code:: bash

   $ pytest


If any tests fail, make sure that you have all the noted dependencies properly installed (specifically that pytorch is installed with MKL support). If so, `shoot me an email <alevitan@mit.edu>`_ and we'll get to the bottom of it.
