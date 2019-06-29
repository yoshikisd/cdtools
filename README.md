# CDTools

Outline:

Description of toolbox

Note about authors and availability

Example usage

Available methods


## Installation

Installation can be done via pip. The only hiccup is that pytorch needs to be compiled with either MKL support (if you intend to do processing on the CPU), or with CUDA support (if you intend to do GPU processing).

For this reason, we recommend using Anaconda python as it handles both cases fairly seamlessly. When using Anaconda python, it is still important to manually install pytorch before installing CDTools, because the pip installer will find and install the wrong version of pytorch (which is listed on pypi).

Therefore, a simple install process can proceed as follows. First, clone the Git repository and navigate to the top-level folder. Then, run:

```console
$ conda install Pytorch -c Pytorch
$ pip install -e .
```

We recommend installing the package in developer mode as above, because the package is under active development. We also suggest installing it into it's own virtual environment as general good practice. Finally, if you would like to integrate it into a larger environment, it is a good idea to install the dependencies via conda before installing the package itself via pip.

If you manage your python installation a different way, the various dependencies that need to be installed are listed in the setup.py file.


## Running the tests

It is a good idea to run the tests after installation to ensure that everything is working, and from time to time as things change on your computer. You can run the tests by installing pytorch, and then running

```console
$ pytest
```

In the top-level directory