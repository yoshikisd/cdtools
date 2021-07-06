import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CDTools",
    version="0.1.1",
    python_requires='>3.4', # we use pathlib, introduced in 3.4
    author="Abe Levitan",
    author_email="alevitan@mit.edu",
    description="Coherent Diffraction Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.mit.edu/scattering/CDTools.git",
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "matplotlib>=2.0", # Matplotlib 2.0 introduces better colormaps and no I'm not sorry
        "python-dateutil",
        "torch>=1.9.0", #1.9.0 implements support for autograd on indexed complex tensors, which we need in order to use complex tensors in the forward models
        "h5py>=2.1"],
    extras_require={
        'tests': ["pytest"],
        'docs': ["sphinx","sphinx-argparse","sphinx_rtd_theme"]
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
