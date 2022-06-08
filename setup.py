import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cdtools",
    version="0.2.0",
    python_requires='>3.7', # recommended minimum version for pytorch
    author="Abe Levitan",
    author_email="alevitan@mit.edu",
    description="Tools for coherent diffractive imaging and ptychography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.mit.edu/scattering/CDTools.git",
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "matplotlib>=2.0", # 2.0 has better colormaps which are used by default
        "python-dateutil",
        "torch>=1.9.0", #1.9.0 supports autograd on indexed complex tensors
        "h5py>=2.1"],
    extras_require={
        'tests': ["pytest"],
        'docs': ["sphinx","sphinx-argparse","sphinx_rtd_theme"]
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

