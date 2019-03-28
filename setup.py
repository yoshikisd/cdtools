import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CDTools",
    version="0.0.1",
    author="Abe Levitan, Madelyn Cain",
    author_email="alevitan@mit.edu",
    description="Coherent Diffraction Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.mit.edu/scattering/CDTools.git",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "dateutil",
        #"pytorch",
        "h5py"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
