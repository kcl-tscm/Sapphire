# Raffy
Ridge-regression Atomistic Force Fields in PYthon



DOI : 10.5281/zenodo.5243000

Use this package to train and validate force fields for single- and multi-element materials.
The force fields are trained using ridge regression on local atomic environment descritptors computed in the "Atomic Cluster Expansion" (ACE) framework.
The package can also be used to classify local atomic environments according to their ACE local descriptor.


## Installation
To install the package, clone the repository and then pip install:

    git clone https://github.com/ClaudioZeni/Raffy
    cd Raffy
    pip install .

The installation process should take 1 to 5 minutes on a standard laptop.


## Examples
Three notebooks are available in the examples folder.
Linear Potential showcases the training and validation of a linear potential for Si.
Trajectory Clustering demonstrates how to use the Raffy package to classify local atomic environments on a sample MD trajectory of a Au nanoparticle.
Hierarchical Clustering Tutorial guides the creation of a hierachical k-means clustering to differentiate local atomic environments in an Au nanoparticle.



## Dependancies
The package uses [ASE](https://pypi.org/project/ase/) to handle .xyz files, [MIR-FLARE](https://github.com/mir-group/flare) to handle local atomic environments, [NUMPY](https://numpy.org/) and [SCIPY](https://www.scipy.org/) for fast computation, and [RAY](https://ray.io/) for multiprocessing.

The package has been tested on Ubuntu 20.04.


## References
If you use RAFFY in your research, or any part of this repository, please cite the following paper:

[1] Claudio Zeni, Kevin Rossi, Aldo Glielmo, and Stefano de Gironcoli, "Compact atomic descriptors enable accurate predictions via linear models", The Journal of Chemical Physics 154, 224112 (2021) https://doi.org/10.1063/5.0052961 


